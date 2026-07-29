"""
Event Hook System

A lightweight event-driven system that fires handlers at key lifecycle points.
Hooks are discovered from ~/.hermes/hooks/ directories, each containing:
  - HOOK.yaml  (metadata: name, description, events list)
  - handler.py (Python handler with async def handle(event_type, context))

Events:
  - gateway:startup     -- Gateway process starts
  - session:start       -- New session created (first message of a new session)
  - session:end         -- Session ends (user ran /new or /reset)
  - session:reset       -- Session reset completed (new session entry created)
  - agent:start         -- Agent begins processing a message
  - agent:step          -- Each turn in the tool-calling loop
  - agent:pre_delivery  -- Decision gate before turn success persistence/delivery
  - agent:end           -- Agent finishes processing
  - command:*           -- Any slash command executed (wildcard match)

Errors in advisory hooks are caught and logged but never block the main
pipeline. ``agent:pre_delivery`` is an authority hook: when declared,
configured, or registered, discovery/import/dispatch failures and missing or
malformed decisions fail closed. Observer silence belongs on ``agent:*``;
only exact ``agent:pre_delivery`` registrations activate the gate.

Context dict passed to ``agent:start`` / ``agent:end`` handlers:
  platform     -- source platform name (e.g. "telegram", "matrix", "slack")
  user_id      -- platform user id of the sender
  chat_id      -- platform chat id (group/DM identifier)
  thread_id    -- Telegram forum-topic id / thread root id (string; empty
                  when not in a thread / topic)
  chat_type    -- "dm" | "group" | "forum" (empty if unknown)
  session_id   -- Hermes session id
  message      -- inbound message text (truncated to 500 chars)

``agent:end`` adds:
  response     -- agent response text (truncated to 500 chars)

``agent:pre_delivery`` receives untruncated ``candidate_final``,
``raw_candidate_final``, and ``original_message`` plus source/message/session
ids, model/provider/API metadata, attempt/empty counters, failure state, and
the current turn's complete ``tool_calls`` / ``tool_results`` with counts.
Handlers return a mapping with ``decision`` set to ``allow``, ``rewrite``,
``continue``, or ``block``. ``rewrite``/``block`` may provide ``response``;
``continue`` may provide ``continuation_prompt``. Only one continuation is
permitted. The hook runs before trajectory/session success persistence, and
registered handlers force buffered assistant delivery so streaming cannot
bypass the decision.

Handlers posting a follow-up into the same Telegram forum-topic should
include ``message_thread_id=int(thread_id)`` when ``chat_type == "forum"``
and ``thread_id`` is non-empty.
"""

import asyncio
import importlib.util
import inspect
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import yaml

from hermes_cli.config import get_hermes_home


HOOKS_DIR = get_hermes_home() / "hooks"
PRE_DELIVERY_EVENT = "agent:pre_delivery"
PRE_DELIVERY_AUTHORITY_ENV = "HERMES_PRE_DELIVERY_AUTHORITY"


class HookAuthorityError(RuntimeError):
    """Raised when a configured decision authority cannot run safely."""


class HookRegistry:
    """
    Discovers, loads, and fires event hooks.

    Usage:
        registry = HookRegistry()
        registry.discover_and_load()
        await registry.emit("agent:start", {"platform": "telegram", ...})
    """

    def __init__(self):
        # event_type -> [handler_fn, ...]
        self._handlers: Dict[str, List[Callable]] = {}
        self._loaded_hooks: List[dict] = []  # metadata for listing
        self._authority_expected: Dict[str, set[str]] = {}
        self._authority_loaded: Dict[str, set[str]] = {}
        self._authority_failures: Dict[str, List[str]] = {}
        self._sdgd_pre_delivery_off = False

    def _expect_authority(self, event_type: str, hook_name: str) -> None:
        self._authority_expected.setdefault(event_type, set()).add(hook_name)

    def _authority_failed(
        self, event_type: str, hook_name: str, reason: str
    ) -> None:
        self._authority_failures.setdefault(event_type, []).append(
            f"{hook_name}: {reason}"
        )

    def _register_configured_authorities(self) -> None:
        """Record authorities whose absence must stop startup.

        The generic setting accepts a comma-separated list of exact hook
        names.  The SDGD compatibility setting is intentionally narrow: only
        observe/enforce requires its named authority; off preserves the
        legacy no-gate behavior.
        """
        configured = [
            name.strip()
            for name in os.getenv(PRE_DELIVERY_AUTHORITY_ENV, "").split(",")
            if name.strip()
        ]
        sdgd_mode_raw = os.getenv("SDGD_HERMES_PRE_DELIVERY_GATE_MODE")
        sdgd_mode = str(sdgd_mode_raw or "").strip().lower()
        self._sdgd_pre_delivery_off = (
            sdgd_mode_raw is not None and sdgd_mode == "off"
        )
        if self._sdgd_pre_delivery_off:
            # The explicit rollback lever wins even if a generic expectation
            # was left behind.  The hook package may remain installed so
            # observe/enforce can be restored without another file mutation.
            configured = [
                name for name in configured if name != "sdgd-pre-delivery"
            ]
        if sdgd_mode in {"observe", "enforce"}:
            configured.append("sdgd-pre-delivery")
        for hook_name in configured:
            self._expect_authority(PRE_DELIVERY_EVENT, hook_name)

    @property
    def loaded_hooks(self) -> List[dict]:
        """Return metadata about all loaded hooks."""
        return list(self._loaded_hooks)

    def _register_builtin_hooks(self) -> None:
        """Register built-in hooks that are always active.

        Currently empty — no shipped built-in hooks. Kept as the extension
        point for future always-on gateway hooks so they drop in without
        re-plumbing discover_and_load().
        """
        return

    def discover_and_load(self) -> None:
        """
        Scan the hooks directory for hook directories and load their handlers.

        Also registers built-in hooks that are always active.

        Each hook directory must contain:
          - HOOK.yaml with at least 'name' and 'events' keys
          - handler.py with a top-level 'handle' function (sync or async)
        """
        self._register_builtin_hooks()
        self._register_configured_authorities()

        if not HOOKS_DIR.exists():
            self.assert_expected_authorities_healthy()
            return

        for hook_dir in sorted(HOOKS_DIR.iterdir()):
            if not hook_dir.is_dir():
                continue

            manifest_path = hook_dir / "HOOK.yaml"
            handler_path = hook_dir / "handler.py"
            hook_name = hook_dir.name

            if not manifest_path.exists():
                continue

            try:
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if not manifest or not isinstance(manifest, dict):
                    print(f"[hooks] Skipping {hook_dir.name}: invalid HOOK.yaml", flush=True)
                    continue

                hook_name = manifest.get("name", hook_dir.name)
                events = manifest.get("events", [])
                if not events:
                    print(f"[hooks] Skipping {hook_name}: no events declared", flush=True)
                    continue

                sdgd_authority_bypassed = (
                    self._sdgd_pre_delivery_off
                    and hook_name == "sdgd-pre-delivery"
                )
                is_pre_delivery_authority = (
                    PRE_DELIVERY_EVENT in events
                    and not sdgd_authority_bypassed
                )
                if is_pre_delivery_authority:
                    self._expect_authority(PRE_DELIVERY_EVENT, hook_name)
                if not handler_path.exists():
                    if is_pre_delivery_authority:
                        self._authority_failed(
                            PRE_DELIVERY_EVENT, hook_name, "handler.py is missing"
                        )
                    continue

                # Dynamically load the handler module.
                # Register in sys.modules BEFORE exec_module so Pydantic /
                # dataclasses / typing introspection can resolve forward
                # references (triggered by `from __future__ import annotations`
                # in the handler). Without this, a handler that declares a
                # Pydantic BaseModel for webhook/event payloads fails at first
                # dispatch with "TypeAdapter ... is not fully defined".
                module_name = f"hermes_hook_{hook_name}"
                spec = importlib.util.spec_from_file_location(
                    module_name, handler_path
                )
                if spec is None or spec.loader is None:
                    print(f"[hooks] Skipping {hook_name}: could not load handler.py", flush=True)
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(module_name, None)
                    raise

                handle_fn = getattr(module, "handle", None)
                if handle_fn is None:
                    print(f"[hooks] Skipping {hook_name}: no 'handle' function found", flush=True)
                    if is_pre_delivery_authority:
                        self._authority_failed(
                            PRE_DELIVERY_EVENT, hook_name, "handle is missing"
                        )
                    continue
                if not callable(handle_fn):
                    print(f"[hooks] Skipping {hook_name}: 'handle' is not callable", flush=True)
                    if is_pre_delivery_authority:
                        self._authority_failed(
                            PRE_DELIVERY_EVENT, hook_name, "handle is not callable"
                        )
                    continue

                # Register the handler for each declared event
                for event in events:
                    if (
                        event == PRE_DELIVERY_EVENT
                        and sdgd_authority_bypassed
                    ):
                        continue
                    self._handlers.setdefault(event, []).append(handle_fn)
                if is_pre_delivery_authority:
                    self._authority_loaded.setdefault(
                        PRE_DELIVERY_EVENT, set()
                    ).add(hook_name)

                self._loaded_hooks.append({
                    "name": hook_name,
                    "description": manifest.get("description", ""),
                    "events": events,
                    "path": str(hook_dir),
                })

                print(f"[hooks] Loaded hook '{hook_name}' for events: {events}", flush=True)

            except Exception as e:
                print(f"[hooks] Error loading hook {hook_dir.name}: {e}", flush=True)
                # A valid manifest may have established the exact authority
                # expectation before import failed.  Preserve that failure so
                # startup cannot silently continue without the gate.
                expected_names = self._authority_expected.get(
                    PRE_DELIVERY_EVENT, set()
                )
                matching = str(locals().get("hook_name") or hook_dir.name)
                if matching in expected_names:
                    self._authority_failed(
                        PRE_DELIVERY_EVENT,
                        matching,
                        f"load failed: {type(e).__name__}: {e}",
                    )

        self.assert_expected_authorities_healthy()

    def authority_expected(self, event_type: str) -> bool:
        """Return whether an exact decision authority is required."""
        return bool(
            self._authority_expected.get(event_type)
            or self._resolve_handlers(event_type, include_wildcards=False)
        )

    def assert_authority_healthy(self, event_type: str) -> None:
        """Refuse an expected authority that is absent or failed discovery."""
        expected = self._authority_expected.get(event_type, set())
        loaded = self._authority_loaded.get(event_type, set())
        failures = list(self._authority_failures.get(event_type, []))
        missing = sorted(expected - loaded)
        if missing:
            failures.append("missing configured authority: " + ", ".join(missing))
        if expected and not self._resolve_handlers(
            event_type, include_wildcards=False
        ):
            failures.append("no exact authority handler registered")
        if failures:
            raise HookAuthorityError(
                f"{event_type} authority unavailable: " + "; ".join(failures)
            )

    def assert_expected_authorities_healthy(self) -> None:
        for event_type in sorted(self._authority_expected):
            self.assert_authority_healthy(event_type)

    async def emit_authority(
        self, event_type: str, context: Optional[Dict[str, Any]] = None,
        *, run_sync_in_executor: bool = False,
    ) -> List[Any]:
        """Dispatch an exact authority without advisory fail-open semantics."""
        self.assert_authority_healthy(event_type)
        handlers = self._resolve_handlers(event_type, include_wildcards=False)
        if not handlers:
            raise HookAuthorityError(
                f"{event_type} authority dispatch found no exact handlers"
            )
        context = {} if context is None else context
        results: List[Any] = []
        for fn in handlers:
            if run_sync_in_executor and not inspect.iscoroutinefunction(fn):
                result = await asyncio.to_thread(fn, event_type, context)
            else:
                result = fn(event_type, context)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                raise HookAuthorityError(
                    f"{event_type} authority handler returned no decision"
                )
            results.append(result)
        if not results:
            raise HookAuthorityError(
                f"{event_type} authority dispatch returned no decisions"
            )
        return results

    def _resolve_handlers(
        self, event_type: str, *, include_wildcards: bool = True
    ) -> List[Callable]:
        """Return all handlers that should fire for ``event_type``.

        Exact matches fire first, followed by wildcard matches (e.g.
        ``command:*`` matches ``command:reset``).
        """
        handlers = list(self._handlers.get(event_type, []))
        if include_wildcards and ":" in event_type:
            base = event_type.split(":")[0]
            wildcard_key = f"{base}:*"
            handlers.extend(self._handlers.get(wildcard_key, []))
        return handlers

    def has_handlers(self, event_type: str) -> bool:
        """Return whether an event has any exact or wildcard handlers."""
        return bool(self._resolve_handlers(event_type))

    def has_exact_handlers(self, event_type: str) -> bool:
        """Return whether an event has explicitly registered handlers."""
        return bool(self._resolve_handlers(event_type, include_wildcards=False))

    async def emit(self, event_type: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Fire all handlers registered for an event, discarding return values.

        Supports wildcard matching: handlers registered for "command:*" will
        fire for any "command:..." event. Handlers registered for a base type
        like "agent" won't fire for "agent:start" -- only exact matches and
        explicit wildcards.

        Args:
            event_type: The event identifier (e.g. "agent:start").
            context:    Optional dict with event-specific data.
        """
        if context is None:
            context = {}

        for fn in self._resolve_handlers(event_type):
            try:
                result = fn(event_type, context)
                # Support both sync and async handlers
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"[hooks] Error in handler for '{event_type}': {e}", flush=True)

    async def emit_collect(
        self,
        event_type: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        raise_exceptions: bool = False,
        exact_only: bool = False,
        run_sync_in_executor: bool = False,
    ) -> List[Any]:
        """Fire handlers and return their non-None return values in order.

        Like :meth:`emit` but captures each handler's return value. Used for
        decision-style hooks (e.g. ``command:<name>`` policies that want to
        allow/deny/rewrite the command before normal dispatch).

        Exceptions from individual handlers are logged but do not abort the
        remaining handlers.
        """
        if context is None:
            context = {}

        results: List[Any] = []
        for fn in self._resolve_handlers(
            event_type, include_wildcards=not exact_only
        ):
            try:
                if run_sync_in_executor and not inspect.iscoroutinefunction(fn):
                    result = await asyncio.to_thread(fn, event_type, context)
                else:
                    result = fn(event_type, context)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    results.append(result)
            except Exception as e:
                if raise_exceptions:
                    raise
                print(f"[hooks] Error in handler for '{event_type}': {e}", flush=True)
        return results
