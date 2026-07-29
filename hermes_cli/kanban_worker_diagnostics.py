"""Fail-closed diagnostics for dispatcher-spawned Kanban workers.

The worker process already knows the exact provider/tool failure before it
exits.  Record that structured diagnosis while the claim is still live so a
platform-specific process reaper cannot replace it with a generic dead-PID
message.
"""

from __future__ import annotations

import os
import re
import json
import tempfile
from pathlib import Path
from typing import Any, Optional


_MODEL_MISMATCH_RE = re.compile(
    r"(?:model[^\n]{0,120}(?:not found|unsupported|not supported|invalid)|"
    r"unknown model|invalid model)",
    re.IGNORECASE,
)
_RETRY_AFTER_RE = re.compile(r"\bretry\s+after\s+(\d{1,9})\s*(?:s|sec(?:ond)?s?)?\b", re.IGNORECASE)
_AUTH_INVALID_RE = re.compile(
    r"\b(?:invalid|expired|revoked)\s+(?:credentials?|token|api[ _-]?key)|"
    r"\b(?:authentication|auth(?:orization)?)\s+(?:failed|required)\b",
    re.IGNORECASE,
)


def provider_failure_snapshot(
    error: BaseException,
    *,
    provider: str,
    model: str,
    reason: str,
    status_code: Any,
    fallback_stage: int,
) -> dict[str, Any]:
    """Return a redacted, structured provider-attempt failure record."""
    raw = str(error or "").lower()
    structured_quota_values: list[str] = []

    def _collect_quota_fields(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = str(key).replace("_", "").lower()
                if normalized_key in {"quotametric", "quotaid"}:
                    structured_quota_values.append(str(item).lower())
                _collect_quota_fields(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect_quota_fields(item)

    for attr in ("body", "details", "error_details"):
        _collect_quota_fields(getattr(error, attr, None))
    response = getattr(error, "response", None)
    if response is not None:
        try:
            _collect_quota_fields(response.json())
        except Exception:
            pass
    quota_text = " ".join(structured_quota_values)
    quota_marker = None
    if (
        re.search(r"paid[_ -]?tier[^\n]{0,160}input[_ -]?token", raw, re.I)
        or all(token in quota_text for token in ("paid", "tier", "input", "token"))
    ):
        quota_marker = "paid_tier_input_token"
    classification = str(reason or "provider_error").strip().lower()
    if _MODEL_MISMATCH_RE.search(raw):
        classification = "model_mismatch"
    return {
        "provider": (provider or "unknown").strip().lower(),
        "model": (model or "unknown").strip(),
        "http_status": status_code if isinstance(status_code, int) else None,
        "classification": classification,
        "quota_marker": quota_marker,
        "fallback_stage": max(0, int(fallback_stage or 0)),
    }


def kanban_provider_retry_ceiling(
    configured_retries: int,
    *,
    is_provider_capacity_failure: bool,
    has_pending_fallback: bool,
) -> int:
    """Bound a Kanban worker once every configured provider is exhausted.

    A fallback still gets one first attempt.  Once no fallback remains, a
    quota/billing failure stops after the current request instead of
    re-sending an 80k-100k-token context through the same exhausted backend.
    Non-Kanban callers do not call this helper and retain normal retry policy.
    """
    try:
        retries = max(1, int(configured_retries))
    except (TypeError, ValueError):
        retries = 1
    if is_provider_capacity_failure and not has_pending_fallback:
        return 1
    return retries


def is_known_disabled_tool(
    requested_tool: str,
    available_tools: Any,
    *,
    registered_tools: Optional[Any] = None,
) -> bool:
    """Distinguish a disabled capability from an arbitrary model typo."""
    name = str(requested_tool or "").strip()
    if not name or name in set(available_tools or []):
        return False
    if registered_tools is None:
        from tools.registry import registry

        registered_tools = registry.get_all_tool_names()
    return name in set(registered_tools or [])


def should_continue_kanban_goal_loop(result: Any) -> bool:
    """A terminally failed attempt must never continue in goal mode."""
    return not (isinstance(result, dict) and bool(result.get("failed")))


def classify_primary_provider_failure(
    exc: BaseException,
    *,
    provider: str,
    fallback_provider: str,
    fallback_model: str,
) -> dict[str, str]:
    """Return a redacted primary-resolution diagnosis for later route-back."""
    primary = (provider or "unknown").strip().lower()
    message = str(exc or "")
    classification = "primary_provider_auth_failed"
    try:
        from hermes_cli.auth import is_rate_limited_auth_error

        if is_rate_limited_auth_error(exc):
            classification = (
                "primary_codex_quota_exhausted"
                if primary == "openai-codex"
                else "primary_provider_quota_exhausted"
            )
    except Exception:
        pass
    if (
        classification == "primary_provider_auth_failed"
        and re.search(r"(?:quota|rate[ _-]?limit|usage limit|429)", message, re.I)
    ):
        classification = (
            "primary_codex_quota_exhausted"
            if primary == "openai-codex"
            else "primary_provider_quota_exhausted"
        )
    return {
        "provider": primary,
        "classification": classification,
        "fallback_provider": (fallback_provider or "").strip().lower(),
        "fallback_model": (fallback_model or "").strip(),
    }


def classify_worker_failure(
    result: dict[str, Any],
    *,
    provider: str,
    model: str,
    profile: str = "",
    primary_failure: Optional[dict[str, str]] = None,
) -> Optional[dict[str, Any]]:
    """Classify a terminal worker result without copying raw provider bodies."""
    if not isinstance(result, dict) or not result.get("failed"):
        return None

    reason = str(result.get("failure_reason") or "").strip().lower()
    current_provider = str(result.get("failure_provider") or provider or "unknown").strip().lower()
    current_model = str(result.get("failure_model") or model or "unknown").strip()
    status = result.get("failure_status_code")
    error = str(result.get("error") or result.get("final_response") or "")
    lowered = error.lower()
    retry_match = _RETRY_AFTER_RE.search(error)
    retry_after_seconds = int(retry_match.group(1)) if retry_match else None

    diagnostic: dict[str, Any] = {
        "profile": (profile or "unknown").strip().lower(),
        "provider": current_provider,
        "model": current_model,
        "http_status": int(status) if isinstance(status, int) else None,
        "primary_failure": dict(primary_failure) if primary_failure else None,
        "provider_failures": list(result.get("provider_failures") or []),
    }
    if retry_after_seconds is not None:
        # This is provider-supplied scheduling metadata, not the provider body.
        diagnostic["retry_after_seconds"] = retry_after_seconds
    if not diagnostic["primary_failure"] and len(diagnostic["provider_failures"]) > 1:
        diagnostic["primary_failure"] = dict(diagnostic["provider_failures"][0])

    if reason == "toolset_mismatch":
        requested = str(result.get("requested_tool") or "unknown")[:120]
        available = sorted(
            str(item)[:120]
            for item in (result.get("available_tools") or [])
            if str(item).strip()
        )
        diagnostic.update(
            category="toolset_mismatch",
            disposition="block_capability",
            requested_tool=requested,
            available_tools=available,
        )
        return diagnostic

    if _MODEL_MISMATCH_RE.search(error):
        diagnostic.update(
            category="fallback_model_mismatch",
            disposition="block_capability",
        )
        return diagnostic

    quota_marker = str(result.get("failure_quota_marker") or "").lower()

    if reason == "billing":
        diagnostic.update(
            category="owner_billing_gate",
            disposition="block_needs_input",
        )
        return diagnostic

    # Credential-invalid signals take precedence over a provider's HTTP
    # status.  Some auth front doors use 429 as a defensive response; never
    # turn an explicit invalid-credential message into a harmless cooldown.
    if reason in {"auth", "authentication"} or _AUTH_INVALID_RE.search(error):
        diagnostic.update(
            category="provider_auth_failed",
            disposition="block_needs_input",
            credentials_state="invalid",
        )
        return diagnostic

    if current_provider == "gemini" and (
        quota_marker == "paid_tier_input_token" or
        ("paid_tier" in lowered and "input_token" in lowered)
    ):
        diagnostic.update(
            category="gemini_paid_tier_input_token_quota",
            disposition="release_with_cooldown",
        )
        return diagnostic

    if current_provider == "gemini" and (
        reason in {"rate_limit", "billing"}
        or "resource_exhausted" in lowered
        or status == 429
    ):
        diagnostic.update(
            category="gemini_request_rate_limited",
            disposition="release_with_cooldown",
        )
        return diagnostic

    if current_provider == "openai-codex" and (
        reason in {"rate_limit", "upstream_rate_limit"}
        or status == 429
        or "codex provider quota exhausted" in lowered
        or "usage limit" in lowered
    ):
        diagnostic.update(
            category="codex_provider_capacity",
            disposition="release_with_cooldown",
            credentials_state="valid",
        )
        return diagnostic

    if reason in {"rate_limit", "upstream_rate_limit"} or status == 429:
        diagnostic.update(
            category="provider_rate_limited",
            disposition="release_with_cooldown",
            credentials_state="unknown",
        )
        return diagnostic

    return None


def _format_reason(diagnostic: dict[str, Any]) -> str:
    parts = [
        str(diagnostic.get("category") or "worker_failure"),
        f"profile={diagnostic.get('profile') or 'unknown'}",
        f"provider={diagnostic.get('provider') or 'unknown'}",
        f"model={diagnostic.get('model') or 'unknown'}",
    ]
    if diagnostic.get("http_status") is not None:
        parts.append(f"http={diagnostic['http_status']}")
    if diagnostic.get("credentials_state"):
        parts.append(f"credentials={diagnostic['credentials_state']}")
    if diagnostic.get("retry_after_seconds") is not None:
        parts.append(f"retry_after_seconds={diagnostic['retry_after_seconds']}")
    if diagnostic.get("requested_tool"):
        parts.append(f"requested_tool={diagnostic['requested_tool']}")
    primary = diagnostic.get("primary_failure")
    if isinstance(primary, dict) and primary.get("classification"):
        parts.append(f"primary={primary['classification']}")
    parts.append("resume=provider_or_toolset_proof_changed")
    return "; ".join(parts)[:1000]


def record_kanban_worker_failure(cli: Any, result: dict[str, Any]) -> bool:
    """Persist an exact terminal diagnosis before the worker process exits."""
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task_id:
        return False
    profile = (os.environ.get("HERMES_PROFILE") or "").strip()
    diagnostic = classify_worker_failure(
        result,
        provider=str(getattr(cli, "provider", "") or ""),
        model=str(getattr(cli, "model", "") or ""),
        profile=profile,
        primary_failure=getattr(cli, "_kanban_primary_provider_failure", None),
    )
    if diagnostic is None:
        return False

    from hermes_cli import kanban_db as kb

    expected_run_id = os.environ.get("HERMES_KANBAN_RUN_ID")
    claim_lock = (os.environ.get("HERMES_KANBAN_CLAIM_LOCK") or "").strip()
    try:
        run_id = int(expected_run_id) if expected_run_id else None
    except ValueError:
        run_id = None
    reason = _format_reason(diagnostic)
    if run_id is None or not claim_lock:
        return False
    worker_pid = os.getpid()
    conn = None
    payload = {
        "task_id": task_id,
        "run_id": run_id,
        "claim_lock": claim_lock,
        "worker_pid": worker_pid,
        "reason": reason,
        "diagnostic": diagnostic,
    }
    try:
        conn = kb.connect()
        disposition = diagnostic["disposition"]
        if disposition == "release_with_cooldown":
            handled = kb.release_rate_limited_worker(
                conn,
                task_id,
                reason=reason,
                metadata=diagnostic,
                expected_run_id=run_id,
                expected_claim_lock=claim_lock,
                expected_worker_pid=worker_pid,
            )
        else:
            block_kind = "needs_input" if disposition == "block_needs_input" else "capability"
            handled = kb.block_worker_failure(
                conn,
                task_id,
                reason=reason,
                kind=block_kind,
                metadata=diagnostic,
                expected_run_id=run_id,
                expected_claim_lock=claim_lock,
                expected_worker_pid=worker_pid,
            )
        if not handled:
            _write_diagnostic_sidecar(payload)
        return handled
    except Exception:
        _write_diagnostic_sidecar(payload)
        raise
    finally:
        if conn is not None:
            conn.close()


def _write_diagnostic_sidecar(payload: dict[str, Any]) -> bool:
    """Atomically retain a worker diagnosis if the board write is unavailable."""
    raw_path = (os.environ.get("HERMES_KANBAN_DIAGNOSTIC_PATH") or "").strip()
    if not raw_path:
        return False
    target = Path(raw_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        return False
    return True
