"""Executable integration coverage for the pre-delivery authority rail."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from agent.pre_delivery import REJECTED_CANDIDATE_PLACEHOLDER
from agent.turn_finalizer import finalize_turn
from gateway.config import GatewayConfig, Platform
from gateway.hooks import HookRegistry
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


class _FinalizerAgent:
    """Minimal concrete finalizer host; persistence remains fully executable."""

    persist_log = []

    def __init__(self, **kwargs):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = kwargs.get("model", "integration-model")
        self.provider = kwargs.get("provider", "integration-provider")
        self.base_url = kwargs.get("base_url", "")
        self.api_mode = kwargs.get("api_mode", "chat_completions")
        self.session_id = kwargs.get("session_id", "sess-authority")
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.tools = []
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.step_callback = None
        self.stream_delta_callback = None
        self.interim_assistant_callback = None
        self.pre_delivery_callback = None
        self.status_callback = None
        self.notice_callback = None
        self.notice_clear_callback = None
        self.event_callback = None
        self.reasoning_config = None
        self.service_tier = None
        self.request_overrides = {}

    def run_conversation(
        self, message, conversation_history=None, task_id=None, **_kwargs
    ):
        history = list(conversation_history or [])
        candidate = (
            "unsafe first completion"
            if message == "perform the original task"
            else "safe completion"
        )
        messages = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": candidate},
        ]
        return finalize_turn(
            self,
            final_response=candidate,
            api_call_count=1,
            interrupted=False,
            failed=False,
            messages=messages,
            conversation_history=history,
            effective_task_id=task_id or "task-1432",
            turn_id=f"turn-{len(self.persist_log)}",
            user_message=message,
            original_user_message=message,
            _should_review_memory=False,
            _turn_exit_reason="text_response(stop)",
        )

    def _handle_max_iterations(self, *_args):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, _messages):
        pass

    def _persist_session(self, messages, _conversation_history):
        self.persist_log.append([dict(message) for message in messages])
        return True

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
        thread_id="77",
    )


def _event():
    return MessageEvent(
        text="perform the original task",
        source=_source(),
        message_id="msg-original-42",
    )


def _runner(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner._agent_cache = None
    runner._agent_cache_lock = None

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-authority",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    return runner


@pytest.mark.asyncio
async def test_real_authority_registry_and_gateway_continuation_are_atomic(
    monkeypatch, tmp_path
):
    """The first candidate is suppressed and identity survives recovery."""
    hooks_dir = tmp_path / "hooks"
    hook_dir = hooks_dir / "integration-gate"
    hook_dir.mkdir(parents=True)
    (hook_dir / "HOOK.yaml").write_text(
        "name: integration-gate\nevents: [agent:pre_delivery]\n"
    )
    (hook_dir / "handler.py").write_text(
        "SEEN = []\n"
        "def handle(event_type, context):\n"
        "    SEEN.append(dict(context))\n"
        "    if context['attempt'] == 0:\n"
        "        return {'decision': 'continue', 'continuation_prompt': 'recover safely'}\n"
        "    return {'decision': 'allow'}\n"
    )
    monkeypatch.setattr("gateway.hooks.HOOKS_DIR", hooks_dir)

    runner = _runner(monkeypatch, tmp_path)
    runner.hooks = HookRegistry()
    runner.hooks.discover_and_load()
    _FinalizerAgent.persist_log = []
    monkeypatch.setattr("run_agent.AIAgent", _FinalizerAgent)
    event = _event()
    source = event.source

    response = await runner._handle_message_with_agent(
        event, source, "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == "safe completion"

    handler = runner.hooks._handlers["agent:pre_delivery"][0]
    seen = handler.__globals__["SEEN"]
    assert [item["attempt"] for item in seen] == [0, 1]
    assert {item["message_id"] for item in seen} == {"msg-original-42"}
    assert {item["source_id"] for item in seen} == {
        "telegram:-1001:77:msg-original-42"
    }
    assert {item["original_message"] for item in seen} == {
        "perform the original task"
    }

    persisted_repr = repr(_FinalizerAgent.persist_log)
    assert "unsafe first completion" not in persisted_repr
    assert REJECTED_CANDIDATE_PLACEHOLDER in persisted_repr
    appended = [
        call.args[1]
        for call in runner.session_store.append_to_transcript.call_args_list
        if len(call.args) > 1
    ]
    assert "unsafe first completion" not in repr(appended)
    assert any(message.get("content") == "safe completion" for message in appended)
