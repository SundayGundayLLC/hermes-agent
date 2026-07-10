from types import SimpleNamespace

import pytest

from agent.pre_delivery import DEGRADED_RESPONSE
from agent.turn_finalizer import finalize_turn


class FakeAgent:
    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.api_mode = "chat_completions"
        self.session_id = "sess-test"
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
        self._response_was_previewed = True
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self.pre_delivery_callback = None
        self.cleanup_calls = 0
        self.trajectory_calls = 0

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        self.trajectory_calls += 1

    def _cleanup_task_resources(self, *_args, **_kwargs):
        self.cleanup_calls += 1

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

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


def test_final_response_closes_tool_tail_before_persistence(monkeypatch):
    """A recovered/previewed final response must be durable in session history.

    Regression for turns where the caller receives a non-empty final_response,
    but the message transcript still ends at a tool result. If persisted that
    way, the next turn reloads a stale/malformed history and can appear to loop
    because the assistant's visible final answer is missing from durable state.
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "I'll check.",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "terminal", "content": "ok"},
    ]

    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content",
    )

    assert result["messages"][-1] == {"role": "assistant", "content": "Done."}
    assert agent.persisted_messages is not None
    assert agent.persisted_messages[-1] == {"role": "assistant", "content": "Done."}


def _finalize_with_gate(agent, messages, response="Candidate complete."):
    return finalize_turn(
        agent,
        final_response=response,
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn-1432",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="text_response(stop)",
    )


def test_answer_only_allow_persists_after_decision(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    captured = []
    agent.pre_delivery_callback = lambda context: (
        captured.append(context) or {"decision": "allow"}
    )
    messages = [
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": "It is noon."},
    ]

    result = _finalize_with_gate(agent, messages, "It is noon.")

    assert result["final_response"] == "It is noon."
    assert captured[0]["tool_call_count"] == 0
    assert captured[0]["candidate_final"] == "It is noon."
    assert agent.persisted_messages[-1]["content"] == "It is noon."


def test_allowed_recovery_appends_after_tool_tail_instead_of_rewriting_call(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    agent.pre_delivery_callback = lambda _context: {"decision": "allow"}
    tool_call = {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [{
            "id": "call-1",
            "function": {"name": "terminal", "arguments": "{}"},
        }],
    }
    messages = [
        {"role": "user", "content": "check"},
        tool_call,
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ]

    result = _finalize_with_gate(agent, messages, "Recovered final.")

    assert tool_call["content"] == "Checking."
    assert result["messages"][-1] == {
        "role": "assistant",
        "content": "Recovered final.",
    }


@pytest.mark.parametrize(
    ("candidate", "decision"),
    [
        (
            "I'll run the checks now.",
            {"decision": "continue", "continuation_prompt": "Run them now."},
        ),
        (
            "Proof: everything passed.",
            {"decision": "continue", "continuation_prompt": "Get real proof."},
        ),
    ],
)
def test_unproven_candidate_continues_without_terminal_persistence(
    monkeypatch, candidate, decision
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    agent.pre_delivery_callback = lambda _context: decision
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": candidate},
    ]

    result = _finalize_with_gate(agent, messages, candidate)

    assert result["pre_delivery_continue"] is True
    assert result["continuation_prompt"] == decision["continuation_prompt"]
    assert agent.persisted_messages is None
    assert agent.trajectory_calls == 0
    assert agent.cleanup_calls == 0
    assert result["messages"][-1]["_pre_delivery_rejected"] is True


def test_valid_blocked_response_can_be_allowed_without_tools(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    agent.pre_delivery_callback = lambda context: (
        {"decision": "block", "response": context["candidate_final"]}
        if "BLOCKED" in context["candidate_final"]
        else {"decision": "block"}
    )
    blocked = "BLOCKED — owner: Marcus; action: authorize login; proof: no token."
    messages = [
        {"role": "user", "content": "publish it"},
        {"role": "assistant", "content": blocked},
    ]

    result = _finalize_with_gate(agent, messages, blocked)

    assert result["final_response"] == blocked
    assert result["completed"] is False
    assert result["pre_delivery_context"]["tool_call_count"] == 0
    assert agent.persisted_messages[-1]["content"] == blocked


def test_rewrite_and_block_are_atomic_with_persisted_assistant(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    for decision, expected in (
        ({"decision": "rewrite", "response": "Verified rewrite."}, "Verified rewrite."),
        ({"decision": "block", "response": "Deterministic degraded."}, "Deterministic degraded."),
    ):
        agent = FakeAgent()
        agent.pre_delivery_callback = lambda _context, decision=decision: decision
        messages = [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "Unverified."},
        ]

        result = _finalize_with_gate(agent, messages, "Unverified.")

        assert result["final_response"] == expected
        assert agent.persisted_messages[-1]["content"] == expected


def test_empty_after_tool_keeps_full_telemetry_and_degrades_atomically(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    captured = []
    agent.pre_delivery_callback = lambda context: (
        captured.append(context)
        or {"decision": "block", "response": "Safe degraded result."}
    )
    messages = [
        {"role": "user", "content": "check it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "terminal", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "terminal",
            "content": "exit=0",
        },
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_terminal_sentinel": True,
        },
    ]

    result = _finalize_with_gate(agent, messages, "(empty)")

    assert captured[0]["is_empty"] is True
    assert captured[0]["tool_call_count"] == 1
    assert captured[0]["tool_result_count"] == 1
    assert result["final_response"] == "Safe degraded result."
    assert agent.persisted_messages[-1]["content"] == "Safe degraded result."
    assert "_empty_terminal_sentinel" not in agent.persisted_messages[-1]


def test_first_empty_candidate_gets_one_unpersisted_recovery(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    captured = []
    agent.pre_delivery_callback = lambda context: (
        captured.append(context)
        or {"decision": "continue", "continuation_prompt": "Recover once."}
    )
    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_terminal_sentinel": True,
        },
    ]

    result = _finalize_with_gate(agent, messages, "(empty)")

    assert captured[0]["is_empty"] is True
    assert result["pre_delivery_continue"] is True
    assert result["continuation_prompt"] == "Recover once."
    assert agent.persisted_messages is None
    assert all(
        not message.get("_empty_terminal_sentinel")
        for message in result["messages"]
    )
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["_pre_delivery_rejected"] is True


def test_callback_error_fails_closed_before_persistence(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()

    def _raise(_context):
        raise RuntimeError("proof validator unavailable")

    agent.pre_delivery_callback = _raise
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "Done."},
    ]

    result = _finalize_with_gate(agent, messages, "Done.")

    assert result["final_response"] == DEGRADED_RESPONSE
    assert result["pre_delivery_decision"]["decision"] == "block"
    assert agent.persisted_messages[-1]["content"] == DEGRADED_RESPONSE
