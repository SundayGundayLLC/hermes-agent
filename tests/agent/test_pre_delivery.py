from agent.pre_delivery import (
    DEGRADED_RESPONSE,
    PreDeliveryDecisionError,
    collect_tool_telemetry,
    compact_decision_summary,
    enforce_continuation_budget,
    reduce_decisions,
    resolve_delivery_modes,
    wait_for_authority_future,
)

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
import pytest


def test_decision_reducer_is_conservative_and_stable():
    decisions = reduce_decisions([
        {"decision": "rewrite", "response": "safe rewrite"},
        {"decision": "allow"},
        {"decision": "block", "response": "blocked with proof"},
    ])

    assert decisions == {
        "decision": "block",
        "response": "blocked with proof",
    }


@pytest.mark.parametrize("value", [None, {}, {"decision": "maybe"}])
def test_malformed_registered_decision_fails_closed(value):
    with pytest.raises(PreDeliveryDecisionError):
        reduce_decisions([] if value is None else [value])


def test_none_only_observer_is_neutral_when_allowed():
    assert reduce_decisions([], allow_empty=True) == {
        "decision": "allow",
        "reason": "pre_delivery_observers_only",
    }


def test_agent_end_summary_is_compact_and_allowlisted():
    summary = compact_decision_summary(
        {
            "decision": "block",
            "reason": "proof_missing",
            "response": "sensitive candidate",
            "proof_bundle_id": "bundle-1",
            "proof_bundle_path": "proof/turn.json",
            "unexpected": "must not leak",
        },
        {
            "attempt": 1,
            "empty_count": 2,
            "tool_call_count": 3,
            "tool_result_count": 2,
            "original_message": "private request",
            "candidate_final": "private candidate",
            "tool_calls": [{"arguments": "secret"}],
            "tool_results": [{"content": "secret"}],
        },
    )

    assert summary == {
        "decision": "block",
        "reason": "proof_missing",
        "attempt": 1,
        "empty_count": 2,
        "tool_call_count": 3,
        "tool_result_count": 2,
        "proof_bundle_id": "bundle-1",
        "proof_bundle_path": "proof/turn.json",
    }


def test_authority_future_is_cancelled_on_timeout():
    future = Future()

    with pytest.raises(FutureTimeoutError):
        wait_for_authority_future(future, timeout=0.01)

    assert future.cancelled() is True


def test_second_continuation_opens_circuit_and_next_turn_resets():
    requested = {"decision": "continue", "continuation_prompt": "keep going"}

    assert enforce_continuation_budget(requested, 0)["decision"] == "continue"
    exhausted = enforce_continuation_budget(requested, 1)
    assert exhausted == {
        "decision": "block",
        "response": DEGRADED_RESPONSE,
        "reason": "pre_delivery_continuation_exhausted",
    }
    # A fresh inbound turn starts again at attempt zero (half-open/reset).
    assert enforce_continuation_budget(requested, 0)["decision"] == "continue"


def test_registered_handler_forces_buffered_assistant_delivery():
    assert resolve_delivery_modes(True, True, True) == (False, False)
    assert resolve_delivery_modes(False, True, True) == (True, True)


def test_full_tool_telemetry_is_scoped_to_current_attempt():
    history = [{"role": "user", "content": "old"}]
    messages = history + [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [{
                "id": "call-1",
                "function": {
                    "name": "terminal",
                    "arguments": '{"command":"pytest"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "terminal",
            "content": '{"exit_code":0,"stdout":"19 passed"}',
        },
    ]

    telemetry = collect_tool_telemetry(messages, history)

    assert telemetry["tool_call_count"] == 1
    assert telemetry["tool_result_count"] == 1
    assert telemetry["tool_calls"][0]["arguments"] == '{"command":"pytest"}'
    assert telemetry["tool_results"][0]["content"] == (
        '{"exit_code":0,"stdout":"19 passed"}'
    )
