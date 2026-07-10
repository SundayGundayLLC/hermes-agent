"""Controller-neutral pre-delivery decision helpers.

The gateway owns hook discovery and async dispatch.  The agent owns the
atomic turn-finalization seam: a rejected candidate must not be recorded as a
completed turn before the hook decides whether to allow, rewrite, continue,
or block it.
"""

from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, Iterable, List, Mapping


PRE_DELIVERY_EVENT = "agent:pre_delivery"
MAX_PRE_DELIVERY_CONTINUATIONS = 1
DEFAULT_CONTINUATION_PROMPT = (
    "[System: The previous candidate was not safe to deliver. Continue the "
    "same task now, use tools when action is required, and return either "
    "verifiable proof or a precise BLOCKED result.]"
)
DEGRADED_RESPONSE = (
    "⚠️ I could not safely verify completion for this turn after one recovery "
    "attempt. No unverified completion claim was delivered. Please retry or "
    "inspect the recorded turn proof."
)
REJECTED_CANDIDATE_PLACEHOLDER = (
    "Candidate withheld by pre-delivery policy; recovery continuation required."
)
PRE_DELIVERY_HOOK_TIMEOUT_SECONDS = 30.0

_VALID_DECISIONS = {"allow", "rewrite", "continue", "block"}
_DECISION_PRIORITY = {"allow": 0, "rewrite": 1, "continue": 2, "block": 3}


class PreDeliveryDecisionError(ValueError):
    """Raised when a registered pre-delivery handler returns bad data."""


def normalize_decision(value: Any) -> Dict[str, Any]:
    """Validate one hook decision without discarding handler metadata."""
    if not isinstance(value, Mapping):
        raise PreDeliveryDecisionError("pre-delivery decision must be a mapping")
    decision = str(value.get("decision") or "").strip().lower()
    if decision not in _VALID_DECISIONS:
        raise PreDeliveryDecisionError(
            "pre-delivery decision must be allow, rewrite, continue, or block"
        )
    result = dict(value)
    result["decision"] = decision
    if decision == "rewrite":
        response = result.get("response")
        if not isinstance(response, str) or not response.strip():
            raise PreDeliveryDecisionError("rewrite requires a non-empty response")
    if decision == "continue":
        prompt = result.get("continuation_prompt", DEFAULT_CONTINUATION_PROMPT)
        if not isinstance(prompt, str) or not prompt.strip():
            raise PreDeliveryDecisionError(
                "continue requires a non-empty continuation_prompt"
            )
        result["continuation_prompt"] = prompt
    if decision == "block":
        response = result.get("response", DEGRADED_RESPONSE)
        if not isinstance(response, str) or not response.strip():
            response = DEGRADED_RESPONSE
        result["response"] = response
    return result


def reduce_decisions(
    values: Iterable[Any], *, allow_empty: bool = False
) -> Dict[str, Any]:
    """Reduce multiple hook results deterministically and conservatively."""
    normalized = [normalize_decision(value) for value in values]
    if not normalized:
        if allow_empty:
            return {
                "decision": "allow",
                "reason": "pre_delivery_observers_only",
            }
        raise PreDeliveryDecisionError(
            "registered pre-delivery handlers returned no decision"
        )
    return max(
        enumerate(normalized),
        key=lambda item: (_DECISION_PRIORITY[item[1]["decision"]], -item[0]),
    )[1]


def compact_decision_summary(
    decision: Mapping[str, Any], context: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the allowlisted, non-sensitive payload exposed to ``agent:end``."""
    summary = {
        "decision": decision.get("decision"),
        "reason": decision.get("reason"),
        "attempt": context.get("attempt"),
        "empty_count": context.get("empty_count", 0),
        "tool_call_count": context.get("tool_call_count", 0),
        "tool_result_count": context.get("tool_result_count", 0),
    }
    for key in ("proof_bundle_id", "proof_bundle_path"):
        value = decision.get(key) or context.get(key)
        if value:
            summary[key] = value
    return summary


def wait_for_authority_future(future: Any, timeout: float) -> Any:
    """Wait for an async authority decision and cancel it on timeout."""
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        future.cancel()
        raise


def enforce_continuation_budget(
    decision: Mapping[str, Any], attempt: int
) -> Dict[str, Any]:
    """Open the per-turn circuit after the single recovery continuation."""
    normalized = normalize_decision(decision)
    if (
        normalized["decision"] == "continue"
        and attempt >= MAX_PRE_DELIVERY_CONTINUATIONS
    ):
        return {
            "decision": "block",
            "response": DEGRADED_RESPONSE,
            "reason": "pre_delivery_continuation_exhausted",
        }
    return normalized


def resolve_delivery_modes(
    handler_registered: bool,
    stream_deltas: bool,
    interim_assistant_messages: bool,
) -> tuple[bool, bool]:
    """Buffer all assistant text while a pre-delivery authority is active."""
    if handler_registered:
        return False, False
    return stream_deltas, interim_assistant_messages


def _tool_call_dict(tool_call: Any) -> Dict[str, Any]:
    if isinstance(tool_call, Mapping):
        function = tool_call.get("function") or {}
        if not isinstance(function, Mapping):
            function = {}
        return {
            "id": tool_call.get("id"),
            "name": function.get("name") or tool_call.get("name"),
            "arguments": function.get("arguments", tool_call.get("arguments")),
        }
    function = getattr(tool_call, "function", None)
    return {
        "id": getattr(tool_call, "id", None),
        "name": getattr(function, "name", None) or getattr(tool_call, "name", None),
        "arguments": getattr(function, "arguments", None),
    }


def collect_tool_telemetry(
    messages: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Return full current-attempt tool call/result telemetry."""
    start = len(conversation_history or [])
    calls: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    for message in messages[start:]:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "assistant":
            calls.extend(
                _tool_call_dict(call) for call in (message.get("tool_calls") or [])
            )
        elif message.get("role") == "tool":
            results.append({
                "tool_call_id": message.get("tool_call_id"),
                "name": message.get("name"),
                "content": message.get("content"),
                "error": message.get("error"),
            })
    return {
        "tool_call_count": len(calls),
        "tool_result_count": len(results),
        "tool_calls": calls,
        "tool_results": results,
    }


def merge_tool_telemetry(*items: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Combine telemetry from bounded continuation attempts in order."""
    calls: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    for item in items:
        if not item:
            continue
        calls.extend(list(item.get("tool_calls") or []))
        results.extend(list(item.get("tool_results") or []))
    return {
        "tool_call_count": len(calls),
        "tool_result_count": len(results),
        "tool_calls": calls,
        "tool_results": results,
    }
