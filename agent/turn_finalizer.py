"""Post-loop turn finalization for ``run_conversation``.

Extracted from ``agent/conversation_loop.py`` as part of the god-file
decomposition campaign (``~/.hermes/plans/god-file-decomposition.md``, Phase 1
step 4 — the post-loop ``TurnFinalizer`` seam). ``run_conversation``'s tail
(everything after the main tool-calling ``while`` loop) is lifted here verbatim:
budget-exhaustion summary, trajectory save, session persist, turn diagnostics,
response transforms, result-dict assembly, steer drain, and the memory/skill
review trigger.

Behavior-neutral: the body is moved unchanged. All ``agent.*`` side effects fire
exactly as before; only the post-loop *locals* are passed in as keyword args, and
the assembled ``result`` dict is returned to ``run_conversation`` which returns it
to the caller. The function is synchronous with a single return — mirroring the
region it replaces (no awaits, no early returns).

Module ``logger`` is imported lazily inside the body (``from
agent.conversation_loop import logger``) so this module never imports
``agent.conversation_loop`` at import time -> no import cycle, and the log records
keep the exact logger name (``"agent.conversation_loop"``).
"""

from __future__ import annotations

import os

from agent.codex_responses_adapter import _summarize_user_message_for_log


def finalize_turn(
    agent,
    *,
    final_response,
    api_call_count,
    interrupted,
    failed,
    messages,
    conversation_history,
    effective_task_id,
    turn_id,
    user_message,
    original_user_message,
    _should_review_memory,
    _turn_exit_reason,
    _pending_verification_response=None,
):
    """Run the post-loop finalization and return the turn ``result`` dict.

    Lifted verbatim from ``run_conversation`` (the region after the main agent
    loop). See module docstring.
    """
    from agent.conversation_loop import logger

    budget_exhausted = (
        api_call_count >= agent.max_iterations
        or agent.iteration_budget.remaining <= 0
    )
    budget_fallback_eligible = (
        budget_exhausted
        and not interrupted
        and not failed
        and str(_turn_exit_reason) in {"unknown", "budget_exhausted"}
    )
    continuation_budget_exhausted = (
        final_response is None
        and bool(_pending_verification_response)
        and budget_fallback_eligible
    )

    iteration_limit_fallback = False
    preserved_verification_fallback = False
    if continuation_budget_exhausted:
        # A verification/continuation gate deliberately withheld a composed
        # answer, then consumed the remaining budget before producing a newer
        # one. Preserve that exact answer instead of replacing it with another
        # fallible model call. The explicit pending value is the provenance
        # guard: unrelated error/recovery exits can never enter this branch.
        final_response = _pending_verification_response
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        iteration_limit_fallback = True
        preserved_verification_fallback = True
    elif final_response is None and budget_fallback_eligible:
        # Budget exhausted — ask the model for a summary via one extra
        # API call with tools stripped.  _handle_max_iterations injects a
        # user message and makes a single toolless request.
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        agent._emit_status(
            f"⚠️ Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
            "— asking model to summarise"
        )
        if not agent.quiet_mode:
            agent._safe_print(
                f"\n⚠️  Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
                "— requesting summary..."
            )
        final_response = agent._handle_max_iterations(messages, api_call_count)
        iteration_limit_fallback = True

    if iteration_limit_fallback:
        # If running as a kanban worker, signal the dispatcher that the
        # worker could not complete (rather than treating it as a
        # protocol violation). This applies whether the user-facing fallback
        # came from the summary call or an explicitly pending continuation;
        # both exhausted the task budget and must advance the failure circuit.
        #
        # We route through ``_record_task_failure(outcome="timed_out")``
        # rather than ``kanban_block`` so this counts toward the dispatcher's
        # consecutive-failure circuit breaker (#29747 gap 2).
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            try:
                from hermes_cli import kanban_db as _kb
                _conn = _kb.connect()
                try:
                    _kb._record_task_failure(
                        _conn,
                        _kanban_task,
                        error=(
                            f"Iteration budget exhausted "
                            f"({api_call_count}/{agent.max_iterations}) — "
                            "task could not complete within the allowed "
                            "iterations"
                        ),
                        outcome="timed_out",
                        release_claim=True,
                        end_run=True,
                        event_payload_extra={
                            "budget_used": api_call_count,
                            "budget_max": agent.max_iterations,
                        },
                    )
                    logger.info(
                        "recorded budget-exhausted failure for task %s (%d/%d)",
                        _kanban_task, api_call_count, agent.max_iterations,
                    )
                finally:
                    try:
                        _conn.close()
                    except Exception:
                        pass
            except Exception:
                logger.warning(
                    "Failed to record budget-exhausted failure for task %s",
                    _kanban_task,
                    exc_info=True,
                )

    # A gateway pre-delivery hook is the last authority over user-visible
    # candidate text.  Run it before trajectory/session success persistence or
    # resource cleanup.  A ``continue`` decision returns the in-memory turn to
    # the gateway without recording the rejected candidate as terminal; the
    # gateway then performs one bounded continuation using these messages.
    _pre_delivery_prepared = False
    _response_transformed = False
    _pre_delivery_decision_result = None
    _pre_delivery_context_result = None
    _pre_delivery_blocked = False
    _pre_delivery_callback = getattr(agent, "pre_delivery_callback", None)
    if callable(_pre_delivery_callback) and not interrupted:
        from agent.pre_delivery import (
            DEGRADED_RESPONSE,
            REJECTED_CANDIDATE_PLACEHOLDER,
            collect_tool_telemetry,
            normalize_decision,
        )

        _raw_candidate_final = final_response

        # Apply the existing final text transforms before the authority hook so
        # neither a verifier footer nor an output plugin can change text after
        # it has been approved.  The legacy no-hook order remains untouched.
        if final_response:
            try:
                _failed_mutations = (
                    getattr(agent, "_turn_failed_file_mutations", None) or {}
                )
                if _failed_mutations and agent._file_mutation_verifier_enabled():
                    _footer = agent._format_file_mutation_failure_footer(
                        _failed_mutations
                    )
                    if _footer:
                        final_response = final_response.rstrip() + "\n\n" + _footer
            except Exception as _ver_err:
                logger.debug("file-mutation verifier footer failed: %s", _ver_err)

        try:
            if agent._turn_completion_explainer_enabled():
                _stripped = (final_response or "").strip()
                _is_empty_terminal = _stripped == "" or _stripped == "(empty)"
                _is_partial_fragment = (
                    not _is_empty_terminal
                    and not preserved_verification_fallback
                    and not str(_turn_exit_reason).startswith("text_response")
                    and len(_stripped) <= 24
                    and _stripped[-1:]
                    not in {".", "!", "?", "。", "！", "？", "`", ")"}
                )
                _is_partial_stream_recovery = (
                    str(_turn_exit_reason) == "partial_stream_recovery"
                )
                if (
                    _is_empty_terminal
                    or _is_partial_fragment
                    or _is_partial_stream_recovery
                ):
                    _explanation = agent._format_turn_completion_explanation(
                        _turn_exit_reason
                    )
                    if _explanation:
                        final_response = (
                            _explanation
                            if _is_empty_terminal
                            else _stripped + "\n\n" + _explanation
                        )
        except Exception as _exp_err:
            logger.debug("turn-completion explainer failed: %s", _exp_err)

        if final_response:
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook

                _transform_results = _invoke_hook(
                    "transform_llm_output",
                    response_text=final_response,
                    session_id=agent.session_id or "",
                    model=agent.model,
                    platform=getattr(agent, "platform", None) or "",
                )
                for _hook_result in _transform_results:
                    if isinstance(_hook_result, str) and _hook_result:
                        final_response = _hook_result
                        _response_transformed = True
                        break
            except Exception as exc:
                logger.warning("transform_llm_output hook failed: %s", exc)

        _tool_telemetry = collect_tool_telemetry(
            messages, conversation_history
        )
        _decision_context = {
            "candidate_final": final_response or "",
            "raw_candidate_final": _raw_candidate_final or "",
            "is_empty": not str(_raw_candidate_final or "").strip()
            or str(_raw_candidate_final or "").strip() == "(empty)",
            "original_message": original_user_message,
            "turn_id": turn_id,
            "task_id": effective_task_id,
            "session_id": agent.session_id or "",
            "model": agent.model,
            "provider": agent.provider,
            "api_mode": getattr(agent, "api_mode", None),
            "base_url": agent.base_url,
            "api_calls": api_call_count,
            "turn_exit_reason": _turn_exit_reason,
            "failed": failed,
            "interrupted": interrupted,
            **_tool_telemetry,
        }
        try:
            _decision = normalize_decision(
                _pre_delivery_callback(_decision_context)
            )
        except Exception as _decision_err:
            logger.error(
                "pre-delivery callback failed closed: %s",
                _decision_err,
                exc_info=True,
            )
            _decision = {
                "decision": "block",
                "response": DEGRADED_RESPONSE,
                "reason": f"pre_delivery_error:{type(_decision_err).__name__}",
            }
        _pre_delivery_decision_result = _decision
        _pre_delivery_context_result = _decision_context

        if _decision["decision"] == "continue":
            # Remove only the private terminal-empty sentinel.  Preserve real
            # tool calls/results for the continuation and its telemetry.
            while (
                messages
                and isinstance(messages[-1], dict)
                and messages[-1].get("_empty_terminal_sentinel")
            ):
                messages.pop()
            _safe_rejected_message = {
                "role": "assistant",
                "content": REJECTED_CANDIDATE_PLACEHOLDER,
                "_pre_delivery_rejected": True,
                "_pre_delivery_status": "rejected_nonterminal",
            }
            if messages and messages[-1].get("role") == "assistant":
                # Rebuild from an allowlist instead of deleting known fields.
                # Provider adapters may add new replay/raw payload keys over
                # time; none may survive into a rejected durable placeholder.
                messages[-1] = _safe_rejected_message
            else:
                messages.append(_safe_rejected_message)

            # The returned history is passed as conversation_history to the
            # bounded recovery run. Persist that exact safe chain now, after
            # the continue decision, so JSONL and SQLite agree on which prefix
            # is durable. The rejected candidate text itself is never stored.
            try:
                _persisted = agent._persist_session(
                    messages, conversation_history
                )
                if _persisted is not True:
                    raise RuntimeError(
                        "session store did not confirm placeholder persistence"
                    )
            except Exception as _continue_persist_err:
                logger.error(
                    "pre-delivery continuation persistence failed closed: %s",
                    _continue_persist_err,
                    exc_info=True,
                )
                _decision = {
                    "decision": "block",
                    "response": DEGRADED_RESPONSE,
                    "reason": "pre_delivery_continuation_persist_failed",
                }
                _pre_delivery_decision_result = _decision
                _pre_delivery_blocked = True
                final_response = DEGRADED_RESPONSE
            else:
                return {
                    "final_response": "",
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "failed": failed,
                    "interrupted": interrupted,
                    "pre_delivery_continue": True,
                    "pre_delivery_decision": _decision,
                    "pre_delivery_context": _decision_context,
                    "continuation_prompt": _decision["continuation_prompt"],
                    "model": agent.model,
                    "provider": agent.provider,
                    "api_mode": getattr(agent, "api_mode", None),
                    "session_id": agent.session_id,
                }

        if _decision["decision"] == "allow":
            # The gateway bridge may apply mandatory transport sanitization
            # before invoking handlers. Persist and return exactly the
            # candidate the authority hook inspected.
            final_response = str(
                _decision_context.get("candidate_final", final_response or "")
            )
        elif _decision["decision"] in {"rewrite", "block"}:
            final_response = _decision["response"]
        if _decision["decision"] == "block":
            _pre_delivery_blocked = True

        # Make the durable assistant row exactly match the approved/replaced
        # candidate.  This also turns an empty terminal sentinel into a normal
        # assistant row without deleting its preceding tool evidence.
        _terminal_assistant = (
            messages[-1]
            if messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "assistant"
            else None
        )
        if _terminal_assistant is None:
            messages.append({"role": "assistant", "content": final_response or ""})
        else:
            _terminal_assistant["content"] = final_response or ""
            _terminal_assistant.pop("_empty_terminal_sentinel", None)
            _terminal_assistant.pop("_empty_recovery_synthetic", None)
        _pre_delivery_prepared = True

    # Determine if conversation completed successfully
    normal_text_response = str(_turn_exit_reason).startswith("text_response(")
    completed = (
        final_response is not None
        and not failed
        and not _pre_delivery_blocked
        and (
            api_call_count < agent.max_iterations
            or normal_text_response
        )
    )

    # Post-loop cleanup must never lose the response.  Trajectory save,
    # resource teardown, and session persistence all touch fallible
    # surfaces — file I/O / JSON serialization (_save_trajectory), remote
    # VM/browser teardown over the network (_cleanup_task_resources), and
    # SQLite writes (_persist_session).  A raise from any of them used to
    # propagate straight out of run_conversation, discarding the partial
    # final_response the caller is waiting for (subprocess wrappers saw an
    # empty stdout with no traceback — #8049).  Each step is now guarded
    # independently so one failure can't skip the others, and any errors
    # are surfaced on the result dict via ``cleanup_errors`` rather than
    # killing the turn.
    _cleanup_errors = []

    # Save trajectory if enabled.  ``user_message`` may be a multimodal
    # list of parts; the trajectory format wants a plain string.
    try:
        agent._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)
    except Exception as _save_err:
        _cleanup_errors.append(f"save_trajectory: {_save_err}")
        logger.error("finalize_turn: _save_trajectory failed: %s", _save_err, exc_info=True)

    # Clean up VM and browser for this task after conversation completes
    try:
        agent._cleanup_task_resources(effective_task_id)
    except Exception as _cleanup_err:
        _cleanup_errors.append(f"cleanup_task_resources: {_cleanup_err}")
        logger.error("finalize_turn: _cleanup_task_resources failed: %s", _cleanup_err, exc_info=True)

    # Persist session to both JSON log and SQLite only after private retry
    # scaffolding has been removed. Otherwise a later user "continue" turn
    # can replay assistant("(empty)") / recovery nudges and fall into the
    # same empty-response loop again.
    try:
        agent._drop_trailing_empty_response_scaffolding(messages)

        # When the turn was interrupted and the last message is a tool
        # result, append a synthetic assistant message to close the
        # tool-call sequence. Without this, the session persists a
        # ``tool → user`` alternation that strict providers (Gemini,
        # Claude) reject, causing them to hallucinate a continuation of
        # the user's message on the next turn (#48879).
        #
        # ``_drop_trailing_empty_response_scaffolding`` only rewinds the
        # tool tail when an empty-response scaffolding flag is present; a
        # clean ``/stop`` interrupt after a successful tool sets no such
        # flag, so the tool result survives as the tail and we close it
        # here instead. On an interrupt ``final_response`` is typically
        # empty, so fall back to an explicit placeholder rather than
        # persisting an empty-content assistant turn.
        if interrupted:
            from agent.message_sanitization import close_interrupted_tool_sequence
            close_interrupted_tool_sequence(messages, final_response)

        # Some recovery/fallback paths return a real final_response without
        # adding a closing assistant message to the transcript (e.g. the
        # partial-stream and prior-turn-content recovery ``break`` sites in
        # ``conversation_loop``). If persisted as-is, the durable session can
        # end at a tool/user message even though the caller — and the gateway
        # platform — already saw a completed assistant response. The next turn
        # then replays a user-only backlog and the model re-answers every
        # "unanswered" message. Close the durable turn at the source, at the
        # single chokepoint every recovery ``break`` flows through, so the
        # invariant "delivered final_response ⇒ assistant row in transcript"
        # holds regardless of which path produced it. (#43849 / #44100)
        if final_response and not interrupted:
            try:
                _tail_role = messages[-1].get("role") if messages else None
            except Exception:
                _tail_role = None
            if _tail_role != "assistant":
                messages.append({"role": "assistant", "content": final_response})

        agent._persist_session(messages, conversation_history)
    except Exception as _persist_err:
        _cleanup_errors.append(f"persist_session: {_persist_err}")
        logger.error("finalize_turn: _persist_session failed: %s", _persist_err, exc_info=True)

    # ── Turn-exit diagnostic log ─────────────────────────────────────
    # Always logged at INFO so agent.log captures WHY every turn ended.
    # When the last message is a tool result (agent was mid-work), log
    # at WARNING — this is the "just stops" scenario users report.
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # Walk back to find the assistant message with the tool call
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and _m.get("tool_calls"):
                _tcs = _m["tool_calls"]
                if _tcs and isinstance(_tcs[0], dict):
                    _last_tool_name = _tcs[-1].get("function", {}).get("name")
                break

    _turn_tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    _resp_len = len(final_response) if final_response else 0
    _budget_used = agent.iteration_budget.used if agent.iteration_budget else 0
    _budget_max = agent.iteration_budget.max_total if agent.iteration_budget else 0

    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, agent.model, api_call_count, agent.max_iterations,
        _budget_used, _budget_max,
        _turn_tool_count, _last_msg_role, _resp_len,
        agent.session_id or "none",
    )

    if _last_msg_role == "tool" and not interrupted:
        # Agent was mid-work — this is the "just stops" case.
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        logger.info(_diag_msg, *_diag_args)

    # File-mutation verifier footer.
    # If one or more ``write_file`` / ``patch`` calls failed during this
    # turn and were never superseded by a successful write to the same
    # path, append an advisory footer to the assistant response.  This
    # catches the specific case — reported by Ben Eng (#15524-adjacent)
    # — where a model issues a batch of parallel patches, half of them
    # fail with "Could not find old_string", and the model summarises
    # the turn claiming every file was edited.  The user then has to
    # manually run ``git status`` to catch the lie.  With this footer
    # the truth is surfaced on every turn, so over-claiming is
    # structurally impossible past the model.
    #
    # Gate: only applied when a real text response exists for this
    # turn and the user didn't interrupt.  Empty/interrupted turns
    # already have other surface text that shouldn't be augmented.
    if final_response and not interrupted and not _pre_delivery_prepared:
        try:
            _failed = getattr(agent, "_turn_failed_file_mutations", None) or {}
            if _failed and agent._file_mutation_verifier_enabled():
                footer = agent._format_file_mutation_failure_footer(_failed)
                if footer:
                    final_response = final_response.rstrip() + "\n\n" + footer
        except Exception as _ver_err:
            logger.debug("file-mutation verifier footer failed: %s", _ver_err)

    # Turn-completion explainer.
    # When a turn ends abnormally after substantive work — empty content
    # after retries, a partial/truncated stream, a still-pending tool
    # result, or an iteration/budget limit — the user otherwise gets a
    # blank or fragmentary response box with no consolidated reason why
    # the agent stopped (#34452).  Surface a single user-visible
    # explanation derived from ``_turn_exit_reason``, mirroring the
    # file-mutation verifier footer pattern above.
    #
    # Gate carefully so healthy turns stay quiet:
    #   - ``text_response(...)`` exits never produce an explanation
    #     (handled inside the formatter), so a terse ``Done.`` is silent.
    #   - We only ACT when there is no genuinely usable reply this turn:
    #     an empty response, the "(empty)" terminal sentinel, or a
    #     suspiciously short partial fragment with no terminating
    #     punctuation (e.g. "The").  A real short answer keeps its text.
    if not interrupted and not _pre_delivery_prepared:
        try:
            if agent._turn_completion_explainer_enabled():
                _stripped = (final_response or "").strip()
                _is_empty_terminal = _stripped == "" or _stripped == "(empty)"
                # A short fragment that is not a normal text_response exit
                # and lacks sentence-ending punctuation is treated as a
                # truncated partial (the "The" case from #34452).
                _is_partial_fragment = (
                    not _is_empty_terminal
                    and not preserved_verification_fallback
                    and not str(_turn_exit_reason).startswith("text_response")
                    and len(_stripped) <= 24
                    and _stripped[-1:] not in {".", "!", "?", "。", "！", "？", "`", ")"}
                )
                _is_partial_stream_recovery = (
                    str(_turn_exit_reason) == "partial_stream_recovery"
                )
                if (
                    _is_empty_terminal
                    or _is_partial_fragment
                    or _is_partial_stream_recovery
                ):
                    _explanation = agent._format_turn_completion_explanation(
                        _turn_exit_reason
                    )
                    if _explanation:
                        if _is_empty_terminal:
                            # Replace the bare "(empty)"/blank sentinel with
                            # the actionable explanation.
                            final_response = _explanation
                        else:
                            # Keep the partial fragment, append the reason so
                            # the user sees both what arrived and why it
                            # stopped.
                            final_response = (
                                _stripped + "\n\n" + _explanation
                            )
        except Exception as _exp_err:
            logger.debug("turn-completion explainer failed: %s", _exp_err)

    # When the pre-delivery path is active, transforms already ran before the
    # authority hook and persistence.  Legacy turns retain the old ordering.

    # Plugin hook: transform_llm_output
    # Fired once per turn after the tool-calling loop completes.
    # Plugins can transform the LLM's output text before it's returned.
    # First hook to return a string wins; None/empty return leaves text unchanged.
    if final_response and not interrupted and not _pre_delivery_prepared:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _transform_results = _invoke_hook(
                "transform_llm_output",
                response_text=final_response,
                session_id=agent.session_id or "",
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
            for _hook_result in _transform_results:
                if isinstance(_hook_result, str) and _hook_result:
                    final_response = _hook_result
                    _response_transformed = True
                    break  # First non-empty string wins
        except Exception as exc:
            logger.warning("transform_llm_output hook failed: %s", exc)

    # Plugin hook: post_llm_call
    # Fired once per turn after the tool-calling loop completes.
    # Plugins can use this to persist conversation data (e.g. sync
    # to an external memory system).
    if final_response and not interrupted:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "post_llm_call",
                session_id=agent.session_id,
                task_id=effective_task_id,
                turn_id=turn_id,
                user_message=original_user_message,
                assistant_response=final_response,
                conversation_history=list(messages),
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("post_llm_call hook failed: %s", exc)

    # Extract reasoning from the CURRENT turn only.  Walk backwards
    # but stop at the user message that started this turn — anything
    # earlier is from a prior turn and must not leak into the reasoning
    # box (confusing stale display; #17055).  Within the current turn
    # we still want the *most recent* non-empty reasoning: many
    # providers (Claude thinking, DeepSeek v4, Codex Responses) emit
    # reasoning on the tool-call step and leave the final-answer step
    # with reasoning=None, so picking only the last assistant would
    # silently drop legitimate same-turn reasoning.
    last_reasoning = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            break  # turn boundary — don't cross into prior turns
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            last_reasoning = msg["reasoning"]
            break

    # Build result with interrupt info if applicable
    result = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": completed,
        "turn_exit_reason": _turn_exit_reason,
        "failed": failed,
        "partial": False,  # True only when stopped due to invalid tool calls
        "interrupted": interrupted,
        "response_transformed": _response_transformed,
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "input_tokens": agent.session_input_tokens,
        "output_tokens": agent.session_output_tokens,
        "cache_read_tokens": agent.session_cache_read_tokens,
        "cache_write_tokens": agent.session_cache_write_tokens,
        "reasoning_tokens": agent.session_reasoning_tokens,
        "prompt_tokens": agent.session_prompt_tokens,
        "completion_tokens": agent.session_completion_tokens,
        "total_tokens": agent.session_total_tokens,
        "last_prompt_tokens": getattr(agent.context_compressor, "last_prompt_tokens", 0) or 0,
        "estimated_cost_usd": agent.session_estimated_cost_usd,
        "cost_status": agent.session_cost_status,
        "cost_source": agent.session_cost_source,
        "session_id": agent.session_id,
    }
    if agent._tool_guardrail_halt_decision is not None:
        result["guardrail"] = agent._tool_guardrail_halt_decision.to_metadata()
    if _pre_delivery_decision_result is not None:
        result["pre_delivery_decision"] = _pre_delivery_decision_result
        result["pre_delivery_context"] = _pre_delivery_context_result
    # Surface any post-loop cleanup failures so the caller can distinguish a
    # clean turn from one whose trajectory/session/resource teardown raised
    # (the response is still returned either way — #8049).
    if _cleanup_errors:
        result["cleanup_errors"] = _cleanup_errors
    # If a /steer landed after the final assistant turn (no more tool
    # batches to drain into), hand it back to the caller so it can be
    # delivered as the next user turn instead of being silently lost.
    _leftover_steer = agent._drain_pending_steer()
    if _leftover_steer:
        result["pending_steer"] = _leftover_steer
    agent._response_was_previewed = False

    # Include interrupt message if one triggered the interrupt
    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message

    # Clear interrupt state after handling
    agent.clear_interrupt()

    # Clear stream callback so it doesn't leak into future calls
    agent._stream_callback = None

    # Check skill trigger NOW — based on how many tool iterations THIS turn used.
    _should_review_skills = False
    if (agent._skill_nudge_interval > 0
            and agent._iters_since_skill >= agent._skill_nudge_interval
            and "skill_manage" in agent.valid_tool_names):
        _should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider: sync the completed turn + queue next prefetch.
    agent._sync_external_memory_for_turn(
        original_user_message=original_user_message,
        final_response=final_response,
        interrupted=interrupted,
        messages=messages,
    )

    # Background memory/skill review — runs AFTER the response is delivered
    # so it never competes with the user's task for model attention.
    if final_response and not interrupted and (_should_review_memory or _should_review_skills):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # Background review is best-effort

    # Note: Memory provider on_session_end() + shutdown_all() are NOT
    # called here — run_conversation() is called once per user message in
    # multi-turn sessions. Shutting down after every turn would kill the
    # provider before the second message. Actual session-end cleanup is
    # handled by the CLI (atexit / /reset) and gateway (session expiry /
    # _reset_session).

    # Plugin hook: on_session_end
    # Fired at the very end of every run_conversation call.
    # Plugins can use this for cleanup, flushing buffers, etc.
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            completed=completed,
            interrupted=interrupted,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_end hook failed: %s", exc)

    return result
