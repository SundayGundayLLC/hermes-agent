from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_worker_diagnostics import (
    classify_primary_provider_failure,
    classify_worker_failure,
    is_known_disabled_tool,
    kanban_provider_retry_ceiling,
    provider_failure_snapshot,
    record_kanban_worker_failure,
    should_continue_kanban_goal_loop,
)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claimed_attempt(conn, *, worker_pid: int) -> tuple[str, int, str]:
    task_id = kb.create_task(conn, title="worker diagnosis", assignee="foreman")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    task = kb.claim_task(conn, task_id, claimer="host:worker")
    assert task is not None and task.current_run_id is not None
    kb._set_worker_pid(conn, task_id, worker_pid)
    return task_id, int(task.current_run_id), str(task.claim_lock)


def test_retry_ceiling_stops_exhausted_final_provider() -> None:
    assert kanban_provider_retry_ceiling(
        3, is_provider_capacity_failure=True, has_pending_fallback=False
    ) == 1
    assert kanban_provider_retry_ceiling(
        3, is_provider_capacity_failure=True, has_pending_fallback=True
    ) == 3
    assert kanban_provider_retry_ceiling(
        3, is_provider_capacity_failure=False, has_pending_fallback=False
    ) == 3


def test_known_disabled_tool_is_not_confused_with_typo() -> None:
    registered = {"terminal", "web_search", "read_file"}
    assert is_known_disabled_tool(
        "terminal", {"web_search"}, registered_tools=registered
    )
    assert not is_known_disabled_tool(
        "termnial", {"web_search"}, registered_tools=registered
    )


def test_failed_goal_worker_never_continues() -> None:
    assert not should_continue_kanban_goal_loop({"failed": True})
    assert should_continue_kanban_goal_loop({"completed": True})


@pytest.mark.parametrize(
    ("result", "provider", "category", "disposition"),
    [
        (
            {"failed": True, "failure_reason": "billing", "error": "credits exhausted"},
            "gemini",
            "owner_billing_gate",
            "block_needs_input",
        ),
        (
            {
                "failed": True,
                "failure_reason": "rate_limit",
                "failure_quota_marker": "paid_tier_input_token",
                "error": "RESOURCE_EXHAUSTED",
            },
            "gemini",
            "gemini_paid_tier_input_token_quota",
            "release_with_cooldown",
        ),
        (
            {
                "failed": True,
                "failure_reason": "rate_limit",
                "failure_status_code": 429,
                "error": "requests per minute exceeded",
            },
            "gemini",
            "gemini_request_rate_limited",
            "release_with_cooldown",
        ),
        (
            {"failed": True, "failure_reason": "unknown", "error": "model is not supported"},
            "gemini",
            "fallback_model_mismatch",
            "block_capability",
        ),
        (
            {
                "failed": True,
                "failure_reason": "toolset_mismatch",
                "requested_tool": "terminal",
                "available_tools": ["web_search"],
            },
            "gemini",
            "toolset_mismatch",
            "block_capability",
        ),
    ],
)
def test_failure_categories_are_exact(result, provider, category, disposition) -> None:
    diagnostic = classify_worker_failure(
        result, provider=provider, model="model", profile="foreman"
    )
    assert diagnostic is not None
    assert diagnostic["category"] == category
    assert diagnostic["disposition"] == disposition


def test_provider_failure_trail_redacts_raw_error() -> None:
    snapshot = provider_failure_snapshot(
        RuntimeError(
            "429 paid_tier generate_content_input_token_count secret-body"
        ),
        provider="gemini",
        model="gemini-3-flash-preview",
        reason="rate_limit",
        status_code=429,
        fallback_stage=1,
    )
    assert snapshot["quota_marker"] == "paid_tier_input_token"
    assert snapshot["classification"] == "rate_limit"
    assert "secret-body" not in json.dumps(snapshot)


def test_provider_failure_uses_structured_gemini_quota_fields() -> None:
    class StructuredQuotaError(RuntimeError):
        body = {
            "error": {
                "details": [
                    {
                        "quotaMetric": "generativelanguage.googleapis.com/generate_content_paid_tier_input_token_count",
                        "quotaId": "GenerateContentInputTokensPerModelPerMinute-PaidTier",
                    }
                ]
            }
        }

        def __str__(self) -> str:
            return "RESOURCE_EXHAUSTED"

    snapshot = provider_failure_snapshot(
        StructuredQuotaError(),
        provider="gemini",
        model="gemini-3-flash-preview",
        reason="rate_limit",
        status_code=429,
        fallback_stage=1,
    )
    assert snapshot["quota_marker"] == "paid_tier_input_token"


def test_primary_codex_quota_is_preserved_with_fallback() -> None:
    primary = classify_primary_provider_failure(
        RuntimeError("429 usage limit reached"),
        provider="openai-codex",
        fallback_provider="gemini",
        fallback_model="gemini-3-flash-preview",
    )
    assert primary["classification"] == "primary_codex_quota_exhausted"
    terminal = classify_worker_failure(
        {
            "failed": True,
            "failure_reason": "rate_limit",
            "failure_status_code": 429,
            "provider_failures": [
                {"provider": "openai-codex", "classification": "rate_limit"},
                {"provider": "gemini", "classification": "rate_limit"},
            ],
        },
        provider="gemini",
        model="gemini-3-flash-preview",
        profile="foreman",
        primary_failure=primary,
    )
    assert terminal is not None
    assert terminal["primary_failure"]["classification"] == "primary_codex_quota_exhausted"
    assert len(terminal["provider_failures"]) == 2


def test_codex_quota_terminal_reason_keeps_retry_and_valid_credentials() -> None:
    diagnostic = classify_worker_failure(
        {
            "failed": True,
            "failure_reason": "rate_limit",
            "failure_status_code": 429,
            "error": "Codex provider quota exhausted (429); retry after 44479s. Credentials are still valid.",
        },
        provider="openai-codex",
        model="gpt-5.6-terra",
        profile="foreman",
    )
    assert diagnostic is not None
    assert diagnostic["category"] == "codex_provider_capacity"
    assert diagnostic["credentials_state"] == "valid"
    assert diagnostic["retry_after_seconds"] == 44479
    assert diagnostic["disposition"] == "release_with_cooldown"


def test_invalid_credentials_are_not_reported_as_capacity() -> None:
    diagnostic = classify_worker_failure(
        {
            "failed": True,
            "failure_reason": "unknown",
            "failure_status_code": 401,
            "error": "Authentication failed: invalid credentials",
        },
        provider="openai-codex",
        model="gpt-5.6-terra",
        profile="foreman",
    )
    assert diagnostic is not None
    assert diagnostic["category"] == "provider_auth_failed"
    assert diagnostic["credentials_state"] == "invalid"
    assert diagnostic["disposition"] == "block_needs_input"


def test_invalid_credentials_override_a_429_capacity_status() -> None:
    diagnostic = classify_worker_failure(
        {
            "failed": True,
            "failure_reason": "auth",
            "failure_status_code": 429,
            "error": "Authentication failed: invalid credentials; retry after 30s.",
        },
        provider="openai-codex",
        model="gpt-5.6-terra",
        profile="foreman",
    )
    assert diagnostic is not None
    assert diagnostic["category"] == "provider_auth_failed"
    assert diagnostic["credentials_state"] == "invalid"
    assert diagnostic["disposition"] == "block_needs_input"


def test_rate_limit_release_is_exact_attempt_bound(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_pid = os.getpid()
    with kb.connect_closing() as conn:
        task_id, run_id, claim_lock = _claimed_attempt(conn, worker_pid=worker_pid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", claim_lock)
    monkeypatch.setenv("HERMES_PROFILE", "foreman")
    cli = SimpleNamespace(
        provider="gemini", model="gemini-3-flash-preview",
        _kanban_primary_provider_failure=None,
    )
    result = {
        "failed": True,
        "failure_reason": "rate_limit",
        "failure_provider": "gemini",
        "failure_model": "gemini-3-flash-preview",
        "failure_status_code": 429,
        "error": "request rate limit",
    }
    assert record_kanban_worker_failure(cli, result)
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "ready"
        assert task.consecutive_failures == 0
        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert run["outcome"] == "rate_limited"
        assert json.loads(run["metadata"])["category"] == "gemini_request_rate_limited"


def test_codex_quota_canary_persists_typed_run_and_event(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker_pid = os.getpid()
    with kb.connect_closing() as conn:
        task_id, run_id, claim_lock = _claimed_attempt(conn, worker_pid=worker_pid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", claim_lock)
    monkeypatch.setenv("HERMES_PROFILE", "foreman")
    cli = SimpleNamespace(provider="openai-codex", model="gpt-5.6-terra", _kanban_primary_provider_failure=None)
    result = {
        "failed": True,
        "failure_reason": "rate_limit",
        "failure_status_code": 429,
        "error": "Codex provider quota exhausted (429); retry after 44479s. Credentials are still valid.",
    }
    assert record_kanban_worker_failure(cli, result)
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "ready"
        run = conn.execute("SELECT outcome, error, metadata FROM task_runs WHERE id=?", (run_id,)).fetchone()
        event = conn.execute("SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    metadata = json.loads(run["metadata"])
    assert run["outcome"] == "rate_limited"
    assert "credentials=valid" in run["error"]
    assert "retry_after_seconds=44479" in run["error"]
    assert metadata["category"] == "codex_provider_capacity"
    assert metadata["retry_after_seconds"] == 44479
    assert event["kind"] == "rate_limited"
    assert json.loads(event["payload"])["credentials_state"] == "valid"


def test_stale_worker_cannot_mutate_new_attempt(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id, run_id, claim_lock = _claimed_attempt(conn, worker_pid=111)
        assert not kb.release_rate_limited_worker(
            conn,
            task_id,
            reason="stale",
            expected_run_id=run_id,
            expected_claim_lock=claim_lock,
            expected_worker_pid=222,
        )
        assert kb.get_task(conn, task_id).status == "running"


def test_fast_failure_sidecar_replays_after_dispatcher_attaches_pid(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_pid = os.getpid()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="fast worker", assignee="foreman"
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        task = kb.claim_task(conn, task_id, claimer="host:fast")
        assert task is not None and task.current_run_id is not None
        run_id = int(task.current_run_id)
        claim_lock = str(task.claim_lock)

    sidecar = kb.worker_logs_dir() / f"{task_id}.{run_id}.diagnostic.json"
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", claim_lock)
    monkeypatch.setenv("HERMES_KANBAN_DIAGNOSTIC_PATH", str(sidecar))
    cli = SimpleNamespace(
        provider="gemini", model="gemini-3-flash-preview",
        _kanban_primary_provider_failure=None,
    )
    result = {
        "failed": True,
        "failure_reason": "rate_limit",
        "failure_status_code": 429,
        "error": "request rate limit",
    }

    # The worker can finish before dispatch_once stores its PID. Exact CAS
    # must fail without losing the diagnosis.
    assert not record_kanban_worker_failure(cli, result)
    assert sidecar.is_file()
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).worker_pid is None
        kb._set_worker_pid(conn, task_id, worker_pid)
        assert kb.replay_worker_diagnostic_sidecars(conn) == [task_id]
        assert kb.get_task(conn, task_id).status == "ready"
    assert not sidecar.exists()


def test_windows_popen_exit_75_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    pid = 99123
    kb._windows_worker_processes.clear()
    kb._recent_worker_exits.pop(pid, None)
    kb._windows_worker_processes[pid] = SimpleNamespace(poll=lambda: 75)
    assert kb._poll_windows_worker_exits() == [pid]
    assert kb._classify_worker_exit(pid) == ("rate_limited", 75)


def test_windows_completed_exit_metadata_is_bounded() -> None:
    kb._windows_worker_processes.clear()
    kb._recent_worker_returncodes.clear()
    for pid in range(100_000, 105_000):
        kb._windows_worker_processes[pid] = SimpleNamespace(poll=lambda: 0)
    assert len(kb._poll_windows_worker_exits()) == 5_000
    assert len(kb._recent_worker_returncodes) <= kb._RECENT_WORKER_EXITS_MAX


def test_diagnostic_write_failure_uses_atomic_sidecar(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker_pid = os.getpid()
    with kb.connect_closing() as conn:
        task_id, run_id, claim_lock = _claimed_attempt(conn, worker_pid=worker_pid)
    sidecar = tmp_path / "worker.diagnostic.json"
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", claim_lock)
    monkeypatch.setenv("HERMES_KANBAN_DIAGNOSTIC_PATH", str(sidecar))
    monkeypatch.setattr(kb, "connect", lambda *a, **k: (_ for _ in ()).throw(OSError("db busy")))
    cli = SimpleNamespace(provider="gemini", model="m", _kanban_primary_provider_failure=None)
    with pytest.raises(OSError, match="db busy"):
        record_kanban_worker_failure(
            cli,
            {
                "failed": True,
                "failure_reason": "rate_limit",
                "failure_status_code": 429,
                "error": "rate limit",
            },
        )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["task_id"] == task_id
    assert payload["run_id"] == run_id
    assert payload["claim_lock"] == claim_lock
