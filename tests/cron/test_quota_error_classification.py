import logging
import sys
import types

import cron.scheduler as cron_scheduler
from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE


def _patch_resolution_only_run(monkeypatch, resolver):
    """Keep run_job on its real provider-resolution path without booting an agent."""
    run_agent = types.ModuleType("run_agent")
    run_agent.AIAgent = object
    monkeypatch.setitem(sys.modules, "run_agent", run_agent)

    hermes_state = types.ModuleType("hermes_state")
    hermes_state.SessionDB = lambda: (_ for _ in ()).throw(RuntimeError("disabled"))
    monkeypatch.setitem(sys.modules, "hermes_state", hermes_state)

    env_loader = types.ModuleType("hermes_cli.env_loader")
    env_loader.load_hermes_dotenv = lambda **kwargs: None
    env_loader.reset_secret_source_cache = lambda: None
    monkeypatch.setitem(sys.modules, "hermes_cli.env_loader", env_loader)

    runtime_provider = types.ModuleType("hermes_cli.runtime_provider")
    runtime_provider.resolve_runtime_provider = resolver
    runtime_provider.format_runtime_provider_error = str
    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", runtime_provider)

    monkeypatch.setattr(cron_scheduler, "_build_job_prompt", lambda *args, **kwargs: "ping")
    monkeypatch.setattr(cron_scheduler, "_resolve_origin", lambda job: None)
    monkeypatch.setattr(cron_scheduler, "_guard_job_credential_exfil", lambda job: None)


def test_cron_quota_exhaustion_is_not_logged_as_auth_failure(monkeypatch, caplog):
    quota_error = AuthError(
        '{"error":{"type":"usage_limit_reached"}}',
        provider="openai-codex",
        code="usage_limit_reached",
        relogin_required=False,
    )

    def _resolve(**kwargs):
        raise quota_error

    _patch_resolution_only_run(monkeypatch, _resolve)
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda config: [])

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        success, _, _, error = cron_scheduler.run_job(
            {
                "id": "quota-job",
                "name": "Quota Test",
                "prompt": "ping",
                "model": "gpt-5-codex",
            }
        )

    assert success is False
    assert "usage_limit_reached" in error
    assert "primary provider quota/rate limit reached" in caplog.text
    assert "primary auth failed" not in caplog.text


def test_cron_quota_exhaustion_keeps_configured_fallback(monkeypatch, caplog):
    quota_error = AuthError(
        "Codex quota exhausted",
        provider="openai-codex",
        code=CODEX_RATE_LIMITED_CODE,
        relogin_required=False,
    )
    resolver_calls = []

    def _resolve(**kwargs):
        resolver_calls.append(kwargs)
        if len(resolver_calls) == 1:
            raise quota_error
        raise RuntimeError("fallback resolution reached")

    _patch_resolution_only_run(monkeypatch, _resolve)
    monkeypatch.setattr(
        cron_scheduler,
        "get_fallback_chain",
        lambda config: [
            {
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4.6",
                "api_key": "fallback-key",
            }
        ],
    )

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        success, _, _, error = cron_scheduler.run_job(
            {
                "id": "quota-fallback-job",
                "name": "Quota Fallback Test",
                "prompt": "ping",
                "model": "gpt-5-codex",
            }
        )

    assert success is False
    assert "Codex quota exhausted" in error
    assert resolver_calls == [
        {"requested": None, "target_model": "gpt-5-codex"},
        {
            "requested": "openrouter",
            "target_model": "anthropic/claude-sonnet-4.6",
            "explicit_api_key": "fallback-key",
        },
    ]
    assert "primary provider quota/rate limit reached" in caplog.text


def test_cron_fallback_replaces_primary_model(monkeypatch):
    quota_error = AuthError(
        "Codex provider quota exhausted (429); retry after 3600s.",
        provider="openai-codex",
        code=CODEX_RATE_LIMITED_CODE,
        relogin_required=False,
    )
    resolver_calls = []

    def _resolve(**kwargs):
        resolver_calls.append(kwargs)
        if len(resolver_calls) == 1:
            raise quota_error
        return {
            "api_key": "gemini-key",
            "base_url": "https://generativelanguage.googleapis.com",
            "provider": "gemini",
            "api_mode": "gemini",
            "command": None,
            "args": [],
        }

    _patch_resolution_only_run(monkeypatch, _resolve)
    captured = []

    class _CaptureAgent:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            raise RuntimeError("stop after agent construction")

    sys.modules["run_agent"].AIAgent = _CaptureAgent
    monkeypatch.setattr(
        cron_scheduler,
        "get_fallback_chain",
        lambda config: [
            {"provider": "gemini", "model": "gemini-3-flash-preview"}
        ],
    )

    success, _, _, error = cron_scheduler.run_job(
        {
            "id": "quota-model-handoff-job",
            "name": "Quota Model Handoff Test",
            "prompt": "ping",
            "model": "gpt-5.6-terra",
        }
    )

    assert success is False
    assert "stop after agent construction" in error
    assert captured[0]["provider"] == "gemini"
    assert captured[0]["model"] == "gemini-3-flash-preview"


def test_cron_skips_fallback_without_model(monkeypatch, caplog):
    quota_error = AuthError(
        "Codex provider quota exhausted (429); retry after 3600s.",
        provider="openai-codex",
        code=CODEX_RATE_LIMITED_CODE,
        relogin_required=False,
    )

    def _resolve(**kwargs):
        raise quota_error

    _patch_resolution_only_run(monkeypatch, _resolve)
    monkeypatch.setattr(
        cron_scheduler,
        "get_fallback_chain",
        lambda config: [{"provider": "gemini"}],
    )

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        success, _, _, error = cron_scheduler.run_job(
            {
                "id": "quota-missing-model-job",
                "name": "Quota Missing Model Test",
                "prompt": "ping",
                "model": "gpt-5.6-terra",
            }
        )

    assert success is False
    assert "quota exhausted" in error
    assert "skipping fallback gemini because no model is configured" in caplog.text


def test_cron_real_auth_failure_keeps_auth_wording(monkeypatch, caplog):
    auth_error = AuthError(
        "Codex token was revoked",
        provider="openai-codex",
        code="invalid_grant",
        relogin_required=True,
    )

    def _resolve(**kwargs):
        raise auth_error

    _patch_resolution_only_run(monkeypatch, _resolve)
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda config: [])

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        success, _, _, _ = cron_scheduler.run_job(
            {
                "id": "auth-job",
                "name": "Auth Test",
                "prompt": "ping",
                "model": "gpt-5-codex",
            }
        )

    assert success is False
    assert "primary auth failed" in caplog.text
    assert "primary provider quota/rate limit reached" not in caplog.text
