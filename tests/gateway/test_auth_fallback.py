"""Test that AuthError triggers fallback provider resolution (#7230)."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class TestResolveRuntimeAgentKwargsAuthFallback:
    """_resolve_runtime_agent_kwargs should try fallback on AuthError."""

    def test_auth_error_tries_fallback(self, tmp_path, monkeypatch):
        """When primary provider raises AuthError, fallback is attempted."""
        from hermes_cli.auth import AuthError

        # Create a config with fallback
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "model:\n  provider: openai-codex\n"
            "fallback_model:\n  provider: openrouter\n"
            "  model: meta-llama/llama-4-maverick\n"
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        call_count = {"n": 0}

        def _mock_resolve(**kwargs):
            call_count["n"] += 1
            # First call = primary path (gateway reads model.provider from
            # config.yaml internally; we simulate the auth failure here).
            # Second call = fallback path with explicit_api_key + explicit_base_url
            # supplied by gateway from fallback_model config.
            if call_count["n"] == 1:
                raise AuthError("Codex token refresh failed with status 401")
            return {
                "api_key": "fallback-key",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "api_mode": "openai_chat",
                "command": None,
                "args": None,
                "credential_pool": None,
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "openrouter"
        assert result["api_key"] == "fallback-key"
        assert result["model"] == "meta-llama/llama-4-maverick"
        # Should have been called at least twice (primary + fallback)
        assert call_count["n"] >= 2

    def test_auth_error_no_fallback_raises(self, tmp_path, monkeypatch):
        """When primary fails and no fallback configured, RuntimeError is raised."""
        from hermes_cli.auth import AuthError

        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  provider: openai-codex\n")

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=AuthError("token expired"),
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            with pytest.raises(RuntimeError):
                _resolve_runtime_agent_kwargs()

    def test_legacy_fallback_is_appended_after_fallback_providers(self, tmp_path, monkeypatch):
        """When both keys exist, the legacy entry still participates in resolution."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: anthropic/claude-sonnet-4.6\n"
            "fallback_model:\n"
            "  provider: nous\n"
            "  model: Hermes-4\n"
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        calls = []

        def _mock_resolve(**kwargs):
            requested = kwargs.get("requested")
            calls.append(requested)
            if requested == "openrouter":
                raise RuntimeError("openrouter unavailable")
            return {
                "api_key": "nous-key",
                "base_url": "https://portal.nousresearch.com/v1",
                "provider": "nous",
                "api_mode": "chat_completions",
                "command": None,
                "args": None,
                "credential_pool": None,
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _try_resolve_fallback_provider

            result = _try_resolve_fallback_provider()

        assert calls == ["openrouter", "nous"]
        assert result["provider"] == "nous"
        assert result["model"] == "Hermes-4"

    def test_quota_fallback_carries_disclosure_metadata(self, tmp_path, monkeypatch):
        """A resolved quota fallback carries safe metadata for post-success disclosure."""
        from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE

        (tmp_path / "config.yaml").write_text(
            "model:\n  provider: openai-codex\n"
            "fallback_providers:\n"
            "  - provider: gemini\n"
            "    model: gemini-3-flash-preview\n"
        )
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        quota_error = AuthError(
            "Codex provider quota exhausted (429); retry after 3600s. Credentials are still valid.",
            provider="openai-codex",
            code=CODEX_RATE_LIMITED_CODE,
            relogin_required=False,
        )

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "gemini-key",
                "base_url": "https://generativelanguage.googleapis.com",
                "provider": "gemini",
                "api_mode": "gemini",
                "command": None,
                "args": None,
                "credential_pool": None,
            },
        ):
            from gateway.run import _try_resolve_fallback_provider
            result = _try_resolve_fallback_provider(primary_error=quota_error)

        assert result["provider"] == "gemini"
        assert result["model"] == "gemini-3-flash-preview"
        notice = result["initial_fallback_notice"]
        assert notice["provider"] == "gemini"
        assert notice["model"] == "gemini-3-flash-preview"
        assert "temporarily using Gemini" in notice["text"]
        assert "provider retry window" in notice["text"]
        assert "metered charges" in notice["text"]


def test_codex_retry_at_text_uses_exact_provider_window():
    from gateway.run import _codex_retry_at_text
    from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE

    error = AuthError(
        "Codex provider quota exhausted (429); retry after 3661s.",
        provider="openai-codex",
        code=CODEX_RATE_LIMITED_CODE,
    )
    now = datetime(2026, 7, 28, 0, 15, 0, tzinfo=timezone.utc)

    assert _codex_retry_at_text(error, now=now) == "2026-07-28 01:16:01 AM UTC"


def test_no_fallback_quota_reply_is_honest_and_cost_visible(monkeypatch):
    from gateway import run as gateway_run
    from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE

    error = AuthError(
        "Codex provider quota exhausted (429); retry after 900s.",
        provider="openai-codex",
        code=CODEX_RATE_LIMITED_CODE,
    )
    wrapped = RuntimeError("provider unavailable")
    wrapped.__cause__ = error
    monkeypatch.setattr(
        gateway_run,
        "_codex_retry_at_text",
        lambda exc: "2026-07-28 12:30:00 AM EDT",
    )

    reply = gateway_run._gateway_runtime_unavailable_reply(wrapped)

    assert "Codex is unavailable until 2026-07-28 12:30:00 AM EDT" in reply
    assert "No usable fallback was available" in reply
    assert "no fallback API call" in reply
    assert "temporarily using Gemini" not in reply


def test_initial_fallback_notice_emits_once_only_after_usable_success():
    from gateway.run import _prepend_successful_initial_fallback_notice

    agent = SimpleNamespace(provider="gemini", model="gemini-3-flash-preview")
    route = {
        "initial_fallback_notice": {
            "id": "gemini:gemini-3-flash-preview:reset",
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "text": "fallback disclosure",
        }
    }

    assert _prepend_successful_initial_fallback_notice(
        agent, route, {"api_calls": 0}, "canned reply"
    ) == "canned reply"
    assert _prepend_successful_initial_fallback_notice(
        agent, route, {"api_calls": 1, "error": "failed"}, "failed reply"
    ) == "failed reply"
    assert _prepend_successful_initial_fallback_notice(
        agent, route, {"api_calls": 1}, "real answer"
    ) == "fallback disclosure\n\nreal answer"
    assert _prepend_successful_initial_fallback_notice(
        agent, route, {"api_calls": 1}, "next answer"
    ) == "next answer"


def test_initial_fallback_notice_rejects_provider_mismatch():
    from gateway.run import _prepend_successful_initial_fallback_notice

    agent = SimpleNamespace(provider="openai-codex", model="gpt-5.6-terra")
    route = {
        "initial_fallback_notice": {
            "id": "gemini:gemini-3-flash-preview:reset",
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "text": "fallback disclosure",
        }
    }

    assert _prepend_successful_initial_fallback_notice(
        agent, route, {"api_calls": 1}, "codex answer"
    ) == "codex answer"


@pytest.mark.parametrize(
    "result",
    [
        {"api_calls": 1, "failed": True},
        {"api_calls": 1, "completed": False},
    ],
)
def test_initial_fallback_notice_rejects_unsuccessful_result_state(result):
    from gateway.run import _prepend_successful_initial_fallback_notice

    agent = SimpleNamespace(provider="gemini", model="gemini-3-flash-preview")
    route = {
        "initial_fallback_notice": {
            "id": "gemini:gemini-3-flash-preview:reset",
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "text": "fallback disclosure",
        }
    }

    assert _prepend_successful_initial_fallback_notice(
        agent, route, result, "unsuccessful reply"
    ) == "unsuccessful reply"
    assert not hasattr(agent, "_gateway_initial_fallback_notice_id")
