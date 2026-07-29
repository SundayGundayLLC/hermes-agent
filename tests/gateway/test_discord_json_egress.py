"""Behavior tests for Discord's final JSON egress boundary."""

from __future__ import annotations

import contextvars
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The canonical runner intentionally starts from ``env -i``.  Windows'
# pathlib ignores HOME when USERPROFILE/LOCALAPPDATA are absent, so give
# collection a temp-scoped profile before importing gateway modules.
_COLLECTION_HERMES_HOME = Path(tempfile.gettempdir()) / "hermes-discord-egress-tests"
_COLLECTION_HERMES_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HERMES_HOME", str(_COLLECTION_HERMES_HOME))
os.environ.setdefault("LOCALAPPDATA", str(_COLLECTION_HERMES_HOME.parent))

from gateway.discord_egress import (
    begin_discord_turn,
    current_message_requests_json,
    end_discord_turn,
    filter_discord_attachment,
    filter_discord_text,
    raw_json_allowed,
)
from gateway.config import PlatformConfig


def _ensure_discord_mock() -> None:
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object,
        button=lambda *args, **kwargs: (lambda fn: fn),
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1,
        green=lambda: 2,
        blue=lambda: 3,
        red=lambda: 4,
        purple=lambda: 5,
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod = MagicMock()
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import (  # noqa: E402
    DiscordAdapter,
    _remember_channel_is_forum,
    _standalone_send,
)


def test_full_process_result_becomes_compact_human_text() -> None:
    raw = (
        '{"operation":"render","success":true,"exit_code":0,'
        '"proof_path":"C:/proof/render.json","payload":{"frames":900}}'
    )
    result = filter_discord_text(raw)
    assert result == (
        "Operation: render. Result: success. Code: 0. "
        "Proof: C:/proof/render.json."
    )
    assert '"payload"' not in result


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Provider failed:\n```json\n"
            '{"provider":"openrouter","status":"error","code":504,'
            '"error":"upstream timeout"}\n```',
            "Provider failed:\nOperation: openrouter. Status: error. "
            "Code: 504. Error: upstream timeout.",
        ),
        (
            "Result follows:\n{\n  \"status\": \"ok\",\n  "
            "\"path\": \"C:/proof/a.json\"\n}",
            "Result follows:\nStatus: ok. Proof: C:/proof/a.json.",
        ),
        (
            r'Provider payload: "{\"status\":\"error\",'
            r'\"error\":\"invalid request\"}"',
            "Provider payload: Status: error. Error: invalid request.",
        ),
    ],
)
def test_fenced_embedded_and_escaped_payloads_are_suppressed(raw: str, expected: str) -> None:
    assert filter_discord_text(raw) == expected


def test_plain_json_path_reference_and_non_json_prose_are_preserved() -> None:
    text = "Proof remains at C:/proof/run.json; MP4 is C:/proof/video.mp4."
    assert filter_discord_text(text) == text


def test_short_inline_json_is_suppressed() -> None:
    assert filter_discord_text('Result: {"status":"ok"}') == "Result: Status: ok."


def test_unlabeled_non_json_code_fence_is_preserved() -> None:
    text = "Example:\n```\nprint('hello')\n```"
    assert filter_discord_text(text) == text


def test_current_message_exception_is_affirmative_and_current_turn_only() -> None:
    assert current_message_requests_json("Return the raw JSON payload") is True
    assert current_message_requests_json("Do not send JSON") is False
    assert current_message_requests_json("The proof path ends in run.json") is False

    binding = begin_discord_turn("Show the response as JSON")
    copied = contextvars.copy_context()
    try:
        assert raw_json_allowed() is True
    finally:
        end_discord_turn(binding)

    assert raw_json_allowed() is False
    # A background task may copy ContextVars during the turn, but ending the
    # turn invalidates its random authorization id.
    assert copied.run(raw_json_allowed) is False


def test_explicit_json_still_redacts_secrets() -> None:
    binding = begin_discord_turn("Return raw JSON")
    try:
        result = filter_discord_text(
            '{"status":"ok","api_key":"sk-secretvalue123456",'
            '"authorization":"Bearer abcdefghijklmnop"}'
        )
    finally:
        end_discord_turn(binding)
    assert '"status":"ok"' in result
    assert "secretvalue" not in result
    assert "abcdefghijklmnop" not in result
    assert result.count("[REDACTED]") == 2


def test_summary_redacts_secrets_inside_proof_handles() -> None:
    result = filter_discord_text(
        '{"status":"ok","url":"https://proof.test/run?token=privatevalue"}'
    )
    assert "privatevalue" not in result
    assert "token=[REDACTED]" in result


def test_classifier_failure_is_fail_closed() -> None:
    with patch(
        "gateway.discord_egress.summarize_json",
        side_effect=RuntimeError("classifier unavailable"),
    ):
        result = filter_discord_text('{"status":"ok","payload":{"raw":true}}')
    assert result == "Structured JSON suppressed because it could not be safely summarized."


def test_json_attachment_is_suppressed_but_safe_files_remain_allowed() -> None:
    blocked = filter_discord_attachment("C:/proof/run.json")
    assert blocked.allowed is False
    assert blocked.replacement_text.endswith("C:/proof/run.json")
    for path in ("clip.mp4", "image.png", "report.pdf"):
        assert filter_discord_attachment(path).allowed is True


def test_explicit_json_attachment_is_validated_and_redacted(tmp_path: Path) -> None:
    proof = tmp_path / "proof.json"
    proof.write_text(
        '{"status":"ok","token":"privatevalue123"}', encoding="utf-8"
    )
    binding = begin_discord_turn("Attach the JSON file")
    try:
        decision = filter_discord_attachment(str(proof))
    finally:
        end_discord_turn(binding)
    assert decision.allowed is True
    assert decision.sanitized_bytes is not None
    assert b"privatevalue123" not in decision.sanitized_bytes
    assert b"[REDACTED]" in decision.sanitized_bytes


@pytest.mark.asyncio
async def test_live_adapter_filters_text_and_stream_edits() -> None:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = SimpleNamespace(id=101)
    edited = SimpleNamespace(edit=AsyncMock())
    channel = SimpleNamespace(
        send=AsyncMock(return_value=sent),
        fetch_message=AsyncMock(return_value=edited),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    raw = '{"operation":"job","status":"ok","proof":"run.json"}'
    send_result = await adapter.send("55", raw)
    edit_result = await adapter.edit_message("55", "101", raw, finalize=True)

    assert send_result.success is True
    assert edit_result.success is True
    delivered = channel.send.await_args.kwargs["content"]
    edited_text = edited.edit.await_args.kwargs["content"]
    assert delivered == "Operation: job. Status: ok. Proof: run.json."
    assert edited_text == delivered


@pytest.mark.asyncio
async def test_live_adapter_replaces_automatic_json_attachment(tmp_path: Path) -> None:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=202)))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    proof = tmp_path / "proof.json"
    proof.write_text('{"status":"ok"}', encoding="utf-8")

    result = await adapter.send_document("55", str(proof))

    assert result.success is True
    kwargs = channel.send.await_args.kwargs
    assert "file" not in kwargs
    assert "Structured JSON attachment suppressed" in kwargs["content"]
    assert str(proof) in kwargs["content"]


@pytest.mark.asyncio
async def test_live_adapter_allows_json_attachment_for_current_explicit_request(tmp_path: Path) -> None:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=203)))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    proof = tmp_path / "proof.json"
    proof.write_text(
        '{"status":"ok","api_key":"privatevalue123"}', encoding="utf-8"
    )

    binding = begin_discord_turn("Attach the JSON file")
    try:
        with patch(
            "plugins.platforms.discord.adapter.discord.File",
            side_effect=lambda fh, filename: SimpleNamespace(
                filename=filename, contents=fh.read()
            ),
        ):
            result = await adapter.send_document("55", str(proof))
    finally:
        end_discord_turn(binding)

    assert result.success is True
    attachment = channel.send.await_args.kwargs["file"]
    assert b"privatevalue123" not in attachment.contents
    assert b"[REDACTED]" in attachment.contents


class _FakeResponse:
    status = 200
    content = None

    def __init__(self, data: dict | None = None):
        self._data = data or {"id": "message-1"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        return self._data

    async def text(self):
        return "ok"


class _FakeSession:
    calls: list[tuple[str, dict]] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse()


@pytest.mark.asyncio
async def test_standalone_rest_sender_filters_text_and_json_files(tmp_path: Path) -> None:
    proof = tmp_path / "proof.json"
    proof.write_text('{"status":"ok"}', encoding="utf-8")
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-safe")
    _remember_channel_is_forum("55", False)
    _FakeSession.calls = []

    with patch("aiohttp.ClientSession", _FakeSession):
        result = await _standalone_send(
            SimpleNamespace(token="test-token"),
            "55",
            '{"operation":"cron","status":"ok","proof":"cron.json"}',
            media_files=[(str(proof), False), (str(pdf), False)],
        )

    assert result["success"] is True
    # One filtered text message plus one safe PDF upload; the JSON file never
    # reaches a multipart POST.
    assert len(_FakeSession.calls) == 2
    text_payload = _FakeSession.calls[0][1]["json"]["content"]
    assert "Operation: cron. Status: ok. Proof: cron.json." in text_payload
    assert "Structured JSON attachment suppressed" in text_payload
    assert str(proof) in text_payload
