"""Fail-closed JSON filtering for Discord's final transport boundary.

The gateway pre-delivery hook remains the first policy layer.  This module is
the last layer used by the Discord adapter and its standalone REST fallback,
so non-agent origins (cron, provider/tool notices, background completions, and
automatic file extraction) receive the same treatment.
"""

from __future__ import annotations

import contextvars
import json
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TURN_TOKEN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "discord_json_egress_turn", default=None
)
_ACTIVE_EXPLICIT_TURNS: set[str] = set()

_JSON_WORD_RE = re.compile(r"\bjson\b", re.IGNORECASE)
_JSON_REQUEST_RE = re.compile(
    r"(?:"
    r"\braw\s+json\b|"
    r"\bjson\s+(?:output|payload|body|response|attachment|file|dump)\b|"
    r"\b(?:return|reply|respond|show|print|send|include|attach|give|provide|emit|dump)"
    r"\b[\s\S]{0,48}\bjson\b|"
    r"\bas\s+(?:raw\s+)?json\b"
    r")",
    re.IGNORECASE,
)
_NEGATED_JSON_RE = re.compile(
    r"(?:do\s+not|don't|dont|without|avoid|never|no)\b[\s\S]{0,32}\bjson\b",
    re.IGNORECASE,
)
_JSON_FENCE_RE = re.compile(
    r"```(?P<label>json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```",
    re.IGNORECASE,
)
_QUOTED_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_SENSITIVE_JSON_VALUE_RE = re.compile(
    r'(?P<prefix>["\'](?:token|access_token|refresh_token|api[_-]?key|secret|password|authorization)'
    r'["\']\s*:\s*["\'])(?P<value>[^"\']*)(?P<suffix>["\'])',
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>\b(?:token|access_token|refresh_token|api[_-]?key|secret|password|authorization)"
    r"\s*[=:]\s*)(?P<value>[^&\s,;]+)",
    re.IGNORECASE,
)
_LIKELY_SECRET_RE = re.compile(
    r"\b(?:sk|xox[baprs]|gh[pousr])[-_][A-Za-z0-9_-]{12,}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DiscordTurnBinding:
    context_token: contextvars.Token
    authorization_id: str | None


@dataclass(frozen=True)
class AttachmentDecision:
    allowed: bool
    replacement_text: str = ""
    sanitized_bytes: bytes | None = None


def current_message_requests_json(message: str | None) -> bool:
    """Return True only for an affirmative JSON request in this message."""
    text = str(message or "")
    if not _JSON_WORD_RE.search(text):
        return False
    affirmative = _JSON_REQUEST_RE.search(text)
    if not affirmative:
        return False
    # A nearby explicit negation wins over broad phrases such as "send JSON".
    for negated in _NEGATED_JSON_RE.finditer(text):
        if negated.start() <= affirmative.end() and negated.end() >= affirmative.start() - 48:
            return False
    return True


def begin_discord_turn(message: str | None) -> DiscordTurnBinding:
    """Bind current-turn JSON intent without retaining the user's message.

    Background tasks copy ContextVars.  The separate active-token set is
    therefore intentional: ending the parent turn invalidates copied tokens,
    so a later process completion cannot inherit an earlier JSON exception.
    """
    authorization_id = None
    if current_message_requests_json(message):
        authorization_id = secrets.token_hex(16)
        _ACTIVE_EXPLICIT_TURNS.add(authorization_id)
    token = _TURN_TOKEN.set(authorization_id)
    return DiscordTurnBinding(token, authorization_id)


def end_discord_turn(binding: DiscordTurnBinding | None) -> None:
    if binding is None:
        return
    if binding.authorization_id:
        _ACTIVE_EXPLICIT_TURNS.discard(binding.authorization_id)
    _TURN_TOKEN.reset(binding.context_token)


def raw_json_allowed() -> bool:
    authorization_id = _TURN_TOKEN.get()
    return bool(
        authorization_id and authorization_id in _ACTIVE_EXPLICIT_TURNS
    )


def redact_sensitive_text(text: str | None) -> str:
    """Redact credential-shaped values even under an explicit JSON exception."""
    value = str(text or "")
    value = _SENSITIVE_JSON_VALUE_RE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]{match.group('suffix')}",
        value,
    )
    value = _BEARER_RE.sub("Bearer [REDACTED]", value)
    value = _SENSITIVE_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]", value
    )
    return _LIKELY_SECRET_RE.sub("[REDACTED]", value)


def _nested_json(value: Any) -> Any:
    current = value
    for _ in range(3):
        if not isinstance(current, str):
            break
        stripped = current.strip()
        if not stripped or stripped[0] not in '{["':
            break
        try:
            decoded = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            break
        if decoded == current:
            break
        current = decoded
    return current


def _walk_mappings(value: Any) -> Iterable[dict[str, Any]]:
    queue: list[Any] = [value]
    seen = 0
    while queue and seen < 200:
        item = queue.pop(0)
        seen += 1
        if isinstance(item, dict):
            yield item
            queue.extend(item.values())
        elif isinstance(item, list):
            queue.extend(item[:50])


def _first_field(value: Any, keys: tuple[str, ...]) -> Any:
    wanted = {key.lower() for key in keys}
    for mapping in _walk_mappings(value):
        for key, field_value in mapping.items():
            if str(key).lower() in wanted and field_value not in (None, "", [], {}):
                return field_value
    return None


def _safe_leaf(value: Any, limit: int = 220) -> str:
    if isinstance(value, (dict, list)):
        return ""
    text = redact_sensitive_text("" if value is None else str(value)).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def summarize_json(value: Any) -> str:
    """Render a compact human summary from an object/array allowlist."""
    value = _nested_json(value)
    if not isinstance(value, (dict, list)):
        raise ValueError("structured JSON root must be an object or array")

    operation = _safe_leaf(_first_field(value, ("operation", "action", "tool", "job", "provider", "name")))
    success_value = _first_field(value, ("success", "ok"))
    status = _safe_leaf(_first_field(value, ("status", "state", "result")))
    exit_code = _safe_leaf(_first_field(value, ("exit_code", "status_code", "http_status", "code")))
    error = _safe_leaf(_first_field(value, ("safe_error", "error", "message", "detail", "reason")))
    proof = _safe_leaf(_first_field(value, ("proof", "proof_path", "artifact", "artifact_path", "path", "message_id", "url")))

    parts: list[str] = []
    if operation:
        parts.append(f"Operation: {operation}.")
    if isinstance(success_value, bool):
        parts.append("Result: success." if success_value else "Result: failed.")
    elif status:
        parts.append(f"Status: {status}.")
    elif isinstance(value, list):
        parts.append(f"Structured result: {len(value)} item(s).")
    else:
        parts.append(f"Structured result: {len(value)} field(s).")
    if exit_code:
        parts.append(f"Code: {exit_code}.")
    if error and error.lower() not in {status.lower(), operation.lower()}:
        parts.append(f"Error: {error}.")
    if proof:
        parts.append(f"Proof: {proof}.")
    return " ".join(parts)


def _decode_whole(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = _nested_json(json.loads(stripped))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, (dict, list)) else None


def _replace_json_fences(text: str) -> tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        value = _decode_whole(match.group("body"))
        if value is None:
            if match.group("label"):
                changed = True
                return "Structured JSON suppressed because it could not be safely summarized."
            return match.group(0)
        changed = True
        return summarize_json(value)

    return _JSON_FENCE_RE.sub(replace, text), changed


def _embedded_candidates(text: str) -> list[tuple[int, int, Any]]:
    decoder = json.JSONDecoder()
    found: list[tuple[int, int, Any]] = []
    occupied_until = -1
    for index, char in enumerate(text):
        if index < occupied_until or char not in "[{":
            continue
        try:
            value, length = decoder.raw_decode(text[index:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        end = index + length
        if isinstance(value, (dict, list)):
            found.append((index, end, value))
            occupied_until = end

    for match in _QUOTED_STRING_RE.finditer(text):
        if any(start <= match.start() < end for start, end, _ in found):
            continue
        try:
            decoded = _nested_json(json.loads(match.group(0)))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(decoded, (dict, list)):
            found.append((match.start(), match.end(), decoded))
    return sorted(found, key=lambda item: item[0])


def filter_discord_text(text: str | None) -> str:
    """Apply the final Discord text policy, preserving ordinary prose/paths."""
    original = str(text or "")
    if raw_json_allowed():
        return redact_sensitive_text(original)
    try:
        whole = _decode_whole(original)
        if whole is not None:
            return summarize_json(whole)

        filtered, fenced = _replace_json_fences(original)
        candidates = _embedded_candidates(filtered)
        if candidates:
            chunks: list[str] = []
            cursor = 0
            for start, end, value in candidates:
                if start < cursor:
                    continue
                chunks.append(filtered[cursor:start])
                chunks.append(summarize_json(value))
                cursor = end
            chunks.append(filtered[cursor:])
            filtered = "".join(chunks)
        if fenced or candidates:
            return redact_sensitive_text(filtered).strip()
        return redact_sensitive_text(original)
    except Exception:
        return "Structured JSON suppressed because it could not be safely summarized."


def filter_discord_attachment(file_path: str, caption: str | None = None) -> AttachmentDecision:
    """Suppress automatic JSON files while preserving safe media/documents."""
    path = str(file_path or "")
    if Path(path).suffix.lower() != ".json":
        return AttachmentDecision(True)
    if raw_json_allowed():
        try:
            raw = Path(path).read_text(encoding="utf-8")
            json.loads(raw)
            return AttachmentDecision(
                True,
                sanitized_bytes=redact_sensitive_text(raw).encode("utf-8"),
            )
        except Exception:
            return AttachmentDecision(
                False,
                "Structured JSON attachment suppressed because it could not be safely sanitized.",
            )
    replacement = "Structured JSON attachment suppressed."
    if path:
        replacement += f" Proof: {path}"
    if caption:
        safe_caption = filter_discord_text(caption)
        if safe_caption:
            replacement = f"{safe_caption}\n{replacement}"
    return AttachmentDecision(False, replacement)
