from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SECRET_PATTERN_SET = "agenomic-secrets/1"
REDACTED = "[REDACTED]"

_B = r"\b"
_WS = r"[\t\n\x0b\x0c\r ]"

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "bearer_token",
        re.compile(_B + r"bearer" + _WS + r"+[A-Za-z0-9\-_.=+/]{20,}", re.ASCII | re.IGNORECASE),
    ),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.ASCII)),
    ("openai_key", re.compile(_B + r"sk-[A-Za-z0-9\-_]{20,}", re.ASCII)),
    ("stripe_key", re.compile(_B + r"[sr]k_(?:live|test)_[A-Za-z0-9]{16,}", re.ASCII)),
    ("aws_access_key", re.compile(_B + r"AKIA[0-9A-Z]{16}" + _B, re.ASCII)),
    ("github_token", re.compile(_B + r"gh[pousr]_[A-Za-z0-9]{36,}", re.ASCII)),
    ("huggingface_token", re.compile(_B + r"hf_[A-Za-z0-9]{20,}", re.ASCII)),
    ("slack_token", re.compile(_B + r"xox[baprs]-[A-Za-z0-9\-]{10,}", re.ASCII)),
    ("google_api_key", re.compile(_B + r"AIza[0-9A-Za-z\-_]{35}", re.ASCII)),
    (
        "jwt",
        re.compile(
            _B + r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}", re.ASCII
        ),
    ),
    ("agenomic_api_key", re.compile(_B + r"agm_[A-Za-z0-9]{24,}", re.ASCII)),
)

_TOKEN_WORDS = frozenset(
    {
        "secret",
        "password",
        "passwd",
        "token",
        "credential",
        "credentials",
        "authorization",
        "cookie",
    }
)
_COLLAPSED_SUFFIXES = (
    "apikey",
    "secret",
    "password",
    "passwd",
    "token",
    "credential",
    "credentials",
    "authorization",
    "cookie",
    "privatekey",
    "accesskey",
    "clientsecret",
)
_NORMALIZED_SUFFIXES = ("_api_key", "_private_key", "_access_key", "_client_secret")


@dataclass(frozen=True)
class SecretFinding:
    pattern: str
    offset: int
    length: int

    def to_dict(self) -> dict[str, Any]:
        return {"pattern": self.pattern, "offset": self.offset, "length": self.length}


def scan(text: str) -> list[SecretFinding]:
    found: list[tuple[int, int, SecretFinding]] = []
    for order, (pattern_id, pattern) in enumerate(SECRET_PATTERNS):
        for match in pattern.finditer(text):
            finding = SecretFinding(pattern_id, match.start(), match.end() - match.start())
            found.append((match.start(), order, finding))
    found.sort(key=lambda item: (item[0], item[1]))
    return [finding for _, _, finding in found]


def scrub(text: str) -> str:
    spans: list[list[Any]] = []
    for finding in scan(text):
        end = finding.offset + finding.length
        if spans and finding.offset <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([finding.offset, end, finding.pattern])
    parts: list[str] = []
    position = 0
    for start, end, pattern_id in spans:
        parts.append(text[position:start] + f"[REDACTED:{pattern_id}]")
        position = end
    parts.append(text[position:])
    return "".join(parts)


def is_secret_shaped_key(key: str) -> bool:
    normalized = "".join(char.lower() if char.isascii() and char.isalnum() else "_" for char in key)
    tokens = [token for token in normalized.split("_") if token]
    collapsed = "".join(tokens)
    return (
        any(token in _TOKEN_WORDS for token in tokens)
        or any(collapsed.endswith(suffix) for suffix in _COLLAPSED_SUFFIXES)
        or normalized == "apikey"
        or normalized.endswith(_NORMALIZED_SUFFIXES)
    )


def scrub_json(value: Any) -> Any:
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, (list, tuple)):
        return [scrub_json(item) for item in value]
    if isinstance(value, Mapping):
        return {
            key: REDACTED if is_secret_shaped_key(str(key)) else scrub_json(item)
            for key, item in value.items()
        }
    return value
