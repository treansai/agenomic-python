from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union, cast

from agenomic.prompts.errors import PromptRefError

MAX_VERSION = 2147483647
PROMPT_ID_PATTERN = re.compile(r"prm_[a-z0-9]+(?:[_-][a-z0-9]+)*", re.ASCII)
ALIAS_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,31}", re.ASCII)
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.ASCII)
VERSION_PATTERN = re.compile(r"[1-9][0-9]{0,9}", re.ASCII)


def is_prompt_id(value: str) -> bool:
    return len(value) <= 64 and PROMPT_ID_PATTERN.fullmatch(value) is not None


def is_uuid(value: str) -> bool:
    return UUID_PATTERN.fullmatch(value) is not None


def _version(value: str) -> Optional[int]:
    if VERSION_PATTERN.fullmatch(value) is None:
        return None
    number = int(value)
    return number if number <= MAX_VERSION else None


@dataclass(frozen=True)
class PromptVersionRef:
    prompt_id: str
    version: int

    def __str__(self) -> str:
        return f"{self.prompt_id}:{self.version}"


@dataclass(frozen=True)
class PromptAliasRef:
    prompt_id: str
    alias: str

    def __str__(self) -> str:
        return f"{self.prompt_id}@{self.alias}"


@dataclass(frozen=True)
class PromptUri:
    workspace_id: str
    prompt_id: str
    version: int

    def __str__(self) -> str:
        return f"agenomic://{self.workspace_id}/prompts/{self.prompt_id}/versions/{self.version}"

    def to_version_ref(self, workspace_id: str) -> PromptVersionRef:
        if workspace_id != self.workspace_id:
            raise _cross_workspace()
        return PromptVersionRef(self.prompt_id, self.version)


PromptReference = Union[PromptVersionRef, PromptAliasRef, PromptUri]


def _invalid(reason: str) -> PromptRefError:
    return PromptRefError(
        "prompt_ref_invalid", 0, f"invalid prompt reference: {reason}", {"reason": reason}
    )


def _cross_workspace() -> PromptRefError:
    return PromptRefError(
        "prompt_ref_cross_workspace", 0, "the prompt reference names another workspace"
    )


def _parse_uri(text: str) -> PromptUri:
    scheme, rest = text.split("://", 1)
    if scheme != "agenomic":
        raise _invalid("unsupported_scheme")
    if "?" in rest or "#" in rest:
        raise _invalid("query_or_fragment")
    if rest == "":
        raise _invalid("empty_segment")
    if rest.endswith("/"):
        raise _invalid("trailing_slash")
    segments = rest.split("/")
    if "" in segments:
        raise _invalid("empty_segment")
    if len(segments) != 5 or segments[1] != "prompts" or segments[3] != "versions":
        raise _invalid("invalid_uri_path")
    if not is_uuid(segments[0]):
        raise _invalid("invalid_workspace")
    if not is_prompt_id(segments[2]):
        raise _invalid("invalid_prompt_id")
    version = _version(segments[4])
    if version is None:
        raise _invalid("invalid_version")
    return PromptUri(segments[0], segments[2], version)


def _parse(text: str) -> Union[PromptReference, str]:
    if text == "":
        raise _invalid("empty_segment")
    if len(text) > 256:
        raise _invalid("too_long")
    for char in text:
        code = ord(char)
        if code < 0x21 or code > 0x7E:
            if 0x09 <= code <= 0x0D or code == 0x20:
                raise _invalid("whitespace")
            raise _invalid("invalid_character")
    if any("A" <= char <= "Z" for char in text):
        raise _invalid("uppercase")
    if "%" in text:
        raise _invalid("percent_encoded_separator")
    if "://" in text:
        return _parse_uri(text)
    if ":" in text and "@" in text:
        raise _invalid("mixed_form")
    if ":" in text:
        prompt_id, raw_version = text.split(":", 1)
        if prompt_id == "" or raw_version == "":
            raise _invalid("empty_segment")
        if not is_prompt_id(prompt_id):
            raise _invalid("invalid_prompt_id")
        version = _version(raw_version)
        if version is None:
            raise _invalid("invalid_version")
        return PromptVersionRef(prompt_id, version)
    if "@" in text:
        prompt_id, alias = text.split("@", 1)
        if prompt_id == "" or alias == "":
            raise _invalid("empty_segment")
        if not is_prompt_id(prompt_id):
            raise _invalid("invalid_prompt_id")
        if ALIAS_PATTERN.fullmatch(alias) is None:
            raise _invalid("invalid_alias")
        return PromptAliasRef(prompt_id, alias)
    if not is_prompt_id(text):
        raise _invalid("invalid_prompt_id")
    return text


def parse_prompt_ref(
    text: str, *, workspace_id: Optional[str] = None, allow_bare_id: bool = False
) -> Union[PromptReference, str]:
    ref = _parse(text)
    if isinstance(ref, PromptUri) and workspace_id is not None and ref.workspace_id != workspace_id:
        raise _cross_workspace()
    if isinstance(ref, str) and not allow_bare_id:
        raise PromptRefError(
            "prompt_ref_unversioned",
            0,
            "a version or an alias is required; there is no implicit latest",
        )
    return ref


def parse_execution_ref(text: str, *, workspace_id: Optional[str] = None) -> PromptReference:
    return cast(PromptReference, parse_prompt_ref(text, workspace_id=workspace_id))
