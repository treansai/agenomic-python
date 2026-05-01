"""Redaction rule definitions."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator


class RedactionMode(str, Enum):
    """How a matched leaf is rewritten."""

    REMOVE = "remove"
    MASK = "mask"
    HASH = "hash"
    TRUNCATE = "truncate"


class RedactionRule(BaseModel):
    """Dotted-path redaction rule with optional wildcards.

    Path syntax:
        - ``a.b.c``  — exact path
        - ``a.*.c``  — single wildcard segment (any key/index)
        - ``a.**.c`` — recursive wildcard (any depth, including zero)

    Example:
        >>> RedactionRule(path="input.user.email", mode=RedactionMode.MASK).path
        'input.user.email'
    """

    model_config = ConfigDict(frozen=True)

    path: str
    mode: RedactionMode
    truncate_length: Optional[int] = None

    @model_validator(mode="after")
    def _validate_truncate(self) -> RedactionRule:
        if self.mode is RedactionMode.TRUNCATE:
            if self.truncate_length is None:
                raise ValueError("truncate_length is required when mode=TRUNCATE")
            if self.truncate_length < 0:
                raise ValueError("truncate_length must be >= 0")
        return self
