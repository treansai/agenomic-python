from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from agentlock.utils import hash_payload, to_jsonable


class RedactionMode(str, Enum):
    REMOVE = "remove"
    MASK = "mask"
    HASH = "hash"


class RedactionRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    mode: RedactionMode = RedactionMode.MASK
    mask: str = "***REDACTED***"


class Redactor:
    def __init__(
        self,
        redact: list[str | RedactionRule] | None = None,
        mode: RedactionMode | str = RedactionMode.MASK,
        mask: str = "***REDACTED***",
    ) -> None:
        resolved_mode = RedactionMode(mode)
        self.rules = [
            item
            if isinstance(item, RedactionRule)
            else RedactionRule(path=item, mode=resolved_mode, mask=mask)
            for item in (redact or [])
        ]

    def redact(self, value: Any) -> Any:
        document = deepcopy(to_jsonable(value))
        for rule in self.rules:
            self._apply_rule(document, rule, rule.path.split("."))
        return document

    def _apply_rule(self, target: Any, rule: RedactionRule, path_parts: list[str]) -> None:
        if not path_parts:
            return

        part = path_parts[0]
        is_leaf = len(path_parts) == 1

        if isinstance(target, dict):
            if part == "*":
                keys = list(target.keys())
            else:
                keys = [part] if part in target else []
            for key in keys:
                if is_leaf:
                    self._replace_mapping_value(target, key, rule)
                else:
                    self._apply_rule(target[key], rule, path_parts[1:])
            return

        if isinstance(target, list):
            indices = self._resolve_list_indices(target, part)
            for index in indices:
                if is_leaf:
                    self._replace_list_value(target, index, rule)
                else:
                    self._apply_rule(target[index], rule, path_parts[1:])

    def _replace_mapping_value(self, target: dict[str, Any], key: str, rule: RedactionRule) -> None:
        if rule.mode is RedactionMode.REMOVE:
            del target[key]
            return
        target[key] = self._transform_value(target[key], rule)

    def _replace_list_value(self, target: list[Any], index: int, rule: RedactionRule) -> None:
        if rule.mode is RedactionMode.REMOVE:
            target[index] = None
            return
        target[index] = self._transform_value(target[index], rule)

    def _transform_value(self, value: Any, rule: RedactionRule) -> Any:
        if rule.mode is RedactionMode.MASK:
            return rule.mask
        if rule.mode is RedactionMode.HASH:
            return f"sha256:{hash_payload(value)}"
        return value

    def _resolve_list_indices(self, target: list[Any], part: str) -> list[int]:
        if part == "*":
            return list(range(len(target)))
        if part.isdigit():
            index = int(part)
            if 0 <= index < len(target):
                return [index]
        return []
