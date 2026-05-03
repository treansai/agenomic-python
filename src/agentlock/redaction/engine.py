"""Redaction engine — apply rules to nested dict/list structures."""

from __future__ import annotations

import copy
import json
from typing import Any

from agentlock.crypto.hashing import blake3_hex
from agentlock.exceptions import RedactionError
from agentlock.redaction.rules import RedactionMode, RedactionRule

_REMOVED = object()  # sentinel for "delete this entry"


class RedactionEngine:
    """Apply redaction rules to a nested dict/list structure.

    Path syntax:
        - ``a.b.c``  — exact path
        - ``a.*.c``  — single wildcard segment (any key/index)
        - ``a.**.c`` — recursive wildcard (any depth, including zero)

    Always operates on a deep copy. Unknown paths are silently skipped.

    Example:
        >>> e = RedactionEngine([RedactionRule(path="user.email", mode=RedactionMode.MASK)])
        >>> e.apply({"user": {"email": "a@b.c"}})
        {'user': {'email': '***'}}
    """

    def __init__(self, rules: list[RedactionRule]) -> None:
        self.rules = list(rules)

    def apply(self, data: Any) -> Any:
        """Return a deep-copied, redacted version of `data`."""
        out = copy.deepcopy(data)
        for rule in self.rules:
            segments = rule.path.split(".")
            out = self._apply_at(out, segments, rule)
        return out

    def _apply_at(self, node: Any, segments: list[str], rule: RedactionRule) -> Any:
        if not segments:
            return self._rewrite_leaf(node, rule)

        head, *rest = segments

        if head == "**":
            # Recursive wildcard: try matching `rest` here, and at every deeper level.
            node = self._apply_at(node, rest, rule)
            if isinstance(node, dict):
                for k in list(node.keys()):
                    node[k] = self._apply_at(node[k], segments, rule)
            elif isinstance(node, list):
                for i in range(len(node)):
                    node[i] = self._apply_at(node[i], segments, rule)
            return node

        if head == "*":
            if isinstance(node, dict):
                for k in list(node.keys()):
                    new_val = self._apply_at(node[k], rest, rule)
                    if new_val is _REMOVED:
                        del node[k]
                    else:
                        node[k] = new_val
            elif isinstance(node, list):
                new_list: list[Any] = []
                for v in node:
                    new_val = self._apply_at(v, rest, rule)
                    if new_val is not _REMOVED:
                        new_list.append(new_val)
                node[:] = new_list
            return node

        # Exact key segment
        if isinstance(node, dict):
            if head in node:
                new_val = self._apply_at(node[head], rest, rule)
                if new_val is _REMOVED:
                    del node[head]
                else:
                    node[head] = new_val
            return node
        if isinstance(node, list):
            try:
                idx = int(head)
            except ValueError:
                return node
            if 0 <= idx < len(node):
                new_val = self._apply_at(node[idx], rest, rule)
                if new_val is _REMOVED:
                    del node[idx]
                else:
                    node[idx] = new_val
            return node
        return node

    @staticmethod
    def _rewrite_leaf(value: Any, rule: RedactionRule) -> Any:
        if rule.mode is RedactionMode.REMOVE:
            return _REMOVED
        if rule.mode is RedactionMode.MASK:
            return "***"
        if rule.mode is RedactionMode.HASH:
            try:
                blob = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
            except (TypeError, ValueError) as e:
                raise RedactionError(f"cannot hash value: {e}") from e
            return f"hash:{blake3_hex(blob)[:16]}"
        if rule.mode is RedactionMode.TRUNCATE:
            if not isinstance(value, str):
                return value
            limit = rule.truncate_length or 0
            return value[:limit]
        return value
