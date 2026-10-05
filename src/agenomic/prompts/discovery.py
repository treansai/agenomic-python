from __future__ import annotations

import ast
import fnmatch
import hashlib
import inspect
import io
import os
import re
import textwrap
import tokenize
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Union

from agenomic._version import __version__
from agenomic.prompts.digest import canonical_json_v1, pointer
from agenomic.prompts.importer import (
    DISCOVERY_SCHEMA,
    ISSUE_MESSAGES,
    ISSUE_SEVERITY,
    SourceEntry,
    SourcePartial,
    SourcePrompt,
    convert_prompt,
)
from agenomic.prompts.secrets import SECRET_PATTERN_SET, scan

__all__ = ["DEFAULT_EXCLUDES", "scan_paths"]

DEFAULT_EXCLUDES = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    ".tox",
    ".nox",
    "__pycache__",
    "node_modules",
    "site-packages",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
)
SCANNER_NAME = "agenomic-python"
PYTHON_GRAMMAR = "3.10"

Pos = tuple[int, int]
Key = tuple[str, str]

_ROOTS = frozenset({"langchain", "langchain_core", "langchain_community", "langchain_classic"})
_GRAPH_ROOTS = frozenset({"langgraph"})
_TEXT_CLASSES = frozenset({"PromptTemplate"})
_CHAT_CLASSES = frozenset({"ChatPromptTemplate"})
_REFUSED_TEXT = frozenset(
    {"FewShotPromptTemplate", "PipelinePromptTemplate", "ImagePromptTemplate", "DictPromptTemplate"}
)
_REFUSED_CHAT = frozenset({"FewShotChatMessagePromptTemplate", "StructuredPrompt"})
_TEMPLATE_ROLES = {
    "SystemMessagePromptTemplate": "system",
    "HumanMessagePromptTemplate": "user",
    "AIMessagePromptTemplate": "assistant",
}
_STATIC_ROLES = {"SystemMessage": "system", "HumanMessage": "user", "AIMessage": "assistant"}
_REFUSED_MESSAGES = frozenset(
    {"ToolMessage", "FunctionMessage", "ChatMessage", "ChatMessagePromptTemplate"}
)
_AGENTS = {
    "create_react_agent": ("langgraph.create_react_agent.prompt", "prompt"),
    "create_agent": ("langchain.create_agent.system_prompt", "system_prompt"),
}
_PROMPT_CLASSES = _TEXT_CLASSES | _CHAT_CLASSES | _REFUSED_TEXT | _REFUSED_CHAT
_MESSAGE_CLASSES = (
    frozenset(_TEMPLATE_ROLES)
    | frozenset(_STATIC_ROLES)
    | _REFUSED_MESSAGES
    | frozenset({"MessagesPlaceholder"})
)
_KNOWN = _PROMPT_CLASSES | _MESSAGE_CLASSES | frozenset(_AGENTS) | frozenset({"StateGraph"})
_CHAT_ROLE_NAMES = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "system": "system",
}
_PROMPT_NAME = re.compile(r"(?:^|_)(?:prompt|template|instructions?|system_message)$", re.I)
_NAME_SUFFIXES = (
    "_system_prompt",
    "_system_message",
    "_prompt_template",
    "_prompt",
    "_template",
    "_instructions",
    "_instruction",
    "_message",
    "_system",
)
_USAGE_WORDS = frozenset(
    {"system", "user", "human", "prompt", "template", "instructions", "instruction", "message"}
)
_SLOT_SEGMENT = re.compile(r"[a-z][a-z0-9_]*", re.ASCII)
_SLOT_PATH = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", re.ASCII)
_HEX40 = re.compile(r"[0-9a-f]{40}", re.ASCII)
_SIMPLE_ESCAPES = {
    "\\": "\\",
    "'": "'",
    '"': '"',
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}


@dataclass(frozen=True)
class _Text:
    value: str
    start: Pos
    chars: Optional[tuple[Pos, ...]] = None
    names: frozenset[Key] = frozenset()
    transforms: tuple[Pos, ...] = ()

    def plus(self, other: _Text) -> _Text:
        chars = (
            self.chars + other.chars if self.chars is not None and other.chars is not None else None
        )
        return _Text(
            self.value + other.value,
            self.start,
            chars,
            self.names | other.names,
            self.transforms + other.transforms,
        )

    def named(self, key: Key) -> _Text:
        return _Text(self.value, self.start, self.chars, self.names | {key}, self.transforms)

    def transformed(self, value: str, chars: Optional[tuple[Pos, ...]], at: Pos) -> _Text:
        return _Text(value, self.start, chars, self.names, (*self.transforms, at))

    def locate(self, offset: int) -> Pos:
        if self.chars is not None and 0 <= offset < len(self.chars):
            return self.chars[offset]
        return self.start


@dataclass(frozen=True)
class _Dynamic:
    code: str
    start: Pos
    texts: tuple[_Text, ...] = ()


Value = Union[_Text, _Dynamic]


@dataclass
class _Draft:
    rel: str
    construct: str
    start: Pos
    end: Pos
    symbol: Optional[str]
    function: Optional[str]
    usage: str
    prompt: Optional[SourcePrompt] = None
    dynamic: Optional[_Dynamic] = None
    texts: dict[str, _Text] = field(default_factory=dict)
    consumed: set[Key] = field(default_factory=set)
    notes: list[tuple[str, Pos]] = field(default_factory=list)
    key: Optional[Key] = None
    node_paths: set[str] = field(default_factory=set)
    node_unresolved: bool = False
    subagent: Optional[str] = None
    hint: Optional[str] = None


@dataclass
class _Parts:
    prompt: Optional[SourcePrompt] = None
    dynamic: Optional[_Dynamic] = None
    texts: dict[str, _Text] = field(default_factory=dict)
    consumed: set[Key] = field(default_factory=set)
    notes: list[tuple[str, Pos]] = field(default_factory=list)


@dataclass
class _Entry:
    entry: Optional[SourceEntry] = None
    text: Optional[_Text] = None
    dynamic: Optional[_Dynamic] = None
    consumed: set[Key] = field(default_factory=set)


@dataclass
class _Agent:
    rel: str
    drafts: list[_Draft] = field(default_factory=list)
    names: set[Key] = field(default_factory=set)


def _kw(call: ast.Call, name: str) -> Optional[ast.expr]:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _arg(call: ast.Call, index: int, name: str) -> Optional[ast.expr]:
    if len(call.args) > index and not isinstance(call.args[index], ast.Starred):
        return call.args[index]
    return _kw(call, name)


def _is_none(node: Optional[ast.expr]) -> bool:
    return node is None or (isinstance(node, ast.Constant) and node.value is None)


def _empty_literal(node: Optional[ast.expr]) -> bool:
    if _is_none(node):
        return True
    if isinstance(node, (ast.Dict, ast.List, ast.Tuple)):
        return not (node.keys if isinstance(node, ast.Dict) else node.elts)
    return False


def _decode_token(token: str, line: int, col: int) -> Optional[tuple[str, list[Pos]]]:
    index = 0
    while index < len(token) and token[index].isalpha():
        index += 1
    prefix = token[:index].lower()
    if "b" in prefix or "f" in prefix:
        return None
    raw = "r" in prefix
    rest = token[index:]
    quote = rest[:3] if rest[:3] in ('"""', "'''") else rest[:1]
    if len(rest) < 2 * len(quote):
        return None
    body = rest[len(quote) : len(rest) - len(quote)]
    cursor = [line, col + index + len(quote)]
    out: list[str] = []
    positions: list[Pos] = []

    def step(chars: str) -> None:
        for char in chars:
            if char == "\n":
                cursor[0] += 1
                cursor[1] = 0
            else:
                cursor[1] += 1

    position = 0
    while position < len(body):
        here = (cursor[0], cursor[1] + 1)
        char = body[position]
        if char != "\\" or raw or position + 1 >= len(body):
            out.append(char)
            positions.append(here)
            step(char)
            position += 1
            continue
        following = body[position + 1]
        width = 2
        produced: Optional[str]
        if following == "\n":
            produced = ""
        elif following in _SIMPLE_ESCAPES:
            produced = _SIMPLE_ESCAPES[following]
        elif following in "01234567":
            end = position + 1
            while end < len(body) and end < position + 4 and body[end] in "01234567":
                end += 1
            produced = chr(int(body[position + 1 : end], 8))
            width = end - position
        elif following in "xuU":
            size = {"x": 2, "u": 4, "U": 8}[following]
            digits = body[position + 2 : position + 2 + size]
            if len(digits) != size or not all(c in "0123456789abcdefABCDEF" for c in digits):
                return None
            produced = chr(int(digits, 16))
            width = 2 + size
        elif following == "N":
            close = body.find("}", position)
            if body[position + 2 : position + 3] != "{" or close < 0:
                return None
            try:
                produced = unicodedata.lookup(body[position + 3 : close])
            except KeyError:
                return None
            width = close + 1 - position
        else:
            produced = "\\"
            width = 1
        for item in produced:
            out.append(item)
            positions.append(here)
        step(body[position : position + width])
        position += width
    return "".join(out), positions


class _Module:
    def __init__(self, rel: str, text: str, tree: ast.Module) -> None:
        self.rel = rel
        self.text = text
        self.tree = tree
        self.lines = text.split("\n")
        self.imports: dict[str, tuple[str, str]] = {}
        self.modules: dict[str, str] = {}
        self.constants: dict[str, ast.expr] = {}
        self.functions: dict[str, ast.AST] = {}
        self.methods: dict[str, list[ast.AST]] = {}
        parts = rel[: -len(".py")].split("/")
        self.package = parts[:-1]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        self.dotted = ".".join(parts)
        self._index()

    def _index(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.modules[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        self.modules[head] = head
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    if node.level - 1 > len(self.package):
                        continue
                    anchor = self.package[: len(self.package) - (node.level - 1)]
                    base = ".".join([*anchor, *([base] if base else [])])
                for alias in node.names:
                    if alias.name != "*":
                        self.imports[alias.asname or alias.name] = (base, alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.methods.setdefault(node.name, []).append(node)
        counts: Counter[str] = Counter()
        rebound: set[str] = set()
        for node in self._module_scope(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                counts[node.id] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                counts[node.name] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    counts[alias.asname or alias.name.split(".")[0]] += 1
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                rebound.update(node.names)
        for statement in self.tree.body:
            target: Optional[ast.expr] = None
            value: Optional[ast.expr] = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value = statement.targets[0], statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                target, value = statement.target, statement.value
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if counts[statement.name] == 1:
                    self.functions[statement.name] = statement
                continue
            if not isinstance(target, ast.Name) or value is None:
                continue
            if counts[target.id] != 1 or target.id in rebound:
                continue
            if isinstance(value, ast.Lambda):
                self.functions[target.id] = value
            else:
                self.constants[target.id] = value

    def _module_scope(self, node: ast.AST) -> Iterator[ast.AST]:
        for child in ast.iter_child_nodes(node):
            yield child
            if isinstance(
                child,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Lambda,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                ),
            ):
                continue
            yield from self._module_scope(child)

    def column(self, line: int, offset: int) -> int:
        if line < 1 or line > len(self.lines):
            return offset
        encoded = self.lines[line - 1].encode("utf-8")
        return len(encoded[:offset].decode("utf-8", errors="ignore"))

    def pos(self, node: ast.AST) -> Pos:
        line = int(getattr(node, "lineno", 1))
        return line, self.column(line, int(getattr(node, "col_offset", 0))) + 1

    def end(self, node: ast.AST) -> Pos:
        line: int = getattr(node, "end_lineno", None) or int(getattr(node, "lineno", 1))
        offset = getattr(node, "end_col_offset", None)
        if offset is None:
            return self.pos(node)
        return line, max(self.column(line, offset), 1)

    def qualname(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            if node.id in self.imports:
                module, attr = self.imports[node.id]
                return f"{module}.{attr}" if module else attr
            return self.modules.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.qualname(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def callee(self, node: ast.AST) -> Optional[tuple[str, Optional[str]]]:
        qualified = self.qualname(node)
        if qualified is None:
            return None
        parts = qualified.split(".")
        if parts[0] not in _ROOTS and parts[0] not in _GRAPH_ROOTS:
            return None
        if len(parts) > 1 and parts[-2] == "hub" and parts[-1] == "pull":
            return "hub.pull", None
        if parts[-1] in _KNOWN:
            return parts[-1], None
        if len(parts) > 1 and parts[-2] in _KNOWN:
            return parts[-2], parts[-1]
        return None

    def literal(self, node: ast.Constant) -> _Text:
        value = node.value
        start = self.pos(node)
        segment = ast.get_source_segment(self.text, node)
        if not isinstance(value, str) or segment is None:
            return _Text(str(value), start)
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO("(" + segment + ")").readline))
        except (tokenize.TokenError, SyntaxError):
            return _Text(value, start)
        pieces: list[str] = []
        positions: list[Pos] = []
        for token in tokens:
            if token.type != tokenize.STRING:
                continue
            row, offset = token.start
            line = start[0] + row - 1
            col = start[1] - 1 + offset - 1 if row == 1 else offset
            decoded = _decode_token(token.string, line, col)
            if decoded is None:
                return _Text(value, start)
            pieces.append(decoded[0])
            positions.extend(decoded[1])
        if "".join(pieces) != value or len(positions) != len(value):
            return _Text(value, start)
        return _Text(value, start, tuple(positions))


class _Scanner:
    def __init__(self) -> None:
        self.modules: dict[str, _Module] = {}
        self.by_dotted: dict[str, _Module] = {}
        self.drafts: list[_Draft] = []
        self.consumed: set[Key] = set()
        self.agents: dict[int, _Agent] = {}
        self.add_nodes: list[tuple[_Module, ast.Call, tuple[str, ...]]] = []

    def register(self, module: _Module) -> None:
        self.modules[module.rel] = module
        self.by_dotted.setdefault(module.dotted, module)
        if module.dotted.startswith("src."):
            self.by_dotted.setdefault(module.dotted[len("src.") :], module)

    def resolve(
        self, module: _Module, name: str, depth: int = 0
    ) -> Optional[tuple[_Module, ast.expr, str]]:
        if depth > 8:
            return None
        if name in module.constants:
            return module, module.constants[name], name
        if name in module.imports:
            source, attr = module.imports[name]
            target = self.by_dotted.get(source)
            if target is not None:
                return self.resolve(target, attr, depth + 1)
        return None

    def function(self, module: _Module, name: str) -> Optional[tuple[_Module, ast.AST]]:
        if name in module.functions:
            return module, module.functions[name]
        if name in module.imports:
            source, attr = module.imports[name]
            target = self.by_dotted.get(source)
            if target is not None and attr in target.functions:
                return target, target.functions[attr]
        return None

    def classify(self, module: _Module, call: ast.Call) -> Optional[tuple[str, str, Optional[str]]]:
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "partial":
            inner = func.value
            if isinstance(inner, ast.Call):
                found = self.classify(module, inner)
                if found is not None and found[0] == "prompt":
                    return "partial", found[1], found[2]
            return None
        callee = module.callee(func)
        if callee is None:
            return None
        name, method = callee
        if name == "hub.pull":
            return "hub", name, None
        if name in _PROMPT_CLASSES:
            return "prompt", name, method
        if name in _MESSAGE_CLASSES:
            return "message", name, method
        if name in _AGENTS and method is None:
            return "agent", name, None
        if name == "StateGraph" and method is None:
            return "graph", name, None
        return None

    def is_template(self, module: _Module, node: ast.expr, stack: frozenset[Key]) -> bool:
        if isinstance(node, ast.Call):
            found = self.classify(module, node)
            return found is not None and found[0] in ("prompt", "partial", "message")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.is_template(module, node.left, stack) or self.is_template(
                module, node.right, stack
            )
        if isinstance(node, ast.Name):
            resolved = self.resolve(module, node.id)
            if resolved is not None and (resolved[0].rel, resolved[2]) not in stack:
                key = (resolved[0].rel, resolved[2])
                return self.is_template(resolved[0], resolved[1], stack | {key})
        return False

    def _dynamic(self, code: str, start: Pos, *values: Value) -> _Dynamic:
        texts: list[_Text] = []
        for value in values:
            if isinstance(value, _Text):
                texts.append(value)
            else:
                texts.extend(value.texts)
        for value in values:
            if isinstance(value, _Dynamic):
                code = value.code
                break
        return _Dynamic(code, start, tuple(texts))

    def text(self, module: _Module, node: ast.expr, stack: frozenset[Key] = frozenset()) -> Value:
        start = module.pos(node)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return module.literal(node)
            return _Dynamic("dynamic_template", start)
        if isinstance(node, ast.JoinedStr):
            pieces = [
                _Text(part.value, start)
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            ]
            if len(pieces) == len(node.values):
                return _Text("".join(piece.value for piece in pieces), start)
            return _Dynamic("python_fstring", start, tuple(pieces))
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                if self.is_template(module, node.left, stack) or self.is_template(
                    module, node.right, stack
                ):
                    return _Dynamic("template_composition", start)
                left = self.text(module, node.left, stack)
                right = self.text(module, node.right, stack)
                if isinstance(left, _Text) and isinstance(right, _Text):
                    return left.plus(right)
                return self._dynamic("dynamic_template", start, left, right)
            if isinstance(node.op, ast.Mod):
                left = self.text(module, node.left, stack)
                return _Dynamic("dynamic_template", start, self._texts(left))
            return _Dynamic("dynamic_template", start)
        if isinstance(node, ast.Call):
            return self._call_text(module, node, stack, start)
        if isinstance(node, ast.Name):
            resolved = self.resolve(module, node.id)
            if resolved is not None:
                key = (resolved[0].rel, resolved[2])
                if key not in stack:
                    value = self.text(resolved[0], resolved[1], stack | {key})
                    return value.named(key) if isinstance(value, _Text) else value
        return _Dynamic("dynamic_template", start)

    def _texts(self, value: Value) -> tuple[_Text, ...]:
        return (value,) if isinstance(value, _Text) else value.texts

    def _call_text(
        self, module: _Module, node: ast.Call, stack: frozenset[Key], start: Pos
    ) -> Value:
        qualified = module.qualname(node.func)
        simple = not node.keywords and not any(isinstance(a, ast.Starred) for a in node.args)
        if qualified in ("textwrap.dedent", "inspect.cleandoc") and simple and len(node.args) == 1:
            inner = self.text(module, node.args[0], stack)
            if isinstance(inner, _Dynamic):
                return inner
            transform = textwrap.dedent if qualified == "textwrap.dedent" else inspect.cleandoc
            return inner.transformed(transform(inner.value), None, start)
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in ("strip", "lstrip", "rstrip") and simple and not node.args:
                inner = self.text(module, node.func.value, stack)
                if isinstance(inner, _Dynamic):
                    return inner
                return self._stripped(inner, attr, start)
            if attr == "format":
                base = self.text(module, node.func.value, stack)
                return _Dynamic("dynamic_template", start, self._texts(base))
        callee = module.callee(node.func)
        if callee == ("hub.pull", None):
            return _Dynamic("remote_prompt", start)
        return _Dynamic("dynamic_template", start)

    def _stripped(self, text: _Text, how: str, at: Pos) -> _Text:
        value = text.value
        left = len(value) - len(value.lstrip()) if how in ("strip", "lstrip") else 0
        right = len(value.rstrip()) if how in ("strip", "rstrip") else len(value)
        right = max(left, right)
        chars = text.chars[left:right] if text.chars is not None else None
        return text.transformed(value[left:right], chars, at)

    def _scalar(
        self, module: _Module, node: ast.expr, stack: frozenset[Key]
    ) -> Optional[tuple[Any]]:
        if isinstance(node, ast.Constant) and not isinstance(node.value, (str, bytes)):
            return (node.value,)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = node.operand
            if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
                return (-operand.value,)
        if isinstance(node, ast.Name):
            resolved = self.resolve(module, node.id)
            if resolved is not None and (resolved[0].rel, resolved[2]) not in stack:
                return self._scalar(
                    resolved[0], resolved[1], stack | {(resolved[0].rel, resolved[2])}
                )
        return None

    def _partials(
        self, module: _Module, node: Optional[ast.expr], parts: _Parts, stack: frozenset[Key]
    ) -> dict[str, SourcePartial]:
        out: dict[str, SourcePartial] = {}
        if _is_none(node):
            return out
        assert node is not None
        if isinstance(node, ast.Name):
            resolved = self.resolve(module, node.id)
            if resolved is not None:
                parts.consumed.add((resolved[0].rel, resolved[2]))
                module, node = resolved[0], resolved[1]
        if not isinstance(node, ast.Dict):
            parts.dynamic = parts.dynamic or _Dynamic("dynamic_template", module.pos(node))
            return out
        for key, value in zip(node.keys, node.values, strict=True):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                parts.dynamic = parts.dynamic or _Dynamic("dynamic_template", module.pos(node))
                continue
            self._partial(module, key.value, value, parts, out, stack)
        return out

    def _partial(
        self,
        module: _Module,
        name: str,
        value: ast.expr,
        parts: _Parts,
        out: dict[str, SourcePartial],
        stack: frozenset[Key],
    ) -> None:
        where = module.pos(value)
        scalar = self._scalar(module, value, stack)
        if scalar is not None:
            out[name] = SourcePartial("value", scalar[0], where)
            return
        if isinstance(value, ast.Lambda):
            out[name] = SourcePartial("callable", where=where)
            return
        if isinstance(value, ast.Name) and self.function(module, value.id) is not None:
            out[name] = SourcePartial("callable", where=where)
            return
        if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
            out[name] = SourcePartial("not_scalar", where=where)
            return
        text = self.text(module, value, stack)
        if isinstance(text, _Text):
            out[name] = SourcePartial("value", text.value, where)
            parts.texts[pointer("/partials", name)] = text
            parts.consumed |= text.names
            parts.notes.extend(("literal_transform_applied", at) for at in text.transforms)
            return
        parts.dynamic = parts.dynamic or text

    def _format(self, module: _Module, call: ast.Call, parts: _Parts, stack: frozenset[Key]) -> str:
        node = _kw(call, "template_format")
        if node is None:
            return "f-string"
        text = self.text(module, node, stack)
        if isinstance(text, _Text):
            return text.value
        parts.dynamic = parts.dynamic or text
        return "f-string"

    def _use_text(self, parts: _Parts, path: str, value: Value) -> Optional[_Text]:
        if isinstance(value, _Dynamic):
            parts.dynamic = parts.dynamic or value
            return None
        parts.texts[path] = value
        parts.consumed |= value.names
        parts.notes.extend(("literal_transform_applied", at) for at in value.transforms)
        return value

    def prompt_parts(
        self,
        module: _Module,
        call: ast.Call,
        name: str,
        method: Optional[str],
        stack: frozenset[Key],
    ) -> _Parts:
        parts = _Parts()
        start = module.pos(call)
        if name in _REFUSED_TEXT or name in _REFUSED_CHAT:
            refused_kind: Literal["text", "chat"] = "chat" if name in _REFUSED_CHAT else "text"
            parts.prompt = SourcePrompt(
                kind=refused_kind, refusal="unsupported_prompt_class", where=start
            )
            return parts
        fmt = self._format(module, call, parts, stack)
        partials = self._partials(module, _kw(call, "partial_variables"), parts, stack)
        parser = not _is_none(_kw(call, "output_parser"))
        if name in _TEXT_CLASSES:
            if method not in (None, "from_template"):
                parts.dynamic = _Dynamic("dynamic_template", start)
                return parts
            node = _arg(call, 0, "template") if method == "from_template" else _kw(call, "template")
            if node is None:
                parts.dynamic = _Dynamic("dynamic_template", start)
                return parts
            text = self._use_text(parts, "/body", self.text(module, node, stack))
            if text is None or parts.dynamic is not None:
                return parts
            parts.prompt = SourcePrompt(
                kind="text",
                template_format=fmt,
                body=text.value,
                partials=partials,
                output_parser=parser,
                where=module.pos(node),
            )
            return parts
        if method == "from_template":
            node = _arg(call, 0, "template")
            if node is None:
                parts.dynamic = _Dynamic("dynamic_template", start)
                return parts
            text = self._use_text(parts, "/body/0/content", self.text(module, node, stack))
            if text is None or parts.dynamic is not None:
                return parts
            entry = SourceEntry("template", "user", text.value, fmt, where=module.pos(node))
            parts.prompt = SourcePrompt(
                kind="chat", entries=(entry,), partials=partials, output_parser=parser, where=start
            )
            return parts
        if method not in (None, "from_messages"):
            parts.dynamic = _Dynamic("dynamic_template", start)
            return parts
        messages = _arg(call, 0, "messages")
        if isinstance(messages, ast.Name):
            resolved = self.resolve(module, messages.id)
            if resolved is not None:
                parts.consumed.add((resolved[0].rel, resolved[2]))
                module, messages = resolved[0], resolved[1]
        if not isinstance(messages, (ast.List, ast.Tuple)):
            parts.dynamic = parts.dynamic or _Dynamic("dynamic_template", start)
            return parts
        entries: list[SourceEntry] = []
        for index, element in enumerate(messages.elts):
            result = self.entry(module, element, fmt, stack)
            parts.consumed |= result.consumed
            if result.text is not None:
                self._use_text(parts, f"/body/{index}/content", result.text)
            if result.dynamic is not None:
                parts.dynamic = parts.dynamic or result.dynamic
                for text in result.dynamic.texts:
                    parts.texts.setdefault(f"/body/{index}/content", text)
            if result.entry is not None:
                entries.append(result.entry)
        if parts.dynamic is None:
            parts.prompt = SourcePrompt(
                kind="chat",
                entries=tuple(entries),
                partials=partials,
                output_parser=parser,
                where=start,
            )
        return parts

    def entry(self, module: _Module, node: ast.expr, fmt: str, stack: frozenset[Key]) -> _Entry:
        where = module.pos(node)
        if isinstance(node, ast.Name):
            resolved = self.resolve(module, node.id)
            if resolved is not None and (resolved[0].rel, resolved[2]) not in stack:
                key = (resolved[0].rel, resolved[2])
                result = self.entry(resolved[0], resolved[1], fmt, stack | {key})
                result.consumed.add(key)
                return result
            return _Entry(dynamic=_Dynamic("dynamic_template", where))
        if isinstance(node, (ast.Tuple, ast.Dict)):
            return self._pair(module, node, fmt, stack, where)
        if isinstance(node, ast.Call):
            found = self.classify(module, node)
            if found is not None and found[0] == "message":
                return self.message(module, node, found[1], found[2], stack)
            if found is not None and found[0] in ("prompt", "partial"):
                return _Entry(
                    SourceEntry("refused", refusal="unsupported_prompt_class", where=where)
                )
        value = self.text(module, node, stack)
        if isinstance(value, _Dynamic):
            return _Entry(dynamic=value)
        return _Entry(SourceEntry("template", "user", value.value, fmt, where=where), value)

    def _pair(
        self,
        module: _Module,
        node: Union[ast.Tuple, ast.Dict],
        fmt: str,
        stack: frozenset[Key],
        where: Pos,
    ) -> _Entry:
        if isinstance(node, ast.Dict):
            keys = [k.value if isinstance(k, ast.Constant) else None for k in node.keys]
            if sorted(str(k) for k in keys) != ["content", "role"]:
                return _Entry(
                    SourceEntry("refused", refusal="unsupported_prompt_class", where=where)
                )
            values = dict(zip(keys, node.values, strict=True))
            role_node, content = values["role"], values["content"]
        else:
            if len(node.elts) != 2:
                return _Entry(
                    SourceEntry("refused", refusal="unsupported_prompt_class", where=where)
                )
            role_node, content = node.elts
        role: Optional[str] = None
        if isinstance(role_node, ast.Constant) and isinstance(role_node.value, str):
            role = role_node.value
        else:
            callee = module.callee(role_node)
            if callee is not None and callee[1] is None:
                role = _STATIC_ROLES.get(callee[0]) or _TEMPLATE_ROLES.get(callee[0])
            if role is None:
                return _Entry(dynamic=_Dynamic("dynamic_template", where))
        if role == "placeholder":
            return self._placeholder_pair(module, content, stack, where)
        mapped = _CHAT_ROLE_NAMES.get(role)
        if isinstance(content, ast.List):
            return _Entry(
                SourceEntry("refused", mapped, refusal="unsupported_content_block", where=where)
            )
        value = self.text(module, content, stack)
        if isinstance(value, _Dynamic):
            return _Entry(dynamic=value)
        if mapped is None:
            refused = SourceEntry(
                "refused", text=value.value, refusal="unsupported_role", where=where
            )
            return _Entry(refused, value)
        return _Entry(SourceEntry("template", mapped, value.value, fmt, where=where), value)

    def _placeholder_pair(
        self, module: _Module, content: ast.expr, stack: frozenset[Key], where: Pos
    ) -> _Entry:
        optional: Any = True
        name_node = content
        if isinstance(content, ast.Tuple) and len(content.elts) == 2:
            name_node = content.elts[0]
            flag = content.elts[1]
            optional = flag.value if isinstance(flag, ast.Constant) else None
        text = self.text(module, name_node, stack)
        if isinstance(text, _Dynamic) or not isinstance(optional, bool):
            return _Entry(dynamic=_Dynamic("dynamic_template", where))
        value = text.value
        if len(value) < 2 or value[0] != "{" or value[-1] != "}":
            return _Entry(
                SourceEntry("refused", refusal="unsupported_placeholder_option", where=where)
            )
        return _Entry(SourceEntry("placeholder", name=value[1:-1], optional=optional, where=where))

    def message(
        self,
        module: _Module,
        call: ast.Call,
        name: str,
        method: Optional[str],
        stack: frozenset[Key],
    ) -> _Entry:
        where = module.pos(call)
        if name == "MessagesPlaceholder":
            variable = _arg(call, 0, "variable_name")
            optional = _kw(call, "optional")
            limit = _kw(call, "n_messages")
            if not (isinstance(variable, ast.Constant) and isinstance(variable.value, str)):
                return _Entry(dynamic=_Dynamic("dynamic_template", where))
            flag = False
            if optional is not None:
                if not (isinstance(optional, ast.Constant) and isinstance(optional.value, bool)):
                    return _Entry(dynamic=_Dynamic("dynamic_template", where))
                flag = optional.value
            count: Optional[int] = None
            if not _is_none(limit):
                if not (isinstance(limit, ast.Constant) and isinstance(limit.value, int)):
                    return _Entry(dynamic=_Dynamic("dynamic_template", where))
                count = limit.value
            return _Entry(
                SourceEntry(
                    "placeholder", name=variable.value, optional=flag, n_messages=count, where=where
                )
            )
        if name in _TEMPLATE_ROLES:
            role = _TEMPLATE_ROLES[name]
            if method != "from_template":
                return _Entry(dynamic=_Dynamic("dynamic_template", where))
            node = _arg(call, 0, "template")
            if node is None:
                return _Entry(dynamic=_Dynamic("dynamic_template", where))
            fmt_parts = _Parts()
            fmt = self._format(module, call, fmt_parts, stack)
            if fmt_parts.dynamic is not None:
                return _Entry(dynamic=fmt_parts.dynamic)
            value = self.text(module, node, stack)
            if isinstance(value, _Dynamic):
                return _Entry(dynamic=value)
            if not _empty_literal(_kw(call, "additional_kwargs")) or not _empty_literal(
                _kw(call, "partial_variables")
            ):
                refused = SourceEntry(
                    "refused",
                    role,
                    value.value,
                    refusal="unsupported_message_attributes",
                    where=where,
                )
                return _Entry(refused, value)
            return _Entry(SourceEntry("template", role, value.value, fmt, where=where), value)
        if name in _STATIC_ROLES:
            role = _STATIC_ROLES[name]
            content = _arg(call, 0, "content")
            if content is None:
                return _Entry(dynamic=_Dynamic("dynamic_template", where))
            if isinstance(content, ast.List):
                return _Entry(
                    SourceEntry("refused", role, refusal="unsupported_content_block", where=where)
                )
            value = self.text(module, content, stack)
            if isinstance(value, _Dynamic):
                return _Entry(dynamic=value)
            attributes = ("name", "additional_kwargs", "tool_calls")
            if any(not _empty_literal(_kw(call, item)) for item in attributes):
                refused = SourceEntry(
                    "refused",
                    role,
                    value.value,
                    refusal="unsupported_message_attributes",
                    where=where,
                )
                return _Entry(refused, value)
            return _Entry(SourceEntry("static", role, value.value, where=where), value)
        text_node = _arg(call, 0, "template" if method == "from_template" else "content")
        other = self.text(module, text_node, stack) if text_node is not None else None
        text = other if isinstance(other, _Text) else None
        refused = SourceEntry(
            "refused",
            text=text.value if text else None,
            refusal="unsupported_role",
            where=where,
        )
        return _Entry(refused, text)

    def draft(
        self,
        module: _Module,
        node: ast.AST,
        construct: str,
        scope: tuple[str, ...],
        symbol: Optional[str],
        usage: str,
        parts: _Parts,
    ) -> _Draft:
        key = (module.rel, symbol) if symbol and not scope and symbol in module.constants else None
        draft = _Draft(
            rel=module.rel,
            construct=construct,
            start=module.pos(node),
            end=module.end(node),
            symbol=symbol,
            function=".".join(scope) or None,
            usage=usage,
            prompt=parts.prompt if parts.dynamic is None else None,
            dynamic=parts.dynamic,
            texts=parts.texts,
            consumed=parts.consumed if parts.dynamic is None else set(),
            notes=parts.notes,
            key=key,
        )
        self.drafts.append(draft)
        if draft.dynamic is None:
            self.consumed |= draft.consumed
        return draft

    def template_draft(
        self,
        module: _Module,
        call: ast.Call,
        scope: tuple[str, ...],
        symbol: Optional[str],
        hint: Optional[str] = None,
    ) -> _Draft:
        found = self.classify(module, call)
        assert found is not None
        kind, name, method = found
        target = call
        extra: list[ast.keyword] = []
        if kind == "partial":
            assert isinstance(call.func, ast.Attribute)
            assert isinstance(call.func.value, ast.Call)
            target = call.func.value
            extra = list(call.keywords)
        parts = self.prompt_parts(module, target, name, method, frozenset())
        if extra and parts.prompt is not None:
            merged = dict(parts.prompt.partials)
            for keyword in extra:
                if keyword.arg is None:
                    parts.dynamic = parts.dynamic or _Dynamic("dynamic_template", module.pos(call))
                    continue
                self._partial(module, keyword.arg, keyword.value, parts, merged, frozenset())
            if parts.dynamic is None:
                parts.prompt = SourcePrompt(
                    kind=parts.prompt.kind,
                    template_format=parts.prompt.template_format,
                    body=parts.prompt.body,
                    entries=parts.prompt.entries,
                    partials=merged,
                    output_parser=parts.prompt.output_parser,
                    refusal=parts.prompt.refusal,
                    where=parts.prompt.where,
                )
        chat = name in _CHAT_CLASSES or name in _REFUSED_CHAT
        construct = "langchain.chat_prompt_template" if chat else "langchain.prompt_template"
        usage = "chat" if chat else _usage_from_symbol(symbol)
        draft = self.draft(module, call, construct, scope, symbol, usage, parts)
        draft.hint = hint
        return draft

    def message_draft(
        self,
        module: _Module,
        call: ast.Call,
        name: str,
        method: Optional[str],
        scope: tuple[str, ...],
        symbol: Optional[str],
    ) -> Optional[_Draft]:
        if name in ("MessagesPlaceholder", "ToolMessage", "FunctionMessage", "ChatMessage"):
            return None
        result = self.message(module, call, name, method, frozenset())
        if name in _STATIC_ROLES and result.dynamic is not None and name != "SystemMessage":
            return None
        parts = _Parts(consumed=set(result.consumed))
        if result.text is not None:
            self._use_text(parts, "/body/0/content", result.text)
        if result.dynamic is not None:
            parts.dynamic = result.dynamic
        elif result.entry is not None:
            parts.prompt = SourcePrompt(
                kind="chat", entries=(result.entry,), where=module.pos(call)
            )
        construct = (
            "langchain.message_prompt_template"
            if name in _TEMPLATE_ROLES or name == "ChatMessagePromptTemplate"
            else "langchain.message"
        )
        role = _TEMPLATE_ROLES.get(name) or _STATIC_ROLES.get(name)
        usage = {"system": "system", "user": "user"}.get(role or "", "other")
        return self.draft(module, call, construct, scope, symbol, usage, parts)

    def agent(
        self,
        module: _Module,
        call: ast.Call,
        name: str,
        scope: tuple[str, ...],
        symbol: Optional[str],
    ) -> Optional[ast.expr]:
        construct, keyword = _AGENTS[name]
        info = _Agent(module.rel)
        self.agents[id(call)] = info
        for node in ast.walk(call):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                resolved = self.resolve(module, node.id)
                if resolved is not None:
                    info.names.add((resolved[0].rel, resolved[2]))
        value = _kw(call, keyword)
        if _is_none(value):
            return None
        assert value is not None
        start = module.pos(value)
        text_node = value
        if isinstance(value, ast.Call):
            found = self.classify(module, value)
            if found is not None and found[0] in ("prompt", "partial"):
                info.drafts.append(self.template_draft(module, value, scope, None, symbol))
                return value
            if found is not None and found[1] == "SystemMessage":
                content = _arg(value, 0, "content")
                text_node = content if content is not None else value
            elif found is not None:
                return None
        elif self.is_template(module, value, frozenset()):
            return None
        parts = _Parts()
        if isinstance(text_node, (ast.Lambda, ast.Attribute)) or (
            isinstance(text_node, ast.Name) and self.function(module, text_node.id) is not None
        ):
            parts.dynamic = _Dynamic("dynamic_template", start)
        else:
            text = self._use_text(parts, "/body", self.text(module, text_node, frozenset()))
            if text is not None:
                parts.prompt = SourcePrompt(kind="text", body=text.value, static=True, where=start)
        draft = self.draft(module, value, construct, scope, None, "system", parts)
        draft.hint = symbol
        info.drafts.append(draft)
        return value

    def hub_draft(
        self, module: _Module, call: ast.Call, scope: tuple[str, ...], symbol: Optional[str]
    ) -> None:
        parts = _Parts(dynamic=_Dynamic("remote_prompt", module.pos(call)))
        self.draft(module, call, "dynamic", scope, symbol, _usage_from_symbol(symbol), parts)

    def walk(self, module: _Module) -> None:
        self._visit_children(module, module.tree, ())

    def _visit_children(self, module: _Module, node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            self._visit(module, child, scope)

    def _visit(self, module: _Module, node: ast.AST, scope: tuple[str, ...]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._visit_children(module, node, (*scope, node.name))
            return
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and self.classify(module, node.value) is not None
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            symbol = (
                targets[0].id if len(targets) == 1 and isinstance(targets[0], ast.Name) else None
            )
            if self._visit_call(module, node.value, scope, symbol):
                return
        if isinstance(node, ast.Call) and self._visit_call(module, node, scope, None):
            return
        self._visit_children(module, node, scope)

    def _visit_call(
        self, module: _Module, call: ast.Call, scope: tuple[str, ...], symbol: Optional[str]
    ) -> bool:
        if isinstance(call.func, ast.Attribute) and call.func.attr == "add_node":
            self.add_nodes.append((module, call, scope))
            return False
        found = self.classify(module, call)
        if found is None:
            return False
        kind, name, method = found
        if kind in ("prompt", "partial"):
            self.template_draft(module, call, scope, symbol)
            return True
        if kind == "message":
            self.message_draft(module, call, name, method, scope, symbol)
            return True
        if kind == "hub":
            self.hub_draft(module, call, scope, symbol)
            return True
        if kind == "agent":
            skipped = self.agent(module, call, name, scope, symbol)
            for child in ast.iter_child_nodes(call):
                if child is not skipped and not (
                    isinstance(child, ast.keyword) and child.value is skipped
                ):
                    self._visit(module, child, scope)
            return True
        return False

    def constants(self, module: _Module) -> None:
        drafted = {draft.key for draft in self.drafts if draft.key is not None}
        for name, expr in module.constants.items():
            key = (module.rel, name)
            if not _PROMPT_NAME.search(name) or key in self.consumed or key in drafted:
                continue
            if isinstance(expr, ast.Call) and self.classify(module, expr) is not None:
                continue
            if isinstance(expr, ast.Name) and self.is_template(module, expr, frozenset({key})):
                continue
            if isinstance(expr, (ast.Dict, ast.List, ast.Set, ast.Tuple, ast.Lambda)):
                continue
            if isinstance(expr, ast.Constant) and not isinstance(expr.value, str):
                continue
            if isinstance(expr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue
            value = self.text(module, expr, frozenset({key}))
            parts = _Parts()
            text = self._use_text(parts, "/body", value)
            construct = "dynamic"
            if text is not None:
                construct = "python.string_constant"
                parts.prompt = SourcePrompt(kind="text", body=text.value, where=module.pos(expr))
            draft = self.draft(module, expr, construct, (), name, _usage_from_symbol(name), parts)
            draft.key = key

    def _graph_names(self, module: _Module) -> tuple[set[str], dict[str, ast.Call], set[str]]:
        graphs: set[str] = set()
        assigned: list[tuple[str, ast.Call]] = []
        for node in ast.walk(module.tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
            ):
                assigned.append((node.targets[0].id, node.value))
        agents: dict[str, ast.Call] = {}
        for name, call in assigned:
            found = self.classify(module, call)
            if found is not None and found[0] == "graph":
                graphs.add(name)
            elif found is not None and found[0] == "agent":
                agents[name] = call
        compiled: set[str] = set()
        for name, call in assigned:
            func = call.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "compile"
                and isinstance(func.value, ast.Name)
                and func.value.id in graphs
            ):
                compiled.add(name)
        return graphs, agents, compiled

    def _body(self, module: _Module, function: ast.AST) -> list[_Draft]:
        start, end = module.pos(function), module.end(function)
        refs: set[Key] = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                resolved = self.resolve(module, node.id)
                if resolved is not None:
                    refs.add((resolved[0].rel, resolved[2]))
        out: list[_Draft] = []
        for draft in self.drafts:
            inside = draft.rel == module.rel and start <= draft.start <= end
            referenced = (draft.key is not None and draft.key in refs) or bool(
                draft.consumed & refs
            )
            if inside or referenced:
                out.append(draft)
        return out

    def link(self) -> None:
        graph_info = {rel: self._graph_names(module) for rel, module in self.modules.items()}
        for module, call, scope in self.add_nodes:
            graphs, agents, compiled = graph_info[module.rel]
            func = call.func
            assert isinstance(func, ast.Attribute)
            if not (isinstance(func.value, ast.Name) and func.value.id in graphs):
                continue
            node_arg = _arg(call, 0, "node")
            action = _arg(call, 1, "action")
            literal = (
                node_arg.value
                if isinstance(node_arg, ast.Constant) and isinstance(node_arg.value, str)
                else None
            )
            if action is None and isinstance(node_arg, ast.Name):
                target = module.functions.get(node_arg.id)
                if target is not None:
                    for draft in self._body(module, target):
                        draft.node_paths.add(node_arg.id)
                continue
            if isinstance(action, ast.Name):
                local = module.functions.get(action.id)
                if literal is not None and local is not None:
                    for draft in self._body(module, local):
                        draft.node_paths.add(literal)
                    continue
                if literal is not None and (action.id in agents or action.id in compiled):
                    self._subagent(module, call, scope, literal, agents.get(action.id))
                    continue
                located = self.function(module, action.id)
                if located is not None:
                    for draft in self._body(located[0], located[1]):
                        draft.node_unresolved = True
                continue
            if isinstance(action, ast.Attribute):
                for method in module.methods.get(action.attr, []):
                    for draft in self._body(module, method):
                        draft.node_unresolved = True

    def _subagent(
        self,
        module: _Module,
        call: ast.Call,
        scope: tuple[str, ...],
        node_path: str,
        agent: Optional[ast.Call],
    ) -> None:
        draft = _Draft(
            rel=module.rel,
            construct="langgraph.subagent_node",
            start=module.pos(call),
            end=module.end(call),
            symbol=None,
            function=".".join(scope) or None,
            usage="other",
            subagent=node_path,
        )
        self.drafts.append(draft)
        info = self.agents.get(id(agent)) if agent is not None else None
        if info is None:
            return
        for linked in info.drafts:
            linked.node_paths.add(node_path)
        for other in self.drafts:
            if other.key is not None and other.key in info.names:
                other.node_paths.add(node_path)


def _usage_from_symbol(symbol: Optional[str]) -> str:
    name = (symbol or "").lower()
    if "system" in name:
        return "system"
    if "instruction" in name:
        return "instructions"
    if "user" in name or "human" in name or "question" in name:
        return "user"
    if "tool" in name or "description" in name:
        return "tool_description"
    return "instructions"


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", value.lower())
    text = re.sub(r"_+", "_", text).strip("_")
    if text and not text[0].isalpha():
        text = "n_" + text
    return text


def _base_name(draft: _Draft) -> str:
    named = draft.symbol or draft.hint
    if named:
        name = named.lower()
        changed = True
        while changed:
            changed = False
            for suffix in _NAME_SUFFIXES:
                if name.endswith(suffix) and len(name) > len(suffix):
                    name = name[: -len(suffix)]
                    changed = True
        slug = _slug(name)
        if slug and slug not in _USAGE_WORDS:
            return slug
    if draft.function:
        slug = _slug(draft.function.split(".")[-1])
        if slug:
            return slug
    stem = _slug(draft.rel.rsplit("/", 1)[-1][: -len(".py")])
    return stem if stem and stem != "n_init" else "prompt"


def _finding(
    text: Optional[_Text], fallback: Pos, pattern: str, offset: int, length: int
) -> dict[str, Any]:
    line, column = text.locate(offset) if text is not None else fallback
    return {"pattern": pattern, "line": line, "column": column, "length": length}


def _report_issue(code: str, at: Pos, message: Optional[str] = None) -> dict[str, Any]:
    return {
        "code": code,
        "severity": ISSUE_SEVERITY.get(code, "error"),
        "line": at[0],
        "column": at[1],
        "message": message or ISSUE_MESSAGES.get(code, f"the prompt cannot be imported: {code}"),
    }


class _Report:
    def __init__(self) -> None:
        self.slots: set[str] = set()
        self.prompt_ids: set[str] = set()

    def slot(self, prefix: str, usage: str) -> Optional[str]:
        if not _SLOT_SEGMENT.fullmatch(prefix):
            return None
        candidate = f"{prefix}.{usage}"
        counter = 2
        while candidate in self.slots:
            candidate = f"{prefix}_{counter}.{usage}"
            counter += 1
        if len(candidate) > 128 or not _SLOT_PATH.fullmatch(candidate):
            return None
        self.slots.add(candidate)
        return candidate

    def prompt_id(self, slot: str) -> str:
        base = ("prm_" + slot.replace(".", "_"))[:64].rstrip("_-")
        candidate = base
        counter = 2
        while candidate in self.prompt_ids:
            suffix = f"_{counter}"
            candidate = base[: 64 - len(suffix)].rstrip("_-") + suffix
            counter += 1
        self.prompt_ids.add(candidate)
        return candidate

    def candidate(self, draft: _Draft) -> dict[str, Any]:
        source = {
            "path": draft.rel,
            "line": draft.start[0],
            "column": draft.start[1],
            "end_line": draft.end[0],
            "end_column": draft.end[1],
            "symbol": draft.symbol,
            "enclosing_function": draft.function,
        }
        identity = {
            "path": draft.rel,
            "line": draft.start[0],
            "column": draft.start[1],
            "construct": draft.construct,
        }
        candidate_id = (
            "cand_" + hashlib.sha256(canonical_json_v1(identity).encode("utf-8")).hexdigest()[:16]
        )
        issues: list[dict[str, Any]] = [_report_issue(code, at) for code, at in draft.notes]
        findings: list[dict[str, Any]] = []
        content: Optional[dict[str, Any]] = None
        digest: Optional[str] = None
        kind: Optional[str] = None
        node_path = next(iter(draft.node_paths)) if len(draft.node_paths) == 1 else None
        if draft.subagent is not None:
            status = "unresolved"
            node_path = draft.subagent
            issues.append(_report_issue("subagent_unmapped", draft.start))
        elif draft.dynamic is not None:
            status = "unresolved"
            issues.append(_report_issue(draft.dynamic.code, draft.dynamic.start))
            for piece in draft.dynamic.texts:
                for found in scan(piece.value):
                    findings.append(
                        _finding(piece, piece.start, found.pattern, found.offset, found.length)
                    )
            if findings:
                first = findings[0]
                issues.append(_report_issue("secret_detected", (first["line"], first["column"])))
        else:
            assert draft.prompt is not None
            conversion = convert_prompt(draft.prompt)
            status = conversion.status
            kind = conversion.prompt_kind
            content = conversion.content
            digest = conversion.content_digest
            for item in conversion.issues:
                at = draft.start
                text = draft.texts.get(item.path or "")
                if text is not None and item.offset is not None:
                    at = text.locate(item.offset)
                elif isinstance(item.where, tuple):
                    at = item.where
                issues.append(_report_issue(item.report_code, at, item.message))
            for secret in conversion.secrets:
                text = draft.texts.get(secret.path)
                findings.append(
                    _finding(text, draft.start, secret.pattern, secret.offset, secret.length)
                )
            if findings:
                issues = [
                    item
                    if item["code"] != "secret_detected"
                    else _report_issue(
                        "secret_detected", (findings[0]["line"], findings[0]["column"])
                    )
                    for item in issues
                ]
        if draft.subagent is None and draft.node_unresolved and node_path is None:
            issues.append(_report_issue("node_unresolved", draft.start))
        if draft.subagent is None and len(draft.node_paths) > 1:
            issues.append(_report_issue("node_unresolved", draft.start))
        proposal: dict[str, Any] = {
            "prompt_id": None,
            "prompt_kind": None,
            "slot_path": None,
            "node_path": node_path,
            "usage": None,
        }
        if draft.subagent is None:
            prefix = _slug(node_path) if node_path is not None else _base_name(draft)
            slot = self.slot(prefix, draft.usage)
            proposal["slot_path"] = slot
            proposal["usage"] = draft.usage
            if status != "unresolved" and slot is not None:
                proposal["prompt_id"] = self.prompt_id(slot)
                proposal["prompt_kind"] = kind
        return {
            "candidate_id": candidate_id,
            "status": status,
            "construct": draft.construct,
            "source": source,
            "proposal": proposal,
            "content": content,
            "content_digest": digest,
            "issues": issues,
            "secret_findings": findings,
        }


def _file_record(rel: str, digest: str, reason: Optional[str]) -> dict[str, Any]:
    return {
        "path": rel,
        "sha256": "sha256:" + digest,
        "status": "scanned" if reason is None else "skipped",
        "skip_reason": reason,
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _excluded(rel: str, patterns: Sequence[str]) -> bool:
    segments = rel.split("/")
    for pattern in patterns:
        if fnmatch.fnmatchcase(rel, pattern):
            return True
        if any(fnmatch.fnmatchcase(segment, pattern) for segment in segments):
            return True
    return False


def _relative(path: Path, root: Path) -> Optional[str]:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _collect(
    paths: Iterable[os.PathLike[str]], root: Path, exclude: Sequence[str]
) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for given in paths:
        start = Path(given)
        start = (start if start.is_absolute() else Path.cwd() / start).resolve()
        if _relative(start, root) is None:
            raise ValueError("every scanned path must be inside root")
        if start.is_file():
            rel = _relative(start, root)
            if rel is not None and start.suffix == ".py":
                found[rel] = start
            continue
        for directory, dirnames, filenames in os.walk(start, followlinks=False):
            dirnames[:] = sorted(
                name for name in dirnames if not any(fnmatch.fnmatchcase(name, p) for p in exclude)
            )
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                candidate = Path(directory) / filename
                if candidate.is_symlink():
                    resolved = candidate.resolve()
                    if _relative(resolved, root) is None or not resolved.is_file():
                        continue
                rel = _relative(Path(directory).resolve() / filename, root)
                if rel is not None:
                    found[rel] = candidate
    return sorted(found.items())


def _decode(data: bytes) -> Optional[str]:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        return io.TextIOWrapper(io.BytesIO(data), encoding=encoding, newline=None).read()
    except (SyntaxError, LookupError, UnicodeDecodeError):
        return None


def _parse(text: str, rel: str) -> Optional[ast.Module]:
    try:
        return ast.parse(text, filename=rel, feature_version=(3, 10))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _label(root: Path, label: Optional[str]) -> str:
    value = label if label is not None else (root.name or "repository")
    invalid = (
        not value
        or len(value) > 128
        or "\0" in value
        or re.match(r"(/|\\|~|[A-Za-z]:)", value) is not None
    )
    if invalid:
        if label is not None:
            raise ValueError("label must be a short name, never an absolute path")
        return "repository"
    return value


def _timestamp(now: Optional[datetime]) -> str:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scan_paths(
    paths: Sequence[os.PathLike[str]],
    *,
    root: os.PathLike[str],
    max_files: int = 4000,
    max_file_bytes: int = 512 * 1024,
    exclude: Sequence[str] = DEFAULT_EXCLUDES,
    label: Optional[str] = None,
    commit: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    if max_files < 1 or max_file_bytes < 1:
        raise ValueError("max_files and max_file_bytes must be positive")
    if commit is not None and not _HEX40.fullmatch(commit):
        raise ValueError("commit must be 40 lowercase hex characters")
    base = Path(root).resolve()
    scanner = _Scanner()
    files: list[dict[str, Any]] = []
    for index, (rel, path) in enumerate(_collect(paths, base, exclude)):
        if _excluded(rel, exclude):
            files.append(_file_record(rel, _hash_file(path), "excluded"))
            continue
        if index >= max_files:
            files.append(_file_record(rel, _hash_file(path), "limit_reached"))
            continue
        if path.stat().st_size > max_file_bytes:
            files.append(_file_record(rel, _hash_file(path), "too_large"))
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        text = _decode(data)
        if text is None:
            files.append(_file_record(rel, digest, "not_utf8"))
            continue
        tree = _parse(text, rel)
        if tree is None:
            files.append(_file_record(rel, digest, "syntax_error"))
            continue
        scanner.register(_Module(rel, text, tree))
        files.append(_file_record(rel, digest, None))
    for module in scanner.modules.values():
        scanner.walk(module)
    for module in scanner.modules.values():
        scanner.constants(module)
    scanner.drafts = [
        draft for draft in scanner.drafts if draft.key is None or draft.key not in scanner.consumed
    ]
    scanner.link()
    scanner.drafts.sort(key=lambda d: (d.rel, d.start, d.construct))
    report = _Report()
    linked = [draft for draft in scanner.drafts if len(draft.node_paths) == 1]
    built = {id(draft): report.candidate(draft) for draft in linked}
    for draft in scanner.drafts:
        if id(draft) not in built:
            built[id(draft)] = report.candidate(draft)
    candidates = [built[id(draft)] for draft in scanner.drafts]
    return {
        "schema": DISCOVERY_SCHEMA,
        "scanner": {
            "name": SCANNER_NAME,
            "version": __version__[:64] or "0",
            "python_grammar": PYTHON_GRAMMAR,
            "secret_patterns": SECRET_PATTERN_SET,
        },
        "root": {
            "label": _label(base, label),
            "vcs": {"kind": "git", "commit": commit} if commit is not None else None,
        },
        "generated_at": _timestamp(now),
        "limits": {"max_files": max_files, "max_file_bytes": max_file_bytes},
        "files": files,
        "candidates": candidates,
    }
