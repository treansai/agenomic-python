from __future__ import annotations

import operator
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, Optional, Union, cast

from agenomic.prompts.digest import (
    CONTENT_SCHEMA,
    check_string,
    ensure_ajs,
    normalized_json,
    pointer,
    prompt_digest,
    sorted_keys,
)
from agenomic.prompts.errors import (
    AjsError,
    PromptTemplateError,
    item_details,
    render_error,
)
from agenomic.prompts.secrets import is_secret_shaped_key, scan

TEMPLATE_FORMAT = "agenomic-fstring/v1"
RENDERER_VERSION = "1"
SUPPORTED_RENDERER_VERSIONS = ("1",)
RENDERED_SCHEMA = "agenomic.rendered_prompt/v1"
CONTENT_MEMBERS = (
    "schema",
    "template_format",
    "renderer_version",
    "kind",
    "body",
    "variables",
    "partials",
    "output_contract",
    "fragments",
)
VARIABLE_TYPES = ("string", "integer", "boolean", "json", "messages")
CHAT_ROLES = ("system", "user", "assistant")
MESSAGE_ROLES = ("system", "user", "assistant", "tool")
MESSAGE_TYPE_ROLES = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
MAX_FRAGMENT_DEPTH = 8
MAX_FRAGMENT_EXPANSIONS = 256
MAX_TEMPLATE_CODE_POINTS = 65536
MAX_CONTENT_BYTES = 262144
MAX_EXPANDED_CODE_POINTS = 1048576
MAX_RENDERED_CODE_POINTS = 4194304
MAX_OUTPUT_CONTRACT_BYTES = 65536
MAX_OUTPUT_CONTRACT_DEPTH = 32
MAX_CHAT_MESSAGES = 256
MAX_VARIABLES = 128
MAX_FRAGMENTS = 32
CONTENT_TOO_LARGE_REASONS = (
    "content_too_large",
    "template_too_large",
    "too_many_messages",
    "too_many_variables",
    "too_many_fragments",
)
NAMED_CONTENT_CODES = {
    "prompt_kind_mismatch": "prompt_kind_mismatch",
    "fragment_cycle": "prompt_fragment_cycle",
    "fragment_depth_exceeded": "prompt_fragment_depth_exceeded",
}

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*", re.ASCII)
_PROMPT_ID = re.compile(r"prm_[a-z0-9]+(?:[_-][a-z0-9]+)*", re.ASCII)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}", re.ASCII)
_DIGITS = re.compile(r"[0-9]+", re.ASCII)
_NAME_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
_SPECIAL = {
    "!": "conversion",
    ":": "format_spec",
    ".": "attribute_access",
    "[": "index_access",
}

FragmentSource = Callable[[str, int, str], Optional[Mapping[str, Any]]]
SecretPolicy = Literal["off", "error"]


@dataclass(frozen=True)
class PromptIssue:
    code: str
    path: Optional[str] = None
    offset: Optional[int] = None
    line: Optional[int] = None
    column: Optional[int] = None
    syntax: Optional[str] = None
    variable: Optional[str] = None
    value_path: Optional[str] = None
    pattern: Optional[str] = None
    fragment: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None:
                out[item.name] = value
        return out


@dataclass(frozen=True)
class RenderedMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class _Origin:
    key: str
    variables: Any


@dataclass(frozen=True)
class Token:
    kind: Literal["literal", "var", "include"]
    text: str
    offset: int = 0
    origin: Optional[_Origin] = None

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "literal":
            return {"t": "literal", "text": self.text}
        return {"t": self.kind, "name": self.text, "offset": self.offset}


class _IssueError(Exception):
    def __init__(self, issue: PromptIssue) -> None:
        super().__init__(issue.code)
        self.issue = issue


def _syntax_issue(syntax: str, offset: int, text: str, **extra: Any) -> PromptIssue:
    before = text[:offset]
    line = before.count("\n") + 1
    column = offset - before.rfind("\n")
    return PromptIssue(
        code="syntax_error", syntax=syntax, offset=offset, line=line, column=column, **extra
    )


class _TemplateSyntaxError(Exception):
    def __init__(self, syntax: str, offset: int, text: str) -> None:
        super().__init__(syntax)
        self.syntax = syntax
        self.offset = offset
        self.text = text

    def issue(self, **extra: Any) -> PromptIssue:
        return _syntax_issue(self.syntax, self.offset, self.text, **extra)


def is_name(value: Any) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 64 and _NAME.fullmatch(value) is not None


def _classify(inner: str, offset: int, text: str) -> str:
    if inner == "":
        raise _TemplateSyntaxError("empty_placeholder", offset, text)
    if "{" in inner:
        raise _TemplateSyntaxError("nested_placeholder", offset, text)
    for char in inner:
        if char in _NAME_CHARS:
            continue
        special = _SPECIAL.get(char)
        if special is not None:
            raise _TemplateSyntaxError(special, offset, text)
        if 0x09 <= ord(char) <= 0x0D or char == " ":
            raise _TemplateSyntaxError("whitespace_in_placeholder", offset, text)
        raise _TemplateSyntaxError("invalid_placeholder_name", offset, text)
    if _DIGITS.fullmatch(inner):
        raise _TemplateSyntaxError("positional_placeholder", offset, text)
    if inner[0] in "0123456789":
        raise _TemplateSyntaxError("invalid_placeholder_name", offset, text)
    if len(inner) > 64:
        raise _TemplateSyntaxError("placeholder_name_too_long", offset, text)
    return inner


def _tokenize(template: str) -> list[Token]:
    out: list[Token] = []
    literal: list[str] = []
    index = 0
    length = len(template)

    def flush() -> None:
        if literal:
            out.append(Token("literal", "".join(literal)))
            literal.clear()

    while index < length:
        char = template[index]
        if char == "{":
            if index + 1 < length and template[index + 1] == "{":
                literal.append("{")
                index += 2
                continue
            close = template.find("}", index + 1)
            if close < 0:
                raise _TemplateSyntaxError("unclosed_brace", index, template)
            inner = template[index + 1 : close]
            flush()
            if inner.startswith(">"):
                name = inner[1:]
                if not is_name(name):
                    raise _TemplateSyntaxError("invalid_fragment_name", index, template)
                out.append(Token("include", name, index))
            else:
                out.append(Token("var", _classify(inner, index, template), index))
            index = close + 1
            continue
        if char == "}":
            if index + 1 < length and template[index + 1] == "}":
                literal.append("}")
                index += 2
                continue
            raise _TemplateSyntaxError("unmatched_closing_brace", index, template)
        literal.append(char)
        index += 1
    flush()
    return out


def tokenize(template: str) -> list[Token]:
    try:
        return _tokenize(template)
    except _TemplateSyntaxError as failure:
        issue = failure.issue().to_dict()
        raise PromptTemplateError(
            "prompt_template_invalid",
            0,
            f"template syntax error: {failure.syntax}",
            item_details(issue),
        ) from failure


def source(tokens: Sequence[Token]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token.kind == "literal":
            parts.append(token.text.replace("{", "{{").replace("}", "}}"))
        elif token.kind == "var":
            parts.append("{" + token.text + "}")
        else:
            parts.append("{>" + token.text + "}")
    return "".join(parts)


def _merge(tokens: Sequence[Token]) -> list[Token]:
    out: list[Token] = []
    for token in tokens:
        if token.kind == "literal":
            if token.text == "":
                continue
            if out and out[-1].kind == "literal":
                out[-1] = Token("literal", out[-1].text + token.text)
                continue
            out.append(Token("literal", token.text))
            continue
        out.append(token)
    return out


def _valid_pin(pin: Any) -> bool:
    if not isinstance(pin, Mapping) or set(pin) != {"content_digest", "prompt_id", "version"}:
        return False
    prompt_id, version, digest = pin["prompt_id"], pin["version"], pin["content_digest"]
    return (
        isinstance(prompt_id, str)
        and len(prompt_id) <= 64
        and _PROMPT_ID.fullmatch(prompt_id) is not None
        and isinstance(version, int)
        and not isinstance(version, bool)
        and 1 <= version <= 2147483647
        and isinstance(digest, str)
        and _SHA256.fullmatch(digest) is not None
    )


def _safe_digest(content: Any) -> Optional[str]:
    try:
        return prompt_digest(content)
    except AjsError:
        return None


def _fragment_kind_issue(entry: Mapping[str, Any], path: str, key: str) -> Optional[PromptIssue]:
    if "prompt_kind" in entry and entry["prompt_kind"] != "fragment":
        return PromptIssue("fragment_not_fragment", path=path, fragment=key)
    content = entry.get("content")
    if not isinstance(content, Mapping) or content.get("kind") != "text":
        return PromptIssue("fragment_not_text", path=path, fragment=key)
    partials = content.get("partials")
    if not isinstance(partials, Mapping) or len(partials) > 0:
        return PromptIssue("fragment_has_partials", path=path, fragment=key)
    if "output_contract" not in content or content["output_contract"] is not None:
        return PromptIssue("fragment_has_output_contract", path=path, fragment=key)
    return None


def _templates(content: Mapping[str, Any]) -> list[tuple[str, Any]]:
    if content.get("kind") == "text":
        return [("/body", content.get("body"))]
    out: list[tuple[str, Any]] = []
    body = content.get("body")
    if isinstance(body, list):
        for index, entry in enumerate(body):
            if isinstance(entry, dict) and "role" in entry:
                out.append((f"/body/{index}/content", entry.get("content")))
    return out


def _json_depth(value: Any) -> int:
    if isinstance(value, list):
        return 1 + max((_json_depth(item) for item in value), default=0)
    if isinstance(value, dict):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    return 0


@dataclass(frozen=True)
class SecretLocation:
    pattern: str
    path: str
    offset: int
    length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "path": self.path,
            "offset": self.offset,
            "length": self.length,
        }


def content_secrets(content: Mapping[str, Any]) -> list[SecretLocation]:
    out: list[SecretLocation] = []

    def push(text: str, path: str) -> None:
        for finding in scan(text):
            out.append(SecretLocation(finding.pattern, path, finding.offset, finding.length))

    def walk(value: Any, path: str) -> None:
        if isinstance(value, str):
            push(value, path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, pointer(path, index))
        elif isinstance(value, dict):
            for key in sorted_keys(value):
                walk(value[key], pointer(path, key))

    kind = content.get("kind")
    body = content.get("body")
    if kind == "text" and isinstance(body, str):
        push(body, "/body")
    if kind == "chat" and isinstance(body, list):
        for index, entry in enumerate(body):
            if isinstance(entry, dict) and isinstance(entry.get("content"), str):
                push(entry["content"], f"/body/{index}/content")
    partials = content.get("partials")
    if isinstance(partials, dict):
        for key in sorted_keys(partials):
            if isinstance(partials[key], str):
                push(partials[key], pointer("/partials", key))
    if content.get("output_contract") is not None:
        walk(content["output_contract"], "/output_contract")
    return out


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: tuple[PromptIssue, ...]
    warnings: tuple[PromptIssue, ...]
    secret_findings: tuple[SecretLocation, ...]
    declared: tuple[str, ...] = ()
    referenced: tuple[str, ...] = ()
    placeholders: tuple[str, ...] = ()
    content_digest: Optional[str] = None
    content: Optional[Mapping[str, Any]] = None
    expanded: Optional[Mapping[str, tuple[Token, ...]]] = None

    @property
    def variables(self) -> dict[str, list[str]]:
        return {
            "declared": list(self.declared),
            "referenced": list(self.referenced),
            "placeholders": list(self.placeholders),
        }

    def raise_for_errors(self, status: int = 0) -> None:
        if not self.errors:
            return
        first = self.errors[0]
        details = item_details(first.to_dict())
        details["errors"] = [issue.to_dict() for issue in self.errors]
        if self.secret_findings:
            details["findings"] = [finding.to_dict() for finding in self.secret_findings]
            raise PromptTemplateError(
                "prompt_secret_detected", status, "the prompt content contains a secret", details
            )
        if first.code in CONTENT_TOO_LARGE_REASONS:
            raise PromptTemplateError(
                "prompt_content_too_large",
                status,
                f"prompt content too large: {first.code}",
                details,
            )
        code = NAMED_CONTENT_CODES.get(first.code, "prompt_template_invalid")
        raise PromptTemplateError(code, status, f"prompt content invalid: {first.code}", details)


class _Validation:
    def __init__(self, fragments: FragmentSource) -> None:
        self.fragments = fragments
        self.errors: list[PromptIssue] = []
        self.warnings: list[PromptIssue] = []
        self.secrets: list[SecretLocation] = []

    def error(self, code: str, **values: Any) -> None:
        self.errors.append(PromptIssue(code, **values))

    def warn(self, code: str, **values: Any) -> None:
        self.warnings.append(PromptIssue(code, **values))

    def lookup(self, pin: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        return self.fragments(pin["prompt_id"], pin["version"], pin["content_digest"])

    def report(self, **values: Any) -> ValidationReport:
        return ValidationReport(
            ok=not self.errors,
            errors=tuple(self.errors),
            warnings=tuple(self.warnings),
            secret_findings=tuple(self.secrets),
            **values,
        )

    def shape(self, raw: Any) -> Optional[dict[str, Any]]:
        if not isinstance(raw, Mapping):
            self.error("invalid_field_type", path="")
            return None
        if "schema" not in raw:
            self.error("missing_field", path="/schema")
            return None
        if raw["schema"] != CONTENT_SCHEMA:
            self.error("unsupported_schema", path="/schema")
            return None
        try:
            content = ensure_ajs(raw)
        except AjsError as failure:
            self.error(failure.reason, value_path=failure.value_path)
            return None
        structural = False
        for member in CONTENT_MEMBERS:
            if member not in content:
                self.error("missing_field", path="/" + member)
                structural = True
        for key in sorted_keys(content):
            if key not in CONTENT_MEMBERS:
                self.error("unknown_field", path=pointer("", key))
                structural = True
        if structural:
            return None
        self.members(content)
        return cast(dict[str, Any], content)

    def members(self, content: dict[str, Any]) -> None:
        if content["template_format"] != TEMPLATE_FORMAT:
            self.error("unsupported_template_format", path="/template_format")
        if content["renderer_version"] != RENDERER_VERSION:
            self.error("unsupported_renderer_version", path="/renderer_version")
        kind, body = content["kind"], content["body"]
        if kind not in ("text", "chat"):
            self.error("invalid_field_type", path="/kind")
        elif kind == "text":
            if not isinstance(body, str):
                self.error("invalid_body", path="/body")
        elif not isinstance(body, list):
            self.error("invalid_body", path="/body")
        elif len(body) == 0:
            self.error("empty_chat", path="/body")
        elif len(body) > MAX_CHAT_MESSAGES:
            self.error("too_many_messages", path="/body")
        variables = content["variables"]
        if not isinstance(variables, dict):
            self.error("invalid_field_type", path="/variables")
        else:
            if len(variables) > MAX_VARIABLES:
                self.error("too_many_variables", path="/variables")
            for name in sorted_keys(variables):
                self.declaration(name, variables[name])
        if not isinstance(content["partials"], dict):
            self.error("invalid_field_type", path="/partials")
        fragments = content["fragments"]
        if not isinstance(fragments, dict):
            self.error("invalid_field_type", path="/fragments")
        elif len(fragments) > MAX_FRAGMENTS:
            self.error("too_many_fragments", path="/fragments")

    def declaration(self, name: str, decl: Any) -> None:
        path = pointer("/variables", name)
        if not is_name(name):
            self.error("invalid_variable_name", path=path)
            return
        if not isinstance(decl, dict):
            self.error("invalid_field_type", path=path)
            return
        for member in ("type", "required"):
            if member not in decl:
                self.error("missing_field", path=pointer(path, member))
        for member in sorted_keys(decl):
            if member not in ("type", "required"):
                self.error("unknown_field", path=pointer(path, member))
        if "type" in decl and decl["type"] not in VARIABLE_TYPES:
            self.error("invalid_variable_type", path=pointer(path, "type"))
        if "required" in decl and not isinstance(decl["required"], bool):
            self.error("invalid_field_type", path=pointer(path, "required"))

    def limits(self, content: dict[str, Any]) -> None:
        for path, text in _templates(content):
            if isinstance(text, str) and len(text) > MAX_TEMPLATE_CODE_POINTS:
                self.error("template_too_large", path=path)
        if len(normalized_json(content).encode("utf-8")) > MAX_CONTENT_BYTES:
            self.error("content_too_large", path="")

    def entries(self, content: dict[str, Any]) -> None:
        if content["kind"] != "chat":
            return
        for index, entry in enumerate(content["body"]):
            path = f"/body/{index}"
            if not isinstance(entry, dict):
                self.error("invalid_message_entry", path=path)
                continue
            has_role, has_placeholder = "role" in entry, "placeholder" in entry
            if has_role and has_placeholder:
                self.error("invalid_message_entry", path=path)
            elif has_role:
                self.role_entry(entry, path)
            elif has_placeholder:
                self.placeholder_entry(entry, path)
            else:
                self.error("invalid_message_entry", path=path)

    def role_entry(self, entry: dict[str, Any], path: str) -> None:
        if set(entry) != {"content", "role"}:
            self.error("invalid_message_entry", path=path)
        elif entry["role"] not in CHAT_ROLES:
            self.error("unsupported_role", path=path + "/role")
        elif isinstance(entry["content"], list):
            self.error("unsupported_content_block", path=path + "/content")
        elif not isinstance(entry["content"], str):
            self.error("invalid_message_entry", path=path + "/content")

    def placeholder_entry(self, entry: dict[str, Any], path: str) -> None:
        if set(entry) != {"optional", "placeholder"}:
            self.error("invalid_message_entry", path=path)
        elif not is_name(entry["placeholder"]):
            self.error("invalid_message_entry", path=path + "/placeholder")
        elif not isinstance(entry["optional"], bool):
            self.error("invalid_message_entry", path=path + "/optional")

    def syntax(self, content: dict[str, Any]) -> dict[str, list[Token]]:
        tokens: dict[str, list[Token]] = {}
        for path, text in _templates(content):
            try:
                tokens[path] = _tokenize(text)
            except _TemplateSyntaxError as failure:
                self.errors.append(failure.issue(path=path))
        return tokens

    def pins(self, content: dict[str, Any]) -> None:
        fragments = content["fragments"]
        for name in sorted_keys(fragments):
            path = pointer("/fragments", name)
            pin = fragments[name]
            if not is_name(name) or not _valid_pin(pin):
                self.error("invalid_fragment_pin", path=path)
                continue
            key = f"{pin['prompt_id']}:{pin['version']}"
            entry = self.lookup(pin)
            if entry is None:
                self.error("fragment_not_found", path=path, fragment=key)
                continue
            if _safe_digest(entry.get("content")) != pin["content_digest"]:
                self.error("fragment_digest_mismatch", path=path, fragment=key)
                continue
            issue = _fragment_kind_issue(entry, path, key)
            if issue is not None:
                self.errors.append(issue)

    def expand(
        self,
        tokens: Sequence[Token],
        fragment_map: Any,
        depth: int,
        stack: tuple[str, ...],
        budget: list[int],
        path: str,
        origin: Optional[_Origin],
    ) -> list[Token]:
        out: list[Token] = []
        origin_key = origin.key if origin is not None else None
        for token in tokens:
            if token.kind != "include":
                out.append(replace(token, origin=origin) if token.kind == "var" else token)
                continue
            if not isinstance(fragment_map, Mapping) or token.text not in fragment_map:
                raise _IssueError(
                    PromptIssue(
                        "fragment_not_declared", path=path, offset=token.offset, fragment=origin_key
                    )
                )
            pin = fragment_map[token.text]
            if not _valid_pin(pin):
                raise _IssueError(
                    PromptIssue("invalid_fragment_pin", path=path, fragment=origin_key)
                )
            key = f"{pin['prompt_id']}:{pin['version']}"
            if key in stack:
                raise _IssueError(PromptIssue("fragment_cycle", path=path, fragment=key))
            if depth + 1 > MAX_FRAGMENT_DEPTH:
                raise _IssueError(PromptIssue("fragment_depth_exceeded", path=path, fragment=key))
            budget[0] += 1
            if budget[0] > MAX_FRAGMENT_EXPANSIONS:
                raise _IssueError(PromptIssue("fragment_expansion_limit", path=path, fragment=key))
            entry = self.lookup(pin)
            if entry is None:
                raise _IssueError(PromptIssue("fragment_not_found", path=path, fragment=key))
            if _safe_digest(entry.get("content")) != pin["content_digest"]:
                raise _IssueError(PromptIssue("fragment_digest_mismatch", path=path, fragment=key))
            issue = _fragment_kind_issue(entry, path, key)
            if issue is not None:
                raise _IssueError(issue)
            fragment = entry["content"]
            body = fragment.get("body")
            if not isinstance(body, str):
                raise _IssueError(PromptIssue("fragment_not_text", path=path, fragment=key))
            try:
                fragment_tokens = _tokenize(body)
            except _TemplateSyntaxError as failure:
                raise _IssueError(failure.issue(path=path, fragment=key)) from failure
            out.extend(
                self.expand(
                    fragment_tokens,
                    fragment.get("fragments"),
                    depth + 1,
                    (*stack, key),
                    budget,
                    path,
                    _Origin(key, fragment.get("variables")),
                )
            )
        merged = _merge(out)
        if len(source(merged)) > MAX_EXPANDED_CODE_POINTS:
            raise _IssueError(PromptIssue("expanded_template_too_large", path=path))
        return merged

    def expansion(
        self, content: dict[str, Any], tokens: dict[str, list[Token]]
    ) -> dict[str, tuple[Token, ...]]:
        budget = [0]
        expanded: dict[str, tuple[Token, ...]] = {}
        for path, _ in _templates(content):
            try:
                expanded[path] = tuple(
                    self.expand(tokens[path], content["fragments"], 0, (), budget, path, None)
                )
            except _IssueError as failure:
                self.errors.append(failure.issue)
                break
        return expanded

    def usage(
        self, content: dict[str, Any], expanded: Mapping[str, tuple[Token, ...]]
    ) -> tuple[set[str], set[str], list[tuple[int, dict[str, Any]]]]:
        declared = content["variables"]
        used: set[str] = set()
        placeholders: set[str] = set()
        entries: list[tuple[int, dict[str, Any]]] = []

        def check(token: Token, path: str) -> None:
            name = token.text
            used.add(name)
            if name not in declared:
                self.error("undeclared_variable", path=path, variable=name)
                return
            if declared[name]["type"] == "messages":
                self.error("messages_outside_placeholder", path=path, variable=name)
                return
            origin = token.origin
            if origin is not None and isinstance(origin.variables, Mapping):
                fragment_decl = origin.variables.get(name, _MISSING)
                if fragment_decl is not _MISSING:
                    fragment_type = (
                        fragment_decl.get("type") if isinstance(fragment_decl, Mapping) else None
                    )
                    if fragment_type != declared[name]["type"]:
                        self.error(
                            "fragment_variable_type_mismatch",
                            path=path,
                            variable=name,
                            fragment=origin.key,
                        )

        if content["kind"] == "text":
            for token in expanded["/body"]:
                if token.kind == "var":
                    check(token, "/body")
        else:
            for index, entry in enumerate(content["body"]):
                if "role" in entry:
                    path = f"/body/{index}/content"
                    for token in expanded[path]:
                        if token.kind == "var":
                            check(token, path)
                    continue
                path = f"/body/{index}"
                name = entry["placeholder"]
                if name not in declared:
                    self.error("undeclared_variable", path=path, variable=name)
                elif declared[name]["type"] != "messages":
                    self.error("placeholder_type_mismatch", path=path, variable=name)
                if name in placeholders:
                    self.error("duplicate_placeholder", path=path, variable=name)
                placeholders.add(name)
                entries.append((index, entry))
        if content["kind"] == "text":
            for name in sorted_keys(declared):
                if declared[name]["type"] == "messages":
                    self.error(
                        "messages_outside_placeholder",
                        path=pointer("/variables", name),
                        variable=name,
                    )
        return used, placeholders, entries

    def requirements(
        self, content: dict[str, Any], entries: list[tuple[int, dict[str, Any]]]
    ) -> None:
        declared, partials = content["variables"], content["partials"]
        for index, entry in entries:
            if declared[entry["placeholder"]]["required"] != (not entry["optional"]):
                self.error(
                    "placeholder_required_mismatch",
                    path=f"/body/{index}",
                    variable=entry["placeholder"],
                )
        for name in sorted_keys(declared):
            decl = declared[name]
            if decl["type"] == "messages":
                continue
            path = pointer("/variables", name)
            if name in partials and decl["required"]:
                self.error("partial_required_mismatch", path=path, variable=name)
            if name not in partials and not decl["required"]:
                self.error("optional_without_partial", path=path, variable=name)

    def partial_values(self, content: dict[str, Any]) -> None:
        declared, partials = content["variables"], content["partials"]
        for name in sorted_keys(partials):
            path = pointer("/partials", name)
            value = partials[name]
            if name not in declared:
                self.error("partial_for_unknown_variable", path=path, variable=name)
                continue
            kind = declared[name]["type"]
            if kind == "messages":
                self.error("partial_for_messages_variable", path=path, variable=name)
                continue
            if isinstance(value, (list, dict)):
                self.error("partial_not_scalar", path=path, variable=name)
                continue
            if not _partial_matches(kind, value):
                self.error("partial_type_mismatch", path=path, variable=name)

    def output_contract(self, content: dict[str, Any]) -> None:
        contract = content["output_contract"]
        if contract is None:
            return
        good = (
            isinstance(contract, dict)
            and set(contract) == {"json_schema", "type"}
            and contract["type"] == "json_schema"
            and isinstance(contract["json_schema"], dict)
        )
        if good:
            schema = contract["json_schema"]
            good = (
                len(normalized_json(schema).encode("utf-8")) <= MAX_OUTPUT_CONTRACT_BYTES
                and _json_depth(schema) <= MAX_OUTPUT_CONTRACT_DEPTH
            )
        if not good:
            self.error("output_contract_invalid", path="/output_contract")


_MISSING = object()


def _partial_matches(kind: str, value: Any) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    return kind == "json"


def validate_content(
    content: Any,
    *,
    fragments: FragmentSource,
    prompt_kind: Optional[str] = None,
) -> ValidationReport:
    check = _Validation(fragments)
    document = check.shape(content)
    if document is None or check.errors:
        return check.report()
    check.limits(document)
    if check.errors:
        return check.report()
    check.secrets = content_secrets(document)
    check.entries(document)
    if check.errors:
        return check.report()
    tokens = check.syntax(document)
    if check.errors:
        return check.report()
    check.pins(document)
    if check.errors:
        return check.report()
    expanded = check.expansion(document, tokens)
    if check.errors:
        return check.report()
    included = {
        token.text for items in tokens.values() for token in items if token.kind == "include"
    }
    for name in sorted_keys(document["fragments"]):
        if name not in included:
            check.warn("fragment_unused", path=pointer("/fragments", name))
    used, placeholders, entries = check.usage(document, expanded)
    if check.errors:
        return check.report()
    declared = document["variables"]
    for name in sorted_keys(declared):
        if name not in used and name not in placeholders:
            check.warn("variable_unused", path=pointer("/variables", name), variable=name)
    check.requirements(document, entries)
    if check.errors:
        return check.report()
    check.partial_values(document)
    if check.errors:
        return check.report()
    check.output_contract(document)
    if check.errors:
        return check.report()
    for finding in check.secrets:
        check.error(
            "secret_detected", pattern=finding.pattern, path=finding.path, offset=finding.offset
        )
    for name in sorted_keys(declared):
        if is_secret_shaped_key(name):
            check.warn(
                "secret_shaped_variable_name", path=pointer("/variables", name), variable=name
            )
    if check.errors:
        return check.report()
    if prompt_kind is not None:
        wanted = "chat" if prompt_kind == "chat" else "text"
        if document["kind"] != wanted:
            check.error("prompt_kind_mismatch", path="/kind")
            return check.report()
    return check.report(
        declared=tuple(sorted_keys(declared)),
        referenced=tuple(sorted(used, key=lambda name: name.encode("utf-16-be"))),
        placeholders=tuple(sorted(placeholders, key=lambda name: name.encode("utf-16-be"))),
        content_digest=prompt_digest(document),
        content=document,
        expanded=expanded,
    )


@dataclass(frozen=True)
class RenderedPrompt:
    kind: Literal["text", "chat"]
    text: Optional[str]
    messages: Optional[list[Any]]
    rendered_hash: str
    content_digest: str
    warnings: tuple[PromptIssue, ...]
    expanded_template: Union[str, list[dict[str, Any]]]
    rendered_document: Mapping[str, Any]


def _fail(code: str, **values: Any) -> _IssueError:
    return _IssueError(PromptIssue(code, **values))


def _message_role(item: Any) -> Optional[str]:
    if isinstance(item, Mapping):
        role = item.get("role")
        return role if isinstance(role, str) else None
    kind = getattr(item, "type", None)
    return MESSAGE_TYPE_ROLES.get(kind) if isinstance(kind, str) else None


def _message_content(item: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get("content")
    return getattr(item, "content", None)


def _check_message(item: Any, path: str, variable: Optional[str]) -> None:
    if isinstance(item, Mapping):
        if "role" not in item or "content" not in item:
            raise _fail("invalid_message_value", value_path=path, variable=variable)
        if item["role"] not in MESSAGE_ROLES:
            raise _fail("invalid_message_value", value_path=path + "/role", variable=variable)
        content = item["content"]
        if not (content is None or isinstance(content, (str, list, tuple))):
            raise _fail("invalid_message_value", value_path=path + "/content", variable=variable)
        return
    kind = getattr(item, "type", None)
    if not isinstance(kind, str) or kind not in MESSAGE_TYPE_ROLES or not hasattr(item, "content"):
        raise _fail("invalid_message_value", value_path=path, variable=variable)


def _check_value(kind: str, value: Any, path: str, name: str) -> Any:
    def ajs(failure: AjsError) -> _IssueError:
        return _fail(failure.reason, variable=name, value_path=failure.value_path)

    if kind == "string":
        if not isinstance(value, str):
            raise _fail("type_mismatch", variable=name, value_path=path)
        try:
            check_string(value, path)
        except AjsError as failure:
            raise ajs(failure) from failure
        return "".join((value,))
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _fail("type_mismatch", variable=name, value_path=path)
        try:
            return ensure_ajs(
                value if isinstance(value, float) else operator.index(value), path=path
            )
        except AjsError as failure:
            raise ajs(failure) from failure
    if kind == "boolean":
        if not isinstance(value, bool):
            raise _fail("type_mismatch", variable=name, value_path=path)
        return value
    if kind == "json":
        try:
            return ensure_ajs(value, path=path)
        except AjsError as failure:
            raise ajs(failure) from failure
    if not isinstance(value, (list, tuple)):
        raise _fail("placeholder_not_list", variable=name, value_path=path)
    for index, item in enumerate(value):
        _check_message(item, pointer(path, index), name)
    return list(value)


def _render_scalar(kind: str, value: Any) -> str:
    if kind == "string":
        return str(value)
    if kind == "integer":
        return str(int(value))
    if kind == "boolean":
        return "true" if value else "false"
    return normalized_json(value)


def _scan_values(declared: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    def scan_at(text: str, path: str, name: str) -> None:
        findings = scan(text)
        if findings:
            raise _fail(
                "secret_in_variables", variable=name, value_path=path, pattern=findings[0].pattern
            )

    def walk(value: Any, path: str, name: str) -> None:
        if isinstance(value, str):
            scan_at(value, path, name)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, pointer(path, index), name)
        elif isinstance(value, dict):
            for key in sorted_keys(value):
                walk(value[key], pointer(path, key), name)

    for name in sorted_keys(declared):
        kind = declared[name]["type"]
        if kind == "string":
            scan_at(values[name], pointer("/variables", name), name)
        elif kind == "json":
            walk(values[name], pointer("/variables", name), name)


def _render(
    report: ValidationReport,
    variables: Mapping[str, Any],
    *,
    strict: bool,
    history: Optional[Sequence[Any]],
    allow_duplicate_system: bool,
    secret_policy: SecretPolicy,
    expect_kind: Optional[str],
) -> RenderedPrompt:
    if report.errors or report.content is None or report.expanded is None:
        raise _IssueError(report.errors[0])
    content = report.content
    expanded = report.expanded
    kind = content["kind"]
    if expect_kind is not None and kind != expect_kind:
        raise _fail("kind_mismatch")
    declared = content["variables"]
    partials = content["partials"]
    warnings = list(report.warnings)
    for key in variables:
        if not isinstance(key, str):
            raise _fail("unknown_variable", variable=str(key))
    for name in sorted_keys(variables):
        if name not in declared:
            if strict:
                raise _fail("unknown_variable", variable=name)
            warnings.append(PromptIssue("strict_disabled", variable=name))
    values: dict[str, Any] = {}
    for name in sorted_keys(declared):
        decl = declared[name]
        if name in variables:
            value = variables[name]
        elif name in partials:
            value = partials[name]
        elif decl["type"] == "messages" and not decl["required"]:
            value = []
        else:
            raise _fail("missing_variable", variable=name)
        values[name] = _check_value(decl["type"], value, pointer("/variables", name), name)
    if secret_policy == "error":
        _scan_values(declared, values)
    body = content["body"]
    has_placeholder = kind == "chat" and any("placeholder" in entry for entry in body)
    if history is not None:
        if kind == "text":
            raise _fail("history_not_supported")
        if has_placeholder:
            raise _fail("history_conflict")
        if not isinstance(history, (list, tuple)):
            raise _fail("invalid_message_value", value_path="/history")
        for index, item in enumerate(history):
            _check_message(item, pointer("/history", index), None)

    def render_tokens(tokens: Sequence[Token]) -> str:
        return "".join(
            token.text
            if token.kind == "literal"
            else _render_scalar(declared[token.text]["type"], values[token.text])
            for token in tokens
        )

    text: Optional[str] = None
    messages: Optional[list[Any]] = None
    document_messages: Optional[list[dict[str, Any]]] = None
    expanded_template: Union[str, list[dict[str, Any]]]
    history_items = list(history or [])
    size = 0
    if kind == "text":
        tokens = expanded["/body"]
        text = render_tokens(tokens)
        expanded_template = source(tokens)
        size = len(text)
    else:
        messages, document_messages, template_entries = [], [], []
        system_contents: set[str] = set()
        for index, entry in enumerate(body):
            if "role" in entry:
                tokens = expanded[f"/body/{index}/content"]
                rendered = render_tokens(tokens)
                messages.append(RenderedMessage(entry["role"], rendered))
                document_messages.append({"role": entry["role"], "content": rendered})
                template_entries.append({"role": entry["role"], "content": source(tokens)})
                if entry["role"] == "system":
                    system_contents.add(rendered)
                size += len(rendered)
            else:
                items = values[entry["placeholder"]]
                messages.extend(items)
                document_messages.append({"placeholder": entry["placeholder"], "count": len(items)})
                template_entries.append(
                    {"placeholder": entry["placeholder"], "optional": entry["optional"]}
                )
        messages.extend(history_items)
        expanded_template = template_entries
        if not allow_duplicate_system:
            for entry in body:
                if "placeholder" not in entry:
                    continue
                for index, item in enumerate(values[entry["placeholder"]]):
                    if _duplicate_system(item, system_contents):
                        raise _fail(
                            "duplicate_system_message",
                            value_path=pointer(pointer("/variables", entry["placeholder"]), index),
                        )
            for index, item in enumerate(history_items):
                if _duplicate_system(item, system_contents):
                    raise _fail("duplicate_system_message", value_path=pointer("/history", index))
    if size > MAX_RENDERED_CODE_POINTS:
        raise _fail("rendered_output_too_large")
    content_digest = report.content_digest or prompt_digest(content)
    document = {
        "schema": RENDERED_SCHEMA,
        "content_digest": content_digest,
        "kind": kind,
        "text": text,
        "messages": document_messages,
        "history_count": len(history_items),
    }
    return RenderedPrompt(
        kind=kind,
        text=text,
        messages=messages,
        rendered_hash=prompt_digest(document),
        content_digest=content_digest,
        warnings=tuple(warnings),
        expanded_template=expanded_template,
        rendered_document=document,
    )


def _duplicate_system(item: Any, system_contents: set[str]) -> bool:
    content = _message_content(item)
    return (
        _message_role(item) == "system" and isinstance(content, str) and content in system_contents
    )


def render_validated(
    report: ValidationReport,
    variables: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = True,
    history: Optional[Sequence[Any]] = None,
    allow_duplicate_system: bool = False,
    secret_policy: SecretPolicy = "off",
    expect_kind: Optional[Literal["text", "chat"]] = None,
) -> RenderedPrompt:
    try:
        return _render(
            report,
            variables or {},
            strict=strict,
            history=history,
            allow_duplicate_system=allow_duplicate_system,
            secret_policy=secret_policy,
            expect_kind=expect_kind,
        )
    except _IssueError as failure:
        raise render_error(failure.issue.to_dict()) from None


def render_content(
    content: Any,
    variables: Optional[Mapping[str, Any]] = None,
    *,
    fragments: FragmentSource,
    strict: bool = True,
    history: Optional[Sequence[Any]] = None,
    allow_duplicate_system: bool = False,
    secret_policy: SecretPolicy = "off",
    expect_kind: Optional[Literal["text", "chat"]] = None,
) -> RenderedPrompt:
    return render_validated(
        validate_content(content, fragments=fragments),
        variables,
        strict=strict,
        history=history,
        allow_duplicate_system=allow_duplicate_system,
        secret_policy=secret_policy,
        expect_kind=expect_kind,
    )
