from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

from agenomic._transport import api_request, segment
from agenomic.exceptions import ApiError
from agenomic.prompts.digest import (
    CONTENT_SCHEMA,
    ensure_ajs,
    pointer,
    prompt_digest,
    sorted_keys,
)
from agenomic.prompts.errors import AjsError, PromptImportError, integrity_error
from agenomic.prompts.render import (
    RENDERER_VERSION,
    TEMPLATE_FORMAT,
    SecretLocation,
    _TemplateSyntaxError,
    _tokenize,
    content_secrets,
    is_name,
    validate_content,
)
from agenomic.prompts.secrets import scan

if TYPE_CHECKING:
    from agenomic._client import Client

PROMPTS_FILE_SCHEMA = "agenomic.prompts_file/v1"
PROMPT_FILE_SCHEMA = "agenomic.prompt_file/v1"
DISCOVERY_SCHEMA = "agenomic.prompt_discovery_report/v1"
PLAN_SCHEMA = "agenomic.prompt_import_plan/v1"
APPLY_ACTIONS = ("create_prompt", "create_version", "reuse_version", "map_slot_only", "skip")
PLAN_ACTIONS = (*APPLY_ACTIONS, "blocked")
CHAT_ROLES = ("system", "user", "assistant")

ISSUE_SEVERITY: dict[str, str] = {
    "unsupported_template_format": "error",
    "unsupported_prompt_class": "error",
    "unsupported_content_block": "error",
    "unsupported_role": "error",
    "unsupported_message_attributes": "error",
    "static_message_escaped": "info",
    "unsupported_placeholder_option": "error",
    "fragment_syntax_in_import": "error",
    "variable_types_defaulted": "info",
    "partial_coerced_to_string": "warning",
    "callable_partial": "error",
    "partial_not_scalar": "error",
    "unused_partial_dropped": "warning",
    "unsupported_optional_variable": "error",
    "output_parser_not_imported": "warning",
    "secret_detected": "error",
    "export_metadata_stale": "info",
    "literal_transform_applied": "info",
    "python_fstring": "error",
    "dynamic_template": "error",
    "remote_prompt": "error",
    "template_composition": "error",
    "subagent_unmapped": "warning",
    "node_unresolved": "info",
}

ISSUE_MESSAGES: dict[str, str] = {
    "unsupported_template_format": "only f-string templates can be imported; rewrite the template "
    "with the agenomic-fstring/v1 grammar",
    "unsupported_prompt_class": "this prompt class cannot be imported; build it from plain "
    "templates and placeholders",
    "unsupported_content_block": "content blocks cannot be imported; use a text template",
    "unsupported_role": "only system, user and assistant messages can be imported",
    "unsupported_message_attributes": "message attributes such as name or additional_kwargs "
    "cannot be imported",
    "static_message_escaped": "braces of a static message were escaped so they stay literal",
    "unsupported_placeholder_option": "this placeholder form cannot be imported",
    "fragment_syntax_in_import": "a variable name starting with > is fragment syntax and cannot "
    "be imported",
    "variable_types_defaulted": "variable types default to string; review them before publishing",
    "partial_coerced_to_string": "a non-string partial was stored as the string LangChain prints",
    "callable_partial": "a callable partial cannot be imported; keep it as a runtime variable in "
    "application code",
    "partial_not_scalar": "only scalar partial values can be imported",
    "unused_partial_dropped": "a partial for a variable no template uses was dropped",
    "unsupported_optional_variable": "an optional variable that is not an optional placeholder "
    "cannot be imported",
    "output_parser_not_imported": "the output parser is code and was not imported",
    "secret_detected": "the template contains a credential; the content is withheld",
    "export_metadata_stale": "the export metadata no longer matches the object; it was imported "
    "structurally",
    "literal_transform_applied": "a standard library transform of a literal was applied",
    "python_fstring": "the template is an f-string, so Python substitutes it before LangChain "
    "sees a template",
    "dynamic_template": "the prompt is built at runtime; map it explicitly or keep it in code",
    "remote_prompt": "the prompt is pulled from a remote hub at runtime; import that prompt "
    "explicitly",
    "template_composition": "templates are composed at runtime; register the composed prompt "
    "explicitly",
    "subagent_unmapped": "the node runs a compiled subgraph or an agent; map it to a child agent",
    "node_unresolved": "the graph node of this prompt could not be determined statically",
}


@dataclass(frozen=True)
class SourceEntry:
    kind: Literal["template", "static", "placeholder", "refused"]
    role: Optional[str] = None
    text: Optional[str] = None
    template_format: str = "f-string"
    name: Optional[str] = None
    optional: bool = False
    n_messages: Optional[int] = None
    refusal: Optional[str] = None
    where: Any = None


@dataclass(frozen=True)
class SourcePartial:
    kind: Literal["value", "callable", "not_scalar"]
    value: Any = None
    where: Any = None


@dataclass(frozen=True)
class SourcePrompt:
    kind: Literal["text", "chat"]
    template_format: str = "f-string"
    body: Optional[str] = None
    static: bool = False
    entries: tuple[SourceEntry, ...] = ()
    partials: Mapping[str, SourcePartial] = field(default_factory=dict)
    optional_variables: tuple[str, ...] = ()
    output_parser: bool = False
    refusal: Optional[str] = None
    where: Any = None


@dataclass(frozen=True)
class ImportIssue:
    code: str
    severity: str
    message: str
    where: Any = None
    path: Optional[str] = None
    offset: Optional[int] = None
    syntax: Optional[str] = None

    @property
    def report_code(self) -> str:
        return self.syntax if self.code == "syntax_error" and self.syntax else self.code


@dataclass(frozen=True)
class Conversion:
    status: Literal["supported", "unsupported", "blocked_secret"]
    prompt_kind: Optional[Literal["text", "chat"]]
    content: Optional[dict[str, Any]]
    content_digest: Optional[str]
    issues: tuple[ImportIssue, ...]
    secrets: tuple[SecretLocation, ...]


def issue(
    code: str,
    *,
    where: Any = None,
    path: Optional[str] = None,
    offset: Optional[int] = None,
    message: Optional[str] = None,
    syntax: Optional[str] = None,
) -> ImportIssue:
    return ImportIssue(
        code=code,
        severity=ISSUE_SEVERITY.get(code, "error"),
        message=message or ISSUE_MESSAGES.get(code, f"the prompt cannot be imported: {code}"),
        where=where,
        path=path,
        offset=offset,
        syntax=syntax,
    )


def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _syntax_message(syntax: str, text: str, offset: int) -> str:
    message = f"template syntax error: {syntax}"
    if text[offset + 1 : offset + 2] in ('"', "'"):
        message += "; write {{ for a literal brace"
    return message


class _Mapping:
    def __init__(self, prompt: SourcePrompt) -> None:
        self.prompt = prompt
        self.issues: list[ImportIssue] = []
        self.secrets: list[SecretLocation] = []
        self.used: list[str] = []
        self.placeholders: dict[str, bool] = {}

    def add(self, code: str, **values: Any) -> None:
        self.issues.append(issue(code, **values))

    def scan(self, text: str, path: str) -> None:
        for finding in scan(text):
            self.secrets.append(
                SecretLocation(finding.pattern, path, finding.offset, finding.length)
            )

    def template(self, text: str, path: str, where: Any) -> None:
        self.scan(text, path)
        try:
            tokens = _tokenize(text)
        except _TemplateSyntaxError as failure:
            self.add(
                "syntax_error",
                where=where,
                path=path,
                offset=failure.offset,
                syntax=failure.syntax,
                message=_syntax_message(failure.syntax, text, failure.offset),
            )
            return
        for token in tokens:
            if token.kind == "include":
                self.add("fragment_syntax_in_import", where=where, path=path, offset=token.offset)
            elif token.kind == "var" and token.text not in self.used:
                self.used.append(token.text)

    def entry(self, index: int, entry: SourceEntry) -> Optional[dict[str, Any]]:
        path = f"/body/{index}"
        if entry.kind == "refused":
            if entry.text is not None:
                self.scan(entry.text, path + "/content")
            self.add(entry.refusal or "unsupported_prompt_class", where=entry.where, path=path)
            return None
        if entry.kind == "placeholder":
            name = entry.name or ""
            if entry.n_messages is not None:
                self.add("unsupported_placeholder_option", where=entry.where, path=path)
                return None
            if not is_name(name):
                self.add("unsupported_placeholder_option", where=entry.where, path=path)
                return None
            self.placeholders[name] = entry.optional
            return {"placeholder": name, "optional": entry.optional}
        text = entry.text or ""
        if entry.role not in CHAT_ROLES:
            self.scan(text, path + "/content")
            self.add("unsupported_role", where=entry.where, path=path)
            return None
        if entry.kind == "static":
            self.scan(text, path + "/content")
            escaped = escape_braces(text)
            if escaped != text:
                self.add("static_message_escaped", where=entry.where, path=path)
            return {"role": entry.role, "content": escaped}
        if entry.template_format != "f-string":
            self.scan(text, path + "/content")
            self.add("unsupported_template_format", where=entry.where, path=path)
            return None
        self.template(text, path + "/content", entry.where)
        return {"role": entry.role, "content": text}

    def partials(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in sorted_keys(self.prompt.partials):
            partial = self.prompt.partials[name]
            path = pointer("/partials", name)
            if partial.kind == "callable":
                self.add("callable_partial", where=partial.where, path=path)
                continue
            value = partial.value
            if name in self.placeholders:
                if partial.kind == "value" and isinstance(value, (list, tuple)) and not value:
                    continue
                self.add("partial_not_scalar", where=partial.where, path=path)
                continue
            if partial.kind == "not_scalar" or isinstance(value, (list, tuple, dict, set)):
                self.add("partial_not_scalar", where=partial.where, path=path)
                continue
            if isinstance(value, str):
                self.scan(value, path)
                text = value
            elif value is None or isinstance(value, (bool, int, float)):
                text = str(value)
                self.add("partial_coerced_to_string", where=partial.where, path=path)
            else:
                self.add("partial_not_scalar", where=partial.where, path=path)
                continue
            if name not in self.used:
                self.add("unused_partial_dropped", where=partial.where, path=path)
                continue
            out[name] = text
        return out

    def run(self) -> Conversion:
        prompt = self.prompt
        if prompt.refusal is not None:
            self.add(prompt.refusal, where=prompt.where)
            return self.finish(None)
        if prompt.template_format != "f-string":
            texts = [prompt.body] if prompt.kind == "text" else [e.text for e in prompt.entries]
            for index, text in enumerate(texts):
                if text is not None:
                    path = "/body" if prompt.kind == "text" else f"/body/{index}/content"
                    self.scan(text, path)
            self.add("unsupported_template_format", where=prompt.where)
            return self.finish(None)
        body: Union[str, list[dict[str, Any]]]
        if prompt.kind == "text":
            text = prompt.body or ""
            if prompt.static:
                self.scan(text, "/body")
                body = escape_braces(text)
                if body != text:
                    self.add("static_message_escaped", where=prompt.where, path="/body")
            else:
                self.template(text, "/body", prompt.where)
                body = text
        else:
            entries: list[dict[str, Any]] = []
            for index, entry in enumerate(prompt.entries):
                mapped = self.entry(index, entry)
                if mapped is not None:
                    entries.append(mapped)
            body = entries
        partials = self.partials()
        for name in prompt.optional_variables:
            if self.placeholders.get(name) is not True:
                self.add("unsupported_optional_variable", where=prompt.where)
        if prompt.output_parser:
            self.add("output_parser_not_imported", where=prompt.where)
        variables: dict[str, dict[str, Any]] = {}
        for name in self.used:
            variables[name] = {"type": "string", "required": name not in partials}
        for name, optional in self.placeholders.items():
            variables[name] = {"type": "messages", "required": not optional}
        if any(spec["type"] == "string" for spec in variables.values()):
            self.add("variable_types_defaulted", where=prompt.where)
        content = {
            "schema": CONTENT_SCHEMA,
            "template_format": TEMPLATE_FORMAT,
            "renderer_version": RENDERER_VERSION,
            "kind": prompt.kind,
            "body": body,
            "variables": variables,
            "partials": partials,
            "output_contract": None,
            "fragments": {},
        }
        return self.finish(content)

    def finish(self, content: Optional[dict[str, Any]]) -> Conversion:
        kind = self.prompt.kind
        if content is not None and not self.secrets:
            for finding in content_secrets(content):
                self.secrets.append(finding)
        if self.secrets:
            first = self.secrets[0]
            self.issues.insert(
                0, issue("secret_detected", where=self.prompt.where, path=first.path)
            )
            return Conversion(
                "blocked_secret", kind, None, None, tuple(self.issues), tuple(self.secrets)
            )
        errors = [item for item in self.issues if item.severity == "error"]
        if content is None or errors:
            return Conversion("unsupported", kind, None, None, tuple(self.issues), ())
        report = validate_content(content, fragments=lambda prompt_id, version, digest: None)
        for item in report.errors:
            self.issues.append(
                issue(item.code, where=self.prompt.where, path=item.path, offset=item.offset)
            )
        for item in report.warnings:
            self.issues.append(
                ImportIssue(
                    code=item.code,
                    severity="warning",
                    message=f"content warning: {item.code}",
                    where=self.prompt.where,
                    path=item.path,
                )
            )
        if report.errors or report.content is None or report.content_digest is None:
            return Conversion("unsupported", kind, None, None, tuple(self.issues), ())
        return Conversion(
            "supported",
            kind,
            dict(report.content),
            report.content_digest,
            tuple(self.issues),
            (),
        )


def convert_prompt(prompt: SourcePrompt) -> Conversion:
    return _Mapping(prompt).run()


def _invalid(code: str, message: str, **extra: Any) -> PromptImportError:
    item: dict[str, Any] = {"code": code}
    item.update({key: value for key, value in extra.items() if value is not None})
    details: dict[str, Any] = {"reason": code, "errors": [item]}
    details.update(item)
    details.pop("code")
    return PromptImportError("prompt_import_invalid", 0, message, details)


_YAML_INT = re.compile(r"[-+]?[0-9]+", re.ASCII)
_YAML_FLOAT = re.compile(
    r"[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)",
    re.ASCII,
)


class _ProfileError(Exception):
    def __init__(self, code: str, mark: Any = None) -> None:
        super().__init__(code)
        self.code = code
        self.line = mark.line + 1 if mark is not None else None
        self.column = mark.column + 1 if mark is not None else None


def _plain_scalar(value: str, mark: Any) -> Any:
    if value in ("true", "false"):
        return value == "true"
    if value in ("null", "~", ""):
        return None
    if _YAML_INT.fullmatch(value):
        return int(value)
    if _YAML_FLOAT.fullmatch(value):
        raise _ProfileError("float_not_allowed", mark)
    return value


def _yaml_events(text: str) -> list[Any]:
    try:
        import yaml
    except ImportError as error:
        raise PromptImportError(
            "yaml_support_not_installed",
            0,
            "YAML input needs the optional dependency: pip install agenomic[yaml]",
        ) from error
    try:
        return list(yaml.parse(text, Loader=yaml.SafeLoader))
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        raise _ProfileError("yaml_syntax_error", mark) from error


def _compose(events: list[Any]) -> Any:
    import yaml

    position = 0
    documents = 0
    result: Any = None

    def node() -> Any:
        nonlocal position
        event = events[position]
        position += 1
        mark = getattr(event, "start_mark", None)
        if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None) is not None:
            raise _ProfileError("yaml_alias_unsupported", mark)
        if getattr(event, "tag", None) is not None:
            raise _ProfileError("yaml_tag_unsupported", mark)
        if isinstance(event, yaml.ScalarEvent):
            return _plain_scalar(event.value, mark) if event.style is None else event.value
        if isinstance(event, yaml.SequenceStartEvent):
            items = []
            while not isinstance(events[position], yaml.SequenceEndEvent):
                items.append(node())
            position += 1
            return items
        if isinstance(event, yaml.MappingStartEvent):
            mapping: dict[str, Any] = {}
            while not isinstance(events[position], yaml.MappingEndEvent):
                key_mark = getattr(events[position], "start_mark", None)
                key = node()
                if not isinstance(key, str):
                    raise _ProfileError("invalid_field_type", key_mark)
                if key == "<<":
                    raise _ProfileError("yaml_alias_unsupported", key_mark)
                if key in mapping:
                    raise _ProfileError("yaml_duplicate_key", key_mark)
                mapping[key] = node()
            position += 1
            return mapping
        raise _ProfileError("yaml_syntax_error", mark)

    while position < len(events):
        event = events[position]
        if isinstance(event, yaml.DocumentStartEvent):
            documents += 1
            if documents > 1:
                raise _ProfileError("yaml_multiple_documents", getattr(event, "start_mark", None))
            position += 1
            result = node()
            continue
        position += 1
    return result


def load_yaml(text: str) -> Any:
    if text.startswith("﻿"):
        text = text[1:]
    try:
        document = _compose(_yaml_events(text))
        return ensure_ajs(document)
    except _ProfileError as failure:
        raise _invalid(
            failure.code,
            f"the YAML document is outside the agenomic-yaml/1 profile: {failure.code}",
            line=failure.line,
            column=failure.column,
        ) from None
    except AjsError as failure:
        raise _invalid(
            failure.reason,
            f"the YAML document is outside the JSON subset: {failure.reason}",
            value_path=failure.value_path,
        ) from None
    except RecursionError:
        raise _invalid("json_too_deep", "the YAML document is nested too deeply") from None


def _refuse_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _ProfileError("duplicate_key")
        out[key] = value
    return out


def load_json(text: str) -> Any:
    try:
        return ensure_ajs(json.loads(text, object_pairs_hook=_refuse_duplicates))
    except _ProfileError as failure:
        raise _invalid(failure.code, f"the JSON document is invalid: {failure.code}") from None
    except AjsError as failure:
        raise _invalid(
            failure.reason,
            f"the JSON document is outside the JSON subset: {failure.reason}",
            value_path=failure.value_path,
        ) from None
    except (ValueError, RecursionError):
        raise _invalid("invalid_json", "the document is not valid JSON") from None


Source = Union[str, bytes, "os.PathLike[str]"]
Format = Optional[Literal["json", "yaml"]]


def parse_document(source: Source, *, format: Format = None) -> Any:
    if isinstance(source, os.PathLike):
        path = Path(source)
        data = path.read_bytes()
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "yaml"
    else:
        data = source.encode("utf-8") if isinstance(source, str) else source
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise _invalid("invalid_unicode", "the document is not UTF-8") from None
    if format is None:
        format = "json" if text.lstrip("﻿ \t\r\n").startswith("{") else "yaml"
    if format == "json":
        return load_json(text[1:] if text.startswith("﻿") else text)
    return load_yaml(text)


def _expect(condition: bool, code: str, message: str, path: str) -> None:
    if not condition:
        raise _invalid(code, message, path=path)


def _check_fragment_graph(prompts: list[Any]) -> None:
    local = {prompt["prompt_id"]: prompt for prompt in prompts}
    edges: dict[str, list[str]] = {}
    for index, prompt in enumerate(prompts):
        targets: list[str] = []
        fragments = prompt["content"].get("fragments") or {}
        _expect(
            isinstance(fragments, dict),
            "invalid_field_type",
            "fragments must be a mapping",
            f"/prompts/{index}/content/fragments",
        )
        for name in sorted_keys(fragments):
            entry = fragments[name]
            path = pointer(f"/prompts/{index}/content/fragments", name)
            _expect(
                isinstance(entry, dict) and isinstance(entry.get("prompt_id"), str),
                "invalid_field_type",
                "a fragment entry needs a prompt_id",
                path,
            )
            if "version" in entry:
                continue
            _expect(
                entry["prompt_id"] in local,
                "fragment_not_found",
                "a fragment without a version must name a prompt of the same file",
                path,
            )
            targets.append(entry["prompt_id"])
        edges[prompt["prompt_id"]] = targets
    state: dict[str, int] = {}
    for start in edges:
        if state.get(start) == 2:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        state[start] = 1
        while stack:
            current, position = stack[-1]
            if position < len(edges[current]):
                stack[-1] = (current, position + 1)
                target = edges[current][position]
                if state.get(target) == 1:
                    raise _invalid(
                        "fragment_cycle",
                        "file-local fragment references form a cycle",
                        prompt_id=target,
                    )
                if state.get(target) != 2:
                    state[target] = 1
                    stack.append((target, 0))
            else:
                state[current] = 2
                stack.pop()


def check_prompts_file(document: Any) -> dict[str, Any]:
    _expect(isinstance(document, dict), "invalid_field_type", "a prompts file is a mapping", "")
    _expect(
        document.get("schema") == PROMPTS_FILE_SCHEMA,
        "unsupported_schema",
        f"the schema member must be {PROMPTS_FILE_SCHEMA}",
        "/schema",
    )
    prompts = document.get("prompts")
    _expect(isinstance(prompts, list), "invalid_field_type", "prompts must be a list", "/prompts")
    seen: set[str] = set()
    for index, prompt in enumerate(prompts):
        path = f"/prompts/{index}"
        _expect(isinstance(prompt, dict), "invalid_field_type", "a prompt is a mapping", path)
        _expect(
            isinstance(prompt.get("prompt_id"), str),
            "missing_field",
            "a prompt needs a prompt_id",
            path + "/prompt_id",
        )
        _expect(
            prompt["prompt_id"] not in seen,
            "duplicate_prompt_id",
            "a prompt id appears twice in the file",
            path + "/prompt_id",
        )
        seen.add(prompt["prompt_id"])
        _expect(
            isinstance(prompt.get("content"), dict),
            "missing_field",
            "a prompt needs a content mapping",
            path + "/content",
        )
    _check_fragment_graph(prompts)
    return cast(dict[str, Any], document)


def load_prompts_file(source: Source, *, format: Format = None) -> dict[str, Any]:
    return check_prompts_file(parse_document(source, format=format))


def complete_content(content: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(content)
    out.setdefault("schema", CONTENT_SCHEMA)
    out.setdefault("template_format", TEMPLATE_FORMAT)
    out.setdefault("renderer_version", RENDERER_VERSION)
    out.setdefault("partials", {})
    out.setdefault("output_contract", None)
    out.setdefault("fragments", {})
    return out


def load_prompt_file(source: Source, *, format: Format = None) -> dict[str, Any]:
    document = parse_document(source, format=format)
    _expect(isinstance(document, dict), "invalid_field_type", "a prompt file is a mapping", "")
    schema = document.get("schema")
    if schema == CONTENT_SCHEMA:
        return {"schema": PROMPT_FILE_SCHEMA, "content": document}
    _expect(
        schema == PROMPT_FILE_SCHEMA,
        "unsupported_schema",
        f"the schema member must be {PROMPT_FILE_SCHEMA} or {CONTENT_SCHEMA}",
        "/schema",
    )
    content = document.get("content")
    _expect(isinstance(content, dict), "missing_field", "a prompt file needs content", "/content")
    kind = document.get("kind")
    if kind is not None:
        wanted = "chat" if kind == "chat" else "text"
        _expect(
            content.get("kind") == wanted,
            "prompt_kind_mismatch",
            "the prompt kind does not fit the content kind",
            "/kind",
        )
    expected = document.get("content_digest")
    if expected is not None:
        actual = prompt_digest(content)
        if actual != expected:
            raise integrity_error(
                "prompt_digest_mismatch",
                "the content digest of the prompt file does not match its content",
                expected=expected,
                actual=actual,
            )
    return cast(dict[str, Any], document)


def valid_label(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 128
        and "\0" not in value
        and re.match(r"(/|\\|~|[A-Za-z]:)", value) is None
    )


def _repo_path(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value != ""
        and not value.startswith("/")
        and "\\" not in value
        and ".." not in value.split("/")
        and not re.match(r"[A-Za-z]:", value)
    )


def check_report(report: Mapping[str, Any]) -> dict[str, Any]:
    try:
        document = ensure_ajs(report)
    except AjsError as failure:
        raise _invalid(
            failure.reason, "the report is outside the JSON subset", value_path=failure.value_path
        ) from None
    _expect(isinstance(document, dict), "invalid_field_type", "a report is a mapping", "")
    _expect(
        document.get("schema") == DISCOVERY_SCHEMA,
        "unsupported_schema",
        f"the schema member must be {DISCOVERY_SCHEMA}",
        "/schema",
    )
    root = document.get("root")
    label = root.get("label") if isinstance(root, dict) else None
    _expect(
        valid_label(label),
        "invalid_field_type",
        "root.label must be a short name, never an absolute path",
        "/root/label",
    )
    files = document.get("files")
    candidates = document.get("candidates")
    _expect(isinstance(files, list), "invalid_field_type", "files must be a list", "/files")
    _expect(
        isinstance(candidates, list),
        "invalid_field_type",
        "candidates must be a list",
        "/candidates",
    )
    for index, entry in enumerate(files):
        _expect(
            isinstance(entry, dict) and _repo_path(entry.get("path")),
            "invalid_field_type",
            "file paths are repository-relative POSIX paths",
            f"/files/{index}/path",
        )
    for index, candidate in enumerate(candidates):
        path = f"/candidates/{index}"
        _expect(isinstance(candidate, dict), "invalid_field_type", "a candidate is a mapping", path)
        source = candidate.get("source")
        _expect(
            isinstance(source, dict) and _repo_path(source.get("path")),
            "invalid_field_type",
            "source paths are repository-relative POSIX paths",
            path + "/source/path",
        )
        content = candidate.get("content")
        if candidate.get("status") != "supported":
            _expect(
                content is None and candidate.get("content_digest") is None,
                "invalid_field_type",
                "only a supported candidate carries content",
                path + "/content",
            )
            continue
        _expect(isinstance(content, dict), "missing_field", "supported content", path + "/content")
        _expect(
            prompt_digest(content) == candidate.get("content_digest"),
            "prompt_digest_mismatch",
            "the content digest of a candidate does not match its content",
            path + "/content_digest",
        )
        _expect(
            not content_secrets(content),
            "secret_detected",
            "a candidate content contains a credential",
            path + "/content",
        )
    outside = {**document, "candidates": [{**item, "content": None} for item in candidates]}
    for path, text in _strings(outside, ""):
        _expect(
            not scan(text),
            "secret_detected",
            "a report member outside the contents contains a credential",
            path,
        )
    return cast(dict[str, Any], document)


def _strings(value: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}/{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield path, key
            yield from _strings(item, pointer(path, key))


def build_import_request(
    report: Mapping[str, Any],
    *,
    agent_id: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"report": check_report(report)}
    if agent_id is not None:
        body["agent_id"] = agent_id
    if options:
        body["options"] = dict(options)
    return body


def plan_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    summary = {action: 0 for action in PLAN_ACTIONS}
    summary["unresolved"] = 0
    for item in items:
        action = item.get("action")
        if action in summary:
            summary[action] += 1
        slot = item.get("slot")
        if isinstance(slot, Mapping) and slot.get("status") == "unresolved":
            summary["unresolved"] += 1
    return summary


def _bad_plan(message: str) -> ApiError:
    return ApiError("invalid_response", 0, f"invalid import plan: {message}")


def verify_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise _bad_plan("schema")
    items = plan.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise _bad_plan("items")
    unsigned = {key: value for key, value in plan.items() if key != "plan_digest"}
    actual = prompt_digest(unsigned)
    if actual != plan.get("plan_digest"):
        raise integrity_error(
            "prompt_digest_mismatch",
            "the plan digest does not match the plan",
            document="import_plan",
            expected=plan.get("plan_digest"),
            actual=actual,
        )
    if plan.get("summary") != plan_summary(items):
        raise _bad_plan("summary")
    for item in items:
        if item.get("action") not in PLAN_ACTIONS:
            raise _bad_plan("action")
        content = item.get("content")
        if content is not None and prompt_digest(content) != item.get("content_digest"):
            raise integrity_error(
                "prompt_digest_mismatch",
                "the content digest of a plan item does not match its content",
                document="import_plan",
                item_id=item.get("item_id"),
            )
        slot = item.get("slot")
        unresolved = isinstance(slot, dict) and slot.get("status") == "unresolved"
        if unresolved and item.get("action") != "skip":
            raise _bad_plan("an unresolved item must be skipped")
    return plan


def default_decisions(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for item in plan["items"]:
        action = item["action"] if item["action"] in APPLY_ACTIONS else "skip"
        decision: dict[str, Any] = {"item_id": item["item_id"], "action": action}
        if action != "skip" and item.get("prompt_id") is not None:
            decision["prompt_id"] = item["prompt_id"]
        if action == "create_version":
            decision["base_version"] = item["base_version"]
        slot = item.get("slot")
        if isinstance(slot, Mapping):
            if slot.get("slot_path") is not None:
                decision["slot_path"] = slot["slot_path"]
            if action == "map_slot_only" and slot.get("subagent_id") is not None:
                decision["subagent_id"] = slot["subagent_id"]
        decisions.append(decision)
    return decisions


def new_idempotency_key(prefix: str = "import-apply") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def build_apply_request(
    plan: Mapping[str, Any],
    *,
    idempotency_key: str,
    items: Optional[Sequence[Mapping[str, Any]]] = None,
    agent_id: Optional[str] = None,
    mode: Literal["publish", "draft"] = "publish",
    declare_slots: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "idempotency_key": idempotency_key,
        "plan_digest": plan["plan_digest"],
        "mode": mode,
        "declare_slots": declare_slots,
        "items": [dict(item) for item in items] if items is not None else default_decisions(plan),
    }
    if agent_id is not None:
        body["agent_id"] = agent_id
    return body


def upload_report(
    client: Client,
    report: Mapping[str, Any],
    *,
    agent_id: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    body = build_import_request(report, agent_id=agent_id, options=options)
    response = api_request(client, "POST", "/v1/prompts/imports", body)
    record = response.body.get("import")
    if not isinstance(record, dict):
        raise ApiError("invalid_response", response.status, "the response carries no import")
    record["plan"] = verify_plan(record.get("plan"))
    if not isinstance(record["plan"].get("plan_id"), str):
        raise ApiError("invalid_response", response.status, "the import plan carries no plan_id")
    record["replayed"] = response.body.get("replayed") is True
    return record


def apply_import(
    client: Client,
    plan: Mapping[str, Any],
    *,
    idempotency_key: str,
    items: Optional[Sequence[Mapping[str, Any]]] = None,
    agent_id: Optional[str] = None,
    mode: Literal["publish", "draft"] = "publish",
    declare_slots: bool = False,
    expected_slots_revision: Optional[int] = None,
) -> dict[str, Any]:
    if declare_slots and expected_slots_revision is None:
        raise ValueError("declare_slots needs expected_slots_revision")
    body = build_apply_request(
        plan,
        idempotency_key=idempotency_key,
        items=items,
        agent_id=agent_id,
        mode=mode,
        declare_slots=declare_slots,
    )
    response = api_request(
        client,
        "POST",
        f"/v1/prompts/imports/{segment(plan['plan_id'])}/apply",
        body,
        if_match=expected_slots_revision if declare_slots else None,
    )
    return response.body
