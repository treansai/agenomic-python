from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    StrictStr,
)

from agenomic.prompts.digest import artifact_set, ensure_ajs, verify_version
from agenomic.prompts.errors import integrity_error
from agenomic.prompts.refs import PromptUri, PromptVersionRef
from agenomic.prompts.render import (
    RenderedMessage,
    RenderedPrompt,
    SecretPolicy,
    ValidationReport,
    render_validated,
    source,
    validate_content,
)

if TYPE_CHECKING:
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

__all__ = [
    "ExecutionBinding",
    "FragmentPin",
    "ManagedPromptVersion",
    "Placeholder",
    "PromptContent",
    "PromptManifest",
    "PromptVersionRecord",
    "RenderedMessage",
    "ResolvedClosure",
    "ResolvedFrom",
    "TemplateMessage",
    "VariableSpec",
]

PromptKind = Literal["text", "chat", "fragment"]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class VariableSpec(_Strict):
    type: Literal["string", "integer", "boolean", "json", "messages"]
    required: bool


class FragmentPin(_Strict):
    prompt_id: str
    version: int
    content_digest: str


class TemplateMessage(_Strict):
    role: Literal["system", "user", "assistant"]
    content: str


class Placeholder(_Strict):
    placeholder: str
    optional: bool


class PromptContent(_Strict):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, populate_by_name=True)

    schema_: Literal["agenomic.prompt_content/v1"] = Field(alias="schema")
    template_format: Literal["agenomic-fstring/v1"]
    renderer_version: Literal["1"]
    kind: Literal["text", "chat"]
    body: Union[str, list[Union[TemplateMessage, Placeholder]]]
    variables: dict[str, VariableSpec]
    partials: dict[str, Union[StrictBool, StrictInt, StrictStr, None]]
    output_contract: Optional[dict[str, Any]]
    fragments: dict[str, FragmentPin]

    def to_document(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)


class PromptManifest(_Strict):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, populate_by_name=True)

    schema_: Literal["agenomic.prompt_manifest/v1"] = Field(alias="schema")
    agent_id: str
    slots: dict[str, FragmentPin]
    children: dict[str, dict[str, Optional[str]]]

    def to_document(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)


@dataclass(frozen=True)
class ResolvedFrom:
    alias: str
    generation: Optional[int] = None


class PromptVersionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    prompt_id: StrictStr
    version: StrictInt
    prompt_kind: Optional[PromptKind] = None
    content_digest: StrictStr
    content: dict[str, Any]
    workspace_id: Optional[str] = None
    parent_version: Optional[int] = None
    change_message: Optional[str] = None
    author: Optional[dict[str, Optional[str]]] = None
    created_at: Optional[datetime] = None

    @property
    def ref(self) -> str:
        return f"{self.prompt_id}:{self.version}"


RecordLookup = Callable[[str, int], Optional[PromptVersionRecord]]


class ManagedPromptVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref: PromptVersionRef
    workspace_id: str
    kind: PromptKind
    content_digest: str
    content: PromptContent
    fragments: dict[str, ManagedPromptVersion]
    parent_version: Optional[int] = None
    author: Optional[dict[str, Optional[str]]] = None
    change_message: Optional[str] = None
    created_at: Optional[datetime] = None
    resolved_from: Optional[ResolvedFrom] = None

    _document: dict[str, Any] = PrivateAttr(default_factory=dict)
    _report: Optional[ValidationReport] = PrivateAttr(default=None)
    _closure: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_record(
        cls,
        record: PromptVersionRecord,
        *,
        workspace_id: str,
        lookup: RecordLookup,
    ) -> ManagedPromptVersion:
        return _build(record, workspace_id, lookup, {})

    @property
    def uri(self) -> PromptUri:
        return PromptUri(self.workspace_id, self.ref.prompt_id, self.ref.version)

    @property
    def variables(self) -> Mapping[str, VariableSpec]:
        return self.content.variables

    @property
    def document(self) -> dict[str, Any]:
        return self._document

    def fragment_source(
        self, prompt_id: str, version: int, digest: str
    ) -> Optional[dict[str, Any]]:
        return self._closure.get(f"{prompt_id}:{version}")

    def expanded_sources(self) -> dict[str, str]:
        expanded = self._validated().expanded or {}
        return {path: source(tokens) for path, tokens in expanded.items()}

    def render(
        self,
        variables: Optional[Mapping[str, Any]] = None,
        *,
        strict: bool = True,
        history: Optional[Sequence[Any]] = None,
        allow_duplicate_system: bool = False,
        secret_policy: SecretPolicy = "off",
    ) -> RenderedPrompt:
        return render_validated(
            self._validated(),
            variables,
            strict=strict,
            history=history,
            allow_duplicate_system=allow_duplicate_system,
            secret_policy=secret_policy,
        )

    def render_text(
        self, variables: Optional[Mapping[str, Any]] = None, *, strict: bool = True
    ) -> str:
        result = render_validated(self._validated(), variables, strict=strict, expect_kind="text")
        return cast(str, result.text)

    def render_messages(
        self,
        variables: Optional[Mapping[str, Any]] = None,
        *,
        strict: bool = True,
        allow_duplicate_system: bool = False,
        secret_policy: SecretPolicy = "off",
    ) -> list[Union[RenderedMessage, object]]:
        result = render_validated(
            self._validated(),
            variables,
            strict=strict,
            allow_duplicate_system=allow_duplicate_system,
            secret_policy=secret_policy,
            expect_kind="chat",
        )
        return list(result.messages or [])

    def compose(
        self,
        variables: Optional[Mapping[str, Any]] = None,
        *,
        history: Sequence[object],
        strict: bool = True,
        allow_duplicate_system: bool = False,
        secret_policy: SecretPolicy = "off",
    ) -> list[Union[RenderedMessage, object]]:
        result = render_validated(
            self._validated(),
            variables,
            strict=strict,
            history=history,
            allow_duplicate_system=allow_duplicate_system,
            secret_policy=secret_policy,
        )
        return list(result.messages or [])

    def to_langchain(self) -> Union[ChatPromptTemplate, PromptTemplate]:
        from agenomic.integrations.langchain_prompts import to_langchain

        return to_langchain(self)

    def verify(self) -> None:
        verify_version(self.record())
        for name, fragment in self.fragments.items():
            pin = self.content.fragments[name]
            if fragment.content_digest != pin.content_digest:
                raise integrity_error(
                    "prompt_digest_mismatch",
                    f"fragment {name} of {self.ref} does not match its pin",
                    ref=str(fragment.ref),
                    expected=pin.content_digest,
                    actual=fragment.content_digest,
                )
            fragment.verify()

    def record(self) -> PromptVersionRecord:
        return PromptVersionRecord(
            prompt_id=self.ref.prompt_id,
            version=self.ref.version,
            prompt_kind=self.kind,
            content_digest=self.content_digest,
            content=self._document,
            workspace_id=self.workspace_id,
            parent_version=self.parent_version,
            change_message=self.change_message,
            author=self.author,
            created_at=self.created_at,
        )

    def closure_records(self) -> list[PromptVersionRecord]:
        seen: dict[str, PromptVersionRecord] = {}
        pending = [self]
        while pending:
            version = pending.pop()
            key = str(version.ref)
            if key in seen:
                continue
            seen[key] = version.record()
            pending.extend(version.fragments.values())
        return [seen[key] for key in sorted(seen)]

    def _validated(self) -> ValidationReport:
        if self._report is None:
            raise ValueError("ManagedPromptVersion must be built with from_record")
        return self._report


def _build(
    record: PromptVersionRecord,
    workspace_id: str,
    lookup: RecordLookup,
    memo: dict[str, ManagedPromptVersion],
) -> ManagedPromptVersion:
    built = memo.get(record.ref)
    if built is not None:
        return built
    verify_version(record)
    document = cast(dict[str, Any], ensure_ajs(record.content))
    closure: dict[str, dict[str, Any]] = {}

    def source(prompt_id: str, version: int, digest: str) -> Optional[dict[str, Any]]:
        found = lookup(prompt_id, version)
        if found is None:
            return None
        entry: dict[str, Any] = {"content": found.content}
        if found.prompt_kind is not None:
            entry["prompt_kind"] = found.prompt_kind
        closure[found.ref] = entry
        return entry

    report = validate_content(document, fragments=source, prompt_kind=record.prompt_kind)
    report.raise_for_errors()
    fragments: dict[str, ManagedPromptVersion] = {}
    for name, pin in document["fragments"].items():
        found = lookup(pin["prompt_id"], pin["version"])
        if found is not None:
            child = _build(found, workspace_id, lookup, memo)
            fragments[name] = child
            closure.update(child._closure)
    version = ManagedPromptVersion(
        ref=PromptVersionRef(record.prompt_id, record.version),
        workspace_id=workspace_id,
        kind=record.prompt_kind or document["kind"],
        content_digest=record.content_digest,
        content=PromptContent.model_validate(document),
        fragments=fragments,
        parent_version=record.parent_version,
        author=record.author,
        change_message=record.change_message,
        created_at=record.created_at,
    )
    version._document = document
    version._report = report
    version._closure = closure
    memo[record.ref] = version
    return version


class ExecutionBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    schema_: Literal["agenomic.execution_binding/v1"] = Field(
        default="agenomic.execution_binding/v1", alias="schema"
    )
    binding_id: str
    workspace_id: str
    agent_id: str
    thread_key: str
    scope: Literal["thread", "execution"]
    release_id: str
    release_name: str
    genome_version: Optional[str]
    prompt_manifest_digest: str
    runtime: dict[str, Any]
    resolved_from: dict[str, Any]
    children: dict[str, dict[str, Any]]
    parent_binding_id: Optional[str] = None
    experiment: Optional[dict[str, Any]] = None
    runtime_client: Optional[dict[str, Any]] = None
    created_at: str
    created_by: Optional[dict[str, Optional[str]]] = None

    def to_document(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)


class ResolvedClosure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_manifest_digest: str
    manifest: dict[str, Any]
    children: dict[str, Any]
    prompts: dict[str, Any]
    prompt_bundle_digest: str

    def artifact_set(self) -> dict[str, Any]:
        return artifact_set(self.model_dump())
