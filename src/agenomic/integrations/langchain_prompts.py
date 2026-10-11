from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from agenomic.prompts.digest import sorted_keys
from agenomic.prompts.errors import PromptRenderError, render_error
from agenomic.prompts.importer import (
    SourceEntry,
    SourcePartial,
    SourcePrompt,
    convert_prompt,
)
from agenomic.prompts.models import ManagedPromptVersion, TemplateMessage
from agenomic.prompts.render import PromptIssue, RenderedMessage, SecretLocation

try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        convert_to_messages,
    )
    from langchain_core.prompts import (
        AIMessagePromptTemplate,
        BasePromptTemplate,
        ChatMessagePromptTemplate,
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
        PromptTemplate,
        SystemMessagePromptTemplate,
    )
    from langchain_core.prompts.chat import BaseChatPromptTemplate
except ImportError as error:
    raise ImportError(
        "langchain-core not installed. Install with: pip install agenomic[langchain]"
    ) from error

_ROLE_TEMPLATES = {
    "system": SystemMessagePromptTemplate,
    "user": HumanMessagePromptTemplate,
    "assistant": AIMessagePromptTemplate,
}
_ROLE_MESSAGES = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}


def to_langchain(version: ManagedPromptVersion) -> Union[ChatPromptTemplate, PromptTemplate]:
    declared = version.content.variables
    for name in sorted_keys(declared):
        if declared[name].type not in ("string", "messages"):
            raise render_error({"code": "unsupported_content", "variable": name})
    sources = version.expanded_sources()
    partials: dict[str, Any] = dict(version.content.partials)
    metadata = {
        "agenomic_prompt_ref": str(version.ref),
        "agenomic_prompt_content_digest": version.content_digest,
    }
    body = version.content.body
    if isinstance(body, str):
        return PromptTemplate.from_template(
            sources["/body"],
            template_format="f-string",
            partial_variables=partials,
            metadata=metadata,
        )
    messages: list[Any] = []
    for index, entry in enumerate(body):
        if isinstance(entry, TemplateMessage):
            messages.append(
                _ROLE_TEMPLATES[entry.role].from_template(
                    sources[f"/body/{index}/content"], template_format="f-string"
                )
            )
        else:
            messages.append(
                MessagesPlaceholder(variable_name=entry.placeholder, optional=entry.optional)
            )
    return ChatPromptTemplate(messages=messages, partial_variables=partials, metadata=metadata)


def to_langchain_messages(rendered: Sequence[Union[RenderedMessage, object]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for index, item in enumerate(rendered):
        if isinstance(item, RenderedMessage):
            out.append(_ROLE_MESSAGES[item.role](content=item.content))
        elif isinstance(item, BaseMessage):
            out.append(item)
        elif isinstance(item, Mapping) and "role" in item and "content" in item:
            out.extend(convert_to_messages([dict(item)]))
        else:
            raise render_error({"code": "type_mismatch", "value_path": f"/messages/{index}"})
    return out


_TEMPLATE_ROLES = {
    SystemMessagePromptTemplate: "system",
    HumanMessagePromptTemplate: "user",
    AIMessagePromptTemplate: "assistant",
}
_STATIC_ROLES = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}

Resolver = Callable[[str], Optional[ManagedPromptVersion]]


@dataclass(frozen=True)
class LangChainImport:
    status: Literal["supported", "unsupported", "blocked_secret"]
    exact: bool
    prompt_kind: Optional[Literal["text", "chat"]]
    content: Optional[dict[str, Any]]
    content_digest: Optional[str]
    issues: tuple[PromptIssue, ...]
    secret_findings: tuple[SecretLocation, ...] = ()


def _structure(obj: Any) -> Any:
    if type(obj) is PromptTemplate:
        return ("text", obj.template, obj.template_format, dict(obj.partial_variables))
    if type(obj) is not ChatPromptTemplate:
        return ("other", id(obj))
    messages: list[Any] = []
    for message in obj.messages:
        if isinstance(message, MessagesPlaceholder):
            messages.append(
                ("placeholder", message.variable_name, message.optional, message.n_messages)
            )
        elif type(message) in _TEMPLATE_ROLES and type(getattr(message, "prompt", None)) is (
            PromptTemplate
        ):
            template: Any = message
            messages.append(
                (
                    type(message).__name__,
                    template.prompt.template,
                    template.prompt.template_format,
                    dict(template.prompt.partial_variables),
                    dict(template.additional_kwargs),
                )
            )
        else:
            messages.append(("other", id(message)))
    return ("chat", tuple(messages), dict(obj.partial_variables))


def _exact(obj: Any, ref: str, digest: str, resolver: Resolver) -> Optional[LangChainImport]:
    resolved = resolver(ref)
    if resolved is None or resolved.content_digest != digest:
        return None
    try:
        exported = to_langchain(resolved)
    except PromptRenderError:
        return None
    if _structure(exported) != _structure(obj):
        return None
    return LangChainImport(
        status="supported",
        exact=True,
        prompt_kind=resolved.content.kind,
        content=dict(resolved.document),
        content_digest=resolved.content_digest,
        issues=(),
    )


def _partial(value: Any) -> SourcePartial:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return SourcePartial("value", value)
    if isinstance(value, list) and not value:
        return SourcePartial("value", [])
    if callable(value):
        return SourcePartial("callable")
    return SourcePartial("not_scalar")


def _template_entry(message: Any, partials: dict[str, Any]) -> SourceEntry:
    role = _TEMPLATE_ROLES[type(message)]
    prompt = message.prompt
    if type(prompt) is not PromptTemplate:
        return SourceEntry("refused", role=role, refusal="unsupported_content_block")
    if message.additional_kwargs:
        return SourceEntry(
            "refused", role=role, text=prompt.template, refusal="unsupported_message_attributes"
        )
    for name, value in prompt.partial_variables.items():
        if name in partials and partials[name] is not value and partials[name] != value:
            return SourceEntry(
                "refused", role=role, text=prompt.template, refusal="unsupported_message_attributes"
            )
        partials[name] = value
    return SourceEntry(
        "template", role=role, text=prompt.template, template_format=prompt.template_format
    )


def _static_entry(message: BaseMessage) -> SourceEntry:
    content = message.content
    text = content if isinstance(content, str) else None
    role = _STATIC_ROLES.get(type(message))
    if role is None:
        return SourceEntry("refused", text=text, refusal="unsupported_role")
    if text is None:
        return SourceEntry("refused", role=role, refusal="unsupported_content_block")
    if message.name or message.additional_kwargs or getattr(message, "tool_calls", None):
        return SourceEntry(
            "refused", role=role, text=text, refusal="unsupported_message_attributes"
        )
    return SourceEntry("static", role=role, text=text)


def _entry(message: Any, partials: dict[str, Any]) -> SourceEntry:
    if isinstance(message, MessagesPlaceholder):
        return SourceEntry(
            "placeholder",
            name=message.variable_name,
            optional=message.optional,
            n_messages=message.n_messages,
        )
    if type(message) in _TEMPLATE_ROLES:
        return _template_entry(message, partials)
    if isinstance(message, ChatMessagePromptTemplate):
        prompt = message.prompt
        text = prompt.template if isinstance(prompt, PromptTemplate) else None
        return SourceEntry("refused", text=text, refusal="unsupported_role")
    if isinstance(message, BaseMessage):
        return _static_entry(message)
    return SourceEntry("refused", refusal="unsupported_prompt_class")


def _source(obj: Any) -> SourcePrompt:
    if not isinstance(obj, BasePromptTemplate):
        raise TypeError("from_langchain takes a LangChain prompt template")
    parser = getattr(obj, "output_parser", None) is not None
    optional = tuple(getattr(obj, "optional_variables", None) or ())
    if type(obj) is PromptTemplate:
        return SourcePrompt(
            kind="text",
            template_format=obj.template_format,
            body=obj.template,
            partials={name: _partial(v) for name, v in obj.partial_variables.items()},
            optional_variables=optional,
            output_parser=parser,
        )
    if type(obj) is ChatPromptTemplate:
        partials: dict[str, Any] = dict(obj.partial_variables)
        entries = tuple(_entry(message, partials) for message in obj.messages)
        return SourcePrompt(
            kind="chat",
            entries=entries,
            partials={name: _partial(v) for name, v in partials.items()},
            optional_variables=optional,
            output_parser=parser,
        )
    kind: Literal["text", "chat"] = "chat" if isinstance(obj, BaseChatPromptTemplate) else "text"
    return SourcePrompt(kind=kind, refusal="unsupported_prompt_class")


def from_langchain(obj: Any, *, resolver: Optional[Resolver] = None) -> LangChainImport:
    metadata = getattr(obj, "metadata", None) or {}
    ref = metadata.get("agenomic_prompt_ref")
    digest = metadata.get("agenomic_prompt_content_digest")
    stale: list[PromptIssue] = []
    if isinstance(ref, str) and isinstance(digest, str) and resolver is not None:
        exact = _exact(obj, ref, digest, resolver)
        if exact is not None:
            return exact
        stale.append(PromptIssue("export_metadata_stale"))
    conversion = convert_prompt(_source(obj))
    issues = tuple(
        stale
        + [
            PromptIssue(item.code, path=item.path, offset=item.offset, syntax=item.syntax)
            for item in conversion.issues
        ]
    )
    return LangChainImport(
        status=conversion.status,
        exact=False,
        prompt_kind=conversion.prompt_kind,
        content=conversion.content,
        content_digest=conversion.content_digest,
        issues=issues,
        secret_findings=conversion.secrets,
    )
