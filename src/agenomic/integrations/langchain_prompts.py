from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Union

from agenomic.prompts.digest import sorted_keys
from agenomic.prompts.errors import render_error
from agenomic.prompts.models import ManagedPromptVersion, TemplateMessage
from agenomic.prompts.render import RenderedMessage

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
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
        PromptTemplate,
        SystemMessagePromptTemplate,
    )
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
