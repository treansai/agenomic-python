from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
from prompt_fakes import chat_content, required, seeded_engine, text_content

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

from agenomic.integrations.langchain_prompts import to_langchain, to_langchain_messages
from agenomic.prompts import (
    ManagedPromptVersion,
    PromptRenderError,
    PromptVersionRecord,
    RenderedMessage,
    prompt_digest,
)

WORKSPACE = "0b6c2f1e-7a44-4c8e-9f1d-2a3b4c5d6e7f"
ROLES = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}
TEMPLATE_ROLES = {
    "SystemMessagePromptTemplate": "system",
    "HumanMessagePromptTemplate": "user",
    "AIMessagePromptTemplate": "assistant",
}


def build(content: dict[str, Any]) -> ManagedPromptVersion:
    record = PromptVersionRecord(
        prompt_id="prm_test",
        version=3,
        prompt_kind=content["kind"],
        content_digest=prompt_digest(content),
        content=content,
    )
    return ManagedPromptVersion.from_record(
        record, workspace_id=WORKSPACE, lookup=lambda p, v: None
    )


def structural_content(template: Any) -> dict[str, Any]:
    partials = dict(template.partial_variables)
    if isinstance(template, PromptTemplate):
        names = set(template.input_variables) | set(partials)
        return text_content(
            template.template,
            {name: {"type": "string", "required": name not in partials} for name in names},
            partials=partials,
        )
    body: list[dict[str, Any]] = []
    variables: dict[str, Any] = {}
    for message in template.messages:
        if isinstance(message, MessagesPlaceholder):
            partials.pop(message.variable_name, None)
            body.append({"placeholder": message.variable_name, "optional": message.optional})
            variables[message.variable_name] = {
                "type": "messages",
                "required": not message.optional,
            }
            continue
        role = TEMPLATE_ROLES[type(message).__name__]
        body.append({"role": role, "content": message.prompt.template})
        for name in message.prompt.input_variables:
            variables[name] = {"type": "string", "required": name not in partials}
    return chat_content(body, variables, partials=partials)


def test_to_langchain_roundtrip_strings() -> None:
    chat = chat_content(
        [
            {"role": "system", "content": "You help {customer} in {{json}} mode, tone {tone}."},
            {"placeholder": "history", "optional": True},
            {"role": "user", "content": "{question}"},
            {"role": "assistant", "content": "Noted."},
        ],
        {
            "customer": required(),
            "history": {"type": "messages", "required": False},
            "question": required(),
            "tone": {"type": "string", "required": False},
        },
        partials={"tone": "formal"},
    )
    version = build(chat)
    template = version.to_langchain()
    assert isinstance(template, ChatPromptTemplate)
    assert template.metadata == {
        "agenomic_prompt_ref": "prm_test:3",
        "agenomic_prompt_content_digest": version.content_digest,
    }
    history = [HumanMessage(content="earlier"), ToolMessage(content="ok", tool_call_id="c1")]
    variables = {"customer": "Acme", "question": "Where?", "history": history}
    produced = template.invoke(variables).to_messages()
    rendered = to_langchain_messages(version.render_messages(variables))
    assert [(type(m), m.content) for m in produced] == [(type(m), m.content) for m in rendered]
    assert produced[0].content == "You help Acme in {json} mode, tone formal."
    assert prompt_digest(structural_content(template)) == version.content_digest
    text = build(
        text_content(
            "Summarize {topic} {{x}}.",
            {"topic": required(), "style": {"type": "string", "required": False}},
            partials={"style": "short"},
        )
    )
    text_template = to_langchain(text)
    assert isinstance(text_template, PromptTemplate)
    assert text_template.format(topic="tea") == text.render_text({"topic": "tea"})


def test_to_langchain_inlines_fragments() -> None:
    planner = seeded_engine().get_version("prm_planner", 1)
    template = to_langchain(planner)
    messages = template.invoke({"customer": "Acme", "question": "Why?"}).to_messages()
    assert messages[0].content == "Plan for Acme. Never share internal notes."


def test_to_langchain_refuses_typed_vars() -> None:
    version = build(
        text_content("{count} {flag}", {"flag": required("boolean"), "count": required("integer")})
    )
    with pytest.raises(PromptRenderError) as raised:
        to_langchain(version)
    assert raised.value.reason == "unsupported_content"
    assert raised.value.details["variable"] == "count"


def test_to_langchain_messages_mapping() -> None:
    passthrough = AIMessage(content="kept")
    messages = to_langchain_messages(
        [
            RenderedMessage("system", "s"),
            RenderedMessage("user", "u"),
            RenderedMessage("assistant", "a"),
            passthrough,
            {"role": "tool", "content": "t", "tool_call_id": "c1"},
        ]
    )
    assert [type(m) for m in messages[:3]] == [SystemMessage, HumanMessage, AIMessage]
    assert messages[3] is passthrough
    assert isinstance(messages[4], ToolMessage)
    with pytest.raises(PromptRenderError) as raised:
        to_langchain_messages([object()])
    assert raised.value.reason == "type_mismatch"


def test_core_prompt_modules_never_import_langchain() -> None:
    code = (
        "import sys, agenomic.prompts, agenomic.integrations; "
        "assert 'langchain_core' not in sys.modules, 'langchain_core loaded'; "
        "assert 'agenomic.integrations.langchain_prompts' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
