from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from prompt_fakes import chat_content, required, seeded_engine, text_content

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from agenomic.integrations.langchain_prompts import (
    LangChainImport,
    from_langchain,
    to_langchain,
    to_langchain_messages,
)
from agenomic.prompts import (
    ManagedPromptVersion,
    PromptRenderError,
    PromptVersionRecord,
    RenderedMessage,
    prompt_digest,
)
from agenomic.prompts.render import render_content

WORKSPACE = "0b6c2f1e-7a44-4c8e-9f1d-2a3b4c5d6e7f"
ROLES = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}


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
    imported = from_langchain(template)
    assert imported.exact is False
    assert imported.content_digest == version.content_digest
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
    unused = from_langchain(text_template)
    assert [item.code for item in unused.issues if item.code == "unused_partial_dropped"]
    used = build(
        text_content(
            "Summarize {topic} {{x}} in {style}.",
            {"topic": required(), "style": {"type": "string", "required": False}},
            partials={"style": "short"},
        )
    )
    assert from_langchain(to_langchain(used)).content_digest == used.content_digest


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
        "import agenomic.prompts.discovery, agenomic.prompts.importer; "
        "assert 'langchain_core' not in sys.modules, 'langchain_core loaded'; "
        "assert 'yaml' not in sys.modules, 'yaml loaded'; "
        "assert 'agenomic.integrations.langchain_prompts' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def no_fragments(prompt_id: str, version: int, digest: str) -> None:
    return None


def as_pairs(messages: list[Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for message in messages:
        if isinstance(message, RenderedMessage):
            out.append((message.role, message.content))
        else:
            out.append((ROLES[type(message)], message.content))
    return out


def rendered(result: LangChainImport, variables: dict[str, Any]) -> Any:
    assert result.content is not None
    output = render_content(result.content, variables, fragments=no_fragments)
    if output.kind == "text":
        return output.text
    return as_pairs(list(output.messages or []))


def issue_codes(result: LangChainImport) -> list[str]:
    return [item.code for item in result.issues]


def test_from_langchain_exact_round_trip_keeps_fragment_pins() -> None:
    planner = seeded_engine().get_version("prm_planner", 1)
    template = to_langchain(planner)
    result = from_langchain(
        template, resolver=lambda ref: planner if ref == "prm_planner:1" else None
    )
    assert result.exact is True
    assert result.status == "supported"
    assert result.prompt_kind == "chat"
    assert result.content == planner.document
    assert result.content_digest == planner.content_digest
    assert result.content["fragments"]["safety"]["prompt_id"] == "prm_safety"
    structural = from_langchain(template)
    assert structural.exact is False
    assert structural.content is not None
    assert structural.content["fragments"] == {}
    assert structural.content_digest != planner.content_digest


def test_from_langchain_stale_metadata_falls_back_to_structure() -> None:
    planner = seeded_engine().get_version("prm_planner", 1)
    template = to_langchain(planner)
    edited = ChatPromptTemplate(
        messages=[*template.messages[:2], ("human", "{question} now")],
        metadata=template.metadata,
    )
    result = from_langchain(edited, resolver=lambda ref: planner)
    assert result.exact is False
    assert issue_codes(result)[0] == "export_metadata_stale"
    assert result.status == "supported"
    missing = from_langchain(template, resolver=lambda ref: None)
    assert missing.exact is False
    assert issue_codes(missing)[0] == "export_metadata_stale"
    other = seeded_engine().get_version("prm_writer", 1)
    swapped = from_langchain(template, resolver=lambda ref: other)
    assert swapped.exact is False


def test_from_langchain_refuses_callable_partial_and_mustache() -> None:
    callable_partial = PromptTemplate.from_template(
        "Today is {today}: {question}", partial_variables={"today": lambda: "monday"}
    )
    result = from_langchain(callable_partial)
    assert result.status == "unsupported"
    assert result.content is None
    assert "callable_partial" in issue_codes(result)
    mustache = PromptTemplate.from_template("Hi {{name}}", template_format="mustache")
    assert issue_codes(from_langchain(mustache)) == ["unsupported_template_format"]
    jinja = PromptTemplate.model_construct(
        template="Hi {{ name }}", template_format="jinja2", input_variables=["name"]
    )
    assert from_langchain(jinja).status == "unsupported"
    spec = PromptTemplate.from_template("Total {amount:>10}")
    failure = from_langchain(spec)
    assert failure.status == "unsupported"
    assert failure.issues[0].code == "syntax_error"
    assert failure.issues[0].syntax == "format_spec"
    with pytest.raises(TypeError):
        from_langchain("plain string")


def test_from_langchain_mapping_rows() -> None:
    def codes_of(template: Any) -> list[str]:
        return issue_codes(from_langchain(template))

    examples = FewShotPromptTemplate(
        examples=[{"q": "a"}],
        example_prompt=PromptTemplate.from_template("{q}"),
        suffix="{input}",
        input_variables=["input"],
    )
    assert codes_of(examples) == ["unsupported_prompt_class"]
    blocks = ChatPromptTemplate.from_messages(
        [HumanMessage(content=[{"type": "text", "text": "hi"}])]
    )
    assert "unsupported_content_block" in codes_of(blocks)
    custom = ChatPromptTemplate.from_messages(
        [ChatMessagePromptTemplate.from_template("{x}", role="critic")]
    )
    assert "unsupported_role" in codes_of(custom)
    tool = ChatPromptTemplate.from_messages([ToolMessage(content="ok", tool_call_id="c1")])
    assert "unsupported_role" in codes_of(tool)
    named = ChatPromptTemplate.from_messages([SystemMessage(content="x", name="ops")])
    assert "unsupported_message_attributes" in codes_of(named)
    kwargs = ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate.from_template("{x}", additional_kwargs={"k": 1})]
    )
    assert "unsupported_message_attributes" in codes_of(kwargs)
    limited = ChatPromptTemplate.from_messages([MessagesPlaceholder("h", n_messages=3)])
    assert "unsupported_placeholder_option" in codes_of(limited)
    optional = PromptTemplate(template="{a}", input_variables=["a"], optional_variables=["a"])
    assert "unsupported_optional_variable" in codes_of(optional)
    parsed = PromptTemplate.from_template("{a}", output_parser=StrOutputParser())
    parsed_result = from_langchain(parsed)
    assert parsed_result.status == "supported"
    assert "output_parser_not_imported" in issue_codes(parsed_result)
    fragment = PromptTemplate(template="{>frag}", input_variables=[">frag"])
    assert "fragment_syntax_in_import" in codes_of(fragment)
    nested = PromptTemplate.from_template("{a}", partial_variables={"a": {"k": 1}})
    assert "partial_not_scalar" in codes_of(nested)
    unused = PromptTemplate.from_template("{a}", partial_variables={"z": "q"})
    unused_result = from_langchain(unused)
    assert unused_result.status == "supported"
    assert "unused_partial_dropped" in issue_codes(unused_result)
    assert unused_result.content is not None
    assert unused_result.content["partials"] == {}


def test_from_langchain_blocks_secrets_without_text() -> None:
    template = PromptTemplate.from_template(
        "Call with Bearer abcdefghijklmnopqrstuvwxyz0123 for {x}"
    )
    result = from_langchain(template)
    assert result.status == "blocked_secret"
    assert result.content is None
    assert result.content_digest is None
    assert issue_codes(result)[0] == "secret_detected"
    assert [(f.pattern, f.path, f.offset, f.length) for f in result.secret_findings] == [
        ("bearer_token", "/body", 10, 37)
    ]
    assert "abcdefghijklmnop" not in repr(result.issues)


def test_import_fidelity_on_mapped_objects() -> None:
    history = [HumanMessage(content="earlier"), AIMessage(content="answer")]
    cases: list[tuple[Any, dict[str, Any]]] = [
        (
            PromptTemplate.from_template(
                "Summarize {ticket} in {style} style {{literal}}.",
                partial_variables={"style": "neutral"},
            ),
            {"ticket": "T-1"},
        ),
        (
            PromptTemplate.from_template(
                "{count} items, rush={rush}, note={note}, ratio={ratio}",
                partial_variables={"count": 3, "rush": True, "note": None, "ratio": 1.5},
            ),
            {},
        ),
        (
            PromptTemplate.from_template("{a} and {b}", partial_variables={"a": "x"}),
            {"a": "override", "b": "y"},
        ),
        (
            ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content="Static {not_a_variable} rules."),
                    ("placeholder", "{history}"),
                    HumanMessagePromptTemplate.from_template("{question}"),
                    AIMessagePromptTemplate.from_template("Noted: {question}"),
                ]
            ),
            {"question": "Where?", "history": history},
        ),
        (
            ChatPromptTemplate(
                messages=[
                    ("system", "Locale {locale}."),
                    MessagesPlaceholder("history"),
                    ("human", "{question}"),
                ],
                partial_variables={"locale": "fr"},
            ),
            {"question": "Quand ?", "history": []},
        ),
    ]
    for template, variables in cases:
        result = from_langchain(template)
        assert result.status == "supported", issue_codes(result)
        if isinstance(template, PromptTemplate):
            assert rendered(result, variables) == template.format(**variables)
        else:
            assert rendered(result, variables) == as_pairs(template.format_messages(**variables))


_CHARS = st.characters(blacklist_characters="{}\x00", blacklist_categories=("Cs",))
_LITERAL = st.text(alphabet=_CHARS, max_size=8)
_VALUE = st.text(alphabet=_CHARS, max_size=6)
_PIECE = st.one_of(
    _LITERAL,
    st.sampled_from(["{{", "}}", "{alpha}", "{beta_2}", "{_gamma}"]),
)


@settings(max_examples=60, deadline=None)
@given(st.lists(_PIECE, max_size=8), _VALUE, _VALUE)
def test_import_fidelity_property(pieces: list[str], first: str, second: str) -> None:
    template_text = "".join(pieces)
    template = PromptTemplate.from_template(template_text)
    result = from_langchain(template)
    if result.status != "supported":
        assert "secret_detected" in issue_codes(result)
        return
    assert result.content is not None
    values = {"alpha": first, "beta_2": second, "_gamma": first + second}
    variables = {name: values[name] for name in result.content["variables"]}
    assert rendered(result, variables) == template.format(**variables)
