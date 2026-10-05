from __future__ import annotations

from typing import Any, Optional

import pytest
from prompt_fakes import chat_content, required, seeded_engine, text_content

from agenomic.prompts import (
    ManagedPromptVersion,
    PromptIntegrityError,
    PromptRenderError,
    PromptTemplateError,
    PromptVersionRecord,
    RenderedMessage,
    prompt_digest,
    render_content,
    validate_content,
)
from agenomic.prompts.models import Placeholder, TemplateMessage
from agenomic.prompts.render import tokenize

WORKSPACE = "0b6c2f1e-7a44-4c8e-9f1d-2a3b4c5d6e7f"


def no_fragments(prompt_id: str, version: int, digest: str) -> Optional[dict[str, Any]]:
    return None


def build(content: dict[str, Any], **fragments: dict[str, Any]) -> ManagedPromptVersion:
    records = {
        key: PromptVersionRecord(
            prompt_id=key.split(":")[0],
            version=int(key.split(":")[1]),
            prompt_kind="fragment",
            content_digest=prompt_digest(value),
            content=value,
        )
        for key, value in fragments.items()
    }
    record = PromptVersionRecord(
        prompt_id="prm_test",
        version=1,
        prompt_kind=content["kind"],
        content_digest=prompt_digest(content),
        content=content,
    )
    return ManagedPromptVersion.from_record(
        record, workspace_id=WORKSPACE, lookup=lambda pid, v: records.get(f"{pid}:{v}")
    )


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list[Any]) -> str:
        self.calls += 1
        return "ok"


class DuckMessage:
    def __init__(self, kind: str, content: Any) -> None:
        self.type = kind
        self.content = content


PLANNER = chat_content(
    [
        {"role": "system", "content": "Plan for {customer}."},
        {"placeholder": "history", "optional": True},
        {"role": "user", "content": "{question}"},
    ],
    {
        "customer": required(),
        "history": {"type": "messages", "required": False},
        "question": required(),
    },
)


def test_missing_variable_fails_before_model_call() -> None:
    model = FakeModel()
    version = build(PLANNER)

    def node(variables: dict[str, Any]) -> str:
        return model.invoke(version.render_messages(variables))

    with pytest.raises(PromptRenderError) as raised:
        node({"customer": "Acme secret value"})
    assert model.calls == 0
    assert raised.value.status == 0
    assert raised.value.reason == "missing_variable"
    assert raised.value.details["variable"] == "question"
    assert "Acme secret value" not in str(raised.value)
    assert "Acme secret value" not in repr(raised.value.details)
    assert node({"customer": "Acme", "question": "Where?"}) == "ok"
    assert model.calls == 1


def test_history_conflict() -> None:
    version = build(PLANNER)
    with pytest.raises(PromptRenderError) as raised:
        version.compose({"customer": "a", "question": "b"}, history=[])
    assert raised.value.reason == "history_conflict"


def test_compose_appends_history_without_placeholder() -> None:
    version = build(
        chat_content([{"role": "system", "content": "Be brief."}], {}),
    )
    history = [{"role": "user", "content": "hi"}, DuckMessage("ai", "hello")]
    messages = version.compose(history=history)
    assert messages[0] == RenderedMessage("system", "Be brief.")
    assert messages[1] is history[0]
    assert messages[2] is history[1]
    rendered = version.render(history=history)
    assert rendered.rendered_document["history_count"] == 2
    with pytest.raises(PromptRenderError) as duplicate:
        version.compose(history=[DuckMessage("system", "Be brief.")])
    assert duplicate.value.details["value_path"] == "/history/0"
    assert version.compose(
        history=[{"role": "system", "content": "Be brief."}], allow_duplicate_system=True
    )[1] == {"role": "system", "content": "Be brief."}
    with pytest.raises(PromptRenderError) as bad:
        version.compose(history=[DuckMessage("developer", "x")])
    assert bad.value.reason == "invalid_message_value"
    with pytest.raises(PromptRenderError) as not_list:
        version.compose(history="hello")
    assert not_list.value.details["value_path"] == "/history"


def test_kind_mismatch_never_flattens() -> None:
    chat = build(PLANNER)
    with pytest.raises(PromptRenderError) as raised:
        chat.render_text({"customer": "a", "question": "b"})
    assert raised.value.reason == "kind_mismatch"
    text = build(text_content("Hello {name}", {"name": required()}))
    with pytest.raises(PromptRenderError) as other:
        text.render_messages({"name": "Ada"})
    assert other.value.reason == "kind_mismatch"
    with pytest.raises(PromptRenderError) as history:
        text.compose({"name": "Ada"}, history=[])
    assert history.value.reason == "history_not_supported"
    assert text.render_text({"name": "Ada"}) == "Hello Ada"


def test_caller_overrides_partial() -> None:
    content = text_content(
        "Answer in {locale}.",
        {"locale": {"type": "string", "required": False}},
        partials={"locale": "en"},
    )
    version = build(content)
    assert version.render_text() == "Answer in en."
    assert version.render_text({"locale": "fr"}) == "Answer in fr."


def test_typed_values_render_canonically() -> None:
    content = text_content(
        "{n} {flag} {payload}",
        {"n": required("integer"), "flag": required("boolean"), "payload": required("json")},
    )
    version = build(content)
    assert version.render_text(
        {"n": 3.0, "flag": False, "payload": {"b": 1, "a": [True, None]}}
    ) == ('3 false {"a":[true,null],"b":1}')
    for value, reason in (
        (True, "type_mismatch"),
        ("3", "type_mismatch"),
        (2.5, "float_not_allowed"),
    ):
        with pytest.raises(PromptRenderError) as raised:
            version.render_text({"n": value, "flag": True, "payload": 1})
        assert raised.value.reason == reason
    with pytest.raises(PromptRenderError) as unicode_error:
        version.render_text({"n": 1, "flag": True, "payload": {"k": "\ud800"}})
    assert unicode_error.value.details["value_path"] == "/variables/payload/k"


def test_placeholder_items_pass_through_by_identity() -> None:
    version = build(PLANNER)
    item = DuckMessage("human", "earlier question")
    messages = version.render_messages({"customer": "a", "question": "b", "history": (item,)})
    assert messages[1] is item
    with pytest.raises(PromptRenderError) as raised:
        version.render_messages({"customer": "a", "question": "b", "history": [object()]})
    assert raised.value.details["value_path"] == "/variables/history/0"
    with pytest.raises(PromptRenderError) as duplicate:
        version.render_messages(
            {"customer": "a", "question": "b", "history": [DuckMessage("system", "Plan for a.")]}
        )
    assert duplicate.value.reason == "duplicate_system_message"
    with pytest.raises(PromptRenderError) as content_error:
        version.render_messages(
            {"customer": "a", "question": "b", "history": [{"role": "user", "content": 3}]}
        )
    assert content_error.value.details["value_path"] == "/variables/history/0/content"


def test_secret_policy_never_echoes_values() -> None:
    version = build(text_content("{note}", {"note": required()}))
    secret = "ghp_" + "a1B2" * 9
    assert version.render_text({"note": secret}) == secret
    with pytest.raises(PromptRenderError) as raised:
        version.render({"note": secret}, secret_policy="error")
    assert raised.value.reason == "secret_in_variables"
    assert raised.value.details["pattern"] == "github_token"
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value.details)


def test_unknown_variables_and_non_string_keys() -> None:
    version = build(text_content("Hi", {}))
    with pytest.raises(PromptRenderError) as raised:
        version.render_text({1: "x"})
    assert raised.value.reason == "unknown_variable"
    result = version.render({"extra": "x"}, strict=False)
    assert [warning.to_dict() for warning in result.warnings] == [
        {"code": "strict_disabled", "variable": "extra"}
    ]


def test_fragments_render_through_the_closure() -> None:
    engine = seeded_engine()
    planner = engine.get_version("prm_planner", 1)
    assert set(planner.fragments) == {"safety"}
    assert planner.fragments["safety"].kind == "fragment"
    assert planner.fragment_source("prm_safety", 1, "") is not None
    messages = planner.render_messages({"customer": "Acme", "question": "Why?"})
    assert messages[0] == RenderedMessage("system", "Plan for Acme. Never share internal notes.")
    assert (
        planner.expanded_sources()["/body/0/content"]
        == "Plan for {customer}. Never share internal notes."
    )
    assert isinstance(planner.content.body[0], TemplateMessage)
    assert isinstance(planner.content.body[1], Placeholder)
    assert planner.variables["question"].required is True
    assert str(planner.uri).endswith("/prompts/prm_planner/versions/1")
    assert [str(record.ref) for record in planner.closure_records()] == [
        "prm_planner:1",
        "prm_safety:1",
    ]
    planner.verify()


def test_verify_detects_tampering() -> None:
    version = build(text_content("Hello", {}))
    version.document["body"] = "Hello!"
    with pytest.raises(PromptIntegrityError) as raised:
        version.verify()
    assert raised.value.code == "prompt_digest_mismatch"


def test_record_digest_mismatch_and_missing_fragment() -> None:
    content = text_content("Hello", {})
    with pytest.raises(PromptIntegrityError):
        ManagedPromptVersion.from_record(
            PromptVersionRecord(
                prompt_id="prm_x", version=1, content_digest="sha256:" + "0" * 64, content=content
            ),
            workspace_id=WORKSPACE,
            lookup=lambda pid, v: None,
        )
    with_fragment = text_content(
        "{>f}",
        fragments={
            "f": {"prompt_id": "prm_f", "version": 1, "content_digest": "sha256:" + "1" * 64}
        },
    )
    with pytest.raises(PromptTemplateError) as raised:
        build(with_fragment)
    assert raised.value.reason == "fragment_not_found"


INVALID_CONTENTS: list[tuple[Any, str]] = [
    ([], "invalid_field_type"),
    ({}, "missing_field"),
    ({"schema": "agenomic.prompt_content/v2"}, "unsupported_schema"),
    ({**text_content("x"), "body": 1.5}, "float_not_allowed"),
    ({**text_content("x"), "renderer_version": "2"}, "unsupported_renderer_version"),
    ({**text_content("x"), "kind": "audio"}, "invalid_field_type"),
    ({**chat_content([]), "body": "x"}, "invalid_body"),
    (chat_content([{"role": "user", "content": "x"}] * 257), "too_many_messages"),
    ({**text_content("x"), "variables": []}, "invalid_field_type"),
    (text_content("x", {f"v{i}": required() for i in range(129)}), "too_many_variables"),
    (text_content("x", {"1x": required()}), "invalid_variable_name"),
    (text_content("x", {"v": "string"}), "invalid_field_type"),
    (text_content("x", {"v": {"type": "string"}}), "missing_field"),
    (text_content("x", {"v": {"type": "string", "required": True, "doc": ""}}), "unknown_field"),
    (text_content("x", {"v": {"type": "float", "required": True}}), "invalid_variable_type"),
    (text_content("x", {"v": {"type": "string", "required": "yes"}}), "invalid_field_type"),
    ({**text_content("x"), "partials": []}, "invalid_field_type"),
    ({**text_content("x"), "fragments": []}, "invalid_field_type"),
    (
        {
            **text_content("x"),
            "fragments": {
                f"f{i}": {
                    "prompt_id": "prm_f",
                    "version": 1,
                    "content_digest": "sha256:" + "0" * 64,
                }
                for i in range(33)
            },
        },
        "too_many_fragments",
    ),
    (text_content("x" * 65537), "template_too_large"),
    (chat_content(["x"]), "invalid_message_entry"),
    (chat_content([{"role": "user", "content": "x", "placeholder": "h"}]), "invalid_message_entry"),
    (chat_content([{"name": "x"}]), "invalid_message_entry"),
    (chat_content([{"role": "user", "content": 3}]), "invalid_message_entry"),
    (chat_content([{"placeholder": "h", "optional": True, "x": 1}]), "invalid_message_entry"),
    (chat_content([{"placeholder": "1h", "optional": True}]), "invalid_message_entry"),
    (chat_content([{"placeholder": "h", "optional": "no"}]), "invalid_message_entry"),
    (
        {
            **text_content("x"),
            "fragments": {"f": {"prompt_id": "prm_f", "version": 0, "content_digest": "x"}},
        },
        "invalid_fragment_pin",
    ),
    (text_content("{x}", {"x": required()}, partials={"x": 1}), "partial_required_mismatch"),
    (
        text_content(
            "x", {}, output_contract={"type": "json_schema", "json_schema": {"a": "x" * 70000}}
        ),
        "output_contract_invalid",
    ),
]


@pytest.mark.parametrize(("content", "code"), INVALID_CONTENTS)
def test_invalid_content_first_error(content: Any, code: str) -> None:
    report = validate_content(content, fragments=no_fragments)
    assert not report.ok
    assert report.errors[0].code == code
    with pytest.raises(PromptTemplateError) as raised:
        report.raise_for_errors()
    assert raised.value.reason == code


def test_template_errors_map_to_top_level_codes() -> None:
    with pytest.raises(PromptTemplateError) as large:
        validate_content(text_content("x" * 65537), fragments=no_fragments).raise_for_errors(400)
    assert large.value.code == "prompt_content_too_large"
    assert large.value.status == 400
    secret = text_content("{oops", {}, partials={})
    secret["body"] = "key sk-" + "a" * 24 + " {"
    with pytest.raises(PromptTemplateError) as found:
        validate_content(secret, fragments=no_fragments).raise_for_errors()
    assert found.value.code == "prompt_secret_detected"
    assert found.value.reason == "syntax_error"
    assert found.value.details["findings"] == [
        {"pattern": "openai_key", "path": "/body", "offset": 4, "length": 27}
    ]
    validate_content(text_content("ok"), fragments=no_fragments).raise_for_errors()


def _pinned(prompt_id: str, content: dict[str, Any]) -> dict[str, Any]:
    return {"prompt_id": prompt_id, "version": 1, "content_digest": prompt_digest(content)}


def _nested_fragments(levels: int) -> tuple[dict[str, Any], dict[str, Any]]:
    entries: dict[str, Any] = {}
    inner = text_content("leaf")
    for level in range(levels, 0, -1):
        prompt_id = f"prm_f{level}"
        entries[f"{prompt_id}:1"] = {"prompt_kind": "fragment", "content": inner}
        inner = text_content("{>n}", fragments={"n": _pinned(prompt_id, inner)})
    return inner, entries


def _fragment_cycle() -> tuple[dict[str, Any], dict[str, Any]]:
    looping = text_content(
        "{>again}",
        fragments={
            "again": {"prompt_id": "prm_loop", "version": 1, "content_digest": "sha256:" + "0" * 64}
        },
    )
    entries = {"prm_loop:1": {"prompt_kind": "fragment", "content": looping}}
    return text_content("{>loop}", fragments={"loop": _pinned("prm_loop", looping)}), entries


@pytest.mark.parametrize(
    ("case", "reason", "code"),
    [
        (
            (chat_content([{"role": "user", "content": "x"}] * 257), {}),
            "too_many_messages",
            "prompt_content_too_large",
        ),
        (
            (text_content("x", {f"v{i}": required() for i in range(129)}), {}),
            "too_many_variables",
            "prompt_content_too_large",
        ),
        (
            (
                {
                    **text_content("x"),
                    "fragments": {
                        f"f{i}": {
                            "prompt_id": "prm_f",
                            "version": 1,
                            "content_digest": "sha256:" + "0" * 64,
                        }
                        for i in range(33)
                    },
                },
                {},
            ),
            "too_many_fragments",
            "prompt_content_too_large",
        ),
        (_nested_fragments(9), "fragment_depth_exceeded", "prompt_fragment_depth_exceeded"),
        (_fragment_cycle(), "fragment_cycle", "prompt_fragment_cycle"),
    ],
)
def test_validation_codes_follow_the_registry(
    case: tuple[dict[str, Any], dict[str, Any]], reason: str, code: str
) -> None:
    content, entries = case
    report = validate_content(content, fragments=lambda pid, v, d: entries.get(f"{pid}:{v}"))
    with pytest.raises(PromptTemplateError) as raised:
        report.raise_for_errors(400)
    assert (raised.value.reason, raised.value.code, raised.value.status) == (reason, code, 400)


def test_eight_nested_fragments_are_valid() -> None:
    content, entries = _nested_fragments(8)
    report = validate_content(content, fragments=lambda pid, v, d: entries.get(f"{pid}:{v}"))
    assert report.ok


def test_prompt_kind_mismatch_and_tokenize_errors() -> None:
    report = validate_content(text_content("x"), fragments=no_fragments, prompt_kind="chat")
    assert report.errors[0].code == "prompt_kind_mismatch"
    with pytest.raises(PromptTemplateError) as raised:
        tokenize("ok\n😀 {x!r}")
    assert raised.value.details["errors"][0] == {
        "code": "syntax_error",
        "syntax": "conversion",
        "offset": 5,
        "line": 2,
        "column": 3,
    }


def test_nested_fragment_errors_carry_the_fragment_ref() -> None:
    inner = text_content("{bad")
    outer = text_content(
        "{>inner}",
        fragments={
            "inner": {
                "prompt_id": "prm_inner",
                "version": 1,
                "content_digest": prompt_digest(inner),
            }
        },
    )
    entries = {
        "prm_inner:1": {"prompt_kind": "fragment", "content": inner},
        "prm_outer:1": {"prompt_kind": "fragment", "content": outer},
    }
    content = text_content(
        "{>outer}",
        fragments={
            "outer": {
                "prompt_id": "prm_outer",
                "version": 1,
                "content_digest": prompt_digest(outer),
            }
        },
    )
    with pytest.raises(PromptRenderError) as raised:
        render_content(content, fragments=lambda pid, v, d: entries.get(f"{pid}:{v}"))
    assert raised.value.details["fragment"] == "prm_inner:1"
    assert raised.value.details["syntax"] == "unclosed_brace"
    undeclared = text_content("{>missing}")
    entries["prm_outer:1"] = {"prompt_kind": "fragment", "content": undeclared}
    content["fragments"]["outer"]["content_digest"] = prompt_digest(undeclared)
    with pytest.raises(PromptRenderError) as missing:
        render_content(content, fragments=lambda pid, v, d: entries.get(f"{pid}:{v}"))
    assert missing.value.reason == "fragment_not_declared"
    assert missing.value.details["fragment"] == "prm_outer:1"
