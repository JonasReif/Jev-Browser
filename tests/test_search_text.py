"""Search text keeps the subject name while dropping destination descriptors."""

from unittest.mock import Mock

import pytest

from jev_ultrafast import model


def test_search_field_uses_subject_name_from_text_helper(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-key")
    monkeypatch.setattr(model, "post_json", lambda *_args: {
        "choices": [{"message": {"content": '{"text": "Widget interface"}'}}],
    })
    context = {
        "goal": "Find the Widget interface reference",
        "field": {"label": "Search", "role": "searchbox", "value": ""},
        "page": {"title": "Docs", "text": ""},
        "recent_actions": [],
    }

    value, _ = model.field_text(context)

    assert value == "Widget"


def test_non_search_field_preserves_text_helper_value(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-key")
    monkeypatch.setattr(model, "post_json", lambda *_args: {
        "choices": [{"message": {"content": '{"text": "Widget interface"}'}}],
    })
    context = {
        "goal": "Name this project Widget interface",
        "field": {"label": "Project name", "role": "textbox", "value": ""},
        "page": {"title": "Projects", "text": ""},
        "recent_actions": [],
    }

    value, _ = model.field_text(context)

    assert value == "Widget interface"


def test_search_field_reformulates_a_query_already_tried(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-key")
    monkeypatch.setattr(model, "post_json", lambda *_args: {
        "choices": [{"message": {"content": '{"text": "JavaScript Promise"}'}}],
    })
    context = {
        "goal": "Find JavaScript Promise reference",
        "field": {"label": "Search", "role": "combobox", "value": ""},
        "page": {"title": "Search results", "text": "No matching result"},
        "recent_actions": [
            {"action": "Search", "kind": "fill", "text": "JavaScript Promise"},
            {"action": "Submit Search", "kind": "click", "text": None},
        ],
    }

    value, _ = model.field_text(context)

    assert value == "Promise"


def test_text_helper_retries_one_empty_provider_response(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-key")
    post = Mock(side_effect=[
        {"choices": [{"message": {"content": None}, "finish_reason": None}]},
        {"choices": [{"message": {"content": '{"text": "Widget"}'}, "finish_reason": "stop"}]},
    ])
    monkeypatch.setattr(model, "post_json", post)

    value, _ = model.field_text({"goal": "Find Widget", "field": {"label": "Search", "role": "searchbox"}})

    assert value == "Widget"
    assert post.call_count == 2


def test_text_helper_retries_malformed_provider_response_before_typing(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-key")
    post = Mock(side_effect=[
        {"choices": [{"message": {"content": '{"text": "Widget"}\nextra prose'}}]},
        {"choices": [{"message": {"content": '{"text": "Widget"}'}}]},
    ])
    monkeypatch.setattr(model, "post_json", post)

    value, _ = model.field_text({"goal": "Find Widget", "field": {"label": "Search", "role": "searchbox"}})

    assert value == "Widget"
    assert post.call_count == 2


def test_text_helper_accepts_json_followed_by_a_stray_fence(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-key")
    post = Mock(return_value={
        "choices": [{"message": {"content": '{"text": "Promise"}\n```'}}],
    })
    monkeypatch.setattr(model, "post_json", post)

    value, _ = model.field_text({"goal": "Find Promise", "field": {"label": "Search", "role": "searchbox"}})

    assert value == "Promise"
    assert post.call_count == 1


def test_text_helper_does_not_retry_explicit_missing_value(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-key")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text": null}'}}]})
    monkeypatch.setattr(model, "post_json", post)

    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find Widget", "field": {"label": "Search", "role": "searchbox"}})

    assert post.call_count == 1


def test_text_helper_retries_null_when_reformulating_a_submitted_search(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-key")
    post = Mock(side_effect=[
        {"choices": [{"message": {"content": '{"text": null}'}}]},
        {"choices": [{"message": {"content": '{"text": "Widget"}'}}]},
    ])
    monkeypatch.setattr(model, "post_json", post)
    context = {
        "goal": "Find the Widget reference",
        "field": {"label": "Search", "role": "combobox", "value": ""},
        "page": {"title": "Search results", "text": "No matching result"},
        "recent_actions": [
            {"action": "Search", "kind": "fill", "text": "Widget catalog"},
            {"action": "Submit Search", "kind": "click", "text": None},
        ],
    }

    value, _ = model.field_text(context)

    assert value == "Widget"
    assert post.call_count == 2


def test_text_helper_does_not_type_the_same_one_word_search_twice(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test-key")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text": "Widget"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = {
        "goal": "Find the Widget reference",
        "field": {"label": "Search", "role": "searchbox", "value": ""},
        "page": {"title": "Search results", "text": "No matching result"},
        "recent_actions": [
            {"action": "Search", "kind": "fill", "text": "Widget"},
            {"action": "Submit Search", "kind": "click", "text": None},
        ],
    }

    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text(context)

    assert post.call_count == 2
