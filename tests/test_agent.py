"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import json
import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.files = {}
    a.values = {}
    a.pending_text = None
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_values_let_type_text_choose_caller_text_in_the_same_request(monkeypatch):
    def post(_url, _key, body):
        q = body["questions"]
        assert q["type_text_value"]["criteria"] == {
            "v1": {"name": "search term", "text": "Gödel"},
            "v2": {"name": "email", "text": "a@b.test"},
        }
        assert body["state"]["values_to_type"] == ["search term", "email"]
        assert "provided values" in q["operation"]["criteria"]["TYPE_TEXT"]
        return {"model": "jev", "answers": {
            "operation": choice(list(q["operation"]["criteria"]), "TYPE_TEXT"),
            "type_text_target": choice(list(q["type_text_target"]["criteria"]), "1"),
            "type_text_value": choice(["v1", "v2"], "v1"),
            "click_target": choice(list(q["click_target"]["criteria"]), "2"),
        }}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    result = model.choose(page(), "Search", [], values={"search term": "Gödel", "email": "a@b.test"})
    assert (result["choice"], result["value"]) == ("e1", "search term")


def test_invented_value_choice_is_rejected(monkeypatch):
    def post(_url, _key, body):
        q = body["questions"]
        return {"model": "jev", "answers": {
            "operation": choice(list(q["operation"]["criteria"]), "TYPE_TEXT"),
            "type_text_target": choice(list(q["type_text_target"]["criteria"]), "1"),
            "type_text_value": choice(["v1", "v7"], "v7"),
        }}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe response"):
        model.choose(page(), "Search", [], values={"search term": "Gödel"})


def test_no_value_head_without_caller_values(monkeypatch):
    bodies = []

    def post(_url, _key, body):
        bodies.append(body)
        q = body["questions"]
        return {"model": "jev", "answers": {
            "operation": choice(list(q["operation"]["criteria"]), "CLICK"),
            "click_target": choice(list(q["click_target"]["criteria"]), "2"),
        }}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    assert model.choose(page(), "Search", [])["value"] is None
    assert "type_text_value" not in bodies[0]["questions"] and "values_to_type" not in bodies[0]["state"]


def test_agent_types_the_chosen_value_without_the_text_model(runner, monkeypatch):
    helper = Mock()
    monkeypatch.setattr(loop, "field_text", helper)
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    runner.values = {"search term": "Gödel"}
    runner.state["decision"] = {**decision(), "value": "search term"}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    helper.assert_not_called()
    assert runner.state["browser"].act.call_args.kwargs["text"] == "Gödel"
    assert runner.state["history"][-1]["text"] == "Gödel" and runner.state["text_calls"] == []


@pytest.mark.parametrize("values", [{"": "x"}, {"email": ""}, {"email": 3}, {"note": "x" * 2001},
                                    {str(i): "x" for i in range(51)}])
def test_agent_rejects_unusable_values_before_opening_a_tab(values, monkeypatch):
    opened = Mock()
    monkeypatch.setattr(loop, "Browser", opened)
    with pytest.raises(ValueError, match="Values must"):
        loop.Agent("https://example.test/", "Search", values=values)
    opened.assert_not_called()


def upload_page():
    state = page()
    upload = {"id": "e9", "kind": "upload", "label": "CV", "role": "file", "value": "", "accept": ".pdf", "node": 30}
    state["actions"].insert(0, upload)
    return state


def test_uploads_are_offered_only_with_caller_files(monkeypatch):
    elements, targets, _ = model.action_space(upload_page()["actions"], uploads=False)
    assert "UPLOAD_FILE" not in targets and all(e["role"] != "file" for e in elements)
    elements, targets, _ = model.action_space(upload_page()["actions"])
    assert targets["UPLOAD_FILE"] == {"1": upload_page()["actions"][0]}
    assert elements[0]["accept"] == ".pdf" and elements[0]["operations"] == ["UPLOAD_FILE"]
    bodies = []

    def post(_url, _key, body):
        bodies.append(body)
        q = body["questions"]
        return {"model": "jev", "answers": {
            "operation": choice(list(q["operation"]["criteria"]), "CLICK"),
            "click_target": choice(list(q["click_target"]["criteria"]), "2"),
        }}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    model.choose(upload_page(), "Apply", [])
    assert "UPLOAD_FILE" not in bodies[0]["questions"]["operation"]["criteria"]
    assert "upload_file" not in bodies[0]["questions"]


def test_upload_chooses_an_observed_input_and_a_caller_file(monkeypatch):
    def post(_url, _key, body):
        q = body["questions"]
        assert q["upload_file"]["criteria"] == {"f1": {"file": "cv.pdf"}, "f2": {"file": "letter.pdf"}}
        assert body["state"]["files_to_upload"] == ["cv.pdf", "letter.pdf"]
        return {"model": "jev", "answers": {
            "operation": choice(list(q["operation"]["criteria"]), "UPLOAD_FILE"),
            "upload_file_target": choice(list(q["upload_file_target"]["criteria"]), "1"),
            "upload_file": choice(["f1", "f2"], "f2"),
            "click_target": choice(list(q["click_target"]["criteria"]), "2"),
        }}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    result = model.choose(upload_page(), "Apply with my cover letter", [], ["cv.pdf", "letter.pdf"])
    assert (result["choice"], result["operation"], result["file"]) == ("e9", "UPLOAD_FILE", "letter.pdf")


def test_invented_file_choice_is_rejected(monkeypatch):
    def post(_url, _key, body):
        q = body["questions"]
        return {"model": "jev", "answers": {
            "operation": choice(list(q["operation"]["criteria"]), "UPLOAD_FILE"),
            "upload_file_target": choice(["1"], "1"),
            "upload_file": choice(["f1", "f9"], "f9"),
        }}

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe response"):
        model.choose(upload_page(), "Apply", [], ["cv.pdf"])


def test_agent_uploads_the_chosen_file_by_its_fixed_path(runner, tmp_path):
    runner.files = {"cv.pdf": str(tmp_path / "cv.pdf")}
    runner.state["page"] = upload_page()
    runner.state["decision"] = {**decision("e9"), "operation": "UPLOAD_FILE", "file": "cv.pdf"}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    args, kwargs = runner.state["browser"].act.call_args
    assert args[0]["id"] == "e9" and kwargs == {"text": "cv.pdf", "file": str(tmp_path / "cv.pdf")}
    assert runner.state["history"][-1]["text"] == "cv.pdf"


def test_agent_requires_existing_distinct_upload_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "cv.pdf").write_text("1")
    (tmp_path / "cv.pdf").write_text("2")
    for files in [[tmp_path / "missing.pdf"], [tmp_path / "cv.pdf", tmp_path / "a" / "cv.pdf"]]:
        with pytest.raises(ValueError, match="distinct names"):
            loop.Agent("https://example.test/", "Apply", files=files)


def test_upload_sets_files_on_the_observed_node_without_a_dialog(monkeypatch):
    import jev_ultrafast.browser as browser

    cdp = Mock(side_effect=[{"result": {"objectId": "obj-1"}}, {}])
    monkeypatch.setattr(browser, "cdp", cdp)
    browser_operation({"operation": "act", "session": "s", "file": "/up/cv.pdf",
                       "action": {"id": "e9", "kind": "upload", "node": 30}})
    assert "nodes.get(30)" in cdp.call_args_list[0].kwargs["expression"]
    assert cdp.call_args_list[1].args == ("DOM.setFileInputFiles",)
    assert cdp.call_args_list[1].kwargs == {"session_id": "s", "files": ["/up/cv.pdf"], "objectId": "obj-1"}


def test_missing_upload_node_is_stale_before_any_input(monkeypatch):
    import jev_ultrafast.browser as browser

    cdp = Mock(return_value={"result": {"type": "object", "subtype": "null"}})
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(StalePage):
        browser_operation({"operation": "act", "session": "s", "file": "/up/cv.pdf",
                           "action": {"id": "e9", "kind": "upload", "node": 30}})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual)["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual)["passed"]


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()
