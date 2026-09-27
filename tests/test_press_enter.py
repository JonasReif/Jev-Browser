"""The local Jev action extension keeps search submission observed and grounded."""

import json
import shutil
import subprocess

import pytest

from jev_ultrafast import browser, model


@pytest.mark.parametrize(("field_type", "label", "value", "expected"), [
    ("search", "Search", "Browser Use", True),
    ("text", "Search or jump to", "Browser Use", True),
    ("search", "Search", "", False),
    ("email", "Enter your email", "me@example.com", False),
])
def test_snapshot_offers_enter_only_for_populated_search_fields(field_type, label, value, expected):
    if not shutil.which("node"):
        pytest.skip("Node is needed to evaluate the DOM snapshot fixture")
    fixture = r"""
const fs = require('fs');
const vm = require('vm');
const [sourcePath, fieldType, label, value] = process.argv.slice(1);
const attrs = {'aria-label': label};
if (fieldType === 'text' && label === 'Search or jump to') attrs.role = 'combobox';
const field = {
  tagName: 'INPUT', type: fieldType, value, readOnly: false, disabled: false,
  isConnected: true, labels: [], childNodes: [], parentElement: {innerText: label},
  getAttribute: key => attrs[key] ?? null,
  closest: () => null, matches: () => false, checkVisibility: () => true,
  contains: other => other === field,
  getBoundingClientRect: () => ({x: 10, y: 10, width: 200, height: 25}),
};
const document = {
  body: {}, documentElement: {scrollHeight: 700}, title: 'Fixture',
  querySelectorAll: () => [field],
  getElementById: () => null,
  elementFromPoint: () => field,
  createTreeWalker: () => ({nextNode: () => null}),
  createRange: () => ({}),
};
const context = {
  document, window: {}, performance: {timeOrigin: 1},
  location: {href: 'https://example.com/'},
  scrollX: 0, scrollY: 0, innerWidth: 1120, innerHeight: 780,
  NodeFilter: {SHOW_TEXT: 4},
};
const state = vm.runInNewContext(fs.readFileSync(sourcePath, 'utf8'), context);
process.stdout.write(JSON.stringify(state.actions));
"""
    result = subprocess.run(
        ["node", "-e", fixture, str(browser.__file__).replace("browser.py", "snapshot.js"),
         field_type, label, value], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    actions = json.loads(result.stdout)
    assert any(action["kind"] == "press_enter" for action in actions) is expected


def test_press_enter_is_a_targeted_operation_on_the_observed_field():
    actions = [
        {"id": "e1", "node": 7, "kind": "fill", "role": "combobox",
         "label": "Search or jump to", "value": "Browser Use"},
        {"id": "e2", "node": 7, "kind": "press_enter", "role": "combobox",
         "label": "Submit Search or jump to with Enter", "value": "Browser Use"},
        {"id": "e3", "node": 8, "kind": "fill", "role": "textbox",
         "label": "Enter your email", "value": ""},
    ]

    elements, targets, controls = model.action_space(actions)

    assert elements[0]["operations"] == ["TYPE_TEXT", "PRESS_ENTER"]
    assert targets["PRESS_ENTER"] == {"1": actions[1]}
    assert "PRESS_ENTER" not in controls
    assert "PRESS_ENTER" not in elements[1]["operations"]


def test_press_enter_executes_only_after_observed_target_is_confirmed(monkeypatch):
    calls = []

    def cdp(method, *, session_id, **parameters):
        calls.append((method, parameters))
        if method == "Runtime.evaluate":
            return {"result": {"value": ({"x": 10, "y": 20} if len(calls) == 1 else True)}}
        return {}

    monkeypatch.setattr(browser, "cdp", cdp)
    action = {"id": "e2", "node": 7, "kind": "press_enter", "label": "Submit search",
              "value": "Browser Use", "role": "combobox"}

    assert browser.browser_operation({"operation": "act", "session": "session",
                                      "action": action}) == {"executed": "e2"}
    assert [(method, args.get("type")) for method, args in calls if method != "Runtime.evaluate"] == [
        ("Input.dispatchMouseEvent", "mousePressed"),
        ("Input.dispatchMouseEvent", "mouseReleased"),
        ("Input.dispatchKeyEvent", "keyDown"),
        ("Input.dispatchKeyEvent", "keyUp"),
    ]
    assert calls[-2][1]["key"] == calls[-1][1]["key"] == "Enter"


def test_press_enter_does_not_send_key_if_field_loses_focus(monkeypatch):
    calls = []

    def cdp(method, *, session_id, **parameters):
        calls.append((method, parameters))
        if method == "Runtime.evaluate":
            return {"result": {"value": ({"x": 10, "y": 20} if len(calls) == 1 else False)}}
        return {}

    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(browser.StalePage, match="lost focus"):
        browser.browser_operation({"operation": "act", "session": "session", "action": {
            "id": "e2", "node": 7, "kind": "press_enter", "label": "Submit search",
            "value": "Browser Use", "role": "combobox",
        }})
    assert "Input.dispatchKeyEvent" not in [method for method, _ in calls]


def test_press_enter_focus_guard_is_valid_javascript(monkeypatch):
    if not shutil.which("node"):
        pytest.skip("Node is needed to parse the Chrome focus guard")
    expressions = []

    def cdp(method, *, session_id, **parameters):
        if method == "Runtime.evaluate":
            expressions.append(parameters["expression"])
            return {"result": {"value": ({"x": 10, "y": 20} if len(expressions) == 1 else True)}}
        return {}

    monkeypatch.setattr(browser, "cdp", cdp)
    browser.browser_operation({"operation": "act", "session": "session", "action": {
        "id": "e2", "node": 7, "kind": "press_enter", "label": "Submit search",
        "value": "Browser Use", "role": "combobox",
    }})
    result = subprocess.run(
        ["node", "-e", "new Function(process.argv[1])", expressions[1]],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_press_enter_rejects_stale_or_covered_target_without_keyboard_input(monkeypatch):
    calls = []

    def cdp(method, *, session_id, **parameters):
        calls.append((method, parameters))
        if method == "Runtime.evaluate":
            return {"result": {"value": None}}
        return {}

    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(browser.StalePage):
        browser.browser_operation({"operation": "act", "session": "session", "action": {
            "id": "e2", "node": 7, "kind": "press_enter", "label": "Submit search",
            "value": "Browser Use", "role": "combobox",
        }})
    assert [method for method, _ in calls] == ["Runtime.evaluate"]
