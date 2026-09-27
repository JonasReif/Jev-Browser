"""Open shadow roots expose observed controls that remain safe to act on."""

import json
import shutil
import subprocess

import pytest

from jev_ultrafast import browser

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(not NODE, reason="Node is needed for DOM fixtures")


def run_node(script, *args):
    result = subprocess.run([NODE, "-e", script, *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_snapshot_observes_button_and_input_inside_open_shadow_root():
    fixture = r"""
const fs = require('fs');
const vm = require('vm');
const [path] = process.argv.slice(1);
const element = (tagName, attrs = {}, y = 10) => ({
  tagName, type: tagName === 'INPUT' ? 'search' : undefined,
  value: '', checked: false, selectedIndex: -1, disabled: false, readOnly: false,
  isConnected: true, labels: [], childNodes: [], parentElement: {innerText: ''},
  getAttribute: key => attrs[key] ?? null,
  closest: () => null, matches: () => false, checkVisibility: () => true,
  getBoundingClientRect: () => ({x: 10, y, width: 200, height: 30}),
});
const button = element('BUTTON', {'aria-label': 'Search the site'});
const input = element('INPUT', {'aria-label': 'Search MDN'}, 60);
button.contains = other => other === button;
input.contains = other => other === input;
const shadowRoot = {
  querySelectorAll: selector => selector === 'input,textarea,select' ? [input] : [button, input],
  getElementById: () => null,
  elementFromPoint: (x, y) => y < 50 ? button : input,
  createTreeWalker: () => ({nextNode: () => null}),
};
const host = element('MDN-SEARCH');
host.shadowRoot = shadowRoot;
const document = {
  body: {}, documentElement: {scrollHeight: 700}, title: 'Fixture',
  querySelectorAll: selector => selector === '*' ? [host] : [],
  getElementById: () => null,
  elementFromPoint: () => host,
  createTreeWalker: () => ({nextNode: () => null}),
  createRange: () => ({}),
};
const context = {
  document, window: {}, performance: {timeOrigin: 1},
  location: {href: 'https://example.com/'},
  scrollX: 0, scrollY: 0, innerWidth: 1120, innerHeight: 780,
  NodeFilter: {SHOW_TEXT: 4},
};
const state = vm.runInNewContext(fs.readFileSync(path, 'utf8'), context);
process.stdout.write(JSON.stringify({actions: state.actions, page_key: state.page_key}));
"""
    state = run_node(fixture, str(browser.__file__).replace("browser.py", "snapshot.js"))
    assert any(a["kind"] == "click" and a["label"] == "Search the site"
               for a in state["actions"])
    assert any(a["kind"] == "fill" and a["label"] == "Search MDN"
               for a in state["actions"])
    assert len(state["page_key"][6]) == 1


def test_snapshot_reads_a_button_name_from_slotted_shadow_text():
    fixture = r"""
const fs = require('fs');
const vm = require('vm');
const slot = {
  nodeType: 1, tagName: 'SLOT', childNodes: [],
  assignedNodes: () => [{nodeType: 3, textContent: 'Search'}],
  getAttribute: () => null,
};
const label = {nodeType: 1, tagName: 'SPAN', childNodes: [slot], getAttribute: () => null};
const button = {
  tagName: 'BUTTON', type: 'button', value: '', isConnected: true,
  labels: [], childNodes: [], parentElement: {innerText: ''},
  getAttribute: key => key === 'aria-labelledby' ? 'label-id' : null,
  getRootNode: () => shadowRoot,
  matches: () => false, checkVisibility: () => true,
  getBoundingClientRect: () => ({x: 10, y: 10, width: 100, height: 30}),
  contains: other => other === button,
};
const shadowRoot = {
  querySelectorAll: selector => selector === 'input,textarea,select' ? [] : [button],
  getElementById: id => id === 'label-id' ? label : null,
  elementFromPoint: () => button,
};
const host = {shadowRoot};
const document = {
  body: {}, documentElement: {scrollHeight: 700}, title: 'Fixture',
  querySelectorAll: selector => selector === '*' ? [host] : [],
  getElementById: () => null,
  elementFromPoint: () => host,
  createTreeWalker: () => ({nextNode: () => null}),
  createRange: () => ({}),
};
const context = {
  document, window: {}, performance: {timeOrigin: 1},
  location: {href: 'https://example.com/'},
  scrollX: 0, scrollY: 0, innerWidth: 1120, innerHeight: 780,
  NodeFilter: {SHOW_TEXT: 4},
};
const state = vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
process.stdout.write(JSON.stringify(state.actions));
"""
    actions = run_node(fixture, str(browser.__file__).replace("browser.py", "snapshot.js"))

    assert any(action["label"] == "Search" for action in actions)


def test_click_hit_test_descends_into_open_shadow_root(monkeypatch):
    expressions = []

    def cdp(method, *, session_id, **parameters):
        if method == "Runtime.evaluate":
            expressions.append(parameters["expression"])
            return {"result": {"value": {"x": 110, "y": 25}}}
        return {}

    monkeypatch.setattr(browser, "cdp", cdp)
    browser.browser_operation({"operation": "act", "session": "test", "action": {
        "id": "e1", "node": 7, "kind": "click", "label": "Search the site",
    }})
    fixture = r"""
const vm = require('vm');
const expression = process.argv[1];
const button = {
  isConnected: true, tagName: 'BUTTON',
  matches: () => false, closest: () => null, checkVisibility: () => true,
  getBoundingClientRect: () => ({x: 10, y: 10, width: 200, height: 30}),
  contains: other => other === button,
};
const host = {shadowRoot: {elementFromPoint: () => button}};
const document = {elementFromPoint: () => host};
const window = {__jevFast: {nodes: new Map([[7, button]])}};
const result = vm.runInNewContext(expression, {document, window, innerWidth: 1120, innerHeight: 780});
process.stdout.write(JSON.stringify(result));
"""
    assert run_node(fixture, expressions[0]) == {"x": 110, "y": 25}


def test_snapshot_does_not_offer_a_covered_control():
    fixture = r"""
const fs = require('fs');
const vm = require('vm');
const button = {
  tagName: 'BUTTON', type: 'button', value: '', isConnected: true,
  labels: [], childNodes: [], parentElement: {innerText: ''},
  getAttribute: key => key === 'aria-label' ? 'Covered Search' : null,
  closest: () => null, matches: () => false, checkVisibility: () => true,
  getBoundingClientRect: () => ({x: 10, y: 10, width: 100, height: 30}),
  contains: other => other === button,
};
const overlay = {tagName: 'DIALOG'};
const document = {
  body: {}, documentElement: {scrollHeight: 700}, title: 'Fixture',
  querySelectorAll: selector => selector === 'input,textarea,select' ? [] : [button],
  getElementById: () => null, elementFromPoint: () => overlay,
  createTreeWalker: () => ({nextNode: () => null}),
  createRange: () => ({}),
};
const context = {
  document, window: {}, performance: {timeOrigin: 1},
  location: {href: 'https://example.com/'},
  scrollX: 0, scrollY: 0, innerWidth: 1120, innerHeight: 780,
  NodeFilter: {SHOW_TEXT: 4},
};
const state = vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
process.stdout.write(JSON.stringify(state.actions));
"""
    actions = run_node(fixture, str(browser.__file__).replace("browser.py", "snapshot.js"))
    assert all(action["label"] != "Covered Search" for action in actions)


@pytest.mark.parametrize("lock_target", ["html", "body"])
def test_focused_search_in_scroll_locked_overlay_offers_text_not_redundant_actions(lock_target):
    fixture = r"""
const fs = require('fs');
const vm = require('vm');
const input = {
  tagName: 'INPUT', type: 'search', value: '', checked: false, selectedIndex: -1,
  disabled: false, readOnly: false, isConnected: true, labels: [], childNodes: [],
  parentElement: {innerText: ''},
  getAttribute: key => key === 'aria-label' ? 'Search' : null,
  matches: () => false, checkVisibility: () => true,
  getBoundingClientRect: () => ({x: 10, y: 10, width: 200, height: 30}),
  contains: other => other === input,
  getRootNode: () => shadowRoot,
};
const shadowRoot = {
  activeElement: input,
  querySelectorAll: selector => selector === 'input,textarea,select' ? [input] : [input],
  getElementById: () => null,
  elementFromPoint: () => input,
  createTreeWalker: () => ({nextNode: () => null}),
};
const host = {shadowRoot};
const document = {
  body: {}, documentElement: {scrollHeight: 2800}, title: 'Results',
  querySelectorAll: selector => selector === '*' ? [host] : [],
  getElementById: () => null,
  elementFromPoint: () => host,
  createTreeWalker: () => ({nextNode: () => null}),
  createRange: () => ({}),
};
const context = {
  document, window: {}, performance: {timeOrigin: 1},
  location: {href: 'https://example.com/search'},
  scrollX: 0, scrollY: 0, innerWidth: 1120, innerHeight: 780,
  getComputedStyle: element => ({overflowY: element ===
    (process.argv[2] === 'body' ? document.body : document.documentElement) ? 'hidden' : 'visible'}),
  NodeFilter: {SHOW_TEXT: 4},
};
const state = vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
process.stdout.write(JSON.stringify(state.actions));
"""
    actions = run_node(fixture, str(browser.__file__).replace("browser.py", "snapshot.js"), lock_target)

    assert any(action["kind"] == "fill" and action["label"] == "Search" for action in actions)
    assert all(action["kind"] != "scroll" for action in actions)
    assert all(action["label"] != "Open Search" for action in actions)


def test_shadow_control_near_top_survives_action_limit_before_later_links():
    fixture = r"""
const fs = require('fs');
const vm = require('vm');
const node = (tagName, label, y) => ({
  tagName, type: tagName === 'BUTTON' ? 'button' : undefined,
  value: '', labels: [], childNodes: [], isConnected: true,
  parentElement: {innerText: ''},
  getAttribute: key => key === 'aria-label' ? label : key === 'href' ? '#' : null,
  matches: () => false, checkVisibility: () => true,
  getBoundingClientRect: () => ({x: 10, y, width: 100, height: 1}),
  contains(other) { return other === this; },
});
const button = node('BUTTON', 'Top Search', 10);
const links = Array.from({length: 260}, (_, i) => node('A', `Later ${i}`, 100 + i * 2));
const shadowRoot = {
  querySelectorAll: selector => selector === 'input,textarea,select' ? [] : [button],
  getElementById: () => null,
  elementFromPoint: () => button,
  createTreeWalker: () => ({nextNode: () => null}),
};
const host = {shadowRoot};
const document = {
  body: {}, documentElement: {scrollHeight: 700}, title: 'Fixture',
  querySelectorAll: selector => selector === '*' ? [host, ...links]
    : selector === 'input,textarea,select' ? [] : links,
  getElementById: () => null,
  elementFromPoint: (_x, y) => y < 50 ? host : links[Math.floor((y - 100) / 2)],
  createTreeWalker: () => ({nextNode: () => null}),
  createRange: () => ({}),
};
const context = {
  document, window: {}, performance: {timeOrigin: 1},
  location: {href: 'https://example.com/'},
  scrollX: 0, scrollY: 0, innerWidth: 1120, innerHeight: 780,
  NodeFilter: {SHOW_TEXT: 4},
};
const state = vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
process.stdout.write(JSON.stringify(state.actions));
"""
    actions = run_node(fixture, str(browser.__file__).replace("browser.py", "snapshot.js"))

    assert any(action["label"] == "Top Search" for action in actions)
    assert any(action["role"] == "link" and action["href"] == "#" for action in actions)


def test_shadow_text_near_top_survives_document_text_limit():
    fixture = r"""
const fs = require('fs');
const vm = require('vm');
const parent = {matches: () => false, parentElement: null, checkVisibility: () => true};
const text = value => ({nodeType: 3, textContent: value, parentElement: parent});
const before = text('Before');
const shadow = text('Shadow detail');
const after = text('Z'.repeat(6000));
const walker = nodes => { let i = 0; return {nextNode: () => nodes[i++] || null}; };
const shadowRoot = {
  querySelectorAll: () => [],
  createTreeWalker: () => walker([shadow]),
};
const host = {nodeType: 1, shadowRoot};
const document = {
  body: {}, documentElement: {scrollHeight: 700}, title: 'Fixture',
  querySelectorAll: selector => selector === '*' ? [host] : [],
  createTreeWalker: (root, mask) => root === shadowRoot ? walker([shadow])
    : walker(mask === 4 ? [before, after] : [before, host, after]),
  createRange: () => ({selectNodeContents: () => {},
                      getBoundingClientRect: () => ({width: 100, height: 10,
                                                     top: 10, bottom: 20, left: 10, right: 110})}),
};
const context = {
  document, window: {}, performance: {timeOrigin: 1},
  location: {href: 'https://example.com/'},
  scrollX: 0, scrollY: 0, innerWidth: 1120, innerHeight: 780,
  NodeFilter: {SHOW_ELEMENT: 1, SHOW_TEXT: 4},
};
const state = vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
process.stdout.write(JSON.stringify(state.text));
"""
    text = run_node(fixture, str(browser.__file__).replace("browser.py", "snapshot.js"))

    assert text.startswith("Before\nShadow detail\n")
