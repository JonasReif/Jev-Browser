"""Offline MCP contracts. A fake agent stands in for Chrome and the models. No paid APIs."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from jev_ultrafast import mcp_server


class FakeAgent:
    instances = []

    def __init__(self, url, goal, outcomes):
        self.outcomes = list(outcomes)
        self.ticks = 0
        self.closed = False
        self.browser = Mock(call=Mock(side_effect=TimeoutError("capture timed out")))
        page = {"url": url, "title": "Start", "text": "x" * 20000, "actions": []}
        self.state = dict(status="ready", history=[], decisions=[], text_calls=[], elapsed_ms=0, page=page)
        FakeAgent.instances.append(self)

    def command(self, name):
        assert name == "tick"
        self.ticks += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        self.state["decisions"].append({})
        if outcome == "DONE":
            self.state["status"] = "done"
        else:
            step = len(self.state["history"]) + 1
            self.state["history"].append({"step": step, "action": outcome, "kind": "click", "usage": {"secret": 1}})
        return self.state

    def close(self):
        self.closed = True


def use_agent(monkeypatch, *outcomes):
    FakeAgent.instances.clear()
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(mcp_server, "Agent", lambda url, goal: FakeAgent(url, goal, outcomes))


def run_task(**kwargs):
    return asyncio.run(mcp_server.browser_task(**{"goal": "Find it", "url": "https://example.test/", **kwargs}))


def test_task_returns_steps_and_final_page_and_closes_its_tab(monkeypatch):
    use_agent(monkeypatch, "Search", "Open result", "DONE")
    ctx = Mock(report_progress=AsyncMock())
    result = run_task(ctx=ctx)
    agent = FakeAgent.instances[0]
    assert result["status"] == "done" and result["error"] is None
    assert [s["action"] for s in result["steps"]] == ["Search", "Open result"]
    assert "usage" not in result["steps"][0]
    assert result["model_calls"] == {"typesafe": 3, "text": 0}
    assert len(result["final_page"]["text"]) == mcp_server.PAGE_TEXT
    assert "not proof" in result["verify"]
    assert agent.closed
    assert [c.kwargs["message"] for c in ctx.report_progress.await_args_list] == ["Search", "Open result"]


def test_failed_step_is_reported_not_retried(monkeypatch):
    use_agent(monkeypatch, "Search", RuntimeError("Dropdown execution was not confirmed"), "DONE")
    result = run_task()
    agent = FakeAgent.instances[0]
    assert result["status"] == "error" and "Dropdown" in result["error"]
    assert agent.ticks == 2 and agent.closed
    assert [s["action"] for s in result["steps"]] == ["Search"]


def test_wall_clock_budget_stops_the_run(monkeypatch):
    use_agent(monkeypatch, *["Scroll"] * 10)
    clock = iter(range(0, 1000, 4))
    monkeypatch.setattr(mcp_server.time, "monotonic", lambda: next(clock))
    result = run_task(max_seconds=10)
    assert result["status"] == "timeout"
    assert 0 < len(result["steps"]) < 10 and FakeAgent.instances[0].closed


def test_screenshot_failure_keeps_the_result(monkeypatch):
    use_agent(monkeypatch, "DONE")
    result, note = run_task(screenshot=True)
    assert result["status"] == "done"
    assert isinstance(note, str) and note == "Screenshot unavailable: capture timed out"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "javascript:alert(1)", "chrome://settings"])
def test_only_web_urls_open_a_tab(monkeypatch, url):
    use_agent(monkeypatch, "DONE")
    with pytest.raises(ValueError):
        run_task(url=url)
    with pytest.raises(ValueError):
        asyncio.run(mcp_server.read_page(url))
    assert not FakeAgent.instances


def test_missing_key_fails_before_opening_a_tab(monkeypatch):
    use_agent(monkeypatch, "DONE")
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        run_task()
    assert not FakeAgent.instances


def test_http_endpoint_is_behind_a_secret_path_and_known_hosts(monkeypatch):
    monkeypatch.setattr(mcp_server.SERVER, "settings", mcp_server.SERVER.settings.model_copy())
    urls = mcp_server.configure_http(9000, ["https://abc.ngrok-free.app/"], "s3cret")
    assert urls == ["http://127.0.0.1:9000/s3cret/mcp", "https://abc.ngrok-free.app/s3cret/mcp"]
    security = mcp_server.SERVER.settings.transport_security
    assert security.enable_dns_rebinding_protection
    assert "abc.ngrok-free.app" in security.allowed_hosts
    assert "https://abc.ngrok-free.app" in security.allowed_origins


def test_tools_are_listed_with_safety_hints():
    tools = {t.name: t for t in asyncio.run(mcp_server.SERVER.list_tools())}
    assert set(tools) == {"browser_task", "read_page"}
    assert tools["read_page"].annotations.readOnlyHint
    assert tools["browser_task"].annotations.destructiveHint
    assert "ctx" not in tools["browser_task"].inputSchema["properties"]
