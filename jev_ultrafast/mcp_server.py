"""MCP server: Claude, ChatGPT, or any MCP client hands one natural-language goal to the Jev agent."""

import argparse
import base64
import contextlib
import os
import secrets
import sys
import time
from pathlib import Path

import anyio
from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from .agent import Agent
from .browser import Browser
from .demo import load_environment
from .model import action_space

PAGE_TEXT = 8000
SERVER = FastMCP(
    "jev-browser",
    instructions=(
        "browser_task drives a real Chrome tab toward one natural-language goal and returns the executed steps "
        "and the final page. A 'done' status is the agent's own judgment: check final_page before reporting "
        "success. read_page observes a page without model calls. The tab uses the host's Chrome profile."
    ),
)
# Browser Harness drives one Chrome through one daemon. Run one owned tab at a time.
LOCK = anyio.Lock()


def quiet(fn, *args):
    # Under stdio, stdout carries the MCP protocol. Library output goes to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        return fn(*args)


async def in_thread(fn, *args):
    return await anyio.to_thread.run_sync(quiet, fn, *args)


def check_url(url):
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")


def page_summary(page):
    return {"url": page["url"], "title": page["title"], "text": page["text"][:PAGE_TEXT]}


def screenshot_of(browser):
    # Optional evidence for the caller. A slow capture must not discard the task result.
    try:
        data = browser.call("Page.captureScreenshot", _response_timeout=10, format="jpeg", quality=72)["data"]
    except Exception as exc:
        return f"Screenshot unavailable: {exc}"
    return Image(data=base64.b64decode(data), format="jpeg")


def task_result(agent, status, error):
    state = agent.state
    return {
        "status": status,
        "error": error,
        "verify": "The agent's DONE is not proof of success. Check final_page against the goal.",
        "elapsed_ms": state["elapsed_ms"],
        "steps": [
            {k: h.get(k) for k in ("step", "action", "kind", "text", "page_changed", "url")} for h in state["history"]
        ],
        "model_calls": {"typesafe": len(state["decisions"]), "text": len(state["text_calls"])},
        "final_page": page_summary(state["page"]),
    }


@SERVER.tool(
    annotations=ToolAnnotations(title="Run a browser task", destructiveHint=True, openWorldHint=True),
)
async def browser_task(goal: str, url: str, max_seconds: int = 120, screenshot: bool = False, ctx: Context = None):
    """Open url in a new Chrome tab and let the Jev agent pursue goal with clicks, typing, and dropdowns.

    goal: one natural-language task, including every value the agent needs (names, dates, search terms).
    url: the http(s) page to start from.
    max_seconds: wall-clock budget; the run stops after the current step once it is exceeded.
    screenshot: also return a JPEG of the final page.

    Returns status (done, blocked, timeout, error), executed steps, model-call counts, and the final page text.
    The agent cannot upload files, handle iframes or pop-up tabs, or solve CAPTCHAs.
    """
    check_url(url)
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise ValueError("TYPESAFE_API_KEY is not set on the MCP server")
    async with LOCK:
        agent = await in_thread(Agent, url, goal)
        try:
            deadline = time.monotonic() + max(5, max_seconds)
            status, error, reported = "timeout", None, 0
            while agent.state["status"] not in {"done", "blocked"} and time.monotonic() < deadline:
                try:
                    state = await in_thread(agent.command, "tick")
                except Exception as exc:
                    # Never retry: the failed step may already have changed the page. Report what ran.
                    status, error = "error", str(exc) or type(exc).__name__
                    break
                if ctx and len(state["history"]) > reported:
                    reported = len(state["history"])
                    with contextlib.suppress(Exception):
                        await ctx.report_progress(reported, message=state["history"][-1]["action"])
            if agent.state["status"] in {"done", "blocked"}:
                status = agent.state["status"]
            result = task_result(agent, status, error)
            image = await in_thread(screenshot_of, agent.browser) if screenshot else None
        finally:
            await in_thread(agent.close)
    return [result, image] if image else result


@SERVER.tool(annotations=ToolAnnotations(title="Read a page", readOnlyHint=True, openWorldHint=True))
async def read_page(url: str, screenshot: bool = False):
    """Open url in a new Chrome tab and return its visible text and the indexed controls Jev would offer.

    No model calls and no clicks. Useful to check what a page offers before writing a goal for browser_task.
    """
    check_url(url)
    async with LOCK:
        browser = await in_thread(Browser, url)
        try:
            page = await in_thread(browser.observe, False)
            image = await in_thread(screenshot_of, browser) if screenshot else None
        finally:
            await in_thread(browser.close)
    result = {**page_summary(page), "elements": action_space(page["actions"])[0]}
    return [result, image] if image else result


def configure_http(port, public_hosts, token):
    SERVER.settings.port = port
    SERVER.settings.streamable_http_path = f"/{token}/mcp"
    hosts = [h.removeprefix("https://").rstrip("/") for h in public_hosts]
    SERVER.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", *hosts],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", *(f"https://{h}" for h in hosts)],
    )
    return [f"http://127.0.0.1:{port}/{token}/mcp", *(f"https://{h}/{token}/mcp" for h in hosts)]


def main():
    parser = argparse.ArgumentParser(description="Jev browser agent as an MCP server.")
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--public-host", action="append", default=[], help="tunnel hostname that forwards to this port (repeatable)"
    )
    args = parser.parse_args()
    load_environment()
    load_environment(Path(__file__).parent.parent / ".env")
    if not args.http:
        SERVER.run("stdio")
        return
    # No accounts: the secret path is the credential. Keep it private; set JEV_MCP_TOKEN to keep it across restarts.
    token = os.environ.get("JEV_MCP_TOKEN") or secrets.token_urlsafe(24)
    for url in configure_http(args.port, args.public_host, token):
        print(f"MCP endpoint: {url}", file=sys.stderr, flush=True)
    SERVER.run("streamable-http")


if __name__ == "__main__":
    main()
