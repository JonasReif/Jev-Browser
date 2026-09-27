"""A stalled model response cannot hold the browser worker indefinitely."""

import time
from contextlib import contextmanager

import pytest

from jev_ultrafast import model


def test_stalled_model_response_is_retried_before_any_action(monkeypatch):
    calls = []
    active = 0
    max_active = 0

    @contextmanager
    def stream(*_args, **_kwargs):
        nonlocal active, max_active
        calls.append(1)
        active += 1
        max_active = max(max_active, active)

        class Response:
            status_code = 200
            is_error = False

            def iter_bytes(self):
                if len(calls) == 1:
                    time.sleep(0.03)
                yield b'{"ok": true}'

        try:
            yield Response()
        finally:
            active -= 1

    monkeypatch.setattr(model, "REQUEST_WALL_TIMEOUT", 0.02, raising=False)
    monkeypatch.setattr(model.CLIENT, "stream", stream)
    monkeypatch.setattr(model.CLIENT, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("The non-streaming request leaves a response read unbounded")
    ))
    before = time.monotonic()
    result = model.post_json("https://example.test/model", "test-key", {"model": "test"})
    elapsed = time.monotonic() - before

    assert result["ok"] is True
    assert result["_jev_transport"] == {"requests": 2, "usage_unknown": 1}
    assert len(calls) == 2
    assert max_active == 1
    assert active == 0
    assert elapsed < 0.5


def test_malformed_model_json_reports_invalid_response_after_retries(monkeypatch):
    calls = []

    @contextmanager
    def stream(*_args, **_kwargs):
        calls.append(1)

        class Response:
            status_code = 200
            is_error = False

            def iter_bytes(self):
                yield b"not json"

        yield Response()

    monkeypatch.setattr(model.CLIENT, "stream", stream)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        model.post_json("https://example.test/model", "test-key", {"model": "test"})

    assert len(calls) == 3
