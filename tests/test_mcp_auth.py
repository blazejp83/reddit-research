"""Unit tests for the MCP bearer-auth ASGI middleware (no network)."""

import asyncio

import pytest

pytest.importorskip("mcp")

from reddit_research.mcp_server import BearerAuthMiddleware  # noqa: E402


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int | None:
        for m in self.messages:
            if m["type"] == "http.response.start":
                return m["status"]
        return None


def _inner_app():
    state = {"called": False}

    async def app(scope, receive, send):
        state["called"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"inner"})

    return app, state


def _call(path: str, headers: list[tuple[bytes, bytes]]):
    app, state = _inner_app()
    mw = BearerAuthMiddleware(app, "s3cret")
    scope = {"type": "http", "path": path, "headers": headers}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    rec = _Recorder()
    asyncio.run(mw(scope, receive, rec.send))
    return rec, state


def test_healthz_is_open_and_does_not_reach_app():
    rec, state = _call("/healthz", [])
    assert rec.status == 200
    assert state["called"] is False


def test_missing_token_rejected():
    rec, state = _call("/mcp", [])
    assert rec.status == 401
    assert state["called"] is False


def test_wrong_token_rejected():
    rec, state = _call("/mcp", [(b"authorization", b"Bearer nope")])
    assert rec.status == 401
    assert state["called"] is False


def test_correct_token_passes_through():
    rec, state = _call("/mcp", [(b"authorization", b"Bearer s3cret")])
    assert rec.status == 200
    assert state["called"] is True
