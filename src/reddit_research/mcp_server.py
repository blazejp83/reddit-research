"""MCP server exposing the reddit-research pipeline as tools.

Exposes the low-level tools (``search_reddit_threads``, ``fetch_reddit_thread``,
``rank_reddit_evidence``) plus the high-level ``answer_from_reddit`` orchestration
tool. Requires the optional ``mcp`` dependency: ``uv pip install '.[mcp]'``.

Transports (via ``REDDIT_RESEARCH_MCP_TRANSPORT``):
- ``stdio`` (default) — local use; the client spawns this process.
- ``http`` / ``streamable-http`` — long-lived network server for container/NAS
  deployment. Bearer-token auth is enforced (``REDDIT_RESEARCH_MCP_TOKEN``); the
  server refuses to start over HTTP without a token so it can't be exposed open.
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from .core import ranking
from .core.config import load_config
from .core.models import Thread
from .core.orchestrator import Orchestrator

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional dep
    raise SystemExit(
        "The MCP server requires the 'mcp' extra. Install with: uv pip install '.[mcp]'"
    ) from exc

mcp = FastMCP("reddit-research")


def _orchestrator() -> Orchestrator:
    return Orchestrator(load_config())


@mcp.tool()
def search_reddit_threads(
    query: str,
    subreddits: list[str] | None = None,
    limit: int = 10,
    recency_days: int | None = None,
) -> dict[str, Any]:
    """Find candidate Reddit threads through the configured search provider."""

    orch = _orchestrator()
    try:
        from .core.models import RunMetadata

        meta = RunMetadata()
        results = orch.search(query, subreddits, limit, recency_days, meta=meta)
        return {
            "results": [r.model_dump() for r in results],
            "metadata": meta.model_dump(),
        }
    finally:
        orch.close()


@mcp.tool()
def fetch_reddit_thread(
    url_or_id: str,
    max_comments: int = 300,
    comment_sort: str = "top",
    include_more_comments: bool = True,
) -> dict[str, Any]:
    """Fetch and normalize one Reddit thread (post + bounded comment tree)."""

    orch = _orchestrator()
    try:
        thread = orch.fetch(
            url_or_id,
            max_comments=max_comments,
            comment_sort=comment_sort,
            include_more_comments=include_more_comments,
        )
        return {"thread": thread.model_dump()}
    finally:
        orch.close()


@mcp.tool()
def rank_reddit_evidence(
    question: str,
    threads: list[dict[str, Any]],
    max_evidence_items: int = 25,
) -> dict[str, Any]:
    """Rank already-fetched thread/comment data against a question."""

    parsed = [Thread.model_validate(t) for t in threads]
    evidence = ranking.rank_evidence(question, parsed, max_evidence_items)
    return {"evidence": [e.model_dump() for e in evidence]}


@mcp.tool()
def answer_from_reddit(
    question: str,
    subreddits: list[str] | None = None,
    limit_threads: int = 8,
    max_comments_per_thread: int = 250,
    include_answer: bool = True,
    recency_days: int | None = None,
    comment_sort: str | None = None,
) -> dict[str, Any]:
    """Run the full pipeline and optionally synthesize a cited answer.

    Set ``recency_days`` for time-sensitive questions (current products, prices,
    software): it limits search to recent threads and favors newer comments.
    ``comment_sort`` is top (default) | new | old | controversial.
    """

    orch = _orchestrator()
    try:
        result = orch.answer(
            question,
            subreddits=subreddits,
            limit_threads=limit_threads,
            max_comments_per_thread=max_comments_per_thread,
            include_answer=include_answer,
            recency_days=recency_days,
            comment_sort=comment_sort,
        )
        return {
            "answer": result.answer,
            "evidence": [e.model_dump() for e in result.evidence],
            "threads_considered": [t.model_dump() for t in result.threads_considered],
            "warnings": result.metadata.warnings,
            "metadata": result.metadata.model_dump(),
        }
    finally:
        orch.close()


class BearerAuthMiddleware:
    """Minimal ASGI middleware enforcing a static bearer token.

    Defense-in-depth for internet-exposed deployments: even behind a reverse
    proxy, the app itself rejects requests without ``Authorization: Bearer
    <token>``. ``GET /healthz`` is left open for container health checks. Non-HTTP
    scopes (lifespan/websocket) pass straight through to the wrapped app.
    """

    def __init__(self, app, token: str, health_path: str = "/healthz") -> None:
        self.app = app
        self.token = token
        self.health_path = health_path

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path") == self.health_path:
            await self._plain(send, 200, b"ok")
            return
        headers = dict(scope.get("headers") or [])
        provided = headers.get(b"authorization", b"").decode()
        if not hmac.compare_digest(provided, f"Bearer {self.token}"):
            await self._plain(send, 401, b"unauthorized")
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _plain(send, status: int, body: bytes) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _transport_security():
    """Build DNS-rebinding (Host/Origin) settings for HTTP transport.

    FastMCP's default only allows localhost Host headers, so a request arriving
    through a reverse proxy (Host = your public domain) is rejected with 421.
    Bearer-token auth is the real access gate here, so host validation is OFF by
    default. Set ``REDDIT_RESEARCH_MCP_ALLOWED_HOSTS`` (comma-separated hostnames)
    to instead enforce a strict allowlist, e.g. ``reddit.example.com``.
    """

    from mcp.server.transport_security import TransportSecuritySettings

    allowed = [
        h.strip()
        for h in os.environ.get("REDDIT_RESEARCH_MCP_ALLOWED_HOSTS", "").split(",")
        if h.strip()
    ]
    if not allowed:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)

    hosts: list[str] = []
    origins: list[str] = []
    for h in allowed:
        hosts += [h, f"{h}:*"]
        origins += [f"https://{h}", f"https://{h}:*", f"http://{h}", f"http://{h}:*"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


def _run_http() -> None:  # pragma: no cover - network entry
    import uvicorn

    token = os.environ.get("REDDIT_RESEARCH_MCP_TOKEN")
    if not token:
        raise SystemExit(
            "REDDIT_RESEARCH_MCP_TOKEN must be set for HTTP transport. Refusing to "
            "expose an unauthenticated MCP server that can spend your API keys."
        )
    host = os.environ.get("REDDIT_RESEARCH_MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("REDDIT_RESEARCH_MCP_PORT", "8000"))
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.transport_security = _transport_security()

    app = BearerAuthMiddleware(mcp.streamable_http_app(), token)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.environ.get("REDDIT_RESEARCH_MCP_LOG_LEVEL", "info"),
    )


def main() -> None:  # pragma: no cover - entry point
    transport = os.environ.get("REDDIT_RESEARCH_MCP_TRANSPORT", "stdio").lower()
    if transport in ("http", "streamable-http", "streamable_http"):
        _run_http()
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run()  # stdio


if __name__ == "__main__":  # pragma: no cover
    main()
