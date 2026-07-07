"""MCP server exposing the reddit-research pipeline as tools.

Exposes the low-level tools (``search_reddit_threads``, ``fetch_reddit_thread``,
``rank_reddit_evidence``) plus the high-level ``answer_from_reddit`` orchestration
tool. Requires the optional ``mcp`` dependency: ``uv pip install '.[mcp]'``.
"""

from __future__ import annotations

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
) -> dict[str, Any]:
    """Run the full pipeline and optionally synthesize a cited answer."""

    orch = _orchestrator()
    try:
        result = orch.answer(
            question,
            subreddits=subreddits,
            limit_threads=limit_threads,
            max_comments_per_thread=max_comments_per_thread,
            include_answer=include_answer,
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


def main() -> None:  # pragma: no cover - stdio entry
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
