"""Render pipeline outputs as JSON or Markdown for the CLI."""

from __future__ import annotations

import json

from .models import (
    AnswerResult,
    EvidenceResult,
    SearchResult,
    Thread,
)


def _json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def render_search(results: list[SearchResult], fmt: str) -> str:
    if fmt == "json":
        return _json([r.model_dump() for r in results])
    if not results:
        return "_No results._"
    lines = ["# Search results", ""]
    for i, r in enumerate(results, 1):
        sub = f"r/{r.subreddit}" if r.subreddit else "reddit"
        lines.append(f"{i}. **{r.title}** ({sub}) — score {r.score:.2f} · {r.source}")
        if r.snippet:
            lines.append(f"   > {r.snippet}")
        lines.append(f"   {r.url}")
        lines.append("")
    return "\n".join(lines)


def render_thread(thread: Thread, fmt: str) -> str:
    if fmt == "json":
        return _json(thread.model_dump())
    lines = [
        f"# {thread.title}",
        "",
        f"r/{thread.subreddit} · score {thread.score} · {thread.num_comments} comments",
        f"{thread.permalink}",
        "",
    ]
    if thread.selftext:
        lines += [thread.selftext.strip(), ""]
    lines.append(f"## Comments ({len(thread.comments)})")
    for c in thread.comments:
        if c.removed:
            continue
        indent = "  " * min(c.depth, 6)
        lines.append(f"{indent}- ({c.score}) {c.body.strip()[:500]}")
    if thread.comments_truncated:
        lines.append("\n_Comment tree truncated by configured limits._")
    return "\n".join(lines)


def render_evidence(result: EvidenceResult, fmt: str) -> str:
    if fmt == "json":
        return _json(result.model_dump())
    lines = ["# Evidence", ""]
    for i, ev in enumerate(result.evidence, 1):
        lines.append(
            f"{i}. [{ev.kind}] r/{ev.subreddit} · relevance {ev.score:.2f} · "
            f"reddit score {ev.reddit_score}"
        )
        lines.append(f"   > {ev.text.strip()[:400]}")
        lines.append(f"   {ev.permalink}")
        lines.append("")
    lines += _render_metadata(result.metadata)
    return "\n".join(lines)


def render_answer(result: AnswerResult, fmt: str) -> str:
    if fmt == "json":
        return _json(result.model_dump())
    lines = []
    if result.answer:
        lines += [result.answer, ""]
    else:
        lines += ["_No synthesized answer (synthesis disabled or unavailable)._", ""]
        lines += render_evidence(
            EvidenceResult(
                evidence=result.evidence,
                threads_considered=result.threads_considered,
                metadata=result.metadata,
            ),
            "markdown",
        ).splitlines()
        return "\n".join(lines)
    lines += _render_metadata(result.metadata)
    return "\n".join(lines)


def _render_metadata(meta) -> list[str]:
    out = ["---", ""]
    out.append(
        f"_threads fetched: {meta.threads_fetched} · comments: {meta.comments_fetched} · "
        f"cache {meta.cache_hits}h/{meta.cache_misses}m_"
    )
    for w in meta.warnings:
        out.append(f"- ⚠ {w}")
    return out
