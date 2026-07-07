"""Pydantic data models shared across the pipeline.

These are the normalized shapes stored in the cache and returned by the CLI and
MCP tools. Keeping them in one place makes the tool/JSON contract explicit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CommentSort = Literal["top", "best", "new", "controversial", "old", "qa"]
ThreadSort = Literal["relevance", "top", "new", "controversial"]
OutputFormat = Literal["json", "markdown"]


class SearchResult(BaseModel):
    """A candidate Reddit thread surfaced by a search provider."""

    title: str
    url: str
    subreddit: str | None = None
    snippet: str = ""
    source: str = "unknown"
    # Provider-normalized relevance in [0, 1]; higher is better.
    score: float = 0.0


class Comment(BaseModel):
    """A normalized Reddit comment."""

    id: str
    parent_id: str | None = None
    body: str = ""
    author: str | None = None
    score: int = 0
    depth: int = 0
    created_utc: float | None = None
    permalink: str | None = None
    # True when the comment was deleted or removed (body is a placeholder).
    removed: bool = False


class Thread(BaseModel):
    """A normalized Reddit submission plus its (bounded) comment tree."""

    id: str
    subreddit: str
    title: str
    url: str
    permalink: str
    selftext: str = ""
    author: str | None = None
    score: int = 0
    upvote_ratio: float | None = None
    created_utc: float | None = None
    num_comments: int = 0
    over_18: bool = False
    comments: list[Comment] = Field(default_factory=list)
    # True when MoreComments were left unexpanded because of configured limits.
    comments_truncated: bool = False
    fetched_utc: float | None = None


class Evidence(BaseModel):
    """A ranked evidence item (a thread or a single comment) for a question."""

    kind: Literal["thread", "comment"]
    thread_id: str
    thread_title: str
    subreddit: str
    text: str
    permalink: str
    # Relevance to the question in [0, 1].
    score: float = 0.0
    # Underlying Reddit vote score of the thread/comment.
    reddit_score: int = 0


class RunMetadata(BaseModel):
    """Observability metadata attached to pipeline responses."""

    search_queries: list[str] = Field(default_factory=list)
    threads_fetched: int = 0
    comments_fetched: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    warnings: list[str] = Field(default_factory=list)


class EvidenceResult(BaseModel):
    """Output of the evidence pipeline (search + fetch + rank)."""

    evidence: list[Evidence] = Field(default_factory=list)
    threads_considered: list[SearchResult] = Field(default_factory=list)
    metadata: RunMetadata = Field(default_factory=RunMetadata)


class AnswerResult(BaseModel):
    """Output of the full answer pipeline, with optional synthesis."""

    answer: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    threads_considered: list[SearchResult] = Field(default_factory=list)
    metadata: RunMetadata = Field(default_factory=RunMetadata)
