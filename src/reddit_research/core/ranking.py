"""Lexical ranking of threads and comments against a question.

Deliberately simple and explainable (per the spec): combine bag-of-words
overlap with light popularity/recency/structure signals, all normalized to
[0, 1]. This is the MVP ranker; semantic reranking can layer on later without
changing the interface.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter

from .models import Comment, Evidence, SearchResult, Thread

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how i if in into is it its of on or
    that the their them then there these they this to was were what when where which who
    why will with you your do does did can could should would about best good vs recommend
    """.split()
)


def _fold(token: str) -> str:
    """Cheap singular/plural folding so 'backups' matches 'backup'.

    Not a real stemmer — just enough to lift recall for the MVP lexical ranker.
    """

    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [
        _fold(t)
        for t in _TOKEN_RE.findall((text or "").lower())
        if t not in _STOPWORDS
    ]


def _overlap(query_tokens: Counter[str], text: str) -> float:
    """Fraction of (weighted) query tokens present in ``text``, in [0, 1]."""

    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    matched = sum(w for tok, w in query_tokens.items() if tok in text_tokens)
    total = sum(query_tokens.values())
    return matched / total if total else 0.0


def _popularity(score: int) -> float:
    """Squash a Reddit vote score into [0, 1] with diminishing returns."""

    if score <= 0:
        return 0.0
    return min(1.0, math.log10(score + 1) / 3.0)  # ~1.0 around 1000 upvotes


def _recency(created_utc: float | None, now: float, half_life_days: float = 365.0) -> float:
    if not created_utc:
        return 0.5
    age_days = max(0.0, (now - created_utc) / 86400)
    return 0.5 ** (age_days / half_life_days)


def rank_threads(
    question: str,
    results: list[SearchResult],
    subreddits: list[str] | None = None,
) -> list[SearchResult]:
    """Re-score search results using lexical overlap + provider rank + allowlist."""

    qtokens = Counter(tokenize(question))
    allow = {s.lower() for s in (subreddits or [])}
    scored: list[SearchResult] = []
    seen_ids: set[str] = set()
    for r in results:
        # Crude crosspost/duplicate suppression by title.
        key = re.sub(r"\W+", "", r.title.lower())[:80]
        if key in seen_ids:
            continue
        seen_ids.add(key)

        lexical = _overlap(qtokens, f"{r.title} {r.snippet}")
        provider = r.score  # already a rank-decay value in (0, 1]
        allow_bonus = 0.15 if (allow and r.subreddit and r.subreddit.lower() in allow) else 0.0
        combined = 0.55 * lexical + 0.35 * provider + allow_bonus
        scored.append(r.model_copy(update={"score": round(min(1.0, combined), 4)}))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


def rank_evidence(
    question: str,
    threads: list[Thread],
    max_evidence_items: int = 25,
    include_threads: bool = True,
) -> list[Evidence]:
    """Rank thread bodies and comments into a flat, cited evidence list."""

    qtokens = Counter(tokenize(question))
    now = time.time()
    items: list[Evidence] = []

    for thread in threads:
        if include_threads:
            body = f"{thread.title}\n{thread.selftext}".strip()
            lexical = _overlap(qtokens, body)
            if lexical > 0:
                combined = (
                    0.6 * lexical
                    + 0.25 * _popularity(thread.score)
                    + 0.15 * _recency(thread.created_utc, now)
                )
                items.append(
                    Evidence(
                        kind="thread",
                        thread_id=thread.id,
                        thread_title=thread.title,
                        subreddit=thread.subreddit,
                        text=(thread.selftext or thread.title)[:2000],
                        permalink=thread.permalink,
                        score=round(min(1.0, combined), 4),
                        reddit_score=thread.score,
                    )
                )

        for c in thread.comments:
            if c.removed or not c.body:
                continue
            lexical = _overlap(qtokens, c.body)
            if lexical <= 0:
                continue
            depth_penalty = 1.0 / (1.0 + 0.35 * c.depth)
            combined = (
                0.6 * lexical
                + 0.25 * _popularity(c.score)
                + 0.15 * _recency(c.created_utc, now)
            ) * depth_penalty
            items.append(
                Evidence(
                    kind="comment",
                    thread_id=thread.id,
                    thread_title=thread.title,
                    subreddit=thread.subreddit,
                    text=c.body[:2000],
                    permalink=c.permalink or thread.permalink,
                    score=round(min(1.0, combined), 4),
                    reddit_score=c.score,
                )
            )

    items.sort(key=lambda x: x.score, reverse=True)
    return _dedupe(items)[:max_evidence_items]


def _dedupe(items: list[Evidence]) -> list[Evidence]:
    """Drop repeated text (double-posts, copy-paste spam), keeping the best-ranked.

    ``items`` must already be sorted best-first. Text is compared after folding
    case, punctuation and whitespace, so trivially-different copies collapse.
    """

    seen: set[str] = set()
    out: list[Evidence] = []
    for ev in items:
        key = " ".join(_TOKEN_RE.findall(ev.text.lower()))
        if key and key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out
