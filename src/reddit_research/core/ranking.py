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


# Relevance vs. community signal. Lexical overlap is a coarse match (any comment
# echoing the question's words scores high), so votes need enough weight to lift
# well-received answers over low-vote ones that merely repeat the question.
W_LEXICAL = 0.5
W_POPULARITY = 0.35
W_RECENCY = 0.15


def _comment_popularity(score: int, thread_max: int) -> float:
    """Blend absolute votes with votes relative to the thread's best comment.

    The relative half lets the standout answers in a small, quiet thread compete
    with comments from a big thread; the absolute half keeps a 3-vote "top"
    comment from looking as validated as a 300-vote one.
    """

    absolute = _popularity(score)
    if score <= 0 or thread_max <= 0:
        return 0.5 * absolute
    relative = math.log10(score + 1) / math.log10(thread_max + 1)
    return 0.5 * absolute + 0.5 * relative


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


# How much of the thread's relevance a comment inherits, by depth. Top-level
# comments answer the post directly ("Boba U4s."), so they inherit the most;
# deep replies drift off-topic and must mostly earn relevance themselves.
_INHERIT_BY_DEPTH = (0.6, 0.35, 0.2)
_INHERIT_DEEP = 0.1
# Query words found only in the parent comment count at this fraction.
PARENT_WEIGHT = 0.5
PARENT_TEXT_CHARS = 500

# Junk / spam heuristics.
MIN_BODY_CHARS = 15  # applies to replies; see _is_junk
MIN_TOP_LEVEL_CHARS = 3
SHORT_BODY_SCORE_EXEMPT = 10  # a terse but well-voted reply is still an answer
SPAM_LATE_DAYS = 30.0
SPAM_LATE_LINK_PENALTY = 0.4
SPAM_LINK_PENALTY = 0.75
NEAR_DUPLICATE_JACCARD = 0.8

# "bot" only as a marked suffix (foo_bot, FooBot), so "Abbot"/"talbot" stay human.
_BOT_AUTHOR_RE = re.compile(r"(?i:^automoderator$)|(?:[-_][Bb][Oo][Tt]|Bot|BOT)$")
_URL_HOST_RE = re.compile(r"https?://([^/\s)\]>\"']+)", re.IGNORECASE)
_REDDIT_HOSTS = ("reddit.com", "redd.it", "redditmedia.com", "redditstatic.com")
# Reference-style sources people cite in good faith; never treated as promo.
_TRUSTED_HOSTS = (
    "nih.gov",
    "doi.org",
    "wikipedia.org",
    "wikimedia.org",
    "pubmed.gov",
    "cochrane.org",
    "cochranelibrary.com",
    "nhs.uk",
    "who.int",
    "arxiv.org",
    "archive.org",
)
_TRUSTED_TLD_RE = re.compile(r"\.(gov|edu|mil)(\.[a-z]{2})?$|\.ac\.[a-z]{2}$")


def _host_matches(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def _has_external_link(text: str) -> bool:
    """True if ``text`` links off-Reddit to something that isn't a reference source."""

    for raw in _URL_HOST_RE.findall(text or ""):
        host = raw.lower().rsplit("@", 1)[-1].split(":", 1)[0].rstrip(".,")
        if _host_matches(host, _REDDIT_HOSTS) or _host_matches(host, _TRUSTED_HOSTS):
            continue
        if _TRUSTED_TLD_RE.search(host):
            continue
        return True
    return False


def _is_junk(c: Comment) -> bool:
    """Comments that are never evidence: bots, mod stickies, and tiny non-answers."""

    if c.author and _BOT_AUTHOR_RE.search(c.author):
        return True
    if c.stickied and c.distinguished:
        return True
    # Short replies are mostly "Thanks!"; a short top-level comment is often a
    # bare product name, which is a real answer in recommendation threads.
    min_chars = MIN_TOP_LEVEL_CHARS if c.depth == 0 else MIN_BODY_CHARS
    if len(c.body.strip()) < min_chars and c.score < SHORT_BODY_SCORE_EXEMPT:
        return True
    return False


def _spam_penalty(c: Comment, thread_created_utc: float | None) -> float:
    """Multiplier in (0, 1] for likely SEO/promo comments.

    An unvoted comment carrying an off-site link is suspicious; one that also
    arrived weeks after the thread went quiet is the classic "necro-post a
    product link for search traffic" pattern. It's a penalty rather than a drop
    because late, helpful, linked answers do exist.
    """

    if c.score > 1 or not _has_external_link(c.body):
        return 1.0
    if (
        c.created_utc
        and thread_created_utc
        and c.created_utc - thread_created_utc > SPAM_LATE_DAYS * 86400
    ):
        return SPAM_LATE_LINK_PENALTY
    return SPAM_LINK_PENALTY


def _inherit_weight(depth: int) -> float:
    if depth < len(_INHERIT_BY_DEPTH):
        return _INHERIT_BY_DEPTH[max(depth, 0)]
    return _INHERIT_DEEP


def _own_overlap(query_tokens: Counter[str], body: str, parent_text: str | None) -> float:
    """Overlap of the comment itself, crediting query words in its parent at reduced weight.

    A reply like "Clicky. Try whites." only makes sense next to the comment it
    answers, so the parent's wording counts toward it (but less than its own).
    """

    own = _overlap(query_tokens, body)
    if not parent_text:
        return own
    with_parent = _overlap(query_tokens, f"{body} {parent_text}")
    return own + PARENT_WEIGHT * (with_parent - own)


def _parent_of(c: Comment, by_id: dict[str, Comment]) -> Comment | None:
    """The parent comment, if it is a usable comment in the fetched set."""

    pid = c.parent_id or ""
    if pid.startswith("t3_"):
        return None
    parent = by_id.get(pid[3:] if pid.startswith("t1_") else pid)
    if parent is None or parent is c or parent.removed or not parent.body:
        return None
    return parent


def rank_evidence(
    question: str,
    threads: list[Thread],
    max_evidence_items: int = 25,
    include_threads: bool = True,
    *,
    max_per_thread: int | None = None,
    recency_half_life_days: float = 365.0,
) -> list[Evidence]:
    """Rank thread bodies and comments into a flat, cited evidence list.

    ``max_per_thread`` caps how many items one thread may contribute so a single
    big thread can't crowd out the others; ``None`` means ~60% of
    ``max_evidence_items`` (at least 3) with 2+ threads, and no cap for one.
    """

    qtokens = Counter(tokenize(question))
    now = time.time()
    items: list[Evidence] = []

    for thread in threads:
        # A comment in an on-topic thread is on-topic by context even when it
        # shares no words with the question (e.g. a bare product name).
        thread_relevance = _overlap(qtokens, f"{thread.title}\n{thread.selftext}")
        if include_threads and thread_relevance > 0:
            combined = (
                W_LEXICAL * thread_relevance
                + W_POPULARITY * _popularity(thread.score)
                + W_RECENCY * _recency(thread.created_utc, now, recency_half_life_days)
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
                    created_utc=thread.created_utc,
                    author=thread.author,
                )
            )

        by_id = {c.id: c for c in thread.comments if c.id}
        thread_max = max((c.score for c in thread.comments if not c.removed), default=0)
        for c in thread.comments:
            if c.removed or not c.body or _is_junk(c):
                continue
            parent = _parent_of(c, by_id)
            parent_text = parent.body if parent else None
            own = _own_overlap(qtokens, c.body, parent_text)
            # Own overlap always counts in full; thread relevance fills part of
            # the remaining gap, so inheriting never lowers a matching comment.
            lexical = own + _inherit_weight(c.depth) * thread_relevance * (1.0 - own)
            if lexical <= 0:
                continue
            depth_penalty = 1.0 / (1.0 + 0.35 * c.depth)
            combined = (
                (
                    W_LEXICAL * lexical
                    + W_POPULARITY * _comment_popularity(c.score, thread_max)
                    + W_RECENCY * _recency(c.created_utc, now, recency_half_life_days)
                )
                * depth_penalty
                * _spam_penalty(c, thread.created_utc)
            )
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
                    parent_text=parent_text[:PARENT_TEXT_CHARS] if parent_text else None,
                    created_utc=c.created_utc,
                    depth=c.depth,
                    author=c.author,
                )
            )

    items.sort(key=lambda x: x.score, reverse=True)
    items = _dedupe(items)
    if max_per_thread is None and len(threads) >= 2:
        max_per_thread = max(3, round(0.6 * max_evidence_items))
    if max_per_thread is not None:
        items = _cap_per_thread(items, max_per_thread)
    return items[:max_evidence_items]


def _cap_per_thread(items: list[Evidence], max_per_thread: int) -> list[Evidence]:
    """Keep at most ``max_per_thread`` items per thread, preserving rank order."""

    counts: Counter[str] = Counter()
    out: list[Evidence] = []
    for ev in items:
        if counts[ev.thread_id] >= max_per_thread:
            continue
        counts[ev.thread_id] += 1
        out.append(ev)
    return out


def _dedupe(items: list[Evidence]) -> list[Evidence]:
    """Drop repeated text (double-posts, copy-paste spam), keeping the best-ranked.

    ``items`` must already be sorted best-first. Texts are compared as token
    sets (case, punctuation and whitespace folded) and count as duplicates at
    Jaccard similarity >= ``NEAR_DUPLICATE_JACCARD``, so lightly reworded copies
    collapse too. Quadratic, which is fine for the few hundred items we rank.
    """

    kept: list[frozenset[str]] = []
    out: list[Evidence] = []
    for ev in items:
        tokens = frozenset(_TOKEN_RE.findall(ev.text.lower()))
        if tokens and any(_jaccard(tokens, other) >= NEAR_DUPLICATE_JACCARD for other in kept):
            continue
        if tokens:
            kept.append(tokens)
        out.append(ev)
    return out


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    # Cheap size bound first: very different lengths can't reach the threshold.
    if min(len(a), len(b)) < NEAR_DUPLICATE_JACCARD * max(len(a), len(b)):
        return 0.0
    return len(a & b) / len(a | b)
