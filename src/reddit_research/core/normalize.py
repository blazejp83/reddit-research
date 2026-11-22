"""Normalize Reddit-API-shaped dicts into our models.

Both the public ``.json`` backend and the archive backends (Arctic Shift /
PullPush) return objects using Reddit's own field names. This module turns those
dicts into :class:`Comment` / :class:`Thread`, so the PRAW backend stays the only
place that deals with PRAW's attribute-access objects.
"""

from __future__ import annotations

from .models import Comment, Thread

REDDIT_BASE = "https://www.reddit.com"
_REMOVED_BODIES = {"[deleted]", "[removed]", ""}


def _abs_permalink(permalink: str | None) -> str | None:
    if not permalink:
        return None
    if permalink.startswith("http"):
        return permalink
    return f"{REDDIT_BASE}{permalink}"


def comment_from_dict(d: dict, depth: int | None = None) -> Comment:
    body = (d.get("body") or "").strip()
    author = d.get("author")
    if author in (None, "[deleted]"):
        author = None
    raw_depth = d.get("depth")
    resolved_depth = depth if depth is not None else (raw_depth if raw_depth is not None else 0)
    return Comment(
        id=d.get("id", ""),
        # Archives occasionally return a numeric parent id; ids are strings to us.
        parent_id=str(d["parent_id"]) if d.get("parent_id") is not None else None,
        body=body,
        author=str(author) if author else None,
        score=int(d.get("score") or 0),
        depth=int(resolved_depth),
        created_utc=d.get("created_utc"),
        permalink=_abs_permalink(d.get("permalink")),
        removed=body in _REMOVED_BODIES,
        stickied=bool(d.get("stickied") or False),
        distinguished=str(d.get("distinguished")) if d.get("distinguished") else None,
    )


def thread_from_dict(
    post: dict,
    comments: list[Comment],
    *,
    truncated: bool,
    fallback_id: str,
    fallback_subreddit: str | None = None,
) -> Thread:
    import time

    permalink = post.get("permalink") or f"/comments/{fallback_id}/"
    author = post.get("author")
    if author in (None, "[deleted]"):
        author = None
    return Thread(
        id=post.get("id", fallback_id),
        subreddit=str(post.get("subreddit") or fallback_subreddit or ""),
        title=post.get("title", "") or "",
        url=post.get("url") or f"{REDDIT_BASE}{permalink}",
        permalink=_abs_permalink(permalink) or f"{REDDIT_BASE}/comments/{fallback_id}/",
        selftext=post.get("selftext", "") or "",
        author=str(author) if author else None,
        score=int(post.get("score") or 0),
        upvote_ratio=post.get("upvote_ratio"),
        created_utc=post.get("created_utc"),
        num_comments=int(post.get("num_comments") or 0),
        over_18=bool(post.get("over_18", False)),
        comments=comments,
        comments_truncated=truncated,
        fetched_utc=time.time(),
    )


def assign_depths_from_parents(comments: list[Comment], link_fullname: str) -> None:
    """Fill in ``depth`` for a flat comment list using ``parent_id`` links.

    Archive backends return comments without reliable depth. Root comments have
    ``parent_id == t3_<link>``; each reply is one deeper than its parent. Orphan
    replies (parent comment not in the set, e.g. dropped by a limit) are known
    to be replies, so they get depth 1 rather than being passed off as roots.
    """

    by_id = {c.id: c for c in comments}
    memo: dict[str, int] = {}

    def depth_of(c: Comment) -> int:
        if c.id in memo:
            return memo[c.id]
        parent = c.parent_id or ""
        if not parent.startswith("t1_"):
            memo[c.id] = 0
            return 0
        parent_id = parent[3:]
        parent_comment = by_id.get(parent_id)
        if parent_comment is None:
            memo[c.id] = 1
            return 1
        memo[c.id] = 0  # guard against cycles in malformed data
        d = depth_of(parent_comment) + 1
        memo[c.id] = d
        return d

    for c in comments:
        c.depth = depth_of(c)
