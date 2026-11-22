"""Archive fetcher: Arctic Shift (primary) with PullPush fallback.

Both are free, keyless community successors to Pushshift and return objects using
Reddit's own field names. This backend needs no Reddit app or approval and works
from any IP (including datacenter), making it a good fallback when the public
``.json`` endpoint 403s or a thread has been deleted (archives retain removed
content). It serves periodically-updated historical data, so very recent threads
may be missing or stale.

Arctic Shift comments are captured seconds after posting and re-fetched roughly a
day later, so scores are only meaningful on the second retrieval. Its flat
``/comments/search`` returns the *newest* comments first, which for a capped
limit means the late, low-vote ones -- the well-voted early comments get cut.
So we pull the whole ``/comments/tree`` (unordered, but complete and nested),
sort it ourselves, and keep the best ``max_comments``.
"""

from __future__ import annotations

import httpx

from .config import RedditConfig
from .errors import RedditError
from .fetch_base import Fetcher
from .models import Comment, Thread
from .normalize import assign_depths_from_parents, comment_from_dict, thread_from_dict
from .urls import parse_ref

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
PULLPUSH_BASE = "https://api.pullpush.io/reddit/search"
# Upper bound on comments pulled from the tree endpoint before we sort and cap.
ARCTIC_TREE_LIMIT = 10000


class ArchiveFetcher(Fetcher):
    name = "arctic"

    def __init__(
        self,
        config: RedditConfig,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._timeout = timeout
        self._client = client

    def _get(self, url: str, params: dict) -> httpx.Response:
        headers = {"User-Agent": self.config.user_agent, "Accept": "application/json"}
        if self._client is not None:
            return self._client.get(url, params=params, headers=headers, timeout=self._timeout)
        return httpx.get(url, params=params, headers=headers, timeout=self._timeout)

    def fetch_thread(
        self,
        url_or_id: str,
        max_comments: int | None = None,
        comment_sort: str | None = None,
        include_more_comments: bool = True,
    ) -> Thread:
        ref = parse_ref(url_or_id)
        if ref is None:
            raise RedditError(f"Could not parse a Reddit submission id from: {url_or_id!r}")
        max_comments = (
            max_comments if max_comments is not None else self.config.max_comments_per_thread
        )

        post: dict | None = None
        comments: list[dict] = []
        errors_seen: list[str] = []

        for source in (self._from_arctic, self._from_pullpush):
            try:
                post, comments = source(ref.submission_id, max_comments)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                errors_seen.append(f"{source.__name__}: {exc}")
                continue
            if post:
                break

        if not post:
            detail = "; ".join(errors_seen) or "no data returned"
            raise RedditError(
                f"Thread {ref.submission_id} not found in archives (Arctic Shift/PullPush): {detail}"
            )

        normalized = [comment_from_dict(c) for c in comments if c.get("id")]
        link_fullname = f"t3_{post.get('id', ref.submission_id)}"
        # Tree comments carry their real depth; flat (PullPush) ones get it from
        # parent links, which must run before the cap drops any parents.
        if not all(c.get("depth") is not None for c in comments):
            assign_depths_from_parents(normalized, link_fullname)
        normalized = sort_comments(normalized, comment_sort or self.config.comment_sort)
        normalized = normalized[: max(max_comments, 0)]
        truncated = int(post.get("num_comments") or 0) > len(normalized)

        return thread_from_dict(
            post,
            normalized,
            truncated=truncated,
            fallback_id=ref.submission_id,
            fallback_subreddit=ref.subreddit,
        )

    # -- sources -----------------------------------------------------------

    def _from_arctic(self, submission_id: str, max_comments: int) -> tuple[dict | None, list[dict]]:
        pr = self._get(f"{ARCTIC_BASE}/posts/ids", {"ids": submission_id})
        pr.raise_for_status()
        posts = pr.json().get("data", [])
        if not posts:
            return None, []
        cr = self._get(
            f"{ARCTIC_BASE}/comments/tree",
            {"link_id": submission_id, "limit": ARCTIC_TREE_LIMIT},
        )
        cr.raise_for_status()
        return posts[0], flatten_tree(cr.json().get("data", []))

    def _from_pullpush(self, submission_id: str, max_comments: int) -> tuple[dict | None, list[dict]]:
        pr = self._get(f"{PULLPUSH_BASE}/submission/", {"ids": submission_id})
        pr.raise_for_status()
        posts = pr.json().get("data", [])
        if not posts:
            return None, []
        cr = self._get(
            f"{PULLPUSH_BASE}/comment/",
            {
                "link_id": submission_id,
                "size": min(max(max_comments, 1), 100),
                "sort_type": "score",
                "sort": "desc",
            },
        )
        cr.raise_for_status()
        comments = cr.json().get("data", [])
        return posts[0], comments


def flatten_tree(items: list[dict], depth: int = 0) -> list[dict]:
    """Flatten Reddit-listing-shaped ``{kind, data}`` trees, stamping ``depth``.

    ``more`` placeholders (only emitted when the limit is hit) are skipped.
    """

    out: list[dict] = []
    for item in items:
        if item.get("kind") != "t1":
            continue
        data = dict(item.get("data") or {})
        replies = data.pop("replies", None)
        data["depth"] = depth
        out.append(data)
        if isinstance(replies, dict):
            children = (replies.get("data") or {}).get("children") or []
            out.extend(flatten_tree(children, depth + 1))
    return out


def sort_comments(comments: list[Comment], sort: str | None) -> list[Comment]:
    """Order a flat comment list like Reddit's sort would (approximately)."""

    sort = (sort or "top").lower()
    if sort == "new":
        return sorted(comments, key=lambda c: c.created_utc or 0, reverse=True)
    if sort == "old":
        return sorted(comments, key=lambda c: c.created_utc or 0)
    if sort == "controversial":
        # No up/down split in archives: a net score near zero is the proxy.
        return sorted(comments, key=lambda c: abs(c.score))
    # top / best / qa: highest score first, shallower first on ties.
    return sorted(comments, key=lambda c: (-c.score, c.depth))
