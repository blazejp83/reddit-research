"""Keyless fetcher using Reddit's public ``.json`` endpoints.

No app registration or OAuth required: it requests
``https://www.reddit.com/comments/<id>.json`` with a descriptive User-Agent and
parses the two-listing response (submission + comment tree). Works from
residential IPs at low volume; Reddit 403s most datacenter/cloud IPs and
rate-limits to ~30 req/min unauthenticated, so requests are lightly paced.

Limitation: without OAuth the ``morechildren`` expansion is unavailable, so deep
"load more comments" branches are dropped and the thread is marked truncated.
Top-of-tree comments (which carry most evidence signal) are preserved.
"""

from __future__ import annotations

import time

import httpx

from .config import RedditConfig
from .errors import RedditError, RATE_LIMITED
from .fetch_base import Fetcher
from .models import Comment, Thread
from .normalize import comment_from_dict, thread_from_dict
from .urls import parse_ref

REDDIT_BASE = "https://www.reddit.com"
# Map our sort names to reddit's query values ("best" is "confidence").
_SORT_MAP = {
    "best": "confidence",
    "top": "top",
    "new": "new",
    "controversial": "controversial",
    "old": "old",
    "qa": "qa",
}


class PublicJsonFetcher(Fetcher):
    name = "json"

    def __init__(
        self,
        config: RedditConfig,
        timeout: float = 20.0,
        min_interval: float = 1.2,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._timeout = timeout
        # Minimum seconds between requests to stay under the unauth rate limit.
        self._min_interval = min_interval
        self._last_request = 0.0
        self._client = client

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _get(self, url: str, params: dict) -> httpx.Response:
        headers = {"User-Agent": self.config.user_agent, "Accept": "application/json"}
        self._throttle()
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
        sort = _SORT_MAP.get((comment_sort or self.config.comment_sort).lower(), "top")

        url = f"{REDDIT_BASE}/comments/{ref.submission_id}.json"
        params = {"sort": sort, "raw_json": 1, "limit": min(max(max_comments, 1), 500)}
        try:
            resp = self._get(url, params)
        except httpx.HTTPError as exc:
            raise RedditError(f"Public JSON request failed: {exc}") from exc

        if resp.status_code == 403:
            raise RedditError(
                "Reddit returned 403 for the public JSON endpoint. This IP is likely a "
                "datacenter/cloud range that Reddit blocks; run from a residential network "
                "or use --backend arctic."
            )
        if resp.status_code == 429:
            raise RedditError(RATE_LIMITED + ": Reddit rate-limited the public JSON endpoint.")
        try:
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RedditError(
                f"Thread {ref.submission_id} unavailable via public JSON: {exc}"
            ) from exc

        if not isinstance(payload, list) or len(payload) < 2:
            raise RedditError(f"Unexpected public JSON shape for {ref.submission_id}.")

        post_children = payload[0].get("data", {}).get("children", [])
        if not post_children:
            raise RedditError(
                f"Thread {ref.submission_id} not found, private, or removed."
            )
        post = post_children[0].get("data", {})

        comment_children = payload[1].get("data", {}).get("children", [])
        comments, truncated = self._flatten(comment_children, max_comments)

        return thread_from_dict(
            post,
            comments,
            truncated=truncated,
            fallback_id=ref.submission_id,
            fallback_subreddit=ref.subreddit,
        )

    def _flatten(self, children: list[dict], max_comments: int) -> tuple[list[Comment], bool]:
        """Depth-first flatten of the nested comment listing, honoring the cap."""

        out: list[Comment] = []
        truncated = False

        def walk(nodes: list[dict], depth: int) -> None:
            nonlocal truncated
            for node in nodes:
                kind = node.get("kind")
                data = node.get("data", {})
                if kind == "more":
                    # Unexpandable without OAuth; note and skip.
                    truncated = True
                    continue
                if kind != "t1":
                    continue
                if len(out) >= max_comments:
                    truncated = True
                    return
                out.append(comment_from_dict(data, depth=depth))
                replies = data.get("replies")
                if isinstance(replies, dict):
                    walk(replies.get("data", {}).get("children", []), depth + 1)

        walk(children, 0)
        return out, truncated
