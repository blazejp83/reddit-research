"""Reddit fetcher built on PRAW.

Authenticates read-only against the official Reddit Data API, fetches a
submission plus a bounded slice of its comment tree, and normalizes everything
into the :mod:`.models` shapes. Deleted/removed comments are kept (marked
``removed``) so citations and thread structure stay intact. ``MoreComments``
expansion is bounded by ``max_comments`` to respect rate limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import RedditConfig
from .errors import RedditError
from .models import Comment, Thread
from .urls import RedditRef, parse_ref

if TYPE_CHECKING:  # pragma: no cover
    import praw

REDDIT_BASE = "https://www.reddit.com"
_REMOVED_BODIES = {"[deleted]", "[removed]", ""}


class RedditFetcher:
    """Wraps a PRAW ``Reddit`` client and normalizes fetched threads."""

    def __init__(self, config: RedditConfig, client: "praw.Reddit | None" = None) -> None:
        self.config = config
        self._client = client

    @property
    def client(self) -> "praw.Reddit":
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> "praw.Reddit":
        try:
            import praw
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RedditError(
                "praw is required for Reddit access. Install with `uv pip install praw`."
            ) from exc

        cfg = self.config
        if not cfg.client_id or not cfg.client_secret:
            raise RedditError(
                "Reddit access requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
            )
        kwargs: dict[str, str] = {
            "client_id": cfg.client_id,
            "client_secret": cfg.client_secret,
            "user_agent": cfg.user_agent,
        }
        if cfg.refresh_token:
            kwargs["refresh_token"] = cfg.refresh_token
        elif cfg.username and cfg.password:
            kwargs["username"] = cfg.username
            kwargs["password"] = cfg.password
        try:
            reddit = praw.Reddit(**kwargs, check_for_updates=False)
            reddit.read_only = True
            return reddit
        except Exception as exc:  # pragma: no cover - network/auth path
            raise RedditError(f"Reddit OAuth failed: {exc}") from exc

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

        max_comments = max_comments if max_comments is not None else self.config.max_comments_per_thread
        comment_sort = comment_sort or self.config.comment_sort

        try:
            submission = self.client.submission(id=ref.submission_id)
            submission.comment_sort = comment_sort
            # Access an attribute to force the lazy fetch and surface 404/403 here.
            _ = submission.title
        except Exception as exc:  # pragma: no cover - network path
            raise RedditError(
                f"Thread {ref.submission_id} unavailable (private, removed, or not found): {exc}"
            ) from exc

        comments, truncated = self._collect_comments(
            submission, max_comments, include_more_comments
        )
        return self._normalize_thread(submission, ref, comments, truncated)

    # -- internals ---------------------------------------------------------

    def _collect_comments(
        self, submission, max_comments: int, include_more: bool
    ) -> tuple[list[Comment], bool]:
        truncated = False
        try:
            if include_more:
                # limit=0 replaces all MoreComments; None keeps them for later.
                submission.comments.replace_more(limit=0)
            else:
                remaining = submission.comments.replace_more(limit=0)
                truncated = bool(remaining)
        except Exception:  # pragma: no cover - network path
            truncated = True

        out: list[Comment] = []
        try:
            flat = submission.comments.list()
        except Exception:  # pragma: no cover - network path
            return out, True

        for praw_comment in flat:
            if len(out) >= max_comments:
                truncated = True
                break
            # Skip leftover MoreComments placeholders.
            if praw_comment.__class__.__name__ == "MoreComments":
                truncated = True
                continue
            out.append(self._normalize_comment(praw_comment))
        return out, truncated

    @staticmethod
    def _normalize_comment(c) -> Comment:
        body = (getattr(c, "body", "") or "").strip()
        removed = body in _REMOVED_BODIES
        author = getattr(c, "author", None)
        parent = getattr(c, "parent_id", None)
        permalink = getattr(c, "permalink", None)
        return Comment(
            id=getattr(c, "id", ""),
            parent_id=parent,
            body=body,
            author=str(author) if author else None,
            score=int(getattr(c, "score", 0) or 0),
            depth=int(getattr(c, "depth", 0) or 0),
            created_utc=getattr(c, "created_utc", None),
            permalink=f"{REDDIT_BASE}{permalink}" if permalink else None,
            removed=removed,
        )

    @staticmethod
    def _normalize_thread(
        submission, ref: RedditRef, comments: list[Comment], truncated: bool
    ) -> Thread:
        import time

        subreddit = ref.subreddit or str(getattr(submission, "subreddit", "") or "")
        permalink = getattr(submission, "permalink", "") or f"/comments/{ref.submission_id}/"
        author = getattr(submission, "author", None)
        return Thread(
            id=getattr(submission, "id", ref.submission_id),
            subreddit=subreddit,
            title=getattr(submission, "title", "") or "",
            url=getattr(submission, "url", "") or f"{REDDIT_BASE}{permalink}",
            permalink=f"{REDDIT_BASE}{permalink}",
            selftext=getattr(submission, "selftext", "") or "",
            author=str(author) if author else None,
            score=int(getattr(submission, "score", 0) or 0),
            upvote_ratio=getattr(submission, "upvote_ratio", None),
            created_utc=getattr(submission, "created_utc", None),
            num_comments=int(getattr(submission, "num_comments", 0) or 0),
            over_18=bool(getattr(submission, "over_18", False)),
            comments=comments,
            comments_truncated=truncated,
            fetched_utc=time.time(),
        )
