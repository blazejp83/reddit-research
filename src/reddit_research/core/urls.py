"""Reddit URL and ID parsing.

Handles the canonical URL shapes the search providers return:

    https://www.reddit.com/r/<sub>/comments/<id>/<slug>/
    https://reddit.com/r/<sub>/comments/<id>
    https://old.reddit.com/r/<sub>/comments/<id>/<slug>/<comment_id>/
    https://redd.it/<id>
    /r/<sub>/comments/<id>/...
    a bare base-36 submission id ("abc123" or "t3_abc123")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

_COMMENTS_RE = re.compile(
    r"/r/(?P<sub>[^/]+)/comments/(?P<id>[a-z0-9]+)"
    r"(?:/[^/]*(?:/(?P<comment_id>[a-z0-9]+))?)?",
    re.IGNORECASE,
)
_SHORT_RE = re.compile(r"^(?:https?://)?redd\.it/(?P<id>[a-z0-9]+)", re.IGNORECASE)
_BARE_ID_RE = re.compile(r"^(?:t3_)?(?P<id>[a-z0-9]{4,10})$", re.IGNORECASE)
_SUBREDDIT_RE = re.compile(r"/r/(?P<sub>[^/]+)", re.IGNORECASE)


@dataclass(frozen=True)
class RedditRef:
    """A parsed reference to a Reddit submission (and optionally a comment)."""

    submission_id: str
    subreddit: str | None = None
    comment_id: str | None = None


def is_reddit_url(value: str) -> bool:
    try:
        host = urlparse(value).hostname or ""
    except ValueError:
        return False
    host = host.lower()
    return host.endswith("reddit.com") or host == "redd.it"


def subreddit_from_url(value: str) -> str | None:
    m = _SUBREDDIT_RE.search(value)
    return m.group("sub").lower() if m else None


def parse_ref(value: str) -> RedditRef | None:
    """Parse a URL, short link, or bare id into a :class:`RedditRef`.

    Returns ``None`` when the input carries no recognizable submission id.
    """

    if not value:
        return None
    value = value.strip()

    m = _COMMENTS_RE.search(value)
    if m:
        return RedditRef(
            submission_id=m.group("id").lower(),
            subreddit=(m.group("sub") or "").lower() or None,
            comment_id=(m.group("comment_id") or "").lower() or None,
        )

    m = _SHORT_RE.match(value)
    if m:
        return RedditRef(submission_id=m.group("id").lower())

    # Only treat as a bare id when it is not URL-shaped.
    if "://" not in value and "/" not in value:
        m = _BARE_ID_RE.match(value)
        if m:
            return RedditRef(submission_id=m.group("id").lower())

    return None
