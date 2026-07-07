"""Backend selection for thread fetching.

Backends:
- ``praw``   — official Reddit Data API via PRAW (needs an approved app).
- ``json``   — keyless public ``.json`` endpoints (residential IP, low volume).
- ``arctic`` — Arctic Shift / PullPush archives (keyless, works from any IP).
- ``auto``   — ``praw`` when Reddit credentials are present, else ``json``.
"""

from __future__ import annotations

from .archive import ArchiveFetcher
from .config import Config
from .errors import RedditError
from .fetch_base import Fetcher
from .public_json import PublicJsonFetcher
from .reddit import RedditFetcher

BACKENDS = ("auto", "praw", "json", "arctic")


def resolve_backend(name: str, config: Config) -> str:
    name = (name or "auto").lower()
    if name != "auto":
        return name
    reddit = config.reddit
    if reddit.client_id and reddit.client_secret:
        return "praw"
    return "json"


def get_fetcher(name: str, config: Config) -> Fetcher:
    backend = resolve_backend(name, config)
    if backend == "praw":
        return RedditFetcher(config.reddit)
    if backend == "json":
        return PublicJsonFetcher(config.reddit)
    if backend == "arctic":
        return ArchiveFetcher(config.reddit)
    raise RedditError(
        f"Unknown fetch backend: {name!r}. Choose one of: {', '.join(BACKENDS)}."
    )
