"""Common interface for thread fetchers.

Backends (``praw``, ``json``, ``arctic``) all implement :meth:`Fetcher.fetch_thread`
with the same signature, so the orchestrator and cache are backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Thread


class Fetcher(ABC):
    #: Short backend identifier, surfaced in warnings/metadata.
    name: str = "fetcher"

    @abstractmethod
    def fetch_thread(
        self,
        url_or_id: str,
        max_comments: int | None = None,
        comment_sort: str | None = None,
        include_more_comments: bool = True,
    ) -> Thread:
        """Fetch and normalize one thread (submission + bounded comment tree)."""
