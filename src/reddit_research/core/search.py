"""Search providers and the query planner.

Providers talk to a search *API* (never scrape result pages) and return
normalized :class:`SearchResult` objects constrained to reddit.com. The query
planner turns a natural-language question plus an optional subreddit allowlist
into concrete ``site:reddit.com`` queries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from .errors import SearchError
from .models import SearchResult
from .urls import is_reddit_url, subreddit_from_url

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"


def plan_queries(query: str, subreddits: list[str] | None = None) -> list[str]:
    """Build ``site:``-scoped search queries from a question.

    With no subreddits, a single ``site:reddit.com`` query is produced. With an
    allowlist, one query per subreddit is produced so each is searched directly.
    """

    query = query.strip()
    if not subreddits:
        return [f"site:reddit.com {query}"]
    return [f"site:reddit.com/r/{sub.strip('/ ')} {query}" for sub in subreddits]


def query_key(query: str, limit: int, recency_days: int | None) -> str:
    """Stable cache key for a planned query."""

    return f"{query}::limit={limit}::recency={recency_days or 'any'}"


class SearchProvider(ABC):
    name = "unknown"

    @abstractmethod
    def _raw_search(self, query: str, limit: int, recency_days: int | None) -> list[dict]:
        """Return raw provider hits as ``{title, url, snippet}`` dicts."""

    def search(
        self,
        query: str,
        limit: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        hits = self._raw_search(query, limit, recency_days)
        results: list[SearchResult] = []
        for i, hit in enumerate(hits):
            url = hit.get("url", "")
            if not url or not is_reddit_url(url):
                continue
            results.append(
                SearchResult(
                    title=hit.get("title", "").strip() or url,
                    url=url,
                    subreddit=subreddit_from_url(url),
                    snippet=(hit.get("snippet") or hit.get("description") or "").strip(),
                    source=self.name,
                    # Rank-decay score in (0, 1]; refined later by the ranker.
                    score=round(1.0 / (1 + i), 4),
                )
            )
            if len(results) >= limit:
                break
        return results


class BraveSearchProvider(SearchProvider):
    name = "brave"

    def __init__(self, api_key: str | None, timeout: float = 15.0) -> None:
        if not api_key:
            raise SearchError(
                "Brave Search requires BRAVE_SEARCH_API_KEY to be set."
            )
        self._api_key = api_key
        self._timeout = timeout

    def _raw_search(self, query: str, limit: int, recency_days: int | None) -> list[dict]:
        params: dict[str, str | int] = {
            "q": query,
            "count": min(max(limit, 1), 20),
        }
        if recency_days is not None:
            # Brave supports coarse freshness buckets; map days to the closest.
            params["freshness"] = _brave_freshness(recency_days)
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        try:
            resp = httpx.get(
                BRAVE_ENDPOINT, params=params, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - network path
            raise SearchError(
                f"Brave Search returned {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:  # pragma: no cover - network path
            raise SearchError(f"Brave Search request failed: {exc}") from exc

        data = resp.json()
        web = (data.get("web") or {}).get("results") or []
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
            for r in web
        ]


class TavilySearchProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str | None, timeout: float = 20.0) -> None:
        if not api_key:
            raise SearchError("Tavily requires TAVILY_API_KEY to be set.")
        self._api_key = api_key
        self._timeout = timeout

    def _raw_search(self, query: str, limit: int, recency_days: int | None) -> list[dict]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": min(max(limit, 1), 20),
            "include_domains": ["reddit.com"],
        }
        if recency_days is not None:
            payload["days"] = recency_days
        try:
            resp = httpx.post(TAVILY_ENDPOINT, json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network path
            raise SearchError(f"Tavily request failed: {exc}") from exc
        data = resp.json()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in data.get("results", [])
        ]


def _brave_freshness(recency_days: int) -> str:
    if recency_days <= 1:
        return "pd"
    if recency_days <= 7:
        return "pw"
    if recency_days <= 31:
        return "pm"
    return "py"


def get_provider(name: str, *, brave_api_key: str | None, tavily_api_key: str | None) -> SearchProvider:
    name = (name or "brave").lower()
    if name == "brave":
        return BraveSearchProvider(brave_api_key)
    if name == "tavily":
        return TavilySearchProvider(tavily_api_key)
    raise SearchError(f"Unknown search provider: {name!r}")
