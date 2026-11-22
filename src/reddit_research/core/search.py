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
from .urls import is_reddit_url, parse_ref, subreddit_from_url

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"


# How many subreddits of an allowlist each query variant is searched in.
VARIANT_SUBREDDITS = 2
# RRF damping constant (the conventional value); larger flattens rank differences.
RRF_K = 60
# The user's own wording is the best statement of intent, so its results count more.
ORIGINAL_QUERY_WEIGHT = 1.5


def plan_queries(
    query: str,
    subreddits: list[str] | None = None,
    variants: list[str] | None = None,
) -> list[str]:
    """Build ``site:``-scoped search queries from a question.

    With no subreddits, a single ``site:reddit.com`` query is produced. With an
    allowlist, one query per subreddit is produced so each is searched directly.

    ``variants`` are alternative phrasings of the question (see
    :mod:`.expand`). The original query's searches always come first, followed
    by each variant's. Without subreddits every variant gets one
    ``site:reddit.com`` query. With subreddits, the original is still searched
    in every subreddit but each variant only in the first
    ``VARIANT_SUBREDDITS`` (2) of the allowlist, so search API usage stays
    bounded at ``len(subreddits) + 2 * len(variants)`` instead of multiplying.
    The list is de-duplicated preserving order.
    """

    def scoped(text: str, subs: list[str] | None) -> list[str]:
        if not subs:
            return [f"site:reddit.com {text}"]
        return [f"site:reddit.com/r/{sub.strip('/ ')} {text}" for sub in subs]

    queries = scoped(query.strip(), subreddits)
    variant_subs = subreddits[:VARIANT_SUBREDDITS] if subreddits else None
    for variant in variants or []:
        variant = variant.strip()
        if variant:
            queries.extend(scoped(variant, variant_subs))
    return list(dict.fromkeys(queries))


def merge_results(
    result_lists: list[list[SearchResult]],
    *,
    primary_lists: int = 1,
) -> list[SearchResult]:
    """Fuse per-query result lists into one ranking (reciprocal rank fusion).

    Each list contributes ``weight / (RRF_K + rank)`` to every thread it
    contains, so a thread found by several queries outranks one found by a
    single query at the same rank. The first ``primary_lists`` lists are the
    original question's results and are weighted ``ORIGINAL_QUERY_WEIGHT``;
    pass a larger value when the original was fanned out over subreddits.

    Threads are identified by submission id so different URL spellings (www/old
    hosts, slugs, comment permalinks, redd.it) merge; unparseable URLs fall back
    to the URL itself. The first-seen title/snippet/url is kept and ``score``
    becomes the fused score normalized into (0, 1] with the best at 1.0.
    """

    fused: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}
    for index, results in enumerate(result_lists):
        weight = ORIGINAL_QUERY_WEIGHT if index < primary_lists else 1.0
        counted: set[str] = set()
        for rank, result in enumerate(results, start=1):
            ref = parse_ref(result.url)
            key = ref.submission_id if ref else result.url
            if key in counted:
                # Same thread twice in one list: only its best rank counts.
                continue
            counted.add(key)
            fused[key] = fused.get(key, 0.0) + weight / (RRF_K + rank)
            first_seen.setdefault(key, result)
    if not fused:
        return []
    best = max(fused.values())
    # sorted() is stable, so ties keep first-seen order.
    ordered = sorted(fused, key=lambda k: fused[k], reverse=True)
    return [
        first_seen[key].model_copy(update={"score": round(fused[key] / best, 4)})
        for key in ordered
    ]


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
