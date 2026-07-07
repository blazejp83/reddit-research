"""The pipeline: search -> fetch -> rank -> (optional) synthesize.

The orchestrator owns the cache and the run-level metadata (queries, counts,
cache hits/misses, warnings). It degrades gracefully: a failing thread fetch
becomes a warning rather than aborting the whole run.
"""

from __future__ import annotations

from .cache import Cache
from .config import Config
from . import errors
from .models import (
    AnswerResult,
    EvidenceResult,
    RunMetadata,
    SearchResult,
    Thread,
)
from . import ranking
from .reddit import RedditFetcher
from .search import get_provider, plan_queries, query_key
from .synth import synthesize
from .urls import parse_ref


class Orchestrator:
    def __init__(
        self,
        config: Config,
        cache: Cache | None = None,
        fetcher: RedditFetcher | None = None,
        provider=None,
    ) -> None:
        self.config = config
        self.cache = cache or Cache(config.cache.path)
        self._fetcher = fetcher
        self._provider = provider

    # -- lazily-built collaborators ---------------------------------------

    @property
    def provider(self):
        if self._provider is None:
            self._provider = get_provider(
                self.config.search.provider,
                brave_api_key=self.config.brave_api_key,
                tavily_api_key=self.config.tavily_api_key,
            )
        return self._provider

    @property
    def fetcher(self) -> RedditFetcher:
        if self._fetcher is None:
            self._fetcher = RedditFetcher(self.config.reddit)
        return self._fetcher

    # -- stages ------------------------------------------------------------

    def search(
        self,
        query: str,
        subreddits: list[str] | None = None,
        limit: int | None = None,
        recency_days: int | None = None,
        meta: RunMetadata | None = None,
    ) -> list[SearchResult]:
        limit = limit or self.config.search.default_limit
        recency_days = recency_days if recency_days is not None else self.config.search.recency_days
        meta = meta or RunMetadata()
        queries = plan_queries(query, subreddits)
        meta.search_queries.extend(queries)

        collected: list[SearchResult] = []
        for q in queries:
            key = query_key(q, limit, recency_days)
            cached = self.cache.get_search_results(key, self.config.cache.search_ttl_days)
            if cached is not None:
                meta.cache_hits += 1
                collected.extend(cached)
                continue
            meta.cache_misses += 1
            try:
                results = self.provider.search(q, limit=limit, recency_days=recency_days)
            except errors.SearchError as exc:
                meta.warnings.append(errors.warning(errors.SEARCH_UNAVAILABLE, str(exc)))
                continue
            self.cache.put_search_results(key, results)
            collected.extend(results)

        ranked = ranking.rank_threads(query, collected, subreddits)
        if len(ranked) < 3:
            meta.warnings.append(
                errors.warning(errors.FEW_RESULTS, f"Only {len(ranked)} candidate threads found.")
            )
        return ranked[:limit]

    def fetch(
        self,
        url_or_id: str,
        max_comments: int | None = None,
        comment_sort: str | None = None,
        include_more_comments: bool = True,
        meta: RunMetadata | None = None,
        use_cache: bool = True,
    ) -> Thread:
        meta = meta or RunMetadata()
        ref = parse_ref(url_or_id)
        if use_cache and ref is not None:
            cached = self.cache.get_thread(ref.submission_id, self.config.cache.thread_ttl_days)
            if cached is not None:
                meta.cache_hits += 1
                meta.threads_fetched += 1
                meta.comments_fetched += len(cached.comments)
                return cached
            meta.cache_misses += 1

        thread = self.fetcher.fetch_thread(
            url_or_id,
            max_comments=max_comments,
            comment_sort=comment_sort,
            include_more_comments=include_more_comments,
        )
        self.cache.put_thread(thread)
        meta.threads_fetched += 1
        meta.comments_fetched += len(thread.comments)
        if thread.comments_truncated:
            meta.warnings.append(
                errors.warning(
                    errors.COMMENTS_TRUNCATED,
                    f"Comments for {thread.id} partially loaded (limit reached).",
                )
            )
        return thread

    def evidence(
        self,
        question: str,
        subreddits: list[str] | None = None,
        limit_threads: int | None = None,
        max_comments_per_thread: int | None = None,
        comment_sort: str | None = None,
        recency_days: int | None = None,
        max_evidence_items: int = 25,
    ) -> EvidenceResult:
        meta = RunMetadata()
        candidates = self.search(
            question, subreddits, limit_threads, recency_days, meta=meta
        )

        threads: list[Thread] = []
        for cand in candidates:
            try:
                threads.append(
                    self.fetch(
                        cand.url,
                        max_comments=max_comments_per_thread,
                        comment_sort=comment_sort,
                        meta=meta,
                    )
                )
            except errors.RedditError as exc:
                meta.warnings.append(errors.warning(errors.THREAD_UNAVAILABLE, str(exc)))

        evidence = ranking.rank_evidence(question, threads, max_evidence_items)
        if not evidence:
            meta.warnings.append(
                errors.warning(errors.FEW_RESULTS, "No relevant evidence extracted.")
            )
        return EvidenceResult(
            evidence=evidence, threads_considered=candidates, metadata=meta
        )

    def answer(
        self,
        question: str,
        subreddits: list[str] | None = None,
        limit_threads: int | None = None,
        max_comments_per_thread: int | None = None,
        comment_sort: str | None = None,
        recency_days: int | None = None,
        max_evidence_items: int = 25,
        include_answer: bool = True,
    ) -> AnswerResult:
        ev = self.evidence(
            question,
            subreddits,
            limit_threads,
            max_comments_per_thread,
            comment_sort,
            recency_days,
            max_evidence_items,
        )
        answer_text: str | None = None
        if include_answer:
            answer_text = synthesize(question, ev.evidence, self.config.synth)
            if answer_text is None:
                ev.metadata.warnings.append(
                    errors.warning(
                        errors.SYNTH_UNAVAILABLE,
                        "Synthesis skipped (no provider configured or synthesis failed).",
                    )
                )
        return AnswerResult(
            answer=answer_text,
            evidence=ev.evidence,
            threads_considered=ev.threads_considered,
            metadata=ev.metadata,
        )

    def close(self) -> None:
        self.cache.close()
