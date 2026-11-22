"""The pipeline: search -> fetch -> rank -> (optional) synthesize.

The orchestrator owns the cache and the run-level metadata (queries, counts,
cache hits/misses, warnings). It degrades gracefully: a failing thread fetch
becomes a warning rather than aborting the whole run.
"""

from __future__ import annotations

import logging

from .cache import Cache
from .config import Config
from . import errors
from .fetch_base import Fetcher
from .fetchers import get_fetcher
from .models import (
    AnswerResult,
    Evidence,
    EvidenceResult,
    RunMetadata,
    SearchResult,
    Thread,
)
from . import ranking
from .expand import expand_queries
from .llm import LLMError
from .rerank import rerank
from .search import (
    get_provider,
    merge_results,
    normalize_safesearch,
    SCOPED_MIN_THREADS,
    plan_queries,
    query_key,
    subreddit_fallback,
    threads_in_subreddit,
)
from .synth import SynthesisError, synthesize
from .urls import parse_ref

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        config: Config,
        cache: Cache | None = None,
        fetcher: Fetcher | None = None,
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
                brave_rate_limit=self.config.search.brave_rate_limit,
            )
        return self._provider

    @property
    def fetcher(self) -> Fetcher:
        if self._fetcher is None:
            self._fetcher = get_fetcher(self.config.reddit.backend, self.config)
        return self._fetcher

    # -- stages ------------------------------------------------------------

    def search(
        self,
        query: str,
        subreddits: list[str] | None = None,
        limit: int | None = None,
        recency_days: int | None = None,
        meta: RunMetadata | None = None,
        safesearch: str | None = None,
    ) -> list[SearchResult]:
        # Validate before spending any LLM/search calls on a bad value.
        safesearch = normalize_safesearch(safesearch)
        limit = limit or self.config.search.default_limit
        recency_days = recency_days if recency_days is not None else self.config.search.recency_days
        meta = meta or RunMetadata()
        variants = self._query_variants(query, meta)
        queries = plan_queries(query, subreddits, variants)
        meta.search_queries.extend(queries)

        result_lists: list[list[SearchResult]] = []
        for q in queries:
            results = self._run_query(q, limit, recency_days, safesearch, meta)
            fallback = subreddit_fallback(q)
            if fallback and len(threads_in_subreddit(results)) < SCOPED_MIN_THREADS:
                # The path-scoped site: filter finds nothing for some subreddits;
                # retry unscoped and keep only that subreddit's threads.
                fallback_query, sub = fallback
                meta.search_queries.append(fallback_query)
                extra = self._run_query(fallback_query, limit, recency_days, safesearch, meta)
                results = threads_in_subreddit(results) + threads_in_subreddit(extra, sub)
            result_lists.append(results)

        if len(result_lists) > 1:
            # Fuse per-query rankings so threads found by several phrasings rise;
            # the original question's lists (one per subreddit) weigh more.
            collected = merge_results(result_lists, primary_lists=max(1, len(subreddits or [])))
        else:
            collected = result_lists[0] if result_lists else []

        ranked = ranking.rank_threads(query, collected, subreddits)
        if len(ranked) < 3:
            meta.warnings.append(
                errors.warning(errors.FEW_RESULTS, f"Only {len(ranked)} candidate threads found.")
            )
        return ranked[:limit]

    def _run_query(
        self,
        q: str,
        limit: int,
        recency_days: int | None,
        safesearch: str,
        meta: RunMetadata,
    ) -> list[SearchResult]:
        """One provider search, through the cache; failures become warnings."""

        key = query_key(q, limit, recency_days, safesearch)
        cached = self.cache.get_search_results(key, self.config.cache.search_ttl_days)
        if cached is not None:
            meta.cache_hits += 1
            return cached
        meta.cache_misses += 1
        try:
            results = self.provider.search(
                q, limit=limit, recency_days=recency_days, safesearch=safesearch
            )
        except errors.SearchError as exc:
            meta.warnings.append(errors.warning(errors.SEARCH_UNAVAILABLE, str(exc)))
            return []
        self.cache.put_search_results(key, results)
        return results

    def _query_variants(self, query: str, meta: RunMetadata) -> list[str]:
        """LLM-written alternative phrasings of the question (empty when off)."""

        n = self.config.search.query_variants
        model = self.config.rerank.model
        if n <= 0 or not model or not self.config.synth.llm_enabled:
            return []
        try:
            return expand_queries(query, api_key=self.config.synth.api_key, model=model, n=n)
        except LLMError as exc:
            logger.warning("Query expansion failed: %s", exc)
            meta.warnings.append(
                errors.warning(
                    errors.EXPANSION_UNAVAILABLE,
                    f"searched with the original question only: {exc}",
                )
            )
            return []

    def _rank(
        self,
        question: str,
        threads: list[Thread],
        n: int,
        recency_days: int | None,
        meta: RunMetadata,
    ) -> list[Evidence]:
        """Lexical prefilter, then LLM rerank when configured (lexical on failure)."""

        # A recency window signals a time-sensitive question: decay old items faster.
        half_life = float(max(recency_days, 30)) if recency_days else 365.0
        model = self.config.rerank.model
        if not model or not self.config.synth.llm_enabled:
            return ranking.rank_evidence(question, threads, n, recency_half_life_days=half_life)

        pool = ranking.rank_evidence(
            question,
            threads,
            max(self.config.rerank.candidates, n),
            recency_half_life_days=half_life,
        )
        try:
            return rerank(
                question, pool, api_key=self.config.synth.api_key, model=model, top_n=n
            )
        except LLMError as exc:
            logger.warning("LLM rerank failed: %s", exc)
            meta.warnings.append(
                errors.warning(errors.RERANK_UNAVAILABLE, f"using keyword ranking: {exc}")
            )
            return pool[:n]

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
        safesearch: str | None = None,
    ) -> EvidenceResult:
        meta = RunMetadata()
        candidates = self.search(
            question, subreddits, limit_threads, recency_days, meta=meta, safesearch=safesearch
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

        evidence = self._rank(question, threads, max_evidence_items, recency_days, meta)
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
        safesearch: str | None = None,
    ) -> AnswerResult:
        # The synthesis model reads (and cites by number) the same list we return,
        # so its size comes from config when an answer will be written.
        if include_answer and self.config.synth.llm_enabled:
            max_evidence_items = self.config.synth.max_evidence
        ev = self.evidence(
            question,
            subreddits,
            limit_threads,
            max_comments_per_thread,
            comment_sort,
            recency_days,
            max_evidence_items,
            safesearch=safesearch,
        )
        answer_text: str | None = None
        if include_answer:
            try:
                answer_text = synthesize(question, ev.evidence, self.config.synth)
                reason = None if answer_text else "no synthesis provider configured"
            except SynthesisError as exc:
                reason = f"synthesis failed: {exc}"
                logger.warning("LLM synthesis failed: %s", exc)
            if reason:
                ev.metadata.warnings.append(errors.warning(errors.SYNTH_UNAVAILABLE, reason))
        return AnswerResult(
            answer=answer_text,
            evidence=ev.evidence,
            threads_considered=ev.threads_considered,
            metadata=ev.metadata,
        )

    def close(self) -> None:
        self.cache.close()
