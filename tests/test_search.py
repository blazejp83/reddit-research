import httpx
import pytest
import respx

from reddit_research.core.models import SearchResult
from reddit_research.core.search import (
    BraveSearchProvider,
    merge_results,
    plan_queries,
    query_key,
)


def test_plan_queries_no_subreddits():
    assert plan_queries("borg vs restic") == ["site:reddit.com borg vs restic"]


def test_plan_queries_with_subreddits():
    qs = plan_queries("backups", ["selfhosted", "datahoarder"])
    assert qs == [
        "site:reddit.com/r/selfhosted backups",
        "site:reddit.com/r/datahoarder backups",
    ]


def test_query_key_is_stable():
    a = query_key("site:reddit.com x", 10, 365)
    b = query_key("site:reddit.com x", 10, 365)
    assert a == b
    assert query_key("site:reddit.com x", 5, None) != a


@respx.mock
def test_brave_provider_normalizes_and_filters():
    payload = {
        "web": {
            "results": [
                {
                    "title": "Best backups",
                    "url": "https://www.reddit.com/r/selfhosted/comments/abc/best",
                    "description": "use borg",
                },
                {  # non-reddit result should be filtered out
                    "title": "Blog post",
                    "url": "https://example.com/blog",
                    "description": "nope",
                },
            ]
        }
    }
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(200, json=payload)
    )
    provider = BraveSearchProvider(api_key="secret-key")
    results = provider.search("backups", limit=10)
    assert len(results) == 1
    r = results[0]
    assert r.subreddit == "selfhosted"
    assert r.source == "brave"
    assert 0 < r.score <= 1


def test_brave_requires_api_key():
    from reddit_research.core.errors import SearchError

    with pytest.raises(SearchError):
        BraveSearchProvider(api_key=None)


def test_plan_queries_variants_none_or_empty_is_unchanged():
    assert plan_queries("borg vs restic", None, None) == ["site:reddit.com borg vs restic"]
    assert plan_queries("borg vs restic", variants=[]) == ["site:reddit.com borg vs restic"]
    assert plan_queries("backups", ["selfhosted", "datahoarder"], []) == [
        "site:reddit.com/r/selfhosted backups",
        "site:reddit.com/r/datahoarder backups",
    ]


def test_plan_queries_variants_without_subreddits():
    qs = plan_queries(
        "what backup tool should I use",
        variants=["best backup software", " borg vs restic ", "best backup software", ""],
    )
    assert qs == [
        "site:reddit.com what backup tool should I use",
        "site:reddit.com best backup software",
        "site:reddit.com borg vs restic",
    ]


def test_plan_queries_variant_equal_to_original_is_deduped():
    assert plan_queries("backups", variants=["backups"]) == ["site:reddit.com backups"]


def test_plan_queries_variants_only_hit_first_two_subreddits():
    qs = plan_queries("backups", ["selfhosted", "datahoarder", "homelab"], ["borg vs restic"])
    assert qs == [
        "site:reddit.com/r/selfhosted backups",
        "site:reddit.com/r/datahoarder backups",
        "site:reddit.com/r/homelab backups",
        "site:reddit.com/r/selfhosted borg vs restic",
        "site:reddit.com/r/datahoarder borg vs restic",
    ]


def _sr(url: str, title: str = "t", snippet: str = "") -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet, source="brave", score=0.5)


def test_merge_results_merges_url_spellings_of_one_thread():
    merged = merge_results(
        [
            [_sr("https://www.reddit.com/r/selfhosted/comments/abc123/best_backups/", "first")],
            [_sr("https://old.reddit.com/r/selfhosted/comments/abc123/best_backups/xyz789/", "second")],
            [_sr("https://redd.it/ABC123", "third")],
        ]
    )
    assert len(merged) == 1
    assert merged[0].title == "first"
    assert merged[0].url == "https://www.reddit.com/r/selfhosted/comments/abc123/best_backups/"
    assert merged[0].score == 1.0


def test_merge_results_multi_query_hit_outranks_single_query_hit():
    a = "https://www.reddit.com/r/x/comments/aaa111/a/"
    b = "https://www.reddit.com/r/x/comments/bbb222/b/"
    c = "https://www.reddit.com/r/x/comments/ccc333/c/"
    filler = "https://www.reddit.com/r/x/comments/fff000/f/"
    # b and c both sit at rank 2 of a variant list; b is also found by another.
    merged = merge_results([[_sr(a)], [_sr(filler), _sr(b)], [_sr(filler), _sr(c)], [_sr(b)]])
    urls = [r.url for r in merged]
    assert urls.index(b) < urls.index(c)


def test_merge_results_original_list_weighs_more():
    a = "https://www.reddit.com/r/x/comments/aaa111/a/"
    b = "https://www.reddit.com/r/x/comments/bbb222/b/"
    merged = merge_results([[_sr(a)], [_sr(b)]])
    assert [r.url for r in merged] == [a, b]
    assert merged[0].score == 1.0
    assert merged[1].score == pytest.approx(1 / 1.5, abs=1e-3)
    # With both lists marked primary they tie and keep first-seen order.
    tied = merge_results([[_sr(a)], [_sr(b)]], primary_lists=2)
    assert [r.score for r in tied] == [1.0, 1.0]


def test_merge_results_scores_normalized_and_sorted():
    urls = [f"https://www.reddit.com/r/x/comments/id{i}00/s/" for i in range(5)]
    merged = merge_results(
        [[_sr(u) for u in urls[:3]], [_sr(u) for u in reversed(urls)], [_sr("https://www.reddit.com/user/someone")]]
    )
    scores = [r.score for r in merged]
    assert len(merged) == 6  # unparseable URL kept, keyed by URL
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0
    assert all(0 < s <= 1 for s in scores)


def test_merge_results_same_thread_twice_in_one_list_counts_once():
    a1 = "https://www.reddit.com/r/x/comments/aaa111/a/"
    a2 = "https://www.reddit.com/r/x/comments/aaa111/a/cmt001/"
    b = "https://www.reddit.com/r/x/comments/bbb222/b/"
    merged = merge_results([[_sr(b), _sr(a1), _sr(a2)]])
    assert [r.url for r in merged] == [b, a1]


def test_merge_results_empty_and_does_not_mutate_inputs():
    assert merge_results([]) == []
    assert merge_results([[], []]) == []
    original = _sr("https://www.reddit.com/r/x/comments/aaa111/a/")
    merge_results([[original]])
    assert original.score == 0.5


def test_rate_limiter_spaces_calls(monkeypatch):
    from reddit_research.core import search

    clock = {"now": 100.0}
    sleeps: list[float] = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(search.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(search.time, "sleep", fake_sleep)

    limiter = search.RateLimiter(2.0)  # 0.5s apart, plus the safety margin
    gap = pytest.approx((1 + search.RATE_LIMIT_MARGIN) / 2.0)
    for _ in range(3):
        limiter.acquire()
    assert sleeps == [gap, gap]

    clock["now"] += 5  # idle long enough: next call goes straight through
    limiter.acquire()
    assert sleeps == [gap, gap]


def test_rate_limiter_zero_disables(monkeypatch):
    from reddit_research.core import search

    monkeypatch.setattr(search.time, "sleep", lambda s: pytest.fail("should not sleep"))
    limiter = search.RateLimiter(0)
    for _ in range(5):
        limiter.acquire()


@respx.mock
def test_brave_sends_per_query_safesearch_and_uses_shared_limiter(monkeypatch):
    from reddit_research.core import search

    calls = []
    monkeypatch.setattr(search._BRAVE_LIMITER, "acquire", lambda: calls.append(1))
    route = respx.get(search.BRAVE_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"web": {"results": []}})
    )
    provider = BraveSearchProvider(api_key="k")
    provider.search("x", safesearch="OFF")
    provider.search("y")
    BraveSearchProvider(api_key="k").search("z", safesearch="strict")
    params = [c.request.url.params["safesearch"] for c in route.calls]
    assert params == ["off", "moderate", "strict"]
    assert len(calls) == 3


def test_brave_rejects_bad_safesearch():
    from reddit_research.core.errors import SearchError

    with pytest.raises(SearchError):
        BraveSearchProvider(api_key="k").search("x", safesearch="maybe")


def test_query_key_safesearch():
    base = query_key("q", 10, None)
    assert query_key("q", 10, None, "moderate") == base  # old cache entries still hit
    assert query_key("q", 10, None, "off") != base
    assert query_key("q", 10, None, "off") != query_key("q", 10, None, "strict")


@respx.mock
def test_brave_retries_after_429(monkeypatch):
    from reddit_research.core import search

    backoffs = []
    monkeypatch.setattr(search._BRAVE_LIMITER, "acquire", lambda: None)
    monkeypatch.setattr(search._BRAVE_LIMITER, "back_off", backoffs.append)
    route = respx.get(search.BRAVE_ENDPOINT).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "2"}, json={}),
            httpx.Response(429, json={}),
            httpx.Response(200, json={"web": {"results": [
                {"title": "t", "url": "https://www.reddit.com/r/a/comments/abc/x/", "description": ""}
            ]}}),
        ]
    )
    results = BraveSearchProvider(api_key="k").search("x")
    assert len(route.calls) == 3
    assert backoffs == [2.0, 2.0]  # Retry-After honoured, then exponential fallback
    assert len(results) == 1


@respx.mock
def test_brave_gives_up_after_max_retries(monkeypatch):
    from reddit_research.core import search
    from reddit_research.core.errors import SearchError

    monkeypatch.setattr(search._BRAVE_LIMITER, "acquire", lambda: None)
    monkeypatch.setattr(search._BRAVE_LIMITER, "back_off", lambda s: None)
    route = respx.get(search.BRAVE_ENDPOINT).mock(return_value=httpx.Response(429, json={}))
    with pytest.raises(SearchError):
        BraveSearchProvider(api_key="k").search("x")
    assert len(route.calls) == search.BRAVE_MAX_RETRIES + 1


def test_subreddit_fallback_rewrites_scoped_queries():
    from reddit_research.core.search import subreddit_fallback

    assert subreddit_fallback("site:reddit.com/r/Swingers best apps") == (
        "site:reddit.com best apps r/Swingers",
        "swingers",
    )
    assert subreddit_fallback("site:reddit.com best apps") is None
    assert subreddit_fallback("best apps") is None


def test_threads_in_subreddit_drops_homepages_and_other_subs():
    from reddit_research.core.search import threads_in_subreddit

    results = [
        SearchResult(title="home", url="https://reddit.com/r/swingers"),
        SearchResult(title="a", url="https://www.reddit.com/r/Swingers/comments/aaa/x/"),
        SearchResult(title="b", url="https://www.reddit.com/r/swingersr4r/comments/bbb/y/"),
    ]
    assert [r.title for r in threads_in_subreddit(results)] == ["a", "b"]
    assert [r.title for r in threads_in_subreddit(results, "swingers")] == ["a"]
