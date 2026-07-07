import httpx
import pytest
import respx

from reddit_research.core.search import (
    BraveSearchProvider,
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
