import httpx
import pytest
import respx

from reddit_research.core.config import RedditConfig
from reddit_research.core.errors import RedditError
from reddit_research.core.public_json import PublicJsonFetcher


def _listing(post, comment_nodes):
    return [
        {"kind": "Listing", "data": {"children": [{"kind": "t3", "data": post}]}},
        {"kind": "Listing", "data": {"children": comment_nodes}},
    ]


def _t1(cid, body, score, replies=None, parent="t3_abc123"):
    data = {
        "id": cid,
        "parent_id": parent,
        "body": body,
        "author": "u",
        "score": score,
        "created_utc": 1710000000,
        "permalink": f"/r/selfhosted/comments/abc123/x/{cid}/",
    }
    if replies:
        data["replies"] = {"data": {"children": replies}}
    return {"kind": "t1", "data": data}


def _fetcher():
    # min_interval=0 disables sleeping in tests.
    return PublicJsonFetcher(RedditConfig(), min_interval=0.0)


@respx.mock
def test_fetch_thread_parses_nested_comments():
    post = {
        "id": "abc123",
        "subreddit": "selfhosted",
        "title": "Backups",
        "selftext": "what do you use?",
        "score": 120,
        "num_comments": 3,
        "permalink": "/r/selfhosted/comments/abc123/backups/",
        "url": "https://www.reddit.com/r/selfhosted/comments/abc123/backups/",
    }
    child = _t1("c2", "reply about restic", 5, parent="t1_c1")
    nodes = [_t1("c1", "borg is great", 40, replies=[child]), {"kind": "more", "data": {}}]
    route = respx.get("https://www.reddit.com/comments/abc123.json").mock(
        return_value=httpx.Response(200, json=_listing(post, nodes))
    )
    thread = _fetcher().fetch_thread("https://www.reddit.com/r/selfhosted/comments/abc123/backups/")
    assert route.called
    assert thread.id == "abc123"
    assert len(thread.comments) == 2
    # Nested reply gets depth 1.
    depths = {c.id: c.depth for c in thread.comments}
    assert depths["c1"] == 0 and depths["c2"] == 1
    assert thread.comments[0].permalink.startswith("https://www.reddit.com")
    # A "more" node marks the tree truncated.
    assert thread.comments_truncated is True


@respx.mock
def test_max_comments_truncates():
    post = {"id": "abc123", "subreddit": "s", "title": "t", "permalink": "/comments/abc123/"}
    nodes = [_t1(f"c{i}", f"backup tip {i}", i) for i in range(10)]
    respx.get("https://www.reddit.com/comments/abc123.json").mock(
        return_value=httpx.Response(200, json=_listing(post, nodes))
    )
    thread = _fetcher().fetch_thread("abc123", max_comments=3)
    assert len(thread.comments) == 3
    assert thread.comments_truncated is True


@respx.mock
def test_403_gives_actionable_error():
    respx.get("https://www.reddit.com/comments/abc123.json").mock(
        return_value=httpx.Response(403, text="blocked")
    )
    with pytest.raises(RedditError) as exc:
        _fetcher().fetch_thread("abc123")
    assert "datacenter" in str(exc.value).lower()


def test_unparseable_id_raises():
    with pytest.raises(RedditError):
        _fetcher().fetch_thread("https://example.com/not-reddit")
