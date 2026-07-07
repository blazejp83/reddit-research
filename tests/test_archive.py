import httpx
import pytest
import respx

from reddit_research.core.archive import ArchiveFetcher
from reddit_research.core.config import RedditConfig
from reddit_research.core.errors import RedditError
from reddit_research.core.normalize import assign_depths_from_parents
from reddit_research.core.models import Comment


ARCTIC = "https://arctic-shift.photon-reddit.com/api"
PULLPUSH = "https://api.pullpush.io/reddit/search"


def _post():
    return {
        "id": "abc123",
        "subreddit": "selfhosted",
        "title": "Backups",
        "selftext": "body",
        "score": 134,
        "num_comments": 2,
        "permalink": "/r/selfhosted/comments/abc123/backups/",
        "url": "https://www.reddit.com/r/selfhosted/comments/abc123/backups/",
    }


def _comment(cid, body, parent="t3_abc123"):
    return {
        "id": cid,
        "parent_id": parent,
        "link_id": "t3_abc123",
        "body": body,
        "author": "u",
        "score": 3,
        "created_utc": 1748148655,
        "permalink": f"/r/selfhosted/comments/abc123/backups/{cid}/",
    }


@respx.mock
def test_arctic_primary_path():
    respx.get(f"{ARCTIC}/posts/ids").mock(return_value=httpx.Response(200, json={"data": [_post()]}))
    respx.get(f"{ARCTIC}/comments/search").mock(
        return_value=httpx.Response(200, json={"data": [_comment("m1", "borg")]})
    )
    thread = ArchiveFetcher(RedditConfig()).fetch_thread("abc123")
    assert thread.id == "abc123"
    assert thread.subreddit == "selfhosted"
    assert len(thread.comments) == 1
    assert thread.comments[0].permalink.startswith("https://www.reddit.com")
    # num_comments (2) > fetched (1) => truncated.
    assert thread.comments_truncated is True


@respx.mock
def test_falls_back_to_pullpush_when_arctic_empty():
    respx.get(f"{ARCTIC}/posts/ids").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(f"{PULLPUSH}/submission/").mock(
        return_value=httpx.Response(200, json={"data": [_post()]})
    )
    respx.get(f"{PULLPUSH}/comment/").mock(
        return_value=httpx.Response(200, json={"data": [_comment("m1", "restic")]})
    )
    thread = ArchiveFetcher(RedditConfig()).fetch_thread("abc123")
    assert thread.comments[0].body == "restic"


@respx.mock
def test_falls_back_when_arctic_errors():
    respx.get(f"{ARCTIC}/posts/ids").mock(return_value=httpx.Response(503))
    respx.get(f"{PULLPUSH}/submission/").mock(
        return_value=httpx.Response(200, json={"data": [_post()]})
    )
    respx.get(f"{PULLPUSH}/comment/").mock(return_value=httpx.Response(200, json={"data": []}))
    thread = ArchiveFetcher(RedditConfig()).fetch_thread("abc123")
    assert thread.id == "abc123"


@respx.mock
def test_not_found_anywhere_raises():
    respx.get(f"{ARCTIC}/posts/ids").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(f"{PULLPUSH}/submission/").mock(return_value=httpx.Response(200, json={"data": []}))
    with pytest.raises(RedditError):
        ArchiveFetcher(RedditConfig()).fetch_thread("abc123")


def test_assign_depths_from_parents():
    comments = [
        Comment(id="c1", parent_id="t3_abc123", body="root"),
        Comment(id="c2", parent_id="t1_c1", body="reply"),
        Comment(id="c3", parent_id="t1_c2", body="deep reply"),
        Comment(id="c4", parent_id="t1_missing", body="orphan"),
    ]
    assign_depths_from_parents(comments, "t3_abc123")
    by_id = {c.id: c.depth for c in comments}
    assert by_id == {"c1": 0, "c2": 1, "c3": 2, "c4": 0}
