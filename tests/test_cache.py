import time

from reddit_research.core.cache import Cache
from reddit_research.core.models import Comment, SearchResult, Thread


def _cache(tmp_path):
    return Cache(tmp_path / "c.sqlite")


def _thread():
    return Thread(
        id="abc123",
        subreddit="selfhosted",
        title="t",
        url="u",
        permalink="p",
        comments=[Comment(id="c1", body="hi", score=1)],
    )


def test_search_roundtrip_and_ttl(tmp_path):
    c = _cache(tmp_path)
    results = [SearchResult(title="x", url="https://reddit.com/r/a/comments/1/x")]
    c.put_search_results("k", results)
    assert c.get_search_results("k", ttl_days=7) is not None
    # Force expiry by rewriting fetched_utc far in the past.
    c.conn.execute("UPDATE search_results SET fetched_utc = ?", (time.time() - 40 * 86400,))
    c.conn.commit()
    assert c.get_search_results("k", ttl_days=7) is None


def test_thread_roundtrip(tmp_path):
    c = _cache(tmp_path)
    c.put_thread(_thread())
    got = c.get_thread("abc123", ttl_days=30)
    assert got is not None
    assert got.comments[0].body == "hi"


def test_purge_removes_old_rows(tmp_path):
    c = _cache(tmp_path)
    c.put_thread(_thread())
    c.conn.execute("UPDATE threads SET fetched_utc = ?", (time.time() - 100 * 86400,))
    c.conn.commit()
    removed = c.purge(older_than_days=30)
    assert removed["threads"] == 1
    assert c.get_thread("abc123", ttl_days=365) is None


def test_stats(tmp_path):
    c = _cache(tmp_path)
    c.put_thread(_thread())
    stats = c.stats()
    assert stats["threads"] == 1
    assert stats["comments"] == 1
