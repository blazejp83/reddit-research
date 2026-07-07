"""Fetcher normalization tests using lightweight PRAW-shaped stand-ins."""

from types import SimpleNamespace

from reddit_research.core.config import RedditConfig
from reddit_research.core.reddit import RedditFetcher


class FakeComments:
    def __init__(self, items):
        self._items = items

    def replace_more(self, limit=0):
        return []

    def list(self):
        return self._items


class FakeSubmission:
    def __init__(self, comments):
        self.id = "abc123"
        self.title = "Homelab backups"
        self.subreddit = "selfhosted"
        self.url = "https://www.reddit.com/r/selfhosted/comments/abc123/x"
        self.permalink = "/r/selfhosted/comments/abc123/x/"
        self.selftext = "body"
        self.author = "someuser"
        self.score = 100
        self.upvote_ratio = 0.97
        self.created_utc = 1710000000
        self.num_comments = 2
        self.over_18 = False
        self.comment_sort = "top"
        self.comments = FakeComments(comments)


class FakeReddit:
    def __init__(self, submission):
        self._submission = submission
        self.read_only = True

    def submission(self, id):
        return self._submission


def _comment(cid, body, score, depth=0, removed_author=False):
    return SimpleNamespace(
        id=cid,
        parent_id="t3_abc123",
        body=body,
        author=None if removed_author else "u",
        score=score,
        depth=depth,
        created_utc=1710001000,
        permalink=f"/r/selfhosted/comments/abc123/x/{cid}/",
    )


def test_normalize_thread_and_comments():
    comments = [
        _comment("c1", "borg is great", 42),
        _comment("c2", "[deleted]", 0, removed_author=True),
    ]
    sub = FakeSubmission(comments)
    fetcher = RedditFetcher(RedditConfig(), client=FakeReddit(sub))
    thread = fetcher.fetch_thread(
        "https://www.reddit.com/r/selfhosted/comments/abc123/x"
    )
    assert thread.id == "abc123"
    assert thread.subreddit == "selfhosted"
    assert thread.permalink.startswith("https://www.reddit.com")
    assert len(thread.comments) == 2
    assert thread.comments[0].permalink.startswith("https://www.reddit.com")
    # Deleted comment is retained but flagged.
    assert thread.comments[1].removed is True


def test_max_comments_truncates():
    comments = [_comment(f"c{i}", f"backup tip {i}", i) for i in range(10)]
    sub = FakeSubmission(comments)
    fetcher = RedditFetcher(RedditConfig(), client=FakeReddit(sub))
    thread = fetcher.fetch_thread("abc123", max_comments=3)
    assert len(thread.comments) == 3
    assert thread.comments_truncated is True
