from reddit_research.core.urls import (
    is_reddit_url,
    parse_ref,
    subreddit_from_url,
)


def test_parse_canonical_url():
    ref = parse_ref(
        "https://www.reddit.com/r/selfhosted/comments/abc123/best_backup_strategy/"
    )
    assert ref is not None
    assert ref.submission_id == "abc123"
    assert ref.subreddit == "selfhosted"
    assert ref.comment_id is None


def test_parse_url_with_comment():
    ref = parse_ref(
        "https://old.reddit.com/r/selfhosted/comments/abc123/slug/def456/"
    )
    assert ref.submission_id == "abc123"
    assert ref.comment_id == "def456"


def test_parse_short_link():
    ref = parse_ref("https://redd.it/xyz789")
    assert ref.submission_id == "xyz789"
    assert ref.subreddit is None


def test_parse_bare_id_and_fullname():
    assert parse_ref("abc123").submission_id == "abc123"
    assert parse_ref("t3_abc123").submission_id == "abc123"


def test_parse_relative_path():
    ref = parse_ref("/r/datahoarder/comments/qq99qq/title")
    assert ref.subreddit == "datahoarder"
    assert ref.submission_id == "qq99qq"


def test_parse_non_reddit_returns_none():
    assert parse_ref("https://example.com/foo/bar") is None
    assert parse_ref("") is None


def test_is_reddit_url():
    assert is_reddit_url("https://www.reddit.com/r/x/comments/1/y")
    assert is_reddit_url("https://redd.it/abc")
    assert not is_reddit_url("https://reddit.evil.com/x")
    assert not is_reddit_url("https://example.com")


def test_subreddit_from_url():
    assert subreddit_from_url("https://www.reddit.com/r/SelfHosted/comments/1/y") == "selfhosted"
    assert subreddit_from_url("https://www.reddit.com/user/foo") is None
