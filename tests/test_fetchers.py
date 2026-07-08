import pytest

from reddit_research.core.archive import ArchiveFetcher
from reddit_research.core.config import load_config
from reddit_research.core.errors import RedditError
from reddit_research.core.fetchers import get_fetcher, resolve_backend
from reddit_research.core.public_json import PublicJsonFetcher
from reddit_research.core.reddit import RedditFetcher


def test_auto_resolves_to_arctic_without_creds():
    cfg = load_config()
    cfg.reddit.client_id = None
    cfg.reddit.client_secret = None
    assert resolve_backend("auto", cfg) == "arctic"


def test_auto_resolves_to_praw_with_creds():
    cfg = load_config()
    cfg.reddit.client_id = "id"
    cfg.reddit.client_secret = "secret"
    assert resolve_backend("auto", cfg) == "praw"


def test_get_fetcher_types():
    cfg = load_config()
    assert isinstance(get_fetcher("json", cfg), PublicJsonFetcher)
    assert isinstance(get_fetcher("arctic", cfg), ArchiveFetcher)
    assert isinstance(get_fetcher("praw", cfg), RedditFetcher)


def test_unknown_backend_raises():
    cfg = load_config()
    with pytest.raises(RedditError):
        get_fetcher("bogus", cfg)
