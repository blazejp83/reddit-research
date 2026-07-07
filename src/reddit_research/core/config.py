"""Configuration loading from environment variables and an optional TOML file.

Precedence (highest first): explicit kwargs -> environment variables -> TOML
config file -> built-in defaults. Secrets only ever come from the environment;
they are never read from or written to the TOML file.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_cache_dir

DEFAULT_CACHE_DIR = Path(user_cache_dir("reddit-research"))
DEFAULT_CACHE_PATH = DEFAULT_CACHE_DIR / "reddit-research.sqlite"
DEFAULT_USER_AGENT = "reddit-research personal research tool (python)"

# Standard locations checked for a config file when none is passed explicitly.
CONFIG_SEARCH_PATHS = [
    Path.cwd() / "reddit-research.toml",
    Path(user_cache_dir("reddit-research")).parent / "reddit-research" / "config.toml",
    Path.home() / ".config" / "reddit-research" / "config.toml",
]


def _expand(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(path))).expanduser()


@dataclass
class SearchConfig:
    provider: str = "brave"
    default_limit: int = 10
    recency_days: int | None = None


@dataclass
class RedditConfig:
    client_id: str | None = None
    client_secret: str | None = None
    username: str | None = None
    password: str | None = None
    refresh_token: str | None = None
    user_agent: str = DEFAULT_USER_AGENT
    max_comments_per_thread: int = 300
    comment_sort: str = "top"


@dataclass
class CacheConfig:
    path: Path = DEFAULT_CACHE_PATH
    search_ttl_days: int = 7
    thread_ttl_days: int = 30


@dataclass
class SynthConfig:
    provider: str | None = None  # e.g. "anthropic"; None disables synthesis
    model: str | None = None
    api_key: str | None = None


@dataclass
class Config:
    search: SearchConfig = field(default_factory=SearchConfig)
    reddit: RedditConfig = field(default_factory=RedditConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    brave_api_key: str | None = None
    tavily_api_key: str | None = None

    # Field names that must be scrubbed from any serialized/logged output.
    SECRET_FIELDS = frozenset(
        {
            "client_secret",
            "password",
            "refresh_token",
            "api_key",
            "brave_api_key",
            "tavily_api_key",
        }
    )


def _load_toml(path: Path | None) -> dict:
    if path is None:
        for candidate in CONFIG_SEARCH_PATHS:
            if candidate.is_file():
                path = candidate
                break
    if path is None or not Path(path).is_file():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def load_config(config_path: str | os.PathLike[str] | None = None) -> Config:
    """Build a :class:`Config` from a TOML file plus environment overrides."""

    data = _load_toml(Path(config_path) if config_path else None)
    search_tbl = data.get("search", {})
    reddit_tbl = data.get("reddit", {})
    cache_tbl = data.get("cache", {})
    synth_tbl = data.get("synth", {})

    env = os.environ

    search = SearchConfig(
        provider=env.get("REDDIT_RESEARCH_PROVIDER", search_tbl.get("provider", "brave")),
        default_limit=int(search_tbl.get("default_limit", 10)),
        recency_days=search_tbl.get("recency_days"),
    )

    reddit = RedditConfig(
        client_id=env.get("REDDIT_CLIENT_ID") or reddit_tbl.get("client_id"),
        client_secret=env.get("REDDIT_CLIENT_SECRET") or reddit_tbl.get("client_secret"),
        username=env.get("REDDIT_USERNAME") or reddit_tbl.get("username"),
        password=env.get("REDDIT_PASSWORD") or reddit_tbl.get("password"),
        refresh_token=env.get("REDDIT_REFRESH_TOKEN") or reddit_tbl.get("refresh_token"),
        user_agent=env.get("REDDIT_USER_AGENT")
        or reddit_tbl.get("user_agent", DEFAULT_USER_AGENT),
        max_comments_per_thread=int(reddit_tbl.get("max_comments_per_thread", 300)),
        comment_sort=reddit_tbl.get("comment_sort", "top"),
    )

    cache_path = env.get("REDDIT_RESEARCH_CACHE") or cache_tbl.get("path", DEFAULT_CACHE_PATH)
    cache = CacheConfig(
        path=_expand(cache_path),
        search_ttl_days=int(cache_tbl.get("search_ttl_days", 7)),
        thread_ttl_days=int(cache_tbl.get("thread_ttl_days", 30)),
    )

    synth = SynthConfig(
        provider=env.get("REDDIT_RESEARCH_SYNTH_PROVIDER") or synth_tbl.get("provider"),
        model=env.get("REDDIT_RESEARCH_SYNTH_MODEL") or synth_tbl.get("model"),
        api_key=env.get("ANTHROPIC_API_KEY") or env.get("OPENAI_API_KEY"),
    )

    return Config(
        search=search,
        reddit=reddit,
        cache=cache,
        synth=synth,
        brave_api_key=env.get("BRAVE_SEARCH_API_KEY"),
        tavily_api_key=env.get("TAVILY_API_KEY"),
    )
