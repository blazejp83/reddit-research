"""Configuration loading from environment variables and an optional TOML file.

Precedence (highest first): real environment variables -> ``.env`` file -> TOML
config file -> built-in defaults. Secrets only ever come from the environment
(directly or via ``.env``); they are never read from or written to the TOML file.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_cache_dir

# Locations checked for a ``.env`` file (most specific first). A real env var
# already set in the process always wins over any file value.
DOTENV_SEARCH_PATHS = [
    Path.cwd() / ".env",
    Path.home() / ".config" / "reddit-research" / ".env",
]

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


def _load_dotenv_files() -> None:
    """Populate ``os.environ`` from ``.env`` files without overriding real vars.

    Files are loaded most-specific first with ``override=False``, so a cwd
    ``.env`` wins over the global one, and any variable already exported in the
    real environment wins over both. No-op if python-dotenv isn't installed.
    """

    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        return
    for path in DOTENV_SEARCH_PATHS:
        if path.is_file():
            load_dotenv(path, override=False)


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
    # Fetch backend: auto | praw | json | arctic (see core.fetchers).
    backend: str = "auto"


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

    _load_dotenv_files()
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
        # Secrets: environment only, never the TOML file.
        client_id=env.get("REDDIT_CLIENT_ID"),
        client_secret=env.get("REDDIT_CLIENT_SECRET"),
        username=env.get("REDDIT_USERNAME"),
        password=env.get("REDDIT_PASSWORD"),
        refresh_token=env.get("REDDIT_REFRESH_TOKEN"),
        # Non-secret behavior: TOML is the home; env var is an optional override.
        user_agent=env.get("REDDIT_USER_AGENT")
        or reddit_tbl.get("user_agent", DEFAULT_USER_AGENT),
        max_comments_per_thread=int(reddit_tbl.get("max_comments_per_thread", 300)),
        comment_sort=reddit_tbl.get("comment_sort", "top"),
        backend=env.get("REDDIT_RESEARCH_BACKEND") or reddit_tbl.get("backend", "auto"),
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
