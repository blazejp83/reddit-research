"""Typer-based CLI for reddit-research.

Commands mirror the pipeline stages: ``search``, ``fetch``, ``evidence``,
``answer``, plus a ``cache`` group for inspection/maintenance. All commands share
config loading and output-format handling; observability flags surface queries,
counts, and warnings on stderr.
"""

from __future__ import annotations

import sys
from typing import Annotated, Optional

import typer

from .core import errors, render
from .core.config import load_config
from .core.orchestrator import Orchestrator

app = typer.Typer(
    add_completion=False,
    help="Answer questions using Reddit discussions as retrieved evidence.",
    no_args_is_help=True,
)
cache_app = typer.Typer(help="Inspect or manage the local cache.")
app.add_typer(cache_app, name="cache")


# -- shared options ---------------------------------------------------------

ConfigOpt = Annotated[
    Optional[str], typer.Option("--config", help="Path to a TOML config file.")
]
FormatOpt = Annotated[
    str, typer.Option("--format", "-f", help="Output format: json or markdown.")
]
SubredditOpt = Annotated[
    Optional[list[str]],
    typer.Option("--subreddit", "-r", help="Restrict to subreddit(s); repeatable."),
]
BackendOpt = Annotated[
    Optional[str],
    typer.Option(
        "--backend",
        help="Fetch backend: auto | praw | json | arctic. Overrides config.",
    ),
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Verbose logging.")]
DebugOpt = Annotated[bool, typer.Option("--debug", help="Debug logging (implies verbose).")]
ShowQueriesOpt = Annotated[bool, typer.Option("--show-queries", help="Print planned search queries.")]
SafesearchOpt = Annotated[
    str,
    typer.Option("--safesearch", help="Search safesearch level: off|moderate|strict (Brave only)."),
]


def _err(msg: str) -> None:
    typer.echo(errors.redact(msg), err=True)


def _emit_meta(meta, verbose: bool, show_queries: bool) -> None:
    if show_queries or verbose:
        for q in meta.search_queries:
            _err(f"query: {q}")
    if verbose:
        _err(
            f"threads={meta.threads_fetched} comments={meta.comments_fetched} "
            f"cache_hits={meta.cache_hits} cache_misses={meta.cache_misses}"
        )
    for w in meta.warnings:
        _err(w)


def _make_orchestrator(config_path: str | None, backend: str | None = None) -> Orchestrator:
    cfg = load_config(config_path)
    if backend:
        cfg.reddit.backend = backend
    return Orchestrator(cfg)


# -- commands ---------------------------------------------------------------


@app.command()
def search(
    query: str,
    subreddit: SubredditOpt = None,
    limit_threads: Annotated[int, typer.Option("--limit-threads", help="Max threads.")] = 10,
    since_days: Annotated[Optional[int], typer.Option("--since-days", help="Recency window in days.")] = None,
    fmt: FormatOpt = "markdown",
    config: ConfigOpt = None,
    verbose: VerboseOpt = False,
    show_queries: ShowQueriesOpt = False,
    safesearch: SafesearchOpt = "moderate",
) -> None:
    """Return candidate Reddit threads for a question."""

    orch = _make_orchestrator(config)
    try:
        meta = _new_meta()
        results = orch.search(
            query, subreddit, limit_threads, since_days, meta=meta, safesearch=safesearch
        )
        typer.echo(render.render_search(results, fmt))
        _emit_meta(meta, verbose, show_queries)
    except errors.RedditResearchError as exc:
        _err(str(exc))
        raise typer.Exit(1)
    finally:
        orch.close()


@app.command()
def fetch(
    url_or_id: str,
    max_comments: Annotated[Optional[int], typer.Option("--max-comments", help="Comment cap.")] = None,
    sort: Annotated[str, typer.Option("--sort", help="Comment sort: top|best|new|controversial|old.")] = "top",
    backend: BackendOpt = None,
    fmt: FormatOpt = "markdown",
    config: ConfigOpt = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the cache.")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Fetch and normalize one Reddit thread."""

    orch = _make_orchestrator(config, backend)
    try:
        meta = _new_meta()
        thread = orch.fetch(
            url_or_id,
            max_comments=max_comments,
            comment_sort=sort,
            meta=meta,
            use_cache=not no_cache,
        )
        typer.echo(render.render_thread(thread, fmt))
        _emit_meta(meta, verbose, False)
    except errors.RedditResearchError as exc:
        _err(str(exc))
        raise typer.Exit(1)
    finally:
        orch.close()


@app.command()
def evidence(
    question: str,
    subreddit: SubredditOpt = None,
    limit_threads: Annotated[int, typer.Option("--limit-threads", help="Max threads.")] = 8,
    max_comments: Annotated[int, typer.Option("--max-comments", help="Comments per thread.")] = 300,
    sort: Annotated[str, typer.Option("--sort", help="Comment sort.")] = "top",
    since_days: Annotated[Optional[int], typer.Option("--since-days")] = None,
    max_evidence: Annotated[int, typer.Option("--max-evidence", help="Max evidence items.")] = 25,
    backend: BackendOpt = None,
    fmt: FormatOpt = "markdown",
    config: ConfigOpt = None,
    verbose: VerboseOpt = False,
    show_queries: ShowQueriesOpt = False,
    safesearch: SafesearchOpt = "moderate",
) -> None:
    """Run search + fetch + ranking and emit structured evidence."""

    orch = _make_orchestrator(config, backend)
    try:
        result = orch.evidence(
            question,
            subreddits=subreddit,
            limit_threads=limit_threads,
            max_comments_per_thread=max_comments,
            comment_sort=sort,
            recency_days=since_days,
            max_evidence_items=max_evidence,
            safesearch=safesearch,
        )
        typer.echo(render.render_evidence(result, fmt))
        _emit_meta(result.metadata, verbose, show_queries)
    except errors.RedditResearchError as exc:
        _err(str(exc))
        raise typer.Exit(1)
    finally:
        orch.close()


@app.command()
def answer(
    question: str,
    subreddit: SubredditOpt = None,
    limit_threads: Annotated[int, typer.Option("--limit-threads")] = 8,
    max_comments: Annotated[int, typer.Option("--max-comments")] = 250,
    sort: Annotated[str, typer.Option("--sort")] = "top",
    since_days: Annotated[Optional[int], typer.Option("--since-days")] = None,
    max_evidence: Annotated[int, typer.Option("--max-evidence")] = 25,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Skip LLM synthesis.")] = False,
    backend: BackendOpt = None,
    fmt: FormatOpt = "markdown",
    config: ConfigOpt = None,
    verbose: VerboseOpt = False,
    show_queries: ShowQueriesOpt = False,
    safesearch: SafesearchOpt = "moderate",
) -> None:
    """Run the full pipeline and optionally synthesize a cited answer."""

    orch = _make_orchestrator(config, backend)
    try:
        result = orch.answer(
            question,
            subreddits=subreddit,
            limit_threads=limit_threads,
            max_comments_per_thread=max_comments,
            comment_sort=sort,
            recency_days=since_days,
            max_evidence_items=max_evidence,
            include_answer=not no_llm,
            safesearch=safesearch,
        )
        typer.echo(render.render_answer(result, fmt))
        _emit_meta(result.metadata, verbose, show_queries)
    except errors.RedditResearchError as exc:
        _err(str(exc))
        raise typer.Exit(1)
    finally:
        orch.close()


@cache_app.command("stats")
def cache_stats(config: ConfigOpt = None) -> None:
    """Show cache row counts and location."""

    orch = _make_orchestrator(config)
    try:
        import json

        typer.echo(json.dumps(orch.cache.stats(), indent=2))
    finally:
        orch.close()


@cache_app.command("purge")
def cache_purge(
    older_than_days: Annotated[float, typer.Option("--older-than-days", help="Delete rows older than N days.")] = 30.0,
    config: ConfigOpt = None,
) -> None:
    """Delete cached rows older than the given age."""

    orch = _make_orchestrator(config)
    try:
        removed = orch.cache.purge(older_than_days)
        typer.echo(f"Purged: {removed}")
    finally:
        orch.close()


def _new_meta():
    from .core.models import RunMetadata

    return RunMetadata()


def main() -> None:  # pragma: no cover - console-script entry
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
