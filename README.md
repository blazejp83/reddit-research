# reddit-research

A personal-use, retrieval-augmented research tool that answers questions using
**Reddit discussions as evidence**. It searches for relevant threads via a search
API, fetches thread/comment data through the official Reddit Data API, ranks the
most useful evidence, and returns structured context (optionally synthesized into
a cited answer).

It is intended for interactive question answering and short-lived local research.
It does **not** train models on Reddit data, bulk-archive Reddit, or redistribute
collected datasets. See [the spec](./reddit-research-mcp-cli-spec.md) for the full
design and non-goals.

## Install

```bash
uv sync                 # core CLI
uv sync --extra mcp     # + MCP server
uv sync --extra dev     # + test tooling
```

## Configure

Credentials come from the environment only (never the config file). Copy
`.env.example` and fill in what you have:

```bash
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT="reddit-research personal tool by u/you"
BRAVE_SEARCH_API_KEY=...          # or TAVILY_API_KEY
# Optional synthesis:
REDDIT_RESEARCH_SYNTH_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
```

Non-secret defaults can live in `reddit-research.toml` (see `reddit-research.example.toml`).

## CLI

```bash
reddit-research search   "best backup strategy for homelab"
reddit-research fetch     "https://www.reddit.com/r/selfhosted/comments/..."
reddit-research evidence  "what do selfhosted users recommend for backups?" -r selfhosted -r datahoarder
reddit-research answer    "what do Reddit users recommend for homelab backups?"
reddit-research cache stats
reddit-research cache purge --older-than-days 30
```

Useful flags: `--subreddit/-r` (repeatable), `--limit-threads`, `--max-comments`,
`--sort`, `--since-days`, `--format json|markdown`, `--no-llm`, `--show-queries`,
`--verbose`.

## MCP server

```bash
reddit-research-mcp        # serves over stdio
```

Exposes `search_reddit_threads`, `fetch_reddit_thread`, `rank_reddit_evidence`,
and the high-level `answer_from_reddit`.

Example MCP client config:

```json
{
  "mcpServers": {
    "reddit-research": {
      "command": "reddit-research-mcp",
      "env": { "REDDIT_CLIENT_ID": "...", "BRAVE_SEARCH_API_KEY": "..." }
    }
  }
}
```

## Architecture

```
question -> query planner -> search provider -> reddit URL extractor
         -> Reddit API fetcher -> cache -> evidence ranker
         -> structured evidence -> optional answer synthesis
```

Package layout under `src/reddit_research/`:

| Module | Responsibility |
| --- | --- |
| `core/config.py` | Env + TOML config, secret handling |
| `core/models.py` | Pydantic data contracts |
| `core/search.py` | Query planner + Brave/Tavily providers |
| `core/reddit.py` | PRAW fetcher + normalization |
| `core/urls.py` | Reddit URL/ID parsing |
| `core/cache.py` | SQLite cache (TTL + purge) |
| `core/ranking.py` | Explainable lexical ranking |
| `core/synth.py` | Optional, provider-agnostic synthesis |
| `core/orchestrator.py` | Pipeline + run metadata/warnings |
| `cli.py` / `mcp_server.py` | Interfaces |

## Develop

```bash
uv run pytest
```

## Status

MVP implemented: `search`/`fetch`/`evidence`/`answer` CLI, Brave + Tavily
providers, PRAW fetcher, SQLite cache, lexical ranking, JSON/Markdown output,
and the MCP server. Later enhancements (async fetching, semantic reranking,
branch summarization) are tracked in the spec.
