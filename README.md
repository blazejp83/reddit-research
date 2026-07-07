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

Configuration splits cleanly in two:

- **Secrets → environment / `.env`** (never the TOML). Copy `.env.example`:

  ```bash
  # Reddit creds — only for the "praw" backend (see Fetch backends below)
  REDDIT_CLIENT_ID=...
  REDDIT_CLIENT_SECRET=...
  BRAVE_SEARCH_API_KEY=...        # or TAVILY_API_KEY (search always needs a key)
  ANTHROPIC_API_KEY=...           # only if you enable synthesis
  ```

- **Non-secret behavior → `reddit-research.toml`** (see `reddit-research.example.toml`):
  `backend`, `user_agent`, search/synth `provider`, comment limits, cache TTLs, etc.

Precedence is `env var > TOML > default`, so any TOML value can be overridden by
its env var when needed — but each setting has one canonical home to avoid
duplication.

## Fetch backends

Reddit gated self-serve API app creation in 2026 (the "Responsible Builder
Policy"), so an approved OAuth app is no longer guaranteed. The fetcher therefore
supports several backends, selectable per-command with `--backend` or via
`[reddit] backend` / `REDDIT_RESEARCH_BACKEND`:

| Backend | Auth | Notes |
| --- | --- | --- |
| `auto` (default) | — | `praw` if Reddit creds are set, else `json` |
| `praw` | Reddit OAuth app | Official API; full comment-tree expansion. Needs an approved app. |
| `json` | none | Reddit's public `.json` endpoints. Works from **residential** IPs at low volume; Reddit 403s most datacenter/cloud IPs and caps ~30 req/min. Deep "load more" branches aren't expandable without OAuth (thread is marked truncated). |
| `arctic` | none | [Arctic Shift](https://arctic-shift.photon-reddit.com/) archive with [PullPush](https://pullpush.io/) fallback. Works from any IP; serves periodically-updated historical data, so very recent threads may be missing. Retains deleted/removed content. |

```bash
reddit-research fetch    <url> --backend json      # keyless, residential IP
reddit-research evidence "..." --backend arctic    # keyless, any IP
```

Compliance note: `json`/`arctic` let you run without an approved app, but using
them to sidestep API approval sits in tension with the spec's "don't circumvent
access controls" non-goal. Intended for genuine personal, low-volume use.

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
| `core/public_json.py` | Keyless `.json` fetcher |
| `core/archive.py` | Arctic Shift / PullPush fetcher |
| `core/fetchers.py` | Backend selection (`auto`/`praw`/`json`/`arctic`) |
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
providers, three fetch backends (`praw`/`json`/`arctic`), SQLite cache, lexical
ranking, JSON/Markdown output, and the MCP server. Later enhancements (async
fetching, semantic reranking, branch summarization) are tracked in the spec.
