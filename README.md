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

A `.env` in the current directory (or `~/.config/reddit-research/.env`) is loaded
automatically. Precedence is `real env var > .env file > TOML > default`, so any
TOML value can be overridden by its env var when needed — but each setting has
one canonical home to avoid duplication.

## Fetch backends

Reddit gated self-serve API app creation in 2026 (the "Responsible Builder
Policy"), so an approved OAuth app is no longer guaranteed. The fetcher therefore
supports several backends, selectable per-command with `--backend` or via
`[reddit] backend` / `REDDIT_RESEARCH_BACKEND`:

| Backend | Auth | Notes |
| --- | --- | --- |
| `auto` (default) | — | `praw` if Reddit creds are set, else `arctic` |
| `arctic` (recommended keyless) | none | [Arctic Shift](https://arctic-shift.photon-reddit.com/) archive with [PullPush](https://pullpush.io/) fallback. Works from any IP/host. Serves periodically-updated historical data, so very recent threads may lag; retains deleted/removed content. |
| `praw` | Reddit OAuth app | Official API; live data + full comment-tree expansion. Needs an approved app. |
| `json` | none | Reddit's public `.json` endpoints. **Largely unusable in 2026**: Reddit fingerprint-blocks the `.json` path (403) for non-browser clients — even from a residential IP, and even with a matching browser User-Agent (it checks the TLS/HTTP2 fingerprint of a current browser). Kept for the rare environment where it still works. |

```bash
reddit-research evidence "..."                 # auto -> arctic (keyless)
reddit-research answer   "..." --backend praw  # live data, needs an approved app
```

Compliance note: `arctic`/`json` let you run without an approved app, but using
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

Exposes `search_reddit_threads`, `fetch_reddit_thread`, `rank_reddit_evidence`,
and the high-level `answer_from_reddit`. Two transports (`REDDIT_RESEARCH_MCP_TRANSPORT`):

**`stdio`** (default) — local use; the client spawns the process:

```json
{
  "mcpServers": {
    "reddit-research": {
      "command": "reddit-research-mcp",
      "env": { "BRAVE_SEARCH_API_KEY": "..." }
    }
  }
}
```

**`http` / `streamable-http`** — a long-lived network server for container/NAS
deployment (see below). A bearer token is **required**; the server refuses to
start over HTTP without `REDDIT_RESEARCH_MCP_TOKEN`, and every request except
`GET /healthz` must send `Authorization: Bearer <token>`.

## Deploy on a NAS (Docker)

The included `Dockerfile` + `docker-compose.yml` run the MCP server over HTTP.

1. **Configure** — create `.env` next to the compose file:

   ```bash
   BRAVE_SEARCH_API_KEY=...                 # search needs a key
   REDDIT_RESEARCH_MCP_TOKEN=$(openssl rand -hex 32)   # required bearer token
   # ANTHROPIC_API_KEY=...                  # only if you enable synthesis
   ```

2. **Run** — build from source:

   ```bash
   docker compose up -d --build
   curl http://127.0.0.1:8000/healthz            # -> ok
   ```

   To deploy a prebuilt image from your own registry instead, set
   `REDDIT_RESEARCH_IMAGE` in `.env` and pull:

   ```bash
   export REGISTRY=your-registry.example.com
   echo "REDDIT_RESEARCH_IMAGE=$REGISTRY/reddit-research-mcp:latest" >> .env
   docker login "$REGISTRY"                      # once
   docker compose pull && docker compose up -d
   ```

   To publish a new image after code changes:

   ```bash
   docker build --provenance=false \
     -t "$REGISTRY/reddit-research-mcp:latest" .
   docker push "$REGISTRY/reddit-research-mcp:latest"
   ```

   The SQLite cache persists in `./data`. The container binds to `127.0.0.1:8000`
   by default, so it's reachable only through the NAS's reverse proxy (change the
   `ports:` mapping to `8000:8000` to expose it on the LAN instead).

3. **Reverse proxy (Synology)** — Control Panel → Login Portal → Advanced →
   Reverse Proxy → Create:
   - Source: `https://reddit-mcp.<your-domain>` (port 443, HTTPS — enables TLS)
   - Destination: `http://localhost:8000`
   - Enable HSTS as desired; the streamable-HTTP transport streams responses, so
     leave response buffering off (the default reverse-proxy behavior is fine).

4. **Connect from Claude Code**:

   ```bash
   claude mcp add --transport http reddit-research \
     https://reddit-mcp.<your-domain>/mcp \
     --header "Authorization: Bearer <your REDDIT_RESEARCH_MCP_TOKEN>"
   ```

Security notes: the token gates outbound calls that spend your API keys, so keep
it secret and prefer a random 32-byte value. The compose file sets the keyless
`arctic` backend (the public `.json` path is fingerprint-blocked by Reddit for
non-browser clients — see Fetch backends). Switch to `praw` only if you have an
approved OAuth app and set its credentials in `.env`.

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
