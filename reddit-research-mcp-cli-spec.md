# Reddit Research MCP/CLI Specification

## Purpose

Build a personal-use research tool that answers user questions using Reddit discussions as retrieved evidence. The tool is retrieval-augmented: it searches for relevant Reddit threads, fetches thread/comment data, ranks useful evidence, and returns structured context for an LLM or a synthesized answer with citations.

This tool must not train, fine-tune, or build a reusable model from Reddit data. It is intended for interactive question answering and short-lived local research workflows.

## Goals

- Accept a natural-language question and gather relevant Reddit threads.
- Discover candidate threads through a search provider, preferably Brave Search API or Tavily.
- Fetch Reddit thread data through the official Reddit Data API where possible.
- Preserve citations to original Reddit posts and comments.
- Provide both a CLI for direct/debug usage and an MCP server for agentic LLM usage.
- Cache fetched data locally to reduce duplicate requests and improve iteration speed.
- Keep LLM synthesis optional and provider-agnostic.

## Non-Goals

- Training, fine-tuning, or embedding Reddit data into a permanent model.
- Bulk Reddit archival.
- Circumventing Reddit, Google, or search-provider access controls.
- Fetching private messages, private communities, deleted content, quarantined communities, or otherwise inaccessible data.
- Redistributing collected Reddit datasets.
- Replacing formal academic access through Reddit for Researchers.

## High-Level Architecture

```text
user question
  -> query planner
  -> search provider
  -> reddit URL extractor
  -> Reddit API fetcher
  -> cache
  -> evidence ranker
  -> structured evidence
  -> optional answer synthesis
```

Suggested package layout:

```text
reddit_research/
  core/
    config.py
    models.py
    search.py
    reddit.py
    ranking.py
    cache.py
    orchestrator.py
    synth.py
  cli.py
  mcp_server.py
  tests/
```

## Data Flow

1. User asks a question.
2. Query planner creates search queries, for example:
   - `site:reddit.com/r/selfhosted backups borg restic`
   - `site:reddit.com/r/BuyItForLife office chair recommendations`
   - `site:reddit.com/r/ExperiencedDevs monorepo migration`
3. Search provider returns URLs and snippets.
4. URL parser extracts Reddit thread identifiers from canonical Reddit URLs.
5. Reddit fetcher retrieves post metadata and comment tree.
6. Cache stores normalized thread/comment data with retrieval timestamp.
7. Ranker selects threads and comments most relevant to the question.
8. Tool returns structured evidence or uses an LLM to produce a cited answer.

## Interfaces

### CLI

Primary commands:

```bash
reddit-research search "best backup strategy for homelab"
reddit-research fetch "https://www.reddit.com/r/selfhosted/comments/..."
reddit-research evidence "what do selfhosted users recommend for backups?"
reddit-research answer "what do selfhosted users recommend for backups?"
reddit-research cache stats
reddit-research cache purge --older-than 30d
```

Command behavior:

- `search`: returns candidate Reddit threads with titles, snippets, scores, and URLs.
- `fetch`: returns normalized post/comment data for one thread.
- `evidence`: runs search, fetch, and ranking, then emits structured evidence as JSON or Markdown.
- `answer`: runs the full pipeline and performs optional LLM synthesis.
- `cache`: inspects or manages local cache.

Useful flags:

```bash
--subreddit selfhosted --subreddit datahoarder
--search-provider brave
--limit-threads 8
--max-comments 300
--sort relevance|top|new|controversial
--since 2024-01-01
--format json|markdown
--no-llm
--cache-ttl 7d
```

### MCP Tools

Expose low-level tools plus one high-level orchestration tool.

#### `search_reddit_threads`

Find candidate Reddit threads through a configured search provider.

Input:

```json
{
  "query": "best backup strategy for homelab",
  "subreddits": ["selfhosted", "datahoarder"],
  "limit": 10,
  "recency_days": 365
}
```

Output:

```json
{
  "results": [
    {
      "title": "Thread title",
      "url": "https://www.reddit.com/r/selfhosted/comments/...",
      "subreddit": "selfhosted",
      "snippet": "Search snippet",
      "source": "brave",
      "score": 0.84
    }
  ]
}
```

#### `fetch_reddit_thread`

Fetch and normalize one Reddit thread.

Input:

```json
{
  "url_or_id": "https://www.reddit.com/r/selfhosted/comments/...",
  "max_comments": 300,
  "comment_sort": "top",
  "include_more_comments": true
}
```

Output:

```json
{
  "thread": {
    "id": "abc123",
    "subreddit": "selfhosted",
    "title": "Thread title",
    "url": "https://www.reddit.com/r/selfhosted/comments/...",
    "permalink": "https://www.reddit.com/r/selfhosted/comments/...",
    "score": 120,
    "created_utc": 1710000000,
    "num_comments": 87,
    "comments": [
      {
        "id": "def456",
        "parent_id": "abc123",
        "body": "Comment body",
        "score": 42,
        "depth": 0,
        "created_utc": 1710001000,
        "permalink": "https://www.reddit.com/r/selfhosted/comments/.../comment/def456"
      }
    ]
  }
}
```

#### `rank_reddit_evidence`

Rank already-fetched thread/comment data against a question.

Input:

```json
{
  "question": "what backup tools do homelab users recommend?",
  "threads": [],
  "max_evidence_items": 25
}
```

Output:

```json
{
  "evidence": [
    {
      "kind": "comment",
      "thread_title": "Thread title",
      "subreddit": "selfhosted",
      "text": "Relevant comment text",
      "permalink": "https://www.reddit.com/...",
      "score": 0.91,
      "reddit_score": 42
    }
  ]
}
```

#### `answer_from_reddit`

Run the full pipeline and optionally synthesize an answer.

Input:

```json
{
  "question": "what do Reddit users recommend for homelab backups?",
  "subreddits": ["selfhosted", "datahoarder"],
  "limit_threads": 8,
  "max_comments_per_thread": 250,
  "include_answer": true
}
```

Output:

```json
{
  "answer": "Cited synthesized answer, if enabled.",
  "evidence": [],
  "threads_considered": [],
  "warnings": []
}
```

## Core Components

### Search Provider

Responsibilities:

- Generate search-provider requests.
- Support domain/subreddit constraints.
- Return normalized search results.
- Avoid scraping search result pages directly.

Initial providers:

- Brave Search API.
- Tavily.

Optional providers:

- Google Programmable Search JSON API for existing users only.
- Manual browser/search import from copied URLs.

### Reddit Fetcher

Responsibilities:

- Authenticate with Reddit OAuth.
- Parse Reddit URLs and IDs.
- Fetch submission metadata.
- Fetch comment trees with bounded expansion.
- Normalize deleted/removed comments.
- Preserve permalinks for citation.
- Respect rate limits and retries.

Suggested Python implementation:

- `praw` for simple synchronous CLI.
- `asyncpraw` if concurrency becomes important.

### Cache

Use SQLite by default.

Tables:

- `search_results`
- `threads`
- `comments`
- `fetch_runs`

Cache policy:

- Store fetched public thread/comment data locally.
- Default TTL: 7 days for search results, 30 days for fetched threads.
- Include retrieval timestamp.
- Provide purge commands.
- Do not treat the cache as a permanent dataset.

### Ranking

Ranking should combine simple, explainable signals before adding heavier semantic ranking.

Thread-level signals:

- Search-provider rank.
- Query/title/snippet lexical match.
- Subreddit allowlist match.
- Recency.
- Reddit score and comment count.
- Duplicate/crosspost suppression.

Comment-level signals:

- Semantic or lexical match to question.
- Reddit score.
- Depth penalty for very deep replies.
- Removal/deletion exclusion.
- Quote/link density.
- Author repetition penalty if author metadata is retained.

Optional semantic ranking:

- Use local embeddings or provider embeddings for ephemeral ranking.
- Do not use embeddings as a permanent Reddit corpus unless the user explicitly accepts that storage model.

### Synthesis

Synthesis is optional and should be separate from retrieval.

Requirements:

- Cite Reddit permalinks.
- Distinguish consensus from isolated anecdotes.
- Surface disagreement and uncertainty.
- Mention when evidence is thin, outdated, or subreddit-specific.
- Avoid presenting Reddit comments as verified facts.

Preferred output shape:

```text
Short answer
What Reddit users seem to agree on
Notable disagreements
Caveats
Sources
```

## Configuration

Environment variables:

```text
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=
REDDIT_USERNAME=        # optional for read-only public access depending on auth mode
REDDIT_PASSWORD=        # optional; prefer OAuth/refresh token when possible
BRAVE_SEARCH_API_KEY=
TAVILY_API_KEY=
REDDIT_RESEARCH_CACHE=~/.cache/reddit-research/reddit-research.sqlite
```

Config file:

```toml
[search]
provider = "brave"
default_limit = 10

[reddit]
max_comments_per_thread = 300
comment_sort = "top"
user_agent = "reddit-research personal tool by u/USERNAME"

[cache]
path = "~/.cache/reddit-research/reddit-research.sqlite"
search_ttl_days = 7
thread_ttl_days = 30
```

## Compliance and Safety Constraints

- Use search APIs rather than scraping Google result pages.
- Do not train or fine-tune a model on Reddit content.
- Do not redistribute raw collected Reddit data.
- Keep personal cache local and purgeable.

## Error Handling

The tool should return structured warnings for:

- Search provider unavailable.
- Reddit OAuth failure.
- Rate limit reached.
- Thread unavailable, private, quarantined, deleted, or removed.
- Comments partially loaded due to configured limits.
- Too few relevant results found.
- LLM synthesis unavailable.

Errors should be actionable and not expose secrets.

## Observability

CLI should support:

```bash
--verbose
--debug
--show-queries
--show-rate-limits
```

MCP responses should include optional metadata:

```json
{
  "metadata": {
    "search_queries": [],
    "threads_fetched": 5,
    "comments_fetched": 812,
    "cache_hits": 3,
    "cache_misses": 2,
    "warnings": []
  }
}
```

## Testing Plan

Unit tests:

- Reddit URL parsing.
- Search query generation.
- Result normalization.
- Ranking behavior.
- Cache TTL and purge behavior.
- Redaction of credentials in errors/logs.

Integration tests:

- Search provider mocked responses.
- Reddit API mocked thread/comment responses.
- End-to-end `evidence` command with fixtures.
- MCP tool request/response schema validation.

Manual tests:

- Ask a product-recommendation question.
- Ask a technical troubleshooting question.
- Ask a question constrained to specific subreddits.
- Fetch a thread with many `MoreComments` entries.
- Verify citations open to the original Reddit locations.

## MVP

1. CLI with `search`, `fetch`, and `evidence`.
2. Brave Search provider.
3. Reddit fetcher using PRAW.
4. SQLite cache.
5. Basic lexical ranking.
6. Markdown and JSON output.
7. MCP server exposing `search_reddit_threads`, `fetch_reddit_thread`, and `answer_from_reddit`.

## Later Enhancements

- Tavily provider.
- Async fetching with concurrency limits.
- Local semantic reranking.
- Comment tree summarization by branch.
- Subreddit quality profiles.
- Query expansion using an LLM.
- Browser extension or share-sheet ingestion for manually selected Reddit URLs.
- UI for inspecting evidence before answer synthesis.

