"""Local SQLite cache for search results and fetched threads.

The cache is an ephemeral performance aid, not a permanent dataset: every row
carries a retrieval timestamp, entries expire by TTL, and ``purge`` / ``clear``
let the user wipe it. Thread bodies and comment trees are stored as normalized
JSON blobs keyed by thread id.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

from .models import SearchResult, Thread

SCHEMA = """
CREATE TABLE IF NOT EXISTS search_results (
    query_key   TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    payload     TEXT NOT NULL,
    fetched_utc REAL NOT NULL,
    PRIMARY KEY (query_key, rank)
);

CREATE TABLE IF NOT EXISTS threads (
    id          TEXT PRIMARY KEY,
    subreddit   TEXT,
    payload     TEXT NOT NULL,
    fetched_utc REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS comments (
    thread_id   TEXT NOT NULL,
    comment_id  TEXT NOT NULL,
    fetched_utc REAL NOT NULL,
    PRIMARY KEY (thread_id, comment_id)
);

CREATE TABLE IF NOT EXISTS fetch_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    detail      TEXT,
    started_utc REAL NOT NULL
);
"""


# Bump when the shape or completeness of cached threads changes, so stale
# entries written by an older fetcher are dropped instead of served for a TTL.
THREAD_FORMAT_VERSION = 2


class Cache:
    """Thin wrapper over a SQLite connection with TTL-aware read helpers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._invalidate_stale_threads()

    def _invalidate_stale_threads(self) -> None:
        (version,) = self.conn.execute("PRAGMA user_version").fetchone()
        if version < THREAD_FORMAT_VERSION:
            self.conn.execute("DELETE FROM threads")
            self.conn.execute("DELETE FROM comments")
            self.conn.execute(f"PRAGMA user_version = {THREAD_FORMAT_VERSION}")
            self.conn.commit()
        self.conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _now() -> float:
        return time.time()

    # -- search results ----------------------------------------------------

    def get_search_results(
        self, query_key: str, ttl_days: int
    ) -> list[SearchResult] | None:
        """Return cached results for ``query_key`` if fresh, else ``None``."""

        cutoff = self._now() - ttl_days * 86400
        rows = self.conn.execute(
            "SELECT payload, fetched_utc FROM search_results "
            "WHERE query_key = ? ORDER BY rank",
            (query_key,),
        ).fetchall()
        if not rows:
            return None
        if any(row["fetched_utc"] < cutoff for row in rows):
            return None
        return [SearchResult.model_validate_json(row["payload"]) for row in rows]

    def put_search_results(
        self, query_key: str, results: Iterable[SearchResult]
    ) -> None:
        now = self._now()
        self.conn.execute("DELETE FROM search_results WHERE query_key = ?", (query_key,))
        self.conn.executemany(
            "INSERT INTO search_results (query_key, rank, payload, fetched_utc) "
            "VALUES (?, ?, ?, ?)",
            [
                (query_key, i, r.model_dump_json(), now)
                for i, r in enumerate(results)
            ],
        )
        self._log_run("search", query_key, now)
        self.conn.commit()

    # -- threads -----------------------------------------------------------

    def get_thread(self, thread_id: str, ttl_days: int) -> Thread | None:
        cutoff = self._now() - ttl_days * 86400
        row = self.conn.execute(
            "SELECT payload, fetched_utc FROM threads WHERE id = ?",
            (thread_id,),
        ).fetchone()
        if row is None or row["fetched_utc"] < cutoff:
            return None
        return Thread.model_validate_json(row["payload"])

    def put_thread(self, thread: Thread) -> None:
        now = self._now()
        self.conn.execute(
            "INSERT OR REPLACE INTO threads (id, subreddit, payload, fetched_utc) "
            "VALUES (?, ?, ?, ?)",
            (thread.id, thread.subreddit, thread.model_dump_json(), now),
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO comments (thread_id, comment_id, fetched_utc) "
            "VALUES (?, ?, ?)",
            [(thread.id, c.id, now) for c in thread.comments],
        )
        self._log_run("thread", thread.id, now)
        self.conn.commit()

    # -- runs --------------------------------------------------------------

    def _log_run(self, kind: str, detail: str, started_utc: float) -> None:
        self.conn.execute(
            "INSERT INTO fetch_runs (kind, detail, started_utc) VALUES (?, ?, ?)",
            (kind, detail, started_utc),
        )

    # -- maintenance -------------------------------------------------------

    def stats(self) -> dict[str, int | str]:
        c = self.conn
        return {
            "path": str(self.path),
            "search_result_rows": c.execute(
                "SELECT COUNT(*) FROM search_results"
            ).fetchone()[0],
            "distinct_queries": c.execute(
                "SELECT COUNT(DISTINCT query_key) FROM search_results"
            ).fetchone()[0],
            "threads": c.execute("SELECT COUNT(*) FROM threads").fetchone()[0],
            "comments": c.execute("SELECT COUNT(*) FROM comments").fetchone()[0],
            "fetch_runs": c.execute("SELECT COUNT(*) FROM fetch_runs").fetchone()[0],
        }

    def purge(self, older_than_days: float) -> dict[str, int]:
        """Delete rows whose retrieval timestamp is older than the cutoff."""

        cutoff = self._now() - older_than_days * 86400
        removed = {}
        for table in ("search_results", "threads", "comments", "fetch_runs"):
            col = "started_utc" if table == "fetch_runs" else "fetched_utc"
            cur = self.conn.execute(f"DELETE FROM {table} WHERE {col} < ?", (cutoff,))
            removed[table] = cur.rowcount
        self.conn.commit()
        return removed

    def clear(self) -> None:
        for table in ("search_results", "threads", "comments", "fetch_runs"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()
