"""Fetch the Reddit threads used by the evidence-ranking eval and store them as JSON.

Run once (needs network); afterwards ``run_eval.py`` is fully offline.

    uv run python eval/fetch_fixtures.py              # fetch threads missing from eval/fixtures/
    uv run python eval/fetch_fixtures.py --force      # re-fetch everything in golden.json
    uv run python eval/fetch_fixtures.py 1ntedtv abc  # fetch specific ids (e.g. label candidates)

Thread ids default to the union of ``thread_ids`` in ``eval/golden.json``. Threads
come from the keyless archive backend (Arctic Shift, PullPush fallback), fetched
sequentially with a pause in between to stay polite.

Re-fetching can change which comments are present (the archive re-scores and the
top-``MAX_COMMENTS`` cut moves), which can orphan labels in golden.json;
``run_eval.py`` warns about label ids missing from the fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from pydantic import ValidationError

from reddit_research.core.archive import ArchiveFetcher
from reddit_research.core.config import RedditConfig
from reddit_research.core.errors import RedditError

EVAL_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = EVAL_DIR / "fixtures"
GOLDEN_PATH = EVAL_DIR / "golden.json"

MAX_COMMENTS = 300
SLEEP_SECONDS = 3.0
RETRIES = 3


def golden_thread_ids() -> list[str]:
    if not GOLDEN_PATH.exists():
        return []
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    ids: list[str] = []
    for q in golden["questions"]:
        for tid in q["thread_ids"]:
            if tid not in ids:
                ids.append(tid)
    return ids


def fetch_one(fetcher: ArchiveFetcher, thread_id: str) -> bool:
    for attempt in range(1, RETRIES + 1):
        try:
            thread = fetcher.fetch_thread(thread_id, max_comments=MAX_COMMENTS)
        except ValidationError as exc:
            # Archive data the normalizer cannot parse; retrying will not help.
            print(f"  {thread_id}: unparseable archive data: {exc.errors()[0]}", file=sys.stderr)
            return False
        except (RedditError, httpx.HTTPError) as exc:
            print(f"  {thread_id}: attempt {attempt}/{RETRIES} failed: {exc}", file=sys.stderr)
            time.sleep(SLEEP_SECONDS * 4 * attempt)
            continue
        path = FIXTURES_DIR / f"{thread.id}.json"
        path.write_text(thread.model_dump_json(indent=1) + "\n", encoding="utf-8")
        print(
            f"  {thread.id}: r/{thread.subreddit} {len(thread.comments)} comments "
            f"(of {thread.num_comments}) -- {thread.title[:70]}"
        )
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("thread_ids", nargs="*", help="ids to fetch (default: all in golden.json)")
    parser.add_argument("--force", action="store_true", help="re-fetch fixtures that already exist")
    args = parser.parse_args()

    ids = args.thread_ids or golden_thread_ids()
    if not ids:
        print("No thread ids given and none found in golden.json.", file=sys.stderr)
        return 2
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fetcher = ArchiveFetcher(RedditConfig())
    failed: list[str] = []
    fetched_any = False
    for tid in ids:
        if not args.force and (FIXTURES_DIR / f"{tid}.json").exists():
            print(f"  {tid}: already present, skipping (use --force to refresh)")
            continue
        if fetched_any:
            time.sleep(SLEEP_SECONDS)
        fetched_any = True
        if not fetch_one(fetcher, tid):
            failed.append(tid)
    if failed:
        print(f"FAILED: {' '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
