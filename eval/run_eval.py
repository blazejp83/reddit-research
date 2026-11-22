"""Offline evaluation of the evidence-ranking stage against a hand-labelled golden set.

    uv run python eval/run_eval.py                         # default "lexical" ranker
    uv run python eval/run_eval.py --json eval/after.json  # save results
    uv run python eval/run_eval.py --compare eval/baseline.json

Loads the thread fixtures in ``eval/fixtures/`` and the labels in
``eval/golden.json``, runs a ranker per question and reports nDCG@10, nDCG@25,
recall@25 of grade-2 comments and junk@25. See ``eval/README.md``.

To evaluate another ranker, add a ``(question, threads) -> list[Evidence]``
callable to ``RANKERS`` and pass ``--ranker NAME``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path

from reddit_research.core.models import Evidence, Thread

EVAL_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = EVAL_DIR / "fixtures"
GOLDEN_PATH = EVAL_DIR / "golden.json"

TOP_K = 25
METRICS = ("ndcg@10", "ndcg@25", "recall2@25", "junk@25")
# For --compare: +1 when higher is better, -1 when lower is better.
DIRECTION = {"ndcg@10": 1, "ndcg@25": 1, "recall2@25": 1, "junk@25": -1}

Ranker = Callable[[str, list[Thread]], list[Evidence]]

# --------------------------------------------------------------------------- rankers


def lexical_ranker(question: str, threads: list[Thread]) -> list[Evidence]:
    """The production ranker with its default settings."""
    from reddit_research.core.ranking import rank_evidence

    return rank_evidence(question, threads, max_evidence_items=TOP_K)


def votes_ranker(question: str, threads: list[Thread]) -> list[Evidence]:
    """Reference floor: ignore the question, take the highest-voted comments."""
    items = [
        Evidence(
            kind="comment",
            thread_id=t.id,
            thread_title=t.title,
            subreddit=t.subreddit,
            text=c.body[:2000],
            permalink=c.permalink or t.permalink,
            score=0.0,
            reddit_score=c.score,
        )
        for t in threads
        for c in t.comments
        if not c.removed and c.body
    ]
    items.sort(key=lambda e: e.reddit_score, reverse=True)
    return items[:TOP_K]


RANKERS: dict[str, Ranker] = {
    "lexical": lexical_ranker,
    "votes": votes_ranker,
}

# --------------------------------------------------------------------------- loading


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def load_thread(thread_id: str) -> Thread:
    path = FIXTURES_DIR / f"{thread_id}.json"
    if not path.exists():
        raise SystemExit(
            f"Missing fixture {path}. Run: uv run python eval/fetch_fixtures.py {thread_id}"
        )
    return Thread.model_validate_json(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- matching

_COMMENT_ID_RE = re.compile(r"/comments/[^/]+/[^/]*/([A-Za-z0-9]+)/?(?:[?#].*)?$")


def comment_id_from_permalink(permalink: str) -> str | None:
    """``.../comments/<thread>/<slug>/<comment_id>/`` -> ``<comment_id>`` (else None)."""
    m = _COMMENT_ID_RE.search(permalink or "")
    return m.group(1) if m else None


def evidence_comment_id(ev: Evidence, threads: dict[str, Thread]) -> str | None:
    """Comment id an evidence item refers to; None for thread-kind items."""
    if ev.kind != "comment":
        return None
    cid = comment_id_from_permalink(ev.permalink)
    if cid:
        return cid
    # Comment without its own permalink: fall back to matching the text.
    thread = threads.get(ev.thread_id)
    for c in thread.comments if thread else []:
        if c.body and (c.body == ev.text or c.body[: len(ev.text)] == ev.text):
            return c.id
    return None


# --------------------------------------------------------------------------- metrics


def _dcg(gains: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg(gains: list[int], all_grades: list[int], k: int) -> float:
    ideal = _dcg(sorted(all_grades, reverse=True)[:k])
    return _dcg(gains[:k]) / ideal if ideal else 0.0


def score_question(question: dict, ranked_ids: list[str | None]) -> dict[str, float]:
    """Metrics for one question given the ranked comment ids (None = unlabelled thread item)."""
    grades = {r["id"]: int(r["grade"]) for r in question["relevant"]}
    junk = {j["id"] for j in question.get("junk", [])}
    ranked_ids = ranked_ids[:TOP_K]

    gains: list[int] = []
    seen: set[str] = set()
    for cid in ranked_ids:
        # A comment returned twice only earns its gain once.
        gains.append(grades.get(cid, 0) if cid and cid not in seen else 0)
        if cid:
            seen.add(cid)

    grade2 = {cid for cid, g in grades.items() if g == 2}
    return {
        "ndcg@10": ndcg(gains, list(grades.values()), 10),
        "ndcg@25": ndcg(gains, list(grades.values()), TOP_K),
        "recall2@25": len(grade2 & seen) / len(grade2) if grade2 else 0.0,
        "junk@25": float(len(junk & seen)),
    }


def evaluate(ranker: Ranker, questions: list[dict]) -> dict:
    per_question: dict[str, dict] = {}
    for q in questions:
        threads = [load_thread(tid) for tid in q["thread_ids"]]
        by_id = {t.id: t for t in threads}
        present = {c.id for t in threads for c in t.comments if not c.removed}
        missing = [r["id"] for r in q["relevant"] if r["id"] not in present]
        if missing:
            print(
                f"warning: {q['id']}: {len(missing)} labelled comment(s) not in fixtures "
                f"({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''})",
                file=sys.stderr,
            )
        evidence = ranker(q["question"], threads)[:TOP_K]
        ranked_ids = [evidence_comment_id(ev, by_id) for ev in evidence]
        metrics = score_question(q, ranked_ids)
        per_question[q["id"]] = {
            "type": q.get("type", ""),
            "returned": len(evidence),
            **{m: round(v, 4) for m, v in metrics.items()},
            "ranked_ids": ranked_ids,
        }
    n = len(per_question) or 1
    mean = {m: round(sum(r[m] for r in per_question.values()) / n, 4) for m in METRICS}
    return {"questions": per_question, "mean": mean}


# --------------------------------------------------------------------------- output


def _fmt(metric: str, value: float) -> str:
    return f"{value:.2f}" if metric == "junk@25" else f"{value:.3f}"


def print_table(result: dict, ranker_name: str) -> None:
    rows = result["questions"]
    width = max([len(qid) for qid in rows] + [len("MEAN")])
    header = f"{'question':<{width}}  {'type':<15}" + "".join(f"{m:>12}" for m in METRICS)
    print(f"ranker: {ranker_name}   questions: {len(rows)}   top-k: {TOP_K}")
    print(header)
    print("-" * len(header))
    for qid, r in rows.items():
        print(f"{qid:<{width}}  {r['type']:<15}" + "".join(f"{_fmt(m, r[m]):>12}" for m in METRICS))
    print("-" * len(header))
    print(f"{'MEAN':<{width}}  {'':<15}" + "".join(f"{_fmt(m, result['mean'][m]):>12}" for m in METRICS))


def print_compare(result: dict, previous: dict, previous_path: str) -> None:
    def delta(metric: str, new: float, old: float) -> str:
        d = new - old
        if abs(d) < 5e-4:
            return f"{'=':>12}"
        mark = "+" if d * DIRECTION[metric] > 0 else "-"  # + better, - worse
        text = f"{d:+.2f}" if metric == "junk@25" else f"{d:+.3f}"
        return f"{text + ' ' + mark:>12}"

    rows = result["questions"]
    old_rows = previous.get("questions", {})
    width = max([len(qid) for qid in rows] + [len("MEAN")])
    header = f"{'question':<{width}}" + "".join(f"{m:>12}" for m in METRICS)
    print()
    print(
        f"delta vs {previous_path} (ranker: {previous.get('ranker', '?')}); "
        "trailing + = better, - = worse"
    )
    print(header)
    print("-" * len(header))
    shared = [qid for qid in rows if qid in old_rows]
    for qid in rows:
        if qid not in old_rows:
            print(f"{qid:<{width}}" + f"{'(new question)':>{12 * len(METRICS)}}")
            continue
        print(f"{qid:<{width}}" + "".join(delta(m, rows[qid][m], old_rows[qid][m]) for m in METRICS))
    print("-" * len(header))
    if shared:
        # Mean over the shared questions only, so added/removed questions do not skew it.
        cells = []
        for m in METRICS:
            new = sum(rows[q][m] for q in shared) / len(shared)
            old = sum(old_rows[q][m] for q in shared) / len(shared)
            cells.append(delta(m, new, old))
        print(f"{'MEAN':<{width}}" + "".join(cells))
    dropped = [qid for qid in old_rows if qid not in rows]
    if len(shared) != len(rows) or dropped:
        print(f"(mean delta over the {len(shared)} shared questions; golden set changed since the dump)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ranker", default="lexical", choices=sorted(RANKERS))
    parser.add_argument("--json", metavar="PATH", help="write results to PATH")
    parser.add_argument("--compare", metavar="PATH", help="print deltas against a previous --json dump")
    parser.add_argument("--question", metavar="ID", action="append", help="only run these question ids")
    args = parser.parse_args()

    questions = load_golden()
    if args.question:
        unknown = set(args.question) - {q["id"] for q in questions}
        if unknown:
            parser.error(f"unknown question id(s): {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q["id"] in args.question]

    result = {"ranker": args.ranker, "top_k": TOP_K, **evaluate(RANKERS[args.ranker], questions)}
    print_table(result, args.ranker)

    if args.compare:
        previous = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print_compare(result, previous, args.compare)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
