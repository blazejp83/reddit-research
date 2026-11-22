# Evidence-ranking eval

A small offline harness for one question: **given the right threads, does the
ranker put the comments a careful reader would want near the top?**

```
uv run python eval/run_eval.py
```

It loads stored Reddit threads (`fixtures/`), runs a ranker per question from the
golden set (`golden.json`), and prints per-question and mean scores. No network,
no LLM, deterministic.

## What it measures

Each question has 2-3 threads and a set of labelled comment ids:

- `relevant`, `grade` 2: directly and substantively answers the question
  (first-hand experience, a specific recommendation with reasoning, a
  well-supported dissent). `grade` 1: useful supporting context.
- `junk`: promo spam / astroturf, AutoModerator and bot notices, exact duplicates.

| metric | meaning |
| --- | --- |
| `ndcg@10`, `ndcg@25` | graded ranking quality (gain `2^grade - 1`, ideal = all labels sorted by grade) |
| `recall2@25` | share of the grade-2 comments that appear in the top 25 |
| `junk@25` | number of junk comments in the top 25 (lower is better) |

Evidence is matched to labels by the comment id at the end of its permalink.
Thread-kind evidence and unlabelled comments count as grade 0.

## What it does not measure

- **Search-stage recall.** The threads are fixed; whether search finds them is out of scope.
- **Synthesis quality.** Nothing downstream of the ranked list is evaluated.
- **Absolute quality.** Labels were made by an AI agent reading the threads, not by
  the repo owner, and they are incomplete: busy threads contain many
  near-interchangeable good comments and only ~16-19 per question are labelled, so
  a ranker that surfaces an equally good unlabelled comment gets no credit. Treat
  absolute numbers cautiously; **the signal is the relative change between two
  runs**, ideally consistent across several questions. With 11 questions a
  single-question swing or a mean nDCG delta of a couple of points is noise.
- Junk labels for the astroturf-heavy questions (`office-chair-under-300`,
  `standing-desk-back-pain`) include judgment calls; suspected-but-unclear promo
  was left unlabelled (neither relevant nor junk).

## Comparing two runs

```
uv run python eval/run_eval.py --json eval/before.json     # before your change
# ...change the ranker...
uv run python eval/run_eval.py --compare eval/before.json  # per-question deltas
```

Deltas are marked `+` (better) or `-` (worse); for `junk@25` lower is better.
Checked-in dumps:

- `baseline.json` - working tree at the time the harness was built (already
  included in-progress ranking changes, so not a clean "before").
- `baseline_head_b218588.json` - the ranker as committed at `b218588`.

## Adding a ranker

Add a `(question: str, threads: list[Thread]) -> list[Evidence]` callable to
`RANKERS` in `run_eval.py`, then `--ranker NAME`. Built in: `lexical`
(`rank_evidence` with defaults) and `votes` (top comments by Reddit score,
ignoring the question - a reference floor).

## Refreshing fixtures

```
uv run python eval/fetch_fixtures.py            # fetch threads in golden.json that are missing
uv run python eval/fetch_fixtures.py --force    # re-fetch all of them
```

Threads come from the keyless archive backend, sequentially, up to 300 comments
each. A refresh can change scores and which comments make the 300 cut;
`run_eval.py` warns when labelled ids are missing from the fixtures. Results
before and after a refresh are not comparable - re-create the baseline.

## Adding a question

1. Pick 2-3 threads (ideally 30+ comments each) and fetch them:
   `uv run python eval/fetch_fixtures.py <id> <id>`.
2. Read the comments in `fixtures/<id>.json` and add an entry to `golden.json`
   with `id`, `type`, `question`, `thread_ids`, `relevant`
   (`{"id", "grade", "thread_id", "note"}`) and `junk` (`{"id", "thread_id", "reason"}`).
   Only `id` and `grade` are used for scoring; the rest is for humans.
3. Label on substance, not votes, and **do not copy the ranker's output** - that
   makes the eval circular. Include good answers that share few words with the
   question; leave jokes and suspected promo unlabelled.
4. Re-create the baseline (`--json`), since means are over all questions
   (`--compare` only averages questions present in both runs).
