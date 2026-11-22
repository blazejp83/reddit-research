from reddit_research.core.models import Comment, SearchResult, Thread
from reddit_research.core import ranking


def test_tokenize_drops_stopwords():
    toks = ranking.tokenize("What is the best backup tool")
    assert "best" not in toks  # stopword
    assert "backup" in toks and "tool" in toks


def _thread(comments):
    return Thread(
        id="t1",
        subreddit="selfhosted",
        title="Backup strategies for homelab",
        url="https://reddit.com/r/selfhosted/comments/t1/x",
        permalink="https://reddit.com/r/selfhosted/comments/t1/x",
        selftext="Looking for backup recommendations",
        score=100,
        num_comments=len(comments),
        comments=comments,
    )


def test_rank_evidence_orders_by_relevance():
    comments = [
        Comment(id="c1", body="I use borg for backups, works great", score=50, depth=0),
        Comment(id="c2", body="unrelated chatter about the weather", score=5, depth=0),
        Comment(id="c3", body="restic is a solid backup tool too", score=30, depth=1),
    ]
    ev = ranking.rank_evidence("best backup tool", [_thread(comments)], max_evidence_items=10)
    texts = [e.text for e in ev]
    # The weather comment shares no query tokens and must be excluded.
    assert all("weather" not in t for t in texts)
    assert ev[0].score >= ev[-1].score


def test_rank_evidence_excludes_removed():
    comments = [
        Comment(id="c1", body="[removed]", score=99, depth=0, removed=True),
        Comment(id="c2", body="backup with restic", score=10, depth=0),
    ]
    ev = ranking.rank_evidence("backup", [_thread(comments)], max_evidence_items=10)
    assert all(e.kind != "comment" or "removed" not in e.text for e in ev)
    assert any(e.text == "backup with restic" for e in ev if e.kind == "comment")


def test_rank_threads_dedupes_and_boosts_allowlist():
    results = [
        SearchResult(title="Backup guide", url="https://reddit.com/r/selfhosted/comments/1/a", subreddit="selfhosted", snippet="borg", score=0.9),
        SearchResult(title="Backup guide", url="https://reddit.com/r/other/comments/2/a", subreddit="other", snippet="dup", score=0.5),
    ]
    ranked = ranking.rank_threads("backup guide", results, subreddits=["selfhosted"])
    # Duplicate title collapsed to one.
    assert len(ranked) == 1
    assert ranked[0].subreddit == "selfhosted"


def test_rank_evidence_dedupes_identical_comments():
    text = "It helps with back pain because you switch positions more often."
    comments = [
        Comment(id=f"d{i}", body=text if i % 2 else text.upper() + "!!", score=1) for i in range(3)
    ] + [Comment(id="u", body="Back pain went away after a walking pad.", score=1)]
    thread = Thread(id="t", subreddit="s", title="Standing desk", url="u", permalink="p", comments=comments)
    ev = ranking.rank_evidence("standing desk back pain", [thread], include_threads=False)
    assert len(ev) == 2
