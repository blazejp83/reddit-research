from reddit_research.core.models import Comment, SearchResult, Thread
from reddit_research.core import ranking
from reddit_research.core.normalize import comment_from_dict


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
    comments_ranked = [e.text for e in ev if e.kind == "comment"]
    # The weather comment shares no query tokens. It is no longer dropped (the
    # thread itself is on-topic, so it inherits some relevance) but it must rank
    # below the comments that actually talk about backups.
    assert "weather" in comments_ranked[-1]
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


def test_well_voted_comment_outranks_low_vote_with_equal_relevance():
    comments = [
        Comment(id="low", body="standing desk helped my back pain", score=1),
        Comment(id="high", body="standing desk fixed my back pain", score=300),
    ]
    thread = Thread(id="t", subreddit="s", title="Desks", url="u", permalink="p", comments=comments)
    ev = ranking.rank_evidence("standing desk back pain", [thread], include_threads=False)
    assert ev[0].reddit_score == 300


DAY = 86400.0


def _t(comments, tid="t", title="Light tactile switches for typing?", selftext="", created_utc=None):
    return Thread(
        id=tid,
        subreddit="s",
        title=title,
        url="u",
        permalink=f"p/{tid}",
        selftext=selftext,
        created_utc=created_utc,
        comments=comments,
    )


def _comment_texts(ev):
    return [e.text for e in ev if e.kind == "comment"]


# -- 1. thread-inherited relevance ------------------------------------------


def test_zero_overlap_answer_survives_in_relevant_thread():
    comments = [Comment(id="a", body="Boba U4T, hands down.", score=40)]
    ev = ranking.rank_evidence("best light tactile switches", [_t(comments)], include_threads=False)
    assert _comment_texts(ev) == ["Boba U4T, hands down."]


def test_zero_overlap_comment_dropped_in_irrelevant_thread():
    comments = [Comment(id="a", body="Boba U4T, hands down.", score=40)]
    thread = _t(comments, title="Weekly photo megathread")
    assert ranking.rank_evidence("best light tactile switches", [thread]) == []


def test_top_level_inherits_more_than_deep_reply():
    comments = [
        Comment(id="a", body="Boba U4T, hands down.", score=5, depth=0),
        Comment(id="b", body="Zealios V2 are great.", score=5, depth=3, parent_id="t1_gone"),
    ]
    ev = ranking.rank_evidence("light tactile switches", [_t(comments)], include_threads=False)
    by_text = {e.text: e for e in ev}
    assert by_text["Boba U4T, hands down."].score > by_text["Zealios V2 are great."].score
    # The gap is bigger than the depth penalty alone would produce.
    assert ranking._inherit_weight(0) > ranking._inherit_weight(1) > ranking._inherit_weight(5) > 0


# -- 2. parent context --------------------------------------------------------


def test_reply_carries_parent_context_and_metadata():
    parent_body = "Do you prefer tactile or clicky switches for typing? " + "x" * 600
    comments = [
        Comment(id="p", body=parent_body, score=3, author="asker", created_utc=100.0),
        Comment(
            id="r",
            parent_id="t1_p",
            body="Honestly go with whites.",
            score=3,
            depth=1,
            author="replier",
            created_utc=200.0,
        ),
        Comment(id="o", parent_id="t1_missing", body="Orphan reply about switches", depth=1),
    ]
    ev = ranking.rank_evidence("tactile switches typing", [_t(comments)], include_threads=False)
    by_text = {e.text: e for e in ev}
    reply = by_text["Honestly go with whites."]
    assert reply.parent_text == parent_body[:500]
    assert (reply.depth, reply.author, reply.created_utc) == (1, "replier", 200.0)
    assert by_text[parent_body].parent_text is None
    assert by_text["Orphan reply about switches"].parent_text is None


def test_parent_text_lifts_reply_with_no_own_overlap():
    # Irrelevant thread title, so nothing is inherited from the thread: the reply
    # can only survive through its parent's wording.
    comments = [
        Comment(id="p", body="Which tactile switches are light enough for typing?", score=2),
        Comment(id="r", parent_id="t1_p", body="Clicky. Try whites instead.", depth=1, score=2),
    ]
    thread = _t(comments, title="Daily chat")
    ev = ranking.rank_evidence("light tactile switches typing", [thread], include_threads=False)
    assert "Clicky. Try whites instead." in _comment_texts(ev)
    q = ranking.Counter(ranking.tokenize("light tactile switches typing"))
    own = ranking._own_overlap(q, "tactile switches", None)
    via_parent = ranking._own_overlap(q, "nope", "tactile switches")
    assert 0 < via_parent < own  # parent words count, but at reduced weight


# -- 3. junk filtering and spam penalty --------------------------------------


def test_comment_from_dict_reads_stickied_and_distinguished():
    c = comment_from_dict({"id": "a", "body": "rules", "stickied": True, "distinguished": "moderator"})
    assert c.stickied is True and c.distinguished == "moderator"
    plain = comment_from_dict({"id": "b", "body": "hi", "distinguished": None})
    assert plain.stickied is False and plain.distinguished is None


def test_junk_comments_are_skipped():
    comments = [
        Comment(id="am", body="Your tactile switches post follows the rules.", author="AutoModerator", score=50),
        Comment(id="b1", body="I am a bot replying about tactile switches.", author="RemindMeBot", score=50),
        Comment(id="b2", body="I am a bot replying about tactile switches too.", author="switch_bot", score=50),
        Comment(
            id="st",
            body="Reminder: tactile switches megathread rules apply.",
            author="a_mod",
            stickied=True,
            distinguished="moderator",
            score=50,
        ),
        Comment(id="sh", body="tactile", score=2, depth=1),  # short reply: noise
        Comment(id="tl", body="Topre", score=1),  # short top-level: a bare recommendation
        Comment(id="ok", body="Holy Pandas", score=120),  # short but well-voted
        Comment(id="hu", body="Tactile switches like Boba U4T are light.", author="abbott_fan", score=4),
    ]
    ev = ranking.rank_evidence("light tactile switches", [_t(comments)], include_threads=False)
    assert sorted(_comment_texts(ev)) == [
        "Holy Pandas",
        "Tactile switches like Boba U4T are light.",
        "Topre",
    ]


def test_spam_penalty_tiers():
    created = 1_000_000.0
    promo = "Great tactile switches at https://shop.example.com/deal"
    late = Comment(id="l", body=promo, score=1, created_utc=created + 90 * DAY)
    early = Comment(id="e", body=promo, score=1, created_utc=created + DAY)
    voted = Comment(id="v", body=promo, score=12, created_utc=created + 90 * DAY)
    assert ranking._spam_penalty(late, created) == ranking.SPAM_LATE_LINK_PENALTY
    assert ranking._spam_penalty(early, created) == ranking.SPAM_LINK_PENALTY
    assert ranking._spam_penalty(voted, created) == 1.0
    assert ranking._spam_penalty(late, None) == ranking.SPAM_LINK_PENALTY  # age unknown
    for url in (
        "https://www.reddit.com/r/x/comments/1/y",
        "https://i.redd.it/abc.png",
        "https://pubmed.ncbi.nlm.nih.gov/123/",
        "https://doi.org/10.1/x",
        "https://en.wikipedia.org/wiki/Desk",
        "https://www.cdc.gov/niosh",
        "http://web.mit.edu/paper",
    ):
        c = Comment(id="s", body=f"see {url} for tactile info", score=1, created_utc=created + 90 * DAY)
        assert ranking._spam_penalty(c, created) == 1.0, url
    # A lookalike host must not pass as reddit.
    fake = Comment(id="f", body="see https://notreddit.com/x", score=0)
    assert ranking._spam_penalty(fake, created) == ranking.SPAM_LINK_PENALTY


def test_late_promo_link_ranks_below_plain_comment():
    created = 1_000_000.0
    comments = [
        Comment(
            id="spam",
            body="Light tactile switches for typing: buy at https://shop.example.com/x",
            score=1,
            created_utc=created + 200 * DAY,
        ),
        Comment(id="real", body="Light tactile ones were fine for me.", score=1, created_utc=created + DAY),
    ]
    ev = ranking.rank_evidence(
        "light tactile switches typing", [_t(comments, created_utc=created)], include_threads=False
    )
    assert [e.text for e in ev][0] == "Light tactile ones were fine for me."
    assert len(ev) == 2  # penalized, not dropped


# -- 4. diversity -------------------------------------------------------------

_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()


def _distinct(i, tid):
    body = f"tactile switches {_WORDS[i]} {tid} variant number {i}"
    return Comment(id=f"{tid}{i}", body=body, score=500 if tid == "big" else 2)


def test_max_per_thread_defaults_and_override():
    big = _t([_distinct(i, "big") for i in range(10)], tid="big")
    small = _t([_distinct(i, "small") for i in range(10)], tid="small")
    ev = ranking.rank_evidence("tactile switches", [big, small], max_evidence_items=10, include_threads=False)
    counts = ranking.Counter(e.thread_id for e in ev)
    assert counts == {"big": 6, "small": 4}  # 60% cap leaves room for the quiet thread

    ev = ranking.rank_evidence(
        "tactile switches", [big, small], max_evidence_items=10, include_threads=False, max_per_thread=2
    )
    assert ranking.Counter(e.thread_id for e in ev) == {"big": 2, "small": 2}

    # A single thread is not capped by default.
    ev = ranking.rank_evidence("tactile switches", [big], max_evidence_items=10, include_threads=False)
    assert len(ev) == 10
    # Tiny limits still allow at least 3 per thread.
    ev = ranking.rank_evidence("tactile switches", [big, small], max_evidence_items=3, include_threads=False)
    assert ranking.Counter(e.thread_id for e in ev) == {"big": 3}


def test_near_duplicates_collapse_keeping_best_ranked():
    base = "I switched to a standing desk last year and my lower back pain mostly went away after a month"
    comments = [
        Comment(id="a", body=base + ".", score=2),
        Comment(id="b", body=base + " honestly!", score=90),
        Comment(id="c", body="My back pain got worse with a standing desk, so I went back to a chair.", score=5),
    ]
    thread = _t(comments, title="Standing desk")
    ev = ranking.rank_evidence("standing desk back pain", [thread], include_threads=False)
    assert len(ev) == 2
    assert ev[0].reddit_score == 90
    assert all(e.reddit_score != 2 for e in ev)


# -- 5. recency ---------------------------------------------------------------


def test_recency_half_life_is_configurable():
    import time

    now = time.time()
    comments = [
        Comment(id="old", body="tactile switches are nice", score=3, created_utc=now - 730 * DAY),
        Comment(id="new", body="tactile switches are okay", score=3, created_utc=now - DAY),
    ]
    thread = _t(comments)

    def gap(**kw):
        ev = ranking.rank_evidence("tactile switches", [thread], include_threads=False, **kw)
        by = {e.created_utc: e.score for e in ev}
        return by[now - DAY] - by[now - 730 * DAY]

    assert gap(recency_half_life_days=30.0) > gap() > gap(recency_half_life_days=36500.0) >= 0
