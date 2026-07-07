from reddit_research.core.cache import Cache
from reddit_research.core.config import load_config
from reddit_research.core.models import Comment, SearchResult, Thread
from reddit_research.core.orchestrator import Orchestrator


class FakeProvider:
    name = "fake"

    def search(self, query, limit=10, recency_days=None):
        return [
            SearchResult(
                title="Best homelab backups",
                url="https://www.reddit.com/r/selfhosted/comments/abc/best",
                subreddit="selfhosted",
                snippet="borg restic backup",
                source="fake",
                score=0.9,
            )
        ]


class FakeFetcher:
    def fetch_thread(self, url_or_id, max_comments=None, comment_sort=None, include_more_comments=True):
        return Thread(
            id="abc",
            subreddit="selfhosted",
            title="Best homelab backups",
            url=url_or_id,
            permalink="https://www.reddit.com/r/selfhosted/comments/abc/best",
            selftext="what do you use for backups?",
            score=120,
            num_comments=2,
            comments=[
                Comment(id="c1", body="I use borg, dedup is great for backups", score=45, depth=0,
                        permalink="https://www.reddit.com/r/selfhosted/comments/abc/best/c1"),
                Comment(id="c2", body="restic to b2 works well", score=30, depth=0),
            ],
        )


def _orch(tmp_path):
    cfg = load_config()
    cfg.cache.path = tmp_path / "c.sqlite"
    return Orchestrator(cfg, cache=Cache(cfg.cache.path), fetcher=FakeFetcher(), provider=FakeProvider())


def test_evidence_pipeline_end_to_end(tmp_path):
    orch = _orch(tmp_path)
    result = orch.evidence("what backup tool do people recommend?", subreddits=["selfhosted"])
    assert result.evidence, "expected some evidence"
    assert result.metadata.threads_fetched == 1
    assert result.metadata.comments_fetched == 2
    assert all(e.permalink for e in result.evidence)
    orch.close()


def test_fetch_uses_cache_on_second_call(tmp_path):
    orch = _orch(tmp_path)
    m1 = orch.evidence("backups", subreddits=["selfhosted"])
    m2 = orch.evidence("backups", subreddits=["selfhosted"])
    # Second run should register cache hits (search + thread).
    assert m2.metadata.cache_hits >= 1
    orch.close()


def test_answer_without_synthesis_warns(tmp_path):
    orch = _orch(tmp_path)
    result = orch.answer("backups", subreddits=["selfhosted"], include_answer=True)
    assert result.answer is None
    assert any("llm_synthesis_unavailable" in w for w in result.metadata.warnings)
    orch.close()
