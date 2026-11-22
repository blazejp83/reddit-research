import httpx
import respx

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


def test_answer_synthesis_failure_reason_in_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secretvalue")
    orch = _orch(tmp_path)
    orch.config.synth.provider = "anthropic"
    orch.config.synth.api_key = "sk-test-secretvalue"
    with respx.mock() as router:
        router.post("https://api.anthropic.com/v1/messages").respond(
            404, json={"type": "error", "error": {"type": "not_found_error", "message": "model: nope"}}
        )
        result = orch.answer("backups", subreddits=["selfhosted"], include_answer=True)
    assert result.answer is None
    warn = next(w for w in result.metadata.warnings if "llm_synthesis_unavailable" in w)
    assert "HTTP 404" in warn and "not_found_error" in warn
    assert "sk-test-secretvalue" not in warn
    orch.close()


def _llm_orch(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secretvalue")
    orch = _orch(tmp_path)
    orch.config.synth.provider = "anthropic"
    orch.config.synth.api_key = "sk-test-secretvalue"
    return orch


def _reply(text):
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


def test_llm_pipeline_expands_reranks_and_synthesizes(tmp_path, monkeypatch):
    orch = _llm_orch(tmp_path, monkeypatch)
    orch.config.synth.max_evidence = 2
    replies = iter(
        [
            _reply('{"queries": ["homelab backup tool", "borg vs restic"]}'),
            _reply('{"ranked": [3, 1]}'),
            _reply("## Short answer\nUse borg [1]."),
        ]
    )
    with respx.mock() as router:
        route = router.post("https://api.anthropic.com/v1/messages").mock(
            side_effect=lambda request: httpx.Response(200, json=next(replies))
        )
        result = orch.answer("backups", include_answer=True)
    assert route.call_count == 3
    assert len(result.metadata.search_queries) == 3  # original + 2 variants
    assert result.answer.startswith("## Short answer")
    assert len(result.evidence) == 2  # synth.max_evidence drives the list size
    assert result.evidence[0].score > result.evidence[1].score
    assert not any("llm_" in w or "expansion" in w for w in result.metadata.warnings)
    orch.close()


def test_llm_failures_degrade_to_keyword_pipeline(tmp_path, monkeypatch):
    orch = _llm_orch(tmp_path, monkeypatch)
    with respx.mock() as router:
        router.post("https://api.anthropic.com/v1/messages").respond(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
        )
        result = orch.evidence("backups")
    assert result.evidence  # keyword ranking still returned
    codes = " ".join(result.metadata.warnings)
    assert "query_expansion_unavailable" in codes and "llm_rerank_unavailable" in codes
    assert "sk-test-secretvalue" not in codes
    orch.close()


def test_rerank_model_off_skips_llm_calls(tmp_path, monkeypatch):
    orch = _llm_orch(tmp_path, monkeypatch)
    orch.config.rerank.model = None
    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.anthropic.com/v1/messages").respond(200, json=_reply("x"))
        result = orch.evidence("backups")
    assert route.call_count == 0
    assert result.evidence
    assert not any("llm_" in w or "expansion" in w for w in result.metadata.warnings)
    orch.close()
