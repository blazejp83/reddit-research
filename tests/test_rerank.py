import json
import time

import httpx
import pytest
import respx

from reddit_research.core.llm import ANTHROPIC_ENDPOINT, LLMError
from reddit_research.core.models import Evidence
from reddit_research.core.rerank import _age, _build_user_prompt, rerank

MODEL = "claude-haiku-5-5"


def _evidence(n: int) -> list[Evidence]:
    return [
        Evidence(
            kind="comment",
            thread_id=f"t{i}",
            thread_title=f"Thread {i}",
            subreddit="coffee",
            text=f"candidate text {i}",
            permalink=f"https://www.reddit.com/r/coffee/comments/t{i}/_/c{i}/",
            score=round(0.9 - i * 0.1, 2),
            reddit_score=i * 10,
        )
        for i in range(1, n + 1)
    ]


def _reply(router: respx.MockRouter, text: str) -> respx.Route:
    return router.post(ANTHROPIC_ENDPOINT).respond(
        200, json={"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}
    )


def _run(text: str, n: int = 5, top_n: int = 3) -> list[Evidence]:
    with respx.mock() as router:
        _reply(router, text)
        return rerank("best grinder?", _evidence(n), api_key="k", model=MODEL, top_n=top_n)


def _ids(items: list[Evidence]) -> list[str]:
    return [ev.thread_id for ev in items]


def test_reorders_per_model_output():
    assert _ids(_run('{"ranked": [4, 1, 5]}')) == ["t4", "t1", "t5"]


def test_tolerates_code_fence_and_prose():
    text = 'Here is my ranking {roughly}:\n```json\n{"ranked": [3, 2, 1]}\n```\nHope that helps.'
    assert _ids(_run(text)) == ["t3", "t2", "t1"]


def test_accepts_numeric_strings():
    assert _ids(_run('{"ranked": ["2", " 5 ", 3.0]}')) == ["t2", "t5", "t3"]


def test_ignores_invalid_and_duplicate_indices():
    text = '{"ranked": [0, 4, 99, 4, -1, "x", null, true, 2.5, 2, 1]}'
    assert _ids(_run(text)) == ["t4", "t2", "t1"]


def test_truncates_to_top_n():
    assert _ids(_run('{"ranked": [5, 4, 3, 2, 1]}', top_n=2)) == ["t5", "t4"]


def test_pads_to_top_n_in_original_order():
    assert _ids(_run('{"ranked": [3]}', top_n=4)) == ["t3", "t1", "t2", "t4"]


def test_top_n_larger_than_candidates_returns_all():
    assert _ids(_run('{"ranked": [2]}', n=3, top_n=10)) == ["t2", "t1", "t3"]


@pytest.mark.parametrize(
    "text",
    ["I could not decide.", '{"ranked": "1, 2"}', '{"order": [1, 2]}', '{"ranked": [1, 2', "[1, 2]"],
)
def test_raises_on_garbage_output(text):
    with pytest.raises(LLMError, match="ranked"):
        _run(text)


def test_raises_when_no_valid_candidate_numbers():
    with pytest.raises(LLMError, match="no valid candidate"):
        _run('{"ranked": [0, 42]}')


def test_truncated_reply_says_so():
    with respx.mock() as router:
        router.post(ANTHROPIC_ENDPOINT).respond(
            200,
            json={"content": [{"type": "text", "text": '{"ranked": [1,'}], "stop_reason": "max_tokens"},
        )
        with pytest.raises(LLMError, match="cut off"):
            rerank("q", _evidence(3), api_key="k", model=MODEL, top_n=2)


def test_raises_on_http_error():
    with respx.mock() as router:
        router.post(ANTHROPIC_ENDPOINT).respond(
            400, json={"error": {"type": "invalid_request_error", "message": "bad model"}}
        )
        with pytest.raises(LLMError, match="HTTP 400"):
            rerank("q", _evidence(3), api_key="k", model=MODEL, top_n=2)


def test_raises_on_network_error():
    with respx.mock() as router:
        router.post(ANTHROPIC_ENDPOINT).mock(side_effect=httpx.ConnectTimeout("slow"))
        with pytest.raises(LLMError, match="ConnectTimeout"):
            rerank("q", _evidence(3), api_key="k", model=MODEL, top_n=2)


def test_missing_api_key_raises_without_http():
    with respx.mock(assert_all_called=False) as router:
        route = _reply(router, '{"ranked": [1]}')
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            rerank("q", _evidence(3), api_key=None, model=MODEL, top_n=2)
        assert not route.called


def test_no_http_call_for_single_item_or_zero_top_n():
    with respx.mock(assert_all_called=False) as router:
        route = _reply(router, '{"ranked": [1]}')
        one = _evidence(1)
        assert rerank("q", one, api_key="k", model=MODEL, top_n=5) == one
        assert rerank("q", [], api_key="k", model=MODEL, top_n=5) == []
        assert rerank("q", _evidence(4), api_key="k", model=MODEL, top_n=0) == []
        assert rerank("q", _evidence(4), api_key="k", model=MODEL, top_n=-2) == []
        assert not route.called


def test_scores_decrease_monotonically_within_unit_interval():
    out = _run('{"ranked": [5, 1]}', n=8, top_n=8)
    scores = [ev.score for ev in out]
    assert scores[0] == 1.0
    assert all(a > b for a, b in zip(scores, scores[1:]))
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_does_not_mutate_inputs():
    evidence = _evidence(4)
    before = [ev.model_copy() for ev in evidence]
    with respx.mock() as router:
        _reply(router, '{"ranked": [4, 3]}')
        out = rerank("q", evidence, api_key="k", model=MODEL, top_n=3)
    assert evidence == before
    assert all(o is not e for o in out for e in evidence)


def test_request_payload():
    with respx.mock() as router:
        route = _reply(router, '{"ranked": [2, 1]}')
        rerank("best grinder?", _evidence(3), api_key="k", model=MODEL, top_n=2, timeout=5.0)
    payload = json.loads(route.calls.last.request.content)
    assert payload["model"] == MODEL
    assert payload["output_config"] == {"effort": "low"}
    assert payload["max_tokens"] == 4000
    assert "reranking step" in payload["system"]
    user = payload["messages"][0]["content"]
    assert "Question: best grinder?" in user
    assert "Select at most 2 of the 3 candidates" in user
    assert "[3] comment | r/coffee | 30 votes" in user
    assert "candidate text 3" in user


def test_prompt_truncates_text_and_shows_optional_fields():
    class RichEvidence(Evidence):
        created_utc: float | None = None
        parent_text: str | None = None

    rich = RichEvidence(
        kind="comment",
        thread_id="t1",
        thread_title="Thread",
        subreddit="coffee",
        text="word " * 400,
        permalink="p",
        reddit_score=7,
        created_utc=time.time() - 800 * 86400,
        parent_text="what   about\nhand grinders? " + "x" * 500,
    )
    prompt = _build_user_prompt("q", [rich, _evidence(1)[0]], top_n=1)
    first, second = prompt.split("[2] ")
    assert "[1] comment | r/coffee | 7 votes | 2 years old" in first
    assert "replying to: what about hand grinders?" in first
    assert len(first) < 1200
    assert "old" not in second and "replying to" not in second


def test_age_buckets():
    now = 1_000_000_000.0
    assert _age(None, now) is None
    assert _age(now - 3600, now) == "under a day old"
    assert _age(now - 86400, now) == "1 day old"
    assert _age(now - 90 * 86400, now) == "2 months old"
    assert _age(now - 5 * 365.25 * 86400, now) == "5 years old"
