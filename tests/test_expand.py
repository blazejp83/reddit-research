import json

import httpx
import pytest
import respx

from reddit_research.core.expand import expand_queries
from reddit_research.core.llm import ANTHROPIC_ENDPOINT, LLMError

QUESTION = "What backup tool should I use for my homelab?"


def _reply(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"},
    )


def _expand(n: int = 3, question: str = QUESTION) -> list[str]:
    return expand_queries(question, api_key="sk-test", model="test-model", n=n)


@respx.mock
def test_parses_plain_json():
    respx.post(ANTHROPIC_ENDPOINT).mock(
        return_value=_reply('{"queries": ["homelab backup software", "borg vs restic"]}')
    )
    assert _expand() == ["homelab backup software", "borg vs restic"]


@respx.mock
def test_parses_fenced_json_with_surrounding_prose():
    text = 'Here you go:\n```json\n{"queries": ["best homelab backup", "restic review"]}\n```\nHope it helps.'
    respx.post(ANTHROPIC_ENDPOINT).mock(return_value=_reply(text))
    assert _expand() == ["best homelab backup", "restic review"]


@respx.mock
def test_parses_unfenced_json_with_stray_prose():
    text = 'Sure! {"queries": ["best homelab backup"]} Let me know.'
    respx.post(ANTHROPIC_ENDPOINT).mock(return_value=_reply(text))
    assert _expand() == ["best homelab backup"]


@respx.mock
def test_dedupes_against_original_and_each_other():
    queries = [
        "  what backup tool should I use for my HOMELAB?  ",
        "borg vs restic",
        "Borg  vs Restic",
        "",
        "   ",
        42,
        "homelab backup software",
    ]
    respx.post(ANTHROPIC_ENDPOINT).mock(return_value=_reply(json.dumps({"queries": queries})))
    assert _expand(n=5) == ["borg vs restic", "homelab backup software"]


@respx.mock
def test_strips_site_operators_and_quotes():
    queries = ["site:reddit.com best homelab backup", 'site:reddit.com/r/homelab "borg" vs restic', "site:reddit.com"]
    respx.post(ANTHROPIC_ENDPOINT).mock(return_value=_reply(json.dumps({"queries": queries})))
    assert _expand() == ["best homelab backup", "borg vs restic"]


@respx.mock
def test_caps_at_n():
    queries = [f"query {i}" for i in range(6)]
    respx.post(ANTHROPIC_ENDPOINT).mock(return_value=_reply(json.dumps({"queries": queries})))
    assert _expand(n=2) == ["query 0", "query 1"]


@respx.mock
def test_n_zero_makes_no_http_call():
    route = respx.post(ANTHROPIC_ENDPOINT).mock(return_value=_reply('{"queries": ["x"]}'))
    assert _expand(n=0) == []
    assert _expand(n=-1) == []
    assert not route.called


@respx.mock
@pytest.mark.parametrize(
    "text",
    ["I cannot help with that.", '{"queries": []}', '{"queries": "borg"}', '{"other": ["x"]}', '{"queries": ["bo'],
)
def test_raises_on_unusable_reply(text):
    respx.post(ANTHROPIC_ENDPOINT).mock(return_value=_reply(text))
    with pytest.raises(LLMError):
        _expand()


@respx.mock
def test_raises_when_only_the_original_comes_back():
    respx.post(ANTHROPIC_ENDPOINT).mock(return_value=_reply(json.dumps({"queries": [QUESTION]})))
    with pytest.raises(LLMError):
        _expand()


@respx.mock
def test_raises_on_http_error():
    respx.post(ANTHROPIC_ENDPOINT).mock(
        return_value=httpx.Response(
            529, json={"error": {"type": "overloaded_error", "message": "Overloaded"}}
        )
    )
    with pytest.raises(LLMError, match="529"):
        _expand()


def test_raises_without_api_key():
    with pytest.raises(LLMError):
        expand_queries(QUESTION, api_key=None, model="test-model", n=3)


@respx.mock
def test_payload_carries_model_effort_and_question():
    route = respx.post(ANTHROPIC_ENDPOINT).mock(return_value=_reply('{"queries": ["borg vs restic"]}'))
    _expand(n=4)
    payload = json.loads(route.calls.last.request.content)
    assert payload["model"] == "test-model"
    assert payload["output_config"]["effort"] == "low"
    assert payload["max_tokens"] == 2000
    assert QUESTION in payload["messages"][0]["content"]
    assert "4" in payload["messages"][0]["content"]
    assert '{"queries"' in payload["system"]
