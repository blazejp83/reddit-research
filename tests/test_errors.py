from reddit_research.core.errors import redact, warning


def test_redact_env_secret(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "supersecretvalue123")
    text = "request failed with key supersecretvalue123 in header"
    assert "supersecretvalue123" not in redact(text)
    assert "***redacted***" in redact(text)


def test_redact_bearer_token():
    text = "Authorization: Bearer abcdEFGH12345678"
    out = redact(text)
    assert "abcdEFGH12345678" not in out


def test_redact_short_env_value_not_replaced(monkeypatch):
    # Very short secret values must not blanket-replace common substrings.
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "ab")
    assert redact("this is about apples") == "this is about apples"


def test_warning_format():
    w = warning("too_few_results", "only 1 found")
    assert w.startswith("[too_few_results]")
