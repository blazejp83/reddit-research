"""Optional, provider-agnostic answer synthesis.

Synthesis is deliberately decoupled from retrieval: it takes ranked evidence and
produces a cited answer in the spec's preferred shape (short answer / agreement
/ disagreements / caveats / sources). It never asserts Reddit comments as fact
and always cites permalinks. If no provider is configured it returns ``None`` so
callers can fall back to returning evidence only.
"""

from __future__ import annotations

from .config import SynthConfig
from .errors import redact
from .llm import LLMError, complete
from .models import Evidence

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
# Room for the full structured answer plus a Sources list of ~25 permalinks.
MAX_OUTPUT_TOKENS = 8000

SYSTEM_PROMPT = """You synthesize answers from Reddit discussion evidence for a personal research tool.

Rules:
- Treat Reddit comments as opinions and anecdotes, NOT verified facts.
- Distinguish broad consensus from isolated anecdotes.
- Surface notable disagreement and uncertainty.
- Note when evidence is thin, outdated, or specific to one subreddit.
- Cite claims using the bracketed source numbers provided, e.g. [3].

Output this structure with Markdown headings:
## Short answer
## What Reddit users seem to agree on
## Notable disagreements
## Caveats
## Sources
(Under Sources, list each cited [n] with its permalink.)"""


def _build_user_prompt(question: str, evidence: list[Evidence]) -> str:
    lines = [f"Question: {question}", "", "Evidence:"]
    for i, ev in enumerate(evidence, start=1):
        header = f"[{i}] r/{ev.subreddit} — {ev.kind} (reddit score {ev.reddit_score})"
        lines.append(header)
        lines.append(f"    permalink: {ev.permalink}")
        lines.append(f"    text: {ev.text.strip()[:1200]}")
        lines.append("")
    return "\n".join(lines)


def synthesize(
    question: str,
    evidence: list[Evidence],
    config: SynthConfig,
    timeout: float = 60.0,
) -> str | None:
    """Return a synthesized cited answer, or ``None`` if synthesis is disabled.

    Returns ``None`` only when no provider is configured (synthesis off by
    choice). Any other reason synthesis can't run -- unsupported provider,
    missing key, API/network failure -- raises :class:`SynthesisError` with a
    redacted, specific reason so the caller can surface it as a warning.
    """

    provider = (config.provider or "").lower()
    if not provider:
        return None
    if not evidence:
        raise SynthesisError("no evidence to synthesize from")
    if provider != "anthropic":
        raise SynthesisError(f"unsupported provider {provider!r} (only 'anthropic' is supported)")
    if not config.api_key:
        raise SynthesisError("provider 'anthropic' is set but ANTHROPIC_API_KEY is empty")

    model = config.model or DEFAULT_ANTHROPIC_MODEL
    try:
        text, truncated = complete(
            api_key=config.api_key,
            model=model,
            system=SYSTEM_PROMPT,
            user=_build_user_prompt(question, evidence),
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=timeout,
        )
    except LLMError as exc:
        raise SynthesisError(str(exc)) from exc
    if truncated:
        text += "\n\n_[Answer truncated: hit the output token limit.]_"
    return text


class SynthesisError(Exception):
    """Synthesis was enabled but could not produce an answer; message is redacted."""

    def __init__(self, message: str) -> None:
        super().__init__(redact(message))
