"""LLM query expansion for search recall.

A single phrasing of a question only matches the threads whose titles happen to
share its words. Asking a model for a few alternative phrasings, in the style
people actually title Reddit posts, lets the search stage find threads the
original wording misses. Expansion is strictly optional: every failure raises
:class:`LLMError` so the caller can keep the single original query and attach a
warning.
"""

from __future__ import annotations

import json
import re

from .llm import LLMError, complete

EXPAND_SYSTEM = """\
You write search queries that find Reddit threads relevant to a user's question.

Your queries are sent to a web search engine that is already restricted to \
reddit.com, alongside the user's original question. The engine mostly matches \
query words against thread titles, so a single phrasing only finds threads \
titled with the same words. Your job is to widen recall: produce alternative \
queries that surface relevant threads the original wording would miss.

Write each query the way it would work in a search box: short (roughly 3 to 8 \
words), keyword-style, lowercase, without filler words or punctuation. Cover \
different angles rather than rewording one angle, for example:
- a bare keyword form of the question
- the way a Redditor would title a post asking for recommendations or sharing \
experience ("best X for Y", "X recommendations", "anyone using X")
- a comparison or verdict form ("X vs Y", "is X worth it", "X review")
- common synonyms, or the specific product, brand or category terms people \
use for the same thing

Stay on the user's topic and keep every constraint that changes which answer \
is right (budget, platform, location, year, use case). Do not add facts, \
product names or constraints that the question does not imply.

Because the search is already scoped to Reddit, do not write "reddit", site: \
operators, quotation marks or other search operators, and only name a \
subreddit if the question itself names it. Each query must differ meaningfully \
from the others and from the original question.

Reply with a JSON object only, no prose or code fences, in this shape:
{"queries": ["first query", "second query"]}"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SITE_RE = re.compile(r"\bsite:\S*", re.IGNORECASE)


def expand_queries(
    question: str,
    *,
    api_key: str | None,
    model: str,
    n: int,
    timeout: float = 20.0,
) -> list[str]:
    """Return up to ``n`` alternative search queries for ``question``.

    The original question is never included; the caller searches it anyway.
    Raises :class:`LLMError` when the call fails or yields nothing usable, so
    the caller can fall back to the original query alone.
    """

    if n <= 0:
        return []
    question = question.strip()
    user = (
        f"Question: {question}\n\n"
        f"Write {n} alternative search {'query' if n == 1 else 'queries'}."
    )
    text, _truncated = complete(
        api_key=api_key,
        model=model,
        system=EXPAND_SYSTEM,
        user=user,
        max_tokens=2000,
        effort="low",
        timeout=timeout,
    )
    queries = _clean(_parse_queries(text), question)
    if not queries:
        raise LLMError("query expansion returned no usable queries")
    return queries[:n]


def _parse_queries(text: str) -> list:
    """Pull the ``queries`` list out of a reply, tolerating fences and prose."""

    candidates = [text]
    candidates.extend(m.group(1) for m in _FENCE_RE.finditer(text))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = data.get("queries")
        if isinstance(data, list):
            return data
    return []


def _clean(raw: list, question: str) -> list[str]:
    """Normalize queries and drop empties and duplicates (incl. the original).

    ``site:`` tokens and quotes are removed because the planner adds its own
    ``site:`` scope and exact-phrase quoting defeats the point of widening.
    """

    seen = {_norm(question)}
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        query = " ".join(_SITE_RE.sub(" ", item).replace('"', " ").split())
        key = _norm(query)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(query)
    return out


def _norm(query: str) -> str:
    return " ".join(query.casefold().split())
