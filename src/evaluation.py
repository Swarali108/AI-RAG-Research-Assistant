"""Evaluation harness (#4).

Two tiers, kept separate on purpose so the cheap one stays free:

1. Retrieval metrics — Hit Rate, MRR, Recall@K, Precision@K. Pure functions,
   no LLM, no cost. A retrieved chunk counts as relevant if it contains one of
   the case's expected key phrases (a lightweight, dependency-free proxy for a
   labeled relevance set).

2. Answer-quality metrics — Faithfulness and Answer Relevance. These need an
   LLM judge, so they are OPT-IN: the caller injects a ``judge_fn`` and is
   charged tokens only when they ask for them. This keeps the project budget-safe.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Sequence


def _contains(text: str, phrases: Sequence[str]) -> List[str]:
    low = (text or "").lower()
    return [p for p in phrases if p.lower() in low]


def query_retrieval_metrics(
    retrieved_texts: Sequence[str], expected_phrases: Sequence[str], k: Optional[int] = None
) -> Dict[str, float]:
    """Per-query retrieval metrics for one labeled case."""
    top = list(retrieved_texts[: (k or len(retrieved_texts))])
    rel_flags = [bool(_contains(txt, expected_phrases)) for txt in top]

    hit = 1.0 if any(rel_flags) else 0.0

    mrr = 0.0
    for rank, flag in enumerate(rel_flags, start=1):
        if flag:
            mrr = 1.0 / rank
            break

    found = set()
    for txt in top:
        for phrase in _contains(txt, expected_phrases):
            found.add(phrase.lower())
    recall = len(found) / len(expected_phrases) if expected_phrases else 0.0
    precision = (sum(rel_flags) / len(top)) if top else 0.0

    return {
        "hit": hit,
        "mrr": round(mrr, 3),
        "recall_at_k": round(recall, 3),
        "precision_at_k": round(precision, 3),
    }


def aggregate_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-query metrics into dataset-level scores."""
    if not per_query:
        return {"hit_rate": 0.0, "mrr": 0.0, "recall_at_k": 0.0, "precision_at_k": 0.0, "cases": 0}

    n = len(per_query)
    avg = lambda key: round(sum(m[key] for m in per_query) / n, 3)
    return {
        "hit_rate": avg("hit"),
        "mrr": avg("mrr"),
        "recall_at_k": avg("recall_at_k"),
        "precision_at_k": avg("precision_at_k"),
        "cases": n,
    }


# --- Opt-in LLM-judge metrics (cost tokens; off by default) ---

_FAITHFULNESS_PROMPT = """You are a strict evaluator. Given the CONTEXT and an ANSWER, rate how well every
claim in the answer is supported by the context. Reply with ONLY a number from 0.0 (unsupported /
hallucinated) to 1.0 (fully grounded).

CONTEXT:
{context}

ANSWER:
{answer}

Faithfulness (0.0-1.0):"""

_RELEVANCE_PROMPT = """You are a strict evaluator. Rate how directly the ANSWER addresses the QUESTION.
Reply with ONLY a number from 0.0 (off-topic) to 1.0 (fully answers the question).

QUESTION:
{question}

ANSWER:
{answer}

Answer relevance (0.0-1.0):"""


def _parse_score(text: str) -> float:
    match = re.search(r"\d+(\.\d+)?", text or "")
    if not match:
        return 0.0
    value = float(match.group(0))
    if value > 1.0:  # tolerate a 0-10 or 0-100 reply
        value = value / 100 if value > 10 else value / 10
    return max(0.0, min(1.0, value))


def judge_faithfulness(answer: str, context: str, judge_fn: Callable[[str], str]) -> float:
    return _parse_score(judge_fn(_FAITHFULNESS_PROMPT.format(context=context, answer=answer)))


def judge_answer_relevance(question: str, answer: str, judge_fn: Callable[[str], str]) -> float:
    return _parse_score(judge_fn(_RELEVANCE_PROMPT.format(question=question, answer=answer)))
