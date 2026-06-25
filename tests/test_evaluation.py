"""Tests for the evaluation harness (#4). Retrieval metrics are pure; the
LLM-judge functions are tested with a stub judge (no API calls)."""

from src.evaluation import (
    _parse_score,
    aggregate_metrics,
    judge_answer_relevance,
    judge_faithfulness,
    query_retrieval_metrics,
)


def test_retrieval_metrics_perfect_first_hit():
    retrieved = ["machine learning powers modern AI", "unrelated text", "more filler"]
    m = query_retrieval_metrics(retrieved, ["machine learning"], k=3)
    assert m["hit"] == 1.0
    assert m["mrr"] == 1.0  # relevant chunk is rank 1
    assert m["recall_at_k"] == 1.0


def test_retrieval_metrics_second_rank_mrr():
    retrieved = ["irrelevant", "deep learning models learn patterns"]
    m = query_retrieval_metrics(retrieved, ["deep learning", "patterns"], k=2)
    assert m["mrr"] == 0.5  # first relevant at rank 2
    assert m["recall_at_k"] == 1.0  # both phrases present in the relevant chunk


def test_retrieval_metrics_miss():
    m = query_retrieval_metrics(["nothing relevant here"], ["quantum entanglement"], k=1)
    assert m["hit"] == 0.0
    assert m["mrr"] == 0.0
    assert m["recall_at_k"] == 0.0


def test_aggregate_metrics_averages():
    agg = aggregate_metrics([
        {"hit": 1.0, "mrr": 1.0, "recall_at_k": 1.0, "precision_at_k": 0.5},
        {"hit": 0.0, "mrr": 0.0, "recall_at_k": 0.0, "precision_at_k": 0.0},
    ])
    assert agg["hit_rate"] == 0.5
    assert agg["cases"] == 2


def test_parse_score_handles_scales():
    assert _parse_score("0.8") == 0.8
    assert _parse_score("Score: 7") == 0.7  # 0-10 scale
    assert _parse_score("85") == 0.85       # 0-100 scale
    assert _parse_score("no number") == 0.0


def test_judge_functions_use_injected_judge():
    stub = lambda prompt: "0.9"
    assert judge_faithfulness("ans", "ctx", stub) == 0.9
    assert judge_answer_relevance("q", "ans", stub) == 0.9
