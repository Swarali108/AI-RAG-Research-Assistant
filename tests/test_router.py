"""Tests for the agentic query router (#6). Web search itself hits the network,
so only the routing logic (pure) is unit-tested here."""

from src.router import route_question


def test_routes_to_web_on_current_event_signals():
    assert route_question("What is the latest news on AI regulation?") == "web"
    assert route_question("current stock price of the company") == "web"


def test_forced_web_overrides():
    assert route_question("explain attention", use_web=True) == "web"


def test_routes_to_memory_on_followups_with_history():
    assert route_question("how does it work?", has_history=True) == "memory"
    assert route_question("tell me more about that", has_history=True) == "memory"


def test_memory_signals_ignored_without_history():
    # no history -> a pronoun question should still go to RAG
    assert route_question("how does it work?", has_history=False) == "rag"


def test_default_is_rag():
    assert route_question("Explain the transformer architecture") == "rag"
