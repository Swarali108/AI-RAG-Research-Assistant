"""Tests for conversation memory + compression (#5). The summarizer is stubbed,
so no LLM calls happen."""

from src.memory import format_history, has_history, prepare_history


def _turns(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"} for i in range(n)]


def test_short_history_is_not_compressed_and_calls_no_summarizer():
    calls = []
    summarizer = lambda text: calls.append(text) or "SUMMARY"
    result = prepare_history(_turns(4), summarizer_fn=summarizer, trigger_after=8)
    assert result["compressed"] is False
    assert result["summary"] == ""
    assert len(result["recent"]) == 4
    assert calls == []  # budget-safe: no summarizer call for short chats


def test_long_history_compresses_older_turns():
    calls = []
    summarizer = lambda text: calls.append(text) or "SUMMARY OF EARLIER"
    result = prepare_history(_turns(12), summarizer_fn=summarizer, keep_recent=4, trigger_after=8)
    assert result["compressed"] is True
    assert result["summary"] == "SUMMARY OF EARLIER"
    assert len(result["recent"]) == 4
    assert len(calls) == 1  # summarizer invoked once for the older turns


def test_long_history_without_summarizer_drops_older_turns():
    result = prepare_history(_turns(12), summarizer_fn=None, keep_recent=4, trigger_after=8)
    assert result["compressed"] is True
    assert result["summary"] == ""
    assert len(result["recent"]) == 4


def test_format_history_and_has_history():
    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    text = format_history(turns)
    assert "USER: hi" in text
    assert "ASSISTANT: hello" in text
    assert has_history(turns) is True
    assert has_history([]) is False
    assert has_history([{"role": "user", "content": "  "}]) is False
