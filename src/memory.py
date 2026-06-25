"""Conversation memory with automatic context compression (#5).

Long chats blow up token usage if every turn is replayed verbatim. This module
keeps the most recent turns intact and, once the history grows past a threshold,
compresses the older turns into a short summary via an injected ``summarizer_fn``.

Budget-safe by design: the summarizer (an LLM call) only runs when the history
actually exceeds ``trigger_after`` turns — short chats cost nothing extra.
"""

from typing import Any, Callable, Dict, List, Optional


def format_history(turns: List[Dict[str, str]]) -> str:
    """Render turns as 'ROLE: content' lines."""
    lines = []
    for turn in turns:
        role = (turn.get("role") or "user").upper()
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


SUMMARY_PROMPT = """Summarize the earlier part of this conversation in 3-5 concise bullet points.
Keep only facts, decisions, and topics needed to understand follow-up questions. No preamble.

CONVERSATION:
{history}

SUMMARY:"""


def prepare_history(
    turns: Optional[List[Dict[str, str]]],
    summarizer_fn: Optional[Callable[[str], str]] = None,
    keep_recent: int = 4,
    trigger_after: int = 8,
) -> Dict[str, Any]:
    """Split history into a compressed summary (older turns) + verbatim recent turns.

    Returns ``{"summary": str, "recent": [...], "compressed": bool, "turns": int}``.
    The summary is empty (and no LLM call is made) until the history exceeds
    ``trigger_after`` turns. If no summarizer is supplied, older turns are dropped
    rather than summarized.
    """
    turns = [t for t in (turns or []) if (t.get("content") or "").strip()]
    total = len(turns)

    if total <= trigger_after:
        return {"summary": "", "recent": turns, "compressed": False, "turns": total}

    older = turns[: total - keep_recent]
    recent = turns[total - keep_recent :]

    if summarizer_fn is not None and older:
        summary = (summarizer_fn(SUMMARY_PROMPT.format(history=format_history(older))) or "").strip()
    else:
        summary = ""

    return {"summary": summary, "recent": recent, "compressed": True, "turns": total}


def has_history(turns: Optional[List[Dict[str, str]]]) -> bool:
    return bool([t for t in (turns or []) if (t.get("content") or "").strip()])
