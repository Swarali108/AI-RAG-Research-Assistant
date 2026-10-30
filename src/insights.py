"""Document insights: key topics and suggested questions.

Two layers, so the panel always has something meaningful to show:

- ``extract_keyphrases`` (free, deterministic): RAKE-style candidate phrases
  split on stopwords and punctuation, scored by frequency with a boost for
  multi-word phrases and acronyms. Replaces the old "most common single word"
  topics that surfaced junk like "its", "does" and "part".
- An LLM pass (prompt + parser here, the call itself lives in ``app.py``)
  turns those candidates plus document excerpts into clean topic labels and
  questions the documents can actually answer.

``fallback_questions`` covers the case where the LLM call fails.
"""

import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

# Function words plus words that are frequent in prose but never a "topic".
STOPWORDS = set("""
a about above according across actually after again against all almost along already also although always am among an and another any anyone anything are around as at away back be became because become becomes been before being below between both but by can cannot could did do does doing done down during each either else enough etc even ever every example examples few first following for from further get gets getting give given gives go goes going good got great had has have having he her here hers him his how however i if in include includes including instead into is it its itself just keep know known last less let like likely made main make makes making many may maybe me might more most mostly much must my need needs never new next no non not now of off often on once one only onto or other others our out over own part parts per perhaps put rather really same see seen several shall she should show shows since so some something sometimes soon still such take takes than that the their them themselves then there these they thing things think this those though three through thus to together too took toward two under until up upon us use used uses using usually very via want was way ways we well were what whatever when where whether which while who whole whom whose why will with within without would yet you your yours yourself
able across actual add added adds allow allows almost along around ask asked asks based basic best better big called case cases certain change chapter clear come comes common consider could different does doing easy either end ends especially even exactly fact far find finds fine form full general generally goes hand happen happens hard help helps high however idea ideas important instance kind kinds large later least level likely little long look looks lot lots low mean means much near nearly number numbers often okay order page pages particular place point points possible pretty quite real reason right said say says section sections set short side simple simply small sort start step steps sure tell term terms thing told top try turn type types understand unit usual value want whole work works world year years yes
""".split())

# Common verbs (all forms) end a phrase, so "RAG reduces hallucinations" yields
# "RAG" and "hallucinations", not one phrase. Words that are just as often nouns
# (store, search, query, answer, test, train) are deliberately left out.
_VERB_STEMS = """accept achieve adapt add adjust allow change analyse analyze apply argue arrive assign
assume attempt avoid become begin believe belong break bring build calculate carry cause check
choose claim combine compare compute concern connect contain continue convert correspond create
decide define demonstrate depend describe design determine develop differ discuss divide enable
encode ensure estimate evaluate examine exist expand expect explain express extend extract fail
feed fetch fill focus follow generate grow guide handle happen hold identify ignore illustrate
imagine imply improve increase indicate influence inform introduce involve know lead learns
leave let limit link lose maintain manage match matter measure meet mention miss move notice
obtain occur offer operate pass perform pick play predict prefer prepare present prevent produce
propose prove provide reach read realize receive recognize reduce refer reflect relate rely
remain remember remove repeat replace represent require respond retrieve return reveal run
scale seem select send separate serve solve specify split spend stand stay suggest supply
support suppose tend transform treat understand update vary write"""
VERBS = set()
for _stem in _VERB_STEMS.split():
    VERBS.update({_stem, _stem + "s", _stem + "es", _stem + "d", _stem + "ed"})
    if _stem.endswith("e"):
        VERBS.update({_stem[:-1] + "es"})
VERBS.update({"knows", "knew", "known", "reads", "made", "makes", "took", "gave", "came",
              "stores", "stored", "searches", "answers", "answered", "trains", "trained",
              "learns", "learned", "means", "meant", "found", "thought", "told", "built",
              "led", "lets", "ran", "sent", "seems", "seemed", "shown", "showed", "written"})

# Words that read as noise on their own but are fine inside a longer phrase.
WEAK_SINGLES = {"model", "models", "data", "system", "systems", "method", "methods",
                "approach", "result", "results", "process", "information", "question",
                "questions", "answer", "answers", "text", "time", "problem", "output",
                "input", "task", "tasks", "figure", "table", "note", "notes"}

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*|[^\sA-Za-z]")


def _is_acronym(word: str) -> bool:
    return 2 <= len(word) <= 6 and word.isupper() and word.isalpha()


def _is_verb(lowered: str) -> bool:
    return lowered in VERBS or lowered.endswith(("ized", "ised", "izes", "ises"))


def _candidate_phrases(text: str) -> List[List[str]]:
    """Split text into runs of content words, breaking on stopwords/punctuation."""
    phrases, current = [], []
    for token in _TOKEN.findall(text or ""):
        word = token.strip("'")
        is_word = bool(word) and word[0].isalpha()
        lowered = word.lower()
        if (not is_word or lowered in STOPWORDS or _is_verb(lowered)
                or (len(word) < 3 and not _is_acronym(word))):
            if current:
                phrases.append(current)
            current = []
            continue
        current.append(word)
        if len(current) == 3:  # cap phrase length; longer runs are usually sentences
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    return phrases


def _display(words: Sequence[str]) -> str:
    return " ".join(w if _is_acronym(w) or any(c.isupper() for c in w[1:]) else w.capitalize()
                    for w in words)


def _key(words: Sequence[str]) -> str:
    # Case-insensitive, naive singular, so "Vector databases" == "vector database".
    return " ".join(w.lower()[:-1] if w.lower().endswith("s") and len(w) > 4 else w.lower()
                    for w in words)


def extract_keyphrases(texts: Sequence[str], top_n: int = 12) -> List[str]:
    """Return up to ``top_n`` display-ready key phrases for the given texts."""
    counts: Counter = Counter()
    standalone: Counter = Counter()  # times a key was the whole phrase, not a sub-span
    spread: Dict[str, set] = defaultdict(set)
    surface: Dict[str, Counter] = defaultdict(Counter)

    for index, text in enumerate(texts):
        for phrase in _candidate_phrases(text):
            standalone[_key(phrase)] += 1
            for size in range(1, len(phrase) + 1):
                for start in range(0, len(phrase) - size + 1):
                    words = phrase[start:start + size]
                    key = _key(words)
                    counts[key] += 1
                    spread[key].add(index)
                    surface[key][tuple(words)] += 1

    min_count = 2 if sum(counts.values()) > 200 else 1
    scored = []
    for key, count in counts.items():
        words = key.split()
        if count < min_count:
            continue
        if len(words) == 1 and (key in WEAK_SINGLES or key.rstrip("s") in WEAK_SINGLES):
            continue
        best_surface = surface[key].most_common(1)[0][0]
        score = count * (1 + 0.8 * (len(words) - 1)) * (1 + 0.15 * len(spread[key]))
        if len(words) == 1:
            # "learning" mostly seen inside "machine/deep learning" is not a topic itself.
            score *= 0.3 + 0.7 * (standalone[key] / count)
        if any(_is_acronym(w) for w in best_surface):
            score *= 1.5
        scored.append((score, key, best_surface))

    scored.sort(key=lambda item: (-item[0], item[1]))
    picked: List[str] = []
    picked_keys: List[str] = []
    for _, key, words in scored:
        # Skip a phrase that is just part of (or contains) one already chosen.
        if any(key in other or other in key for other in picked_keys):
            continue
        picked.append(_display(words))
        picked_keys.append(key)
        if len(picked) >= top_n:
            break
    return picked


INSIGHTS_PROMPT = """You are analysing documents a user uploaded to a research assistant.

Candidate key phrases found by keyword analysis (may contain noise):
{candidates}

Document excerpts:
{excerpts}

Return ONLY a JSON object, no prose, in exactly this shape:
{{"topics": ["...", "..."], "questions": ["...", "..."]}}

Rules:
- "topics": up to 10 key topics the documents actually cover, each 1-4 words, specific
  noun phrases in Title Case (keep acronyms like RAG or LLM uppercase). No generic words.
- "questions": exactly 5 specific questions a learner would ask that these documents can
  answer. Each under 90 characters. Vary them: overview, how-it-works, comparison,
  limitations, application.{avoid}"""


def build_insights_prompt(candidates: Sequence[str], excerpts: Sequence[str],
                          avoid_questions: Optional[Sequence[str]] = None) -> str:
    avoid = ""
    if avoid_questions:
        avoid = "\n- Do NOT repeat or paraphrase these earlier questions: " + "; ".join(avoid_questions)
    return INSIGHTS_PROMPT.format(
        candidates=", ".join(candidates) or "(none)",
        excerpts="\n---\n".join(excerpts),
        avoid=avoid,
    )


def sample_excerpts(texts: Sequence[str], max_words: int = 2500) -> List[str]:
    """Evenly spaced excerpts across the whole document, within a word budget."""
    if not texts:
        return []
    per_chunk = 180
    wanted = max(1, max_words // per_chunk)
    step = max(1, len(texts) // wanted)
    picked = texts[::step][:wanted]
    return [" ".join(t.split()[:per_chunk]) for t in picked]


def parse_insights(raw: str) -> Optional[Dict[str, List[str]]]:
    """Parse the model's JSON (tolerating code fences or stray prose)."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    topics = [str(t).strip() for t in data.get("topics", []) if str(t).strip()]
    questions = [str(q).strip() for q in data.get("questions", []) if str(q).strip()]
    if not topics and not questions:
        return None
    return {"topics": topics[:10], "questions": questions[:5]}


def fallback_questions(topics: Sequence[str]) -> List[str]:
    """Template questions built from extracted topics (used if the LLM call fails)."""
    questions = ["What are the key ideas in these documents?"]
    templates = [
        "What is {} and why does it matter?",
        "How does {} work, step by step?",
        "How is {} related to {}?",
        "What limitations of {} are mentioned?",
    ]
    for template in templates:
        needed = template.count("{}")
        if len(topics) < needed:
            continue
        index = len(questions) - 1
        args = [topics[(index + offset) % len(topics)] for offset in range(needed)]
        questions.append(template.format(*args))
    generic = [
        "Summarize these documents in five bullet points.",
        "What are the most important definitions here?",
        "What limitations or open questions are mentioned?",
        "What would be good interview questions on this material?",
    ]
    for question in generic:
        if len(questions) >= 5:
            break
        questions.append(question)
    return questions[:5]


def build_insights(keyphrases: Sequence[str], llm_result: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
    """Combine keyphrases with an (optional) parsed LLM result."""
    keyphrases = list(keyphrases)
    if llm_result and llm_result.get("topics"):
        topics = llm_result["topics"]
        source = "llm"
    else:
        topics = keyphrases
        source = "keyphrases"
    questions = (llm_result or {}).get("questions") or fallback_questions(topics)
    return {"topics": topics, "questions": questions, "source": source, "keyphrases": keyphrases}
