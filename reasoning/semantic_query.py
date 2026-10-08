"""Semantic query generation: interpreted meaning -> a search query.

This module answers the question *"what should Atlas actually search for?"* from
the **interpreted meaning** of a request rather than from its surface string.

The distinction matters. String surgery over the raw prompt produces queries that
carry the user's sentence rather than their information need, and it fails on any
phrasing it was not written for:

    "Who took the title this year?"   -> "this year"          (no subject at all)
    "Who became champion?"            -> "who became champion?" (raw prompt leaked)
    "Who won yesterday's game?"       -> "won yesterday's game" (no event named)

A meaning-derived query is built from the *components* the understanding layer
already resolved, so paraphrases converge and an unfamiliar entity needs no new
rule:

    relation = event_result
    subject  = "championship"
    temporal = this_year / 2026
    -> "2026 championship winner"

The builder is deterministic, has no tool access, and never invents an entity: it
composes only the parts the semantic reading actually resolved. When it cannot
compose anything better than the subject it returns the subject, so a caller
always gets a query it can use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Question words and auxiliaries that are never part of a search query. These are
#: *syntax*, not topics: they are the frame the user wrapped their subject in.
_QUESTION_FRAME_RE = re.compile(
    r"^\s*(?:please\s+|hey\s+atlas[,\s]+|can\s+you\s+|could\s+you\s+|would\s+you\s+|"
    r"will\s+you\s+|i\s+need\s+to\s+know\s+|i\s+want\s+to\s+know\s+|"
    r"i'd\s+like\s+to\s+know\s+|let\s+me\s+know\s+|tell\s+me\s+|"
    r"find\s+out\s+|check\s+|verify\s+|do\s+you\s+know\s+)*"
    r"(?:what|who|whom|whose|which|when|where|why|how|is|are|was|were|do|does|did|"
    r"can|could|would|should|will|has|have|had)\b\s*",
    re.IGNORECASE,
)

#: Copulas and light verbs that carry no information need.
_LIGHT_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|does|do|did|will|would|"
    r"should|could|can|going|got|get)\b",
    re.IGNORECASE,
)

#: Articles, possessives, and connective filler.
_FILLER_RE = re.compile(
    r"\b(?:the|a|an|of|for|about|regarding|on|in|at|to|from|with|by|please|"
    r"me|my|us|our|your|right|now|currently|actually|really|just)\b",
    re.IGNORECASE,
)

#: Verbs that describe the *act of asking*, not the subject of the question. They
#: are stripped so the query is about the topic.
_ASKING_VERB_RE = re.compile(
    r"\b(?:happened|happening|going\s+on|went\s+down|telling|told|asked|asking|"
    r"know|knowing|checking|checked|finding|found)\b",
    re.IGNORECASE,
)

#: The *research verb frame*: the user's instruction to Atlas to look something up.
#: It is the relation ("go and find out"), not the subject, so it must not appear
#: in a search query. Stripping it here means "Search the web for the latest
#: Python release" yields a query about Python rather than about searching.
_RESEARCH_VERB_RE = re.compile(
    r"\b(?:search|find|look\s+up|look|google|browse|research|fetch|pull\s+up|"
    r"give|show|get|tell|send|bring)\b",
    re.IGNORECASE,
)

#: Sources named as the *place* to look, which are not the subject either.
_SEARCH_PLACE_RE = re.compile(
    r"\b(?:the\s+web|web|the\s+internet|internet|online|youtube|google|bing|"
    r"wikipedia|the\s+news|the\s+latest\s+news)\b",
    re.IGNORECASE,
)

#: Result relations and the query suffix each licenses. A result relation asks for
#: *who/what took the outcome*, so the useful query names the event plus "winner"
#: (or the outcome noun), which is what a search engine answers well.
_RESULT_SUFFIX = "winner"

#: Outcome nouns whose *own* wording is a better suffix than the generic "winner"
#: (\"who was crowned champion\" -> \"champion\", \"what was the final score\").
_OUTCOME_NOUN_SUFFIXES: dict[str, str] = {
    "champion": "champion",
    "champions": "champion",
    "championship": "championship winner",
    "title": "title winner",
    "crown": "champion",
    "trophy": "winner",
    "medal": "medal winner",
    "gold": "gold medal winner",
    "score": "score",
    "scoreline": "score",
    "result": "result",
    "results": "results",
    "outcome": "result",
    "victor": "winner",
    "victory": "winner",
    "standings": "standings",
}

#: Verbs that mean the query should ask about a *current value/state*, which a
#: search engine answers best with the noun phrase plus a recency marker.
_VALUE_MARKERS: tuple[str, ...] = ("price", "cost", "rate", "value", "worth", "level")

_WS_RE = re.compile(r"\s+")


@dataclass
class SemanticQuery:
    """A search query derived from interpreted meaning, with its provenance."""

    query: str = ""
    #: The components that composed it (for the reasoning/diagnostics trace).
    components: list[str] = field(default_factory=list)
    #: True when the query is meaning-derived rather than a cleaned prompt echo.
    semantic: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "components": list(self.components),
            "semantic": self.semantic,
            "notes": list(self.notes),
        }


def build_query(
    *,
    question: str,
    subject: str = "",
    relation: str = "",
    event_result: bool = False,
    outcome_noun: str = "",
    temporal_period: str = "",
    temporal_relation: str = "",
    comparative: bool = False,
    criterion: str = "",
    candidate_set: str = "",
    value_seeking: bool = False,
) -> SemanticQuery:
    """Compose a search query from the interpreted components of a request.

    Parameters are all *already resolved meaning*: the builder does not parse the
    question for them (it only falls back to a cleaned question when no component
    is available). ``relation`` is a free-text description of what the user wants
    done; the named flags are the structural facts the understanding layer proved.
    """

    components: list[str] = []
    notes: list[str] = []

    core = (subject or "").strip()
    if core:
        components.append(f"subject={core}")

    # A subject the *builder* receives is already meaning-derived, but it may still
    # carry the outcome word twice when the interpreter's topic reading and the ask
    # overlap ("championship championship"). Punctuation attached to the subject
    # ("iphone?") is the user's sentence punctuation, not part of a name, so it is
    # removed here, before any composition.
    core = _drop_repeated_words(_strip_trailing_punctuation(_collapse_repeats(core)))
    # A subject that is only a *pronoun referring to Atlas* ("you") names no
    # external entity: the question is a self-query and there is nothing to search
    # for. Composing a query from it would search for the word "you".
    if _is_self_subject(core):
        components.append("self_reference")
        notes.append("the subject refers to Atlas; no external query applies")
        return SemanticQuery(query="", components=components, semantic=True, notes=notes)

    # An *explanation* is about its subject, not about a time: "Why did the 2024
    # Lakers win?" is answered about the Lakers, so a resolved year does not
    # qualify the query (the year is context for the explanation, not a filter on
    # the answer).
    explanatory = _is_explanation(question)

    # A deictic time window as the subject ("What happened last night?") is not a
    # thing to search for on its own; without an event, the honest query is the
    # *window plus the news framing*, which is what a current-events lookup is.
    # A window-only subject ("What happened last night?") is a *current-events*
    # request: the useful query asks for the news covering that window, not for the
    # window itself, which a search engine would read as a literal date string. The
    # window check is independent of the event-result relation: either way there is
    # no entity to look up, only a time to report on.
    window_as_subject = bool(core and _is_temporal_only(core))
    if window_as_subject:
        components.append(f"time_window={core}")
        composed = _join([_window_phrase(core), "news"])
        components.append("window_lookup")
        notes.append("no entity named; querying for news covering the window")
        return SemanticQuery(query=composed, components=components, semantic=True, notes=notes)

    composed = ""
    if event_result and not core:
        # The request asks for an event outcome but names no event ("Who became
        # champion?", "Which team won the final?"). There is nothing to search for
        # and no honest query to compose, so the builder says so instead of
        # manufacturing a generic one: the caller must ask which event, or answer
        # from the conversation, rather than run a meaningless search.
        components.append("subject_missing")
        notes.append("event outcome requested but no event was named; no query possible")
        return SemanticQuery(query="", components=components, semantic=True, notes=notes)

    if event_result:
        # "who won X" / "who took the title this year" -> "<event> winner". The
        # *event* (subject) is what a search engine needs; the relation supplies the
        # outcome word.
        suffix = _OUTCOME_NOUN_SUFFIXES.get((outcome_noun or "").strip().casefold(), _RESULT_SUFFIX)
        subject_part = _window_phrase(core) if window_as_subject else _drop_repeated_words(core)
        composed = _drop_repeated_words(_join([subject_part, suffix]))
        components.append("relation=event_result")
        if suffix:
            components.append(f"suffix={suffix}")
    elif comparative:
        # A ranking over candidates ("the most famous Minecraft YouTuber") composes
        # the candidate set with the criterion. A *temporal* superlative ("the latest
        # iPhone", "the newest Pixel", "the latest Python release") names one thing
        # or one release, not a candidate set to rank, so it is composed like an
        # ordinary subject and the resolved period qualifies it. The temporal
        # relation is read from the request's own words rather than inferred, so the
        # decision does not depend on which noun happens to follow "latest".
        temporal_superlative = (
            _is_temporal_relation(temporal_relation)
            or (outcome_noun or "").strip().casefold() in {"latest", "newest"}
        )
        if temporal_superlative:
            composed = core
            components.append("relation=current_item")
        else:
            base = candidate_set or core
            composed = _join([base, criterion or ""])
            components.append("relation=ranking")
            if criterion:
                components.append(f"criterion={criterion}")
    elif value_seeking:
        # "What is Bitcoin trading at?" -> "Bitcoin price". The value noun is the
        # *property* asked about and the subject is the thing it is a property of;
        # the query reads "<subject> <property>" ("Brent crude price"), never the
        # other way round.
        marker = next((word for word in _VALUE_MARKERS if word in (relation or "").casefold()), "price")
        subject_part = _reorder_value_subject(core, marker)
        composed = _join([subject_part, marker])
        components.append("relation=value")
    else:
        composed = core

    # Temporal period qualifies the query rather than replacing it: "Baku Masters"
    # -> "2026 Baku Masters winner". It is appended only when there is a real
    # subject to qualify, so a request with no named event ("What happened last
    # night?") does not produce a year-only query that searches for the date.
    period = (temporal_period or "").strip()
    if period and composed and not explanatory and not _is_temporal_only(composed) \
            and period.casefold() not in composed.casefold():
        year = re.search(r"\b(?:19|20)\d{2}\b", period)
        # A period like "last 30 days from 2026-10-08" or "evening of 2026-10-07"
        # is not a useful search term; the year is. A bare calendar period that
        # carries no year (an open span) is skipped rather than pasted in.
        qualifier = year.group(0) if year else (period if re.fullmatch(r"[A-Za-z ]{3,30}", period) else "")
        if qualifier and qualifier.casefold() not in composed.casefold():
            # The period *qualifies* the subject ("2024 Baku Masters winner") rather
            # than replacing or heading it, so it is placed before the head noun the
            # relation contributed, not prefixed to the phrase.
            composed = _reorder_qualifier(composed, qualifier)
            components.append(f"temporal={qualifier}")

    composed = _tidy(composed)

    # Fall back to a cleaned form of the question when nothing was composed. The
    # cleaning strips the *question frame* and the *research instruction* (not the
    # topic), so the result is still about the user's subject rather than their
    # sentence. A request that named no resolvable subject at all yields no query,
    # which is the honest outcome: the caller must ask which event, or answer from
    # the conversation, rather than search for a stray pronoun.
    if not composed and not window_as_subject:
        candidate = _clean_question(question)
        if candidate and not _is_self_subject(candidate):
            composed = candidate
            notes.append("no semantic component was available; used a cleaned question")
        else:
            notes.append("no resolvable subject; no query composed")
    else:
        notes.append("query composed from interpreted meaning")

    # A composed query is cleaned *after* composition so no user punctuation or
    # left-over frame word reaches a search tool ("iphone? 2026" -> "iphone 2026").
    composed = _tidy(composed)

    return SemanticQuery(
        query=composed,
        components=components,
        semantic=bool(components),
        notes=notes,
    )


def _join(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _is_temporal_relation(relation: str) -> bool:
    """True when a temporal relation names *recency* rather than a time point.

    A relation may carry a completion suffix ("latest_completed",
    "most_recent_completed"), so it is matched on its base form rather than by exact
    equality. Recency relations select *the newest instance of one thing*, which is
    why they are not a ranking over a candidate set.
    """

    base = (relation or "").casefold()
    for suffix in ("_completed", "_ongoing"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base in {"latest", "newest", "most_recent", "current", "currently", "now", "recently", "recent"}


def _strip_trailing_punctuation(text: str) -> str:
    """Remove sentence punctuation attached to the end of a subject."""

    return re.sub(r"[?!.,;:]+$", "", (text or "").strip()).strip()


def _drop_repeated_words(text: str) -> str:
    """Drop a word that already appeared, so a phrase is not mirrored.

    Used on a *query* the builder composed, where a repeated content word means
    the components overlapped ("championship championship winner"). It is not a
    general de-duplication of meaning: order is preserved and the first occurrence
    of each distinct word is kept.
    """

    seen: set[str] = set()
    out: list[str] = []
    for word in (text or "").split():
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(word)
    return " ".join(out)


def _is_self_subject(text: str) -> bool:
    """True when a subject is only a reference to Atlas, not an external entity."""

    lowered = (text or "").casefold().strip(" ?.,'")
    if not lowered:
        return False
    words = set(re.findall(r"[a-z']+", lowered))
    # "you", "your capabilities", "what can you do": every content word refers to
    # Atlas, so there is nothing external to search for.
    return bool(words) and words <= {
        "you", "your", "yours", "yourself", "u", "atlas", "atlas's",
        "can", "do", "what", "are", "is", "abilities", "capabilities",
        "tools", "features", "able", "to",
    }


def _reorder_value_subject(subject: str, marker: str) -> str:
    """Return the phrase naming the thing whose *value* was asked for.

    "the price of a barrel of Brent crude" states the property first; the useful
    query is about the commodity, so a leading "<property> of" frame is removed and
    the filler words between the commodity words are dropped, leaving a compact
    noun phrase ("Brent crude") for the value to attach to.
    """

    lowered = (subject or "").strip()
    # Strip a leading "<property> of" repeatedly, so "the price of a barrel of"
    # reduces to the commodity rather than only its first layer.
    property_words = "|".join(
        re.escape(word) for word in _VALUE_MARKERS + ("cost", "value", "rate", "worth")
    )
    for _ in range(3):
        stripped = re.sub(
            r"^(?:the\s+)?(?:" + property_words + r")\s+of\s+(?:a\s+|an\s+|the\s+)?",
            "",
            lowered,
            flags=re.IGNORECASE,
        ).strip()
        if stripped == lowered:
            break
        lowered = stripped
    if not lowered:
        return (subject or "").strip()
    return _WS_RE.sub(" ", lowered).strip()


def _is_explanation(question: str) -> bool:
    """True when the request asks for a reason or a mechanism, not a fact."""

    return bool(re.match(r"^\s*(?:why|how|explain|describe)\b", (question or "").casefold()))


def _reorder_qualifier(composed: str, qualifier: str) -> str:
    """Place a temporal qualifier where a query reads naturally.

    A year qualifies the *event* in a result query ("2024 Baku Masters winner"),
    so the qualifier is moved directly in front of the head noun that follows the
    event ("winner"/"champion") rather than being prefixed or appended to the
    end. When the query has no outcome word, it qualifies the whole phrase and is
    appended.
    """

    words = composed.split()
    if not words:
        return qualifier
    for index, word in enumerate(words):
        if word.casefold() in {
            "winner", "champion", "championship", "result", "score", "standings",
        }:
            return " ".join(words[:index] + [qualifier] + words[index:])
    return " ".join(words + [qualifier])


def _collapse_repeats(text: str) -> str:
    """Collapse an immediately repeated content word ("championship championship").

    Only *adjacent* duplicates are collapsed. A general de-duplication would be
    wrong here: it would delete a legitimate second mention ("the 2026 Baku
    Masters" appears once, but "winner of the 2010 World Cup final" has two distinct
    content words that must both survive).
    """

    words = (text or "").split()
    out: list[str] = []
    for word in words:
        if out and out[-1].casefold() == word.casefold():
            continue
        out.append(word)
    return " ".join(out)


def _tidy(text: str) -> str:
    value = _RESEARCH_VERB_RE.sub(" ", text or "")
    value = _SEARCH_PLACE_RE.sub(" ", value)
    value = _FILLER_RE.sub(" ", value)
    value = _ASKING_VERB_RE.sub(" ", value)
    value = _LIGHT_VERB_RE.sub(" ", value)
    value = _WS_RE.sub(" ", value).strip(" ?.,;:!'-'\"")
    return value


def _clean_question(question: str) -> str:
    """Strip the question *frame* from a raw question, keeping the topic."""

    value = (question or "").strip()
    value = _QUESTION_FRAME_RE.sub("", value, count=1)
    value = _RESEARCH_FRAME_RE.sub("", value, count=1)
    value = _EXPLAIN_FRAME_RE.sub("", value, count=1)
    value = _INDIRECT_FRAME_RE.sub("", value, count=1)
    value = _tidy(value)
    # A cleaned question that still reads as a question (a leftover interrogative
    # or auxiliary) is not a usable query: it would hand a search engine the user's
    # sentence. Reducing it to its content words is the best meaning-preserving
    # option, and an empty result is returned so the *caller* decides the fallback
    # rather than sending a question to a search tool.
    if re.match(r"\s*(?:what|who|which|whom|whose|when|where|why|how|is|are|was|were|"
                r"do|does|did|can|could|would|should|will|has|have|had)\b", value):
        value = ""
    return value


#: A leading research/retrieval instruction ("search the web for X", "look up X").
#: The *frame* is the user telling Atlas to look something up; the payload after it
#: is the subject, so the frame is removed and "the web" with it.
_RESEARCH_FRAME_RE = re.compile(
    r"^\s*(?:please\s+)?(?:search|find|look\s+up|look|google|browse|research|"
    r"fetch|pull\s+up|get|show\s+me|give\s+me)\s+"
    r"(?:the\s+web|the\s+internet|online|up)?\s*(?:for|about|on)?\s*",
    re.IGNORECASE,
)

#: An *explanation* frame ("explain recursion", "describe a bloom filter"). It asks
#: Atlas to say what the subject is, so the subject is what remains after the verb.
_EXPLAIN_FRAME_RE = re.compile(
    r"^\s*(?:please\s+)?(?:explain|describe|define|summarize|summarise|"
    r"clarify|outline)\s+(?:to\s+me\s+|me\s+)?(?:how\s+|what\s+|why\s+)?"
    r"(?:the\s+|a\s+|an\s+)?",
    re.IGNORECASE,
)

#: An indirect request frame ("I need to know X", "can you check X", "find out X").
#: The frame hands Atlas a question; the payload after it is the subject.
_INDIRECT_FRAME_RE = re.compile(
    r"^\s*(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"(?:check|verify|confirm|see|find\s+out|look\s+up|tell\s+me|let\s+me\s+know)\s+"
    r"|(?:find\s+out|check|verify|confirm|see|look\s+up)\s+"
    r"|i\s+(?:need|want|would\s+like)\s+to\s+know\s+"
    r"|i'd\s+like\s+to\s+know\s+"
    r"|let\s+me\s+know\s+"
    r"|(?:do\s+you\s+)?(?:know|happen\s+to\s+know)\s+)",
    re.IGNORECASE,
)


def _is_temporal_only(subject: str) -> bool:
    """True when a \"subject\" is really just a time window.

    The test is lexical *containment* of a window phrase in a short span: a subject
    that is nothing but a window ("last night", "this year", "three days ago") is
    a current-events reference, whereas a window attached to an entity ("the Baku
    Masters 2026", "the latest Python release") is a qualified subject and is not.
    A determiner or a content noun beside the window means there is something to
    look up, so it is not window-only.
    """

    from reasoning.semantic_analysis import _DEICTIC_TIME_NOUNS

    lowered = subject.casefold().strip(" ?.,")
    if not lowered:
        return False
    matched = [
        term for term in _DEICTIC_TIME_NOUNS
        if lowered == term or lowered.endswith(" " + term) or lowered.startswith(term + " ")
    ]
    if not matched:
        return False
    # Remove the window phrase and whatever bare filler surrounds it: what remains
    # is either nothing (window-only) or a real subject (a qualified window).
    residue = lowered
    for term in sorted(matched, key=len, reverse=True):
        residue = residue.replace(term, " ")
    residue = _FILLER_RE.sub(" ", residue)
    residue = _LIGHT_VERB_RE.sub(" ", residue)
    residue = _WS_RE.sub(" ", residue).strip(" .,?")
    return not residue


def _window_phrase(window: str) -> str:
    cleaned = _tidy(window)
    return cleaned or "latest"
