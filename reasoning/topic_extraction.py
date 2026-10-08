"""Semantic topic extraction and query normalization.

This module is the *semantic boundary* between a natural-language instruction and
the concrete search/research query Atlas hands to a tool. It exists because a
request such as::

    research about sykrim and write it in notepad

must not become::

    web.research("about sykrim")

The words "research" and "about" are *instruction* language (an action verb and a
connector), not semantic content. The query should be the subject alone::

    web.research("Skyrim")

Design rules (deliberately semantic, not a phrase table):

* The parser reasons about the *grammatical role* of a span (action verb,
  connector, metalinguistic head noun, qualifier, content) rather than matching
  whole sentence templates.
* Connector words (``about``, ``on``, ``regarding``, ``concerning``,
  ``related to``, ``re:``) and "meta" head nouns (``information``, ``info``,
  ``details``, ``facts``, ``overview``) are stripped **only when they introduce
  the subject**, never when they are part of a noun phrase that carries meaning.
  ``research the concept of a story about skyrim`` keeps the inner "about".
* A content-type noun that merely describes the *kind* of thing wanted
  ("find videos about skyrim") is separated from the subject ("skyrim") so it
  does not contaminate the query.
* Typos are corrected only when the correction is confident (a known entity, or
  an edit distance of 1 against a token the same request otherwise establishes);
  otherwise the original term is preserved.

The result is a :class:`TopicReading` carrying the raw span, the normalized
subject, the research query, and the qualifiers/content-type split. The
interpreter consumes it; nothing here executes, plans, or calls a model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from reasoning.semantic_resolution import (
    REAL_WORD_GUARD,
    EntityResolver,
    Resolution,
    application_names,
    bounded_edit_distance,
    web_platforms,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grammar vocabulary (roles, not templates)
# ---------------------------------------------------------------------------

#: Intro connectors that mark the *start* of the subject after an action verb.
#: Order matters for regex alternation: longer multi-word forms first.
CONNECTORS: tuple[str, ...] = (
    "on the topic of", "in relation to", "with regard to", "related to",
    "with respect to", "on the subject of", "when it comes to",
    "about", "regarding", "concerning", "re:",
    # "on" is a connector at the *start* of a subject ("research on skyrim")
    # but also a legitimate preposition inside one ("effects of X on Y"), so it
    # is only stripped as a lead-in, never mid-phrase (see _strip_leading_connector).
    "on",
)

#: "Meta" head nouns that describe the *kind of information wanted* rather than
#: the subject. ``information about skyrim`` has subject ``skyrim``; the noun
#: "information" is framing. Kept separate from :data:`CONNECTORS` so the
#: stripper can require that a connector follow (or that the noun be terminal).
META_NOUNS: tuple[str, ...] = (
    "information", "info", "details", "facts", "overview", "background",
    "summary", "points", "explanation",
)

#: Determiners dropped from the *front* of a subject ("the history of X" ->
#: "history of X") only when a bare determiner would otherwise dominate.
LEADING_DETERMINERS: tuple[str, ...] = ("the", "a", "an", "some", "any")

#: Hedging/qualifier words that may legitimately belong to a subject
#: ("popular builds for skyrim", "best skyrim mods"). Recorded, never stripped.
QUALIFIER_WORDS: tuple[str, ...] = (
    "best", "top", "popular", "latest", "newest", "recent", "good", "great",
    "important", "key", "main", "beginner", "advanced", "common", "useful",
    "survival", "hardcore",
)

#: Placement / delivery verbs that terminate the subject span.
_CLAUSE_VERBS: tuple[str, ...] = (
    "write", "save", "put", "store", "type", "paste", "copy", "add", "insert",
    "place", "export", "dump", "drop", "open", "launch", "start", "create",
    "make", "display", "print", "send", "share", "email",
)

#: Action/search verbs that *frame* a request. When they lead the span, they are
#: instruction language, not subject content. Only a leading occurrence is
#: removed, so "how to research a topic" keeps its inner verb.
_ACTION_LEADS: tuple[str, ...] = (
    "look up", "search for", "find out", "research", "search", "find",
    "look for", "google", "browse", "fetch", "retrieve", "show", "get",
    "pull up", "tell", "give", "explain", "describe",
)

#: Delivery verbs used to cut an explicit "... and <verb> ..." clause. Kept
#: narrower than the trailing-clause list so an interrogative subject keeps its
#: inner verbs ("how to make a resume" is one subject).
_AND_CLAUSE_VERBS: tuple[str, ...] = (
    "write", "save", "put", "store", "type", "paste", "copy", "add",
    "insert", "place", "export", "dump", "drop", "open", "launch",
    "summarize", "summarise", "condense", "explain", "describe", "print",
    "send", "share", "email",
)
_AND_CLAUSE_RE = re.compile(
    r"\s*,?\s*and\s+(?:" + "|".join(re.escape(v) for v in _AND_CLAUSE_VERBS) + r")\b",
    re.IGNORECASE,
)

#: Real English words that must never be "corrected" to a known entity. A real
#: English word ("mode" -> "code") is not a typo; correcting it changes the
#: meaning of the request, which the confidence policy forbids. The vocabulary
#: itself lives in :mod:`reasoning.semantic_resolution` (one shared guard), and
#: is a damage-prevention *cache*, never the intelligence layer: the actual
#: resolution comes from the vocabulary, external evidence, or a validated model
#: suggestion (see :class:`reasoning.semantic_resolution.EntityResolver`).
_COMMON_WORDS: frozenset[str] = REAL_WORD_GUARD

#: Transformation verbs that, when they *report* the subject, terminate it.
_TRANSFORM_VERBS: tuple[str, ...] = (
    "summarize", "summarise", "condense", "extract", "abstract", "digest",
    "simplify", "shorten", "translate", "rewrite", "explain", "describe",
)

#: Destination prepositions that terminate a subject span when followed by a
#: known application or web platform, e.g. "about skyrim in notepad",
#: "videos about skyrim on youtube". The platform/app guard is what keeps a
#: legitimate "on" subject ("research the effects of X on Y") intact.
_DESTINATION_PREPS: tuple[str, ...] = (
    "in", "into", "onto", "to", "using", "within", "inside", "on",
)

_KNOWN_APPLICATIONS_CACHE: frozenset[str] | None = None
_WEB_PLATFORMS_CACHE: frozenset[str] | None = None


def known_applications() -> frozenset[str]:
    """Destination names, sourced from the live application registry.

    The list is not maintained here: :func:`reasoning.semantic_resolution.application_names`
    reads it from the launch capability's alias table, so a destination phrase
    does not need a matching sentence pattern in this module.
    """

    global _KNOWN_APPLICATIONS_CACHE
    if _KNOWN_APPLICATIONS_CACHE is None:
        _KNOWN_APPLICATIONS_CACHE = frozenset(application_names())
    return _KNOWN_APPLICATIONS_CACHE


def known_web_platforms() -> frozenset[str]:
    """Web platforms (a source, not a destination) usable as sentence breaks."""

    global _WEB_PLATFORMS_CACHE
    if _WEB_PLATFORMS_CACHE is None:
        _WEB_PLATFORMS_CACHE = frozenset(web_platforms())
    return _WEB_PLATFORMS_CACHE

#: Interrogative/how-to openers whose inner verbs belong to the subject rather
#: than marking a separate delivery clause ("how to make a resume").
_OPEN_QUESTION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:how\s+to\b|how\s+do\b|how\s+does\b|how\s+can\b|"
    r"what\s+is\b|what\s+are\b|why\s+\b|explain\s+how\b)",
    re.IGNORECASE,
)

#: Informational lead-ins that introduce a subject without naming a search verb.
_INFO_LEAD_RE = re.compile(
    r"^\s*(?:please\s+)?(?:tell\s+me\s+about|tell\s+me|give\s+me\s+(?:information\s+|info\s+|details\s+|facts\s+)?about|"
    r"show\s+me\s+about|what\s+about|how\s+about|information\s+about|info\s+about|"
    r"information\s+on|details\s+about|facts\s+about|anything\s+about|everything\s+about)\b\s*",
    re.IGNORECASE,
)

#: A leading web platform followed by a conjunction/preposition, e.g.
#: "youtube for X", "on youtube", "google X". The trailing "for"/"on" is part
#: of the platform phrase, not the subject.
_PLATFORM_LEAD_RE = re.compile(
    r"^\s*(?:on\s+|in\s+|at\s+)?(?:youtube|google|bing|the\s+web|the\s+internet|online|wikipedia)\b"
    r"(?:\s+(?:for|about|on|of|to|and|,))?\s*",
    re.IGNORECASE,
)

_CONNECTOR_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(c) for c in CONNECTORS) + r")\b",
    re.IGNORECASE,
)
# A metalinguistic head noun optionally followed by a connector.
_META_HEAD_RE = re.compile(
    r"^\s*(?:the|some|any|more|a|an)?\s*(?:" + "|".join(re.escape(n) for n in META_NOUNS) + r")s?\b"
    r"(?:\s+(?:about|on|regarding|concerning|of|for|related to|into))?",
    re.IGNORECASE,
)
_ACTION_LEAD_RE = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|will\s+you\s+|let\s+me\s+)*"
    r"(?:" + "|".join(re.escape(v) for v in _ACTION_LEADS) + r")\b"
    r"(?:\s+(?:me|us|up))?"
    r"(?:\s+for)?",
    re.IGNORECASE,
)

_TRAILING_CLAUSE_RE = re.compile(
    r"\s+(?:and\s+)?(?:" + "|".join(re.escape(v) for v in _CLAUSE_VERBS + _TRANSFORM_VERBS) + r")\b",
    re.IGNORECASE,
)
_DESTINATION_RE = re.compile(
    r"\s+(?:" + "|".join(re.escape(p) for p in _DESTINATION_PREPS) + r")\s+"
    r"([A-Za-z][A-Za-z0-9._+-]*(?:\s+[A-Za-z][A-Za-z0-9._+-]*){0,3})",
    re.IGNORECASE,
)


@dataclass
class TopicReading:
    """Structured result of semantic topic extraction for one request span."""

    #: The subject as it appeared in the text (before normalization), e.g.
    #: "about sykrim" or "videos about skyrim". Kept for diagnostics/provenance.
    raw_topic: str = ""
    #: The connector stripped (e.g. "about"), for observability. Empty if none.
    connector: str = ""
    #: The metalinguistic head noun stripped (e.g. "information"). Empty if none.
    stripped_head: str = ""
    #: The content-type noun that described the *kind* of thing ("videos").
    content_noun: str = ""
    #: The clean subject: "skyrim", "history of skyrim", "how x works".
    subject: str = ""
    #: The subject after confident normalization (typo correction/casing).
    normalized: str = ""
    #: The query to hand to a research/search tool. Equals ``normalized`` unless
    #: a qualifier was deliberately preserved.
    query: str = ""
    #: Qualifier words retained from the subject (e.g. "best", "survival").
    qualifiers: list[str] = field(default_factory=list)
    #: True when a connector was present and stripped from the subject.
    had_connector: bool = False
    #: A short, inspectable reason for the normalization decision.
    notes: list[str] = field(default_factory=list)
    #: The full resolution record for the subject (method, confidence, evidence,
    #: alternatives). ``None`` when no resolution was attempted.
    resolution: Resolution | None = None

    @property
    def is_empty(self) -> bool:
        return not self.subject.strip()

    @property
    def needs_normalization(self) -> bool:
        return bool(self.normalized and self.normalized.casefold() != self.subject.casefold())

    @property
    def ambiguous(self) -> bool:
        return bool(self.resolution is not None and self.resolution.ambiguous)

    @property
    def confidence(self) -> float:
        return float(self.resolution.confidence) if self.resolution is not None else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_topic": self.raw_topic,
            "connector": self.connector,
            "stripped_head": self.stripped_head,
            "content_noun": self.content_noun,
            "subject": self.subject,
            "normalized": self.normalized,
            "query": self.query,
            "qualifiers": list(self.qualifiers),
            "had_connector": self.had_connector,
            "notes": list(self.notes),
            "resolution": self.resolution.to_dict() if self.resolution is not None else {},
        }

    @classmethod
    def from_dict(cls, data: Any) -> "TopicReading":
        """Rebuild a reading from its serialized form.

        Used when a later stage needs the reading the entity pass already made,
        so the query builder consumes one reading instead of re-parsing the text.
        Unknown keys are ignored and wrong types fall back to the defaults, so a
        malformed record yields an empty reading rather than an exception.
        """

        if not isinstance(data, dict):
            return cls()

        def text(key: str) -> str:
            value = data.get(key)
            return str(value).strip() if isinstance(value, (str, int, float)) else ""

        return cls(
            raw_topic=text("raw_topic"),
            connector=text("connector"),
            stripped_head=text("stripped_head"),
            content_noun=text("content_noun"),
            subject=text("subject"),
            normalized=text("normalized"),
            query=text("query"),
            qualifiers=[
                str(item) for item in (data.get("qualifiers") or []) if str(item).strip()
            ] if isinstance(data.get("qualifiers"), (list, tuple)) else [],
            had_connector=bool(data.get("had_connector")),
            notes=[
                str(item) for item in (data.get("notes") or []) if str(item).strip()
            ] if isinstance(data.get("notes"), (list, tuple)) else [],
        )


def normalize_query(
    query: str,
    *,
    known_entities: Iterable[str] | None = None,
) -> tuple[str, bool]:
    """Clean a research query that may carry instruction language.

    Used both when an interpreter-level query is suspect and during bounded
    recovery, when a tool reports that a query returned nothing usable. It
    removes leading connectors and framing head nouns and applies confident
    normalization, returning ``(cleaned, changed)``. ``changed`` is False when
    the query is already clean, so a caller can avoid a pointless retry.
    """

    original = re.sub(r"\s+", " ", (query or "").strip())
    if not original:
        return "", False
    # One shared normalizer for every caller (this module's adapters, the
    # reasoning engine's retrieval retry, and the web research tool).
    from reasoning.semantic_resolution import normalize_research_query

    result = normalize_research_query(original, vocabulary=_vocabulary(known_entities))
    cleaned = (result.query or original).strip()
    return cleaned, cleaned.casefold() != original.casefold()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_topic(
    text: str,
    *,
    content_noun: str = "",
    known_entities: Iterable[str] | None = None,
    evidence: Iterable[str] = (),
    resolver: EntityResolver | None = None,
) -> TopicReading:
    """Extract the semantic subject from a request span.

    ``text`` may be the whole request or (preferably) the span already isolated
    after the action verb. ``content_noun`` is the captured content-type noun
    (e.g. "videos", "poem") which is separated from the subject rather than
    deleted, so "find videos about skyrim" yields content_noun="videos" and
    subject="skyrim".

    ``known_entities`` supplies canonical spellings for confident typo
    correction (e.g. a locally-known entity "Skyrim"). ``evidence`` supplies
    external spans (search titles/snippets) that may *themselves* contain the
    spelling the user meant, which is how a mistyped proper noun that Atlas has
    never seen before is resolved. When neither is supplied, no correction is
    attempted and the subject is preserved verbatim.
    """

    reading = TopicReading()
    body = (text or "").strip()
    lead = _ACTION_LEAD_RE.match(body)
    if not _open_question_frame(body) and lead and lead.end() > 0 and body[lead.end():].strip():
        body = body[lead.end():].strip()
    # Now strip the remaining instruction language: informational lead-ins and a
    # leading web platform, then reduce to the subject span.
    if not _open_question_frame(body):
        info_lead = _INFO_LEAD_RE.match(body)
        if info_lead and info_lead.end() > 0 and body[info_lead.end():].strip():
            # Record the framing noun ("information", "details") so the reading
            # still shows *what kind* of instruction was removed.
            phrase = info_lead.group(0).strip().casefold()
            reading.stripped_head = re.sub(
                r"\b(?:about|on|of|for|regarding|concerning)\b", "", phrase
            ).strip()
            body = body[info_lead.end():].strip()
        platform_lead = _PLATFORM_LEAD_RE.match(body)
        if platform_lead and platform_lead.end() > 0 and body[platform_lead.end():].strip():
            body = body[platform_lead.end():].strip()
    # Reduce to the span that can carry the subject (cut delivery clauses and a
    # destination preposition naming a known application).
    body = _trim_to_subject_span(body)
    # The raw subject is the span after delivery clauses are cut but before
    # connector/head-noun stripping, so an interpretation problem is always
    # inspectable from the reading ("about skyrim", not the whole request).
    reading.raw_topic = body.strip()

    # 1. Split the content-type noun off the subject.
    body, content_noun_found, notes = _split_content_noun(body, content_noun)
    reading.content_noun = content_noun_found
    reading.notes.extend(notes)

    # 2. Strip a leading metalinguistic head noun ("information about X").
    body, head, head_notes = _strip_meta_head(body)
    if head:
        reading.stripped_head = reading.stripped_head or head
    reading.notes.extend(head_notes)

    # 3. Strip a *leading* connector ("about X"). Only the leading one: an
    #    "about" inside the remaining noun phrase is semantic content.
    body, connector = _strip_leading_connector(body)
    # A meta head noun may follow the verb ("research information about X"):
    # strip it again after the lead so both orders reduce to the subject.
    if not connector:
        body, head2, more_notes = _strip_meta_head(body)
        if head2:
            reading.stripped_head = reading.stripped_head or head2
            reading.notes.extend(more_notes)
    reading.connector = connector
    reading.had_connector = bool(connector)

    # 4. Drop a bare leading determiner only when the remainder is still whole.
    subject = _strip_leading_determiner(body)

    subject = subject.strip(" ,.;:!?\"'")
    reading.subject = subject
    reading.qualifiers = [w for w in subject.casefold().split() if w in QUALIFIER_WORDS]

    # 5. Resolve the subject at the semantic level: a confident correction comes
    #    from the one shared resolver (vocabulary, local corpus, bounded lexical
    #    distance, external evidence, validated model suggestion). An uncertain
    #    term is preserved rather than silently invented.
    active = resolver or EntityResolver(vocabulary=_vocabulary(known_entities))
    reading.resolution = active.resolve_subject(
        subject,
        context=reading.raw_topic or (text or ""),
        evidence=evidence,
        kind="topic",
    )
    reading.normalized = (reading.resolution.value(subject) or subject).strip()
    reading.notes.extend(reading.resolution.notes)
    if reading.resolution.ambiguous:
        reading.notes.append("subject may refer to more than one thing")
    reading.query = reading.normalized
    return reading


def normalize_subject(
    subject: str,
    *,
    known_entities: Iterable[str] | None = None,
    evidence: Iterable[str] = (),
    resolver: EntityResolver | None = None,
) -> tuple[str, list[str]]:
    """Normalize a subject for a research query without changing its meaning.

    Returns ``(normalized, notes)``. This is a thin adapter over the shared
    :class:`reasoning.semantic_resolution.EntityResolver`, so topic
    normalization, destination normalization, and research-query normalization
    are the same mechanism with the same guard rails: a token is replaced only
    when the candidate is supported (known, one edit away, or backed by
    evidence), a real English word is never "corrected", and an uncertain term
    is preserved rather than invented.
    """

    active = resolver or EntityResolver(vocabulary=_vocabulary(known_entities))
    resolution = active.resolve_subject(subject, evidence=evidence, kind="topic")
    return resolution.value(subject), list(resolution.notes)


def _vocabulary(known_entities: Iterable[str] | None) -> list[str]:
    names: list[str] = []
    for entity in known_entities or ():
        text = str(entity or "").strip()
        if not text:
            continue
        names.append(text)
        # A multi-word known name also contributes its significant tokens, so a
        # single mistyped word inside a subject can still be matched
        # ("Skryim mods" against a known "Skyrim").
        for token in re.findall(r"[A-Za-z0-9]+", text):
            if len(token) >= 4:
                names.append(token)
    return list(dict.fromkeys(names))


def _apply_known_corrections(text: str, entities: list[str], notes: list[str]) -> str:
    """Legacy adapter: normalize a subject against known entity spellings.

    Retained for compatibility only. It delegates to the one canonical resolver
    (:class:`reasoning.semantic_resolution.EntityResolver`) instead of running a
    second, independent correction algorithm, so the two paths can never
    disagree about what the user meant.
    """

    resolution = EntityResolver(vocabulary=_vocabulary(entities)).resolve_subject(text)
    notes.extend(resolution.notes)
    return resolution.value(text)


def _bounded_edit_distance(a: str, b: str, *, limit: int = 1) -> int | None:
    """Legacy alias for :func:`reasoning.semantic_resolution.bounded_edit_distance`."""

    return bounded_edit_distance(a, b, limit=limit)


# ---------------------------------------------------------------------------
# Span reduction helpers
# ---------------------------------------------------------------------------


def _trim_to_subject_span(text: str) -> str:
    """Reduce a full request to the span that can carry the subject.

    Cuts trailing delivery/transformation clauses ("... and write it in
    notepad") and cuts at a destination preposition that precedes a known
    application, so the subject never absorbs the destination.
    """

    body = (text or "").strip()
    # 1. Cut at the first reporting clause verb ("... and write it ..."). A
    #    request that *opens* with an interrogative/how-to frame keeps its own
    #    verbs ("how to make a resume" is one subject), but an explicit
    #    "... and write/save ..." clause is still delivery, so that is cut.
    if _open_question_frame(body):
        # In a question frame, cut only at an explicit "and <delivery verb>".
        explicit = _AND_CLAUSE_RE.search(body)
        if explicit:
            body = body[: explicit.start()]
    else:
        clause = _TRAILING_CLAUSE_RE.search(body)
        if clause:
            body = body[: clause.start()]
    # 2. Cut at a destination preposition followed by a known application
    #    ("... about skyrim in notepad"). Only when a real app is named, so
    #    "... about the weather in tokyo" keeps "tokyo".
    for match in _DESTINATION_RE.finditer(body):
        tail = match.group(1).strip().casefold()
        if (
            any(re.search(rf"\b{re.escape(app)}\b", tail) for app in known_applications())
            or any(re.search(rf"\b{re.escape(host)}\b", tail) for host in known_web_platforms())
        ):
            body = body[: match.start()]
            break
    return body.strip()


def _split_content_noun(body: str, content_noun: str) -> tuple[str, str, list[str]]:
    """Separate a leading content-type noun from the subject it modifies."""

    notes: list[str] = []
    text = body.strip()
    lowered = text.casefold()
    # A content noun that leads a "connector" phrase ("videos about skyrim")
    # describes the kind of result, not the subject.
    if content_noun:
        pattern = re.compile(
            rf"^\s*(?:the|some|any|a|an)?\s*{re.escape(content_noun)}s?\b"
            rf"(?:\s+(?:of|about|on|regarding|concerning|for))?",
            re.IGNORECASE,
        )
        match = pattern.match(text)
        if match and match.end() > 0:
            remainder = text[match.end():].strip()
            if remainder:
                notes.append(f"separated content noun '{content_noun}' from subject")
                return remainder, content_noun, notes
    # Fallback: the request opens with the content noun ("videos about X") even
    # when the caller did not pass it.
    generic = re.match(
        r"^\s*(?:the|some|any|a|an)?\s*([a-z]+(?:s|es)?)\s+"
        r"(?:about|on|regarding|concerning|of|for)\s+(.+)$",
        lowered,
    )
    if generic and content_noun and generic.group(1).startswith(content_noun.casefold()):
        remainder = re.match(
            r"^\s*(?:the|some|any|a|an)?\s*[a-z]+(?:s|es)?\s+(.+)", text, re.IGNORECASE
        )
        if remainder:
            notes.append(f"separated content noun '{content_noun}' from subject")
            return remainder.group(1).strip(), content_noun, notes
    return text, "", notes


def _strip_meta_head(body: str) -> tuple[str, str, list[str]]:
    """Strip a leading metalinguistic head noun ("information about X")."""

    notes: list[str] = []
    text = body.strip()
    match = _META_HEAD_RE.match(text)
    if not match:
        return text, "", notes
    remainder = text[match.end():].strip()
    # Only strip when a real subject follows; a request that is *only* the head
    # noun ("look up information") keeps it so the ambiguity is preserved.
    if not remainder or len(remainder.split()) < 1:
        return text, "", notes
    head = match.group(0).strip().casefold()
    stripped = head.split()[-1] if head.split() else head
    notes.append(f"stripped framing noun '{stripped}'")
    return remainder, stripped, notes


def _strip_leading_connector(body: str) -> tuple[str, str]:
    """Strip a leading connector, preserving inner connectors.

    "on" is special: it is a connector when it introduces a subject ("on
    skyrim") but a preposition inside one ("the effects of X on Y"). Only a
    leading "on" followed by a bare subject (not "the ...") is stripped, so a
    legitimate noun phrase is never truncated.
    """

    text = body.strip()
    match = _CONNECTOR_RE.match(text)
    if not match:
        return text, ""
    connector = match.group(0).strip().casefold()
    remainder = text[match.end():].strip()
    # Never reduce the subject to nothing: "research about" alone keeps "about".
    if not remainder:
        return text, ""
    # A bare "on the <noun>" is a prepositional phrase, not a connector lead-in.
    if connector == "on" and remainder.casefold().startswith("the "):
        return text, ""
    return remainder, connector


def _open_question_frame(text: str) -> bool:
    return bool(_OPEN_QUESTION_RE.match(text or ""))


def _strip_leading_determiner(body: str) -> str:
    text = body.strip()
    words = text.split()
    # A leading determiner is dropped only from a *compound* subject, where it
    # reads as scaffolding ("about the history of skyrim" -> "history of
    # skyrim"). A short subject keeps its article, because there the determiner
    # is part of the noun phrase the user wrote ("a haiku about the ocean" -> "the
    # ocean", not "ocean").
    if len(words) > 3 and words[0].casefold() in LEADING_DETERMINERS:
        return " ".join(words[1:])
    return text
