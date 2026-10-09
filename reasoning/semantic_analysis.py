"""Deterministic structural analysis of a request's *shape*.

This module is the offline half of semantic understanding. It answers questions
about the **structure** of a request rather than matching a phrase table:

* Is this a comparative/ranking question (a superlative or an explicit
  comparison over a candidate set)?
* What criterion is it ranked by, and is that criterion subjective?
* Is there an objective proxy that would make a subjective criterion answerable?
* Does the request depend on the current world, or on stable knowledge?
* Is the request about *this machine* rather than the world?
* Does the subject come from an earlier turn (anaphora)?

Every answer is derived from grammar and from the *presence or absence of a
resolved subject*, never from `if message.startswith("who is")` or
`if "search" in message`. Word lists do appear, but only as *linguistic
vocabulary* (a set of superlative morphemes, a set of comparison markers) whose
job is to classify grammatical structure - the same way a parser needs a lexicon
of determiners. They never select a capability, a source, or an intent: that
decision is made later from the finished :class:`~reasoning.semantic_request.
SemanticRequest`.

Nothing here executes tools or mutates state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Superlative morphemes and their comparative direction. These describe the
#: *grammatical form* of a ranking question; they are not intent keywords. A
#: bare adjective ("famous", "popular") is deliberately absent: the superlative
#: form is "most famous"/"most popular", and listing the adjective alone would
#: make the compound-criterion guard misfire.
_SUPERLATIVES: tuple[str, ...] = (
    "most", "biggest", "largest", "greatest", "highest", "top", "best",
    "smallest", "lowest", "worst", "leading", "number one", "#1",
    "no. 1", "ranked first", "richest", "wealthiest", "strongest",
    "fastest", "oldest", "youngest", "newest", "latest", "fewest",
    "least", "winningest", "dominant", "dominates", "ranked",
)

#: The individual words inside ``_SUPERLATIVES``, used to avoid composing a
#: nonsense criterion like "most best" while still reading "most famous".
_SUPERLATIVE_WORDS: frozenset[str] = frozenset(
    word for term in _SUPERLATIVES for word in term.split()
)

#: Comparative markers that turn a request into an explicit comparison.
_COMPARATIVES: tuple[str, ...] = (
    "more than", "less than", "bigger than", "larger than", "smaller than",
    "compared to", "compare", "versus", " vs ", "difference between",
    "greater than", "fewer than", "better than", "worse than", "as big as",
)

#: Periphrastic comparatives: the *irregular/synthetic* comparatives whose form is
#: the comparison itself ("better", "worse", "faster"). A graded adjective is the
#: same semantic relation as "more X than", so recognising the morphology keeps
#: "Which laptop is better?" and "Which laptop is best?" on one code path.
#:
#: Note the deliberate omission: bare *dimensional* comparatives ("smaller",
#: "bigger", "newer", "older", "faster", "slower", "lighter", "heavier") are
#: **not** here. "Would that still work with a smaller GPU?" uses "smaller" as a
#: scale modifier on a hypothetical, not as a request to rank candidates. A
#: dimensional adjective only becomes a comparison together with a *choice frame*
#: (a non-specific head noun such as "which laptop" or "what phone"), which
#: :func:`_is_choice_comparison` recognises. That is the difference between
#: "Which laptop is smaller?" (rank) and "with a smaller GPU?" (sizing).
_GRADED_COMPARATIVES: tuple[str, ...] = (
    "better", "worse", "smarter", "cheaper", "pricier", "stronger", "weaker",
    "quieter", "most reliable", "more reliable", "more powerful", "less reliable",
)

#: Dimensional comparatives/superlatives: the *scale* vocabulary. These become a
#: ranking only when the request is a choice over candidates.
_DIMENSIONAL_COMPARATIVES: tuple[str, ...] = (
    "smaller", "bigger", "larger", "newer", "older", "faster", "slower",
    "lighter", "heavier", "taller", "shorter", "wider", "thinner", "cheapest",
    "most expensive", "least expensive",
)

#: A *choice frame*: an interrogative over a non-specific head noun ("which
#: laptop", "what phone", "which one"). It is the grammatical marker that the
#: user is choosing between candidates of that kind, which is what turns a
#: dimensional adjective into a ranking.
_CHOICE_FRAME_RE = re.compile(
    r"\b(?:which|what)\s+(?:kind\s+of\s+|type\s+of\s+|sort\s+of\s+)?"
    r"(?:one|ones|option|options|choice|choice\b|[a-z]+)\b",
    re.IGNORECASE,
)


def _is_choice_comparison(lowered: str) -> bool:
    """True when the request is a choice over candidates of some kind.

    "Which laptop is smaller?" and "Which phone is best?" choose between
    candidates; "with a smaller GPU" and "a faster build" modify a scale. The
    distinguishing structure is an interrogative over a non-specific head noun.
    """

    return bool(_CHOICE_FRAME_RE.search(lowered))


def _dimensional_comparative(lowered: str) -> str:
    """Return a dimensional comparative ("smaller", "newer"), if any."""

    padded = " " + re.sub(r"[^\w\s']+", " ", lowered) + " "
    for term in _DIMENSIONAL_COMPARATIVES:
        if f" {term} " in padded:
            return term
    return ""


#: Result nouns that can appear as the *subject* of a "the outcome went to X"
#: frame. The frame is what makes it an event result; the noun is the outcome.
_RESULT_SUBJECT_FRAME_RE = re.compile(
    r"\b(?:the|a)\s+(?:" + "|".join(
        re.escape(noun) for noun in (
            "title", "crown", "trophy", "championship", "championships", "medal",
            "gold", "victory", "final", "finals", "grand final", "prize",
            "belt", "cup", "trophy",
        )
    ) + r")\s+(?:went|goes|go|belongs|belong|is|was|are|were)\s+(?:to|with)\b",
    re.IGNORECASE,
)


def _graded_comparative(lowered: str) -> str:
    """Return the comparative adjective present, if any (morphology, not keywords).

    Punctuation is stripped first so a trailing "?" or "." does not hide a final
    adjective ("Which laptop is better?").
    """

    padded = " " + re.sub(r"[^\w\s']+", " ", lowered) + " "
    for term in _GRADED_COMPARATIVES:
        if f" {term} " in padded:
            return term
    return ""

#: Marker words that introduce the *criterion* a ranking is judged by.
_CRITERION_LEADS: tuple[str, ...] = (
    "by", "in terms of", "based on", "when it comes to", "for",
)

#: Criterion vocabulary the user may name, mapped to the objective proxy that
#: makes it answerable. The key is the user's *criterion*; the value is a
#: measurable stand-in. A criterion with no proxy stays subjective.
_CRITERION_PROXIES: dict[str, str] = {
    "famous": "subscriber count",
    "most famous": "subscriber count",
    "popular": "subscriber count",
    "most popular": "subscriber count",
    "influential": "follower count",
    "biggest": "subscriber count",
    "largest": "subscriber count",
    "top": "subscriber count",
    "best": "",
    "richest": "net worth",
    "wealthiest": "net worth",
    "oldest": "founding date",
    "newest": "release date",
    "fastest": "measured benchmark",
    "strongest": "measured benchmark",
}

#: Criterion names that are *inherently* subjective: no measurement settles them,
#: so a proxy may be offered but the reading must stay marked as ambiguous and
#: Atlas must say whose criterion it used.
_SUBJECTIVE_CRITERIA: frozenset[str] = frozenset(
    {
        "best", "greatest", "favourite", "favorite", "coolest", "funniest",
        "most fun", "most beautiful", "most important", "most influential",
        "most famous", "most popular", "most talented", "smartest",
    }
)

#: Linguistic markers of a *stable* fact: definitions, mechanisms, and
#: classifications. Their presence lowers the freshness requirement.
_STABLE_QUESTION_HEADS: tuple[str, ...] = (
    "what is", "what are", "what does", "define", "definition of",
    "how does", "how do", "why does", "why is", "explain",
)

#: Nouns whose value is *intrinsically* time-bound: an answer about them can
#: change at any moment, so it is always fresh.
#:
#: Deliberately *excluded*: "release", "version", and "news". They are the HEAD of
#: a named thing ("the latest Python release", "the Firefox version", "the news
#: about X"), not a live value of that thing. Treating them as intrinsically
#: current made the interpreter raise the freshness of an ordinary lookup and, in
#: turn, let the query builder replace the subject with a bare year. A request is
#: fresh because of what it *asks for* (a price, a winner), not because of the
#: noun that happens to be the object.
_INTRINSICALLY_CURRENT: tuple[str, ...] = (
    "president", "prime minister", "ceo", "chancellor", "governor", "mayor",
    "leader", "king", "queen", "pope", "champion", "championship", "winner", "record holder",
    "price", "stock", "score", "weather", "temperature",
    "population", "ranking", "rank", "subscribers", "views", "followers",
    "net worth", "exchange rate", "interest rate", "schedule",
)

#: Event-result vocabulary: the *relations* that ask for the factual outcome of an
#: event (who won, who took the title, what was the score). These words describe a
#: semantic relation, not a topic: any event can be the object of "won", so an
#: unfamiliar competition needs no new rule. Two classes are distinguished because
#: they license a different search query:
#:
#: * RESULT_OF - a verb/participle that names the outcome ("won", "crowned",
#:   "defeated"). The *object* of the verb is the event.
#: * RESULT_NOUN - a noun that names the outcome and can stand alone as the
#:   relation ("winner", "champion", "title", "score", "result").
_EVENT_RESULT_VERBS: tuple[str, ...] = (
    "won", "wins", "win", "won by", "beat", "beats", "defeated", "defeats",
    "crowned", "crowns", "took", "takes", "taken", "claimed", "claims",
    "secured", "secures", "captured", "captures", "clinched", "clinches",
    "went to", "emerged", "emerges", "triumphed", "survived", "prevailed",
    "ended up", "came out on top", "took home", "picked up", "earned", "snagged",
)

#: Nouns that name the outcome itself; paired with a verb or standing alone.
_EVENT_RESULT_NOUNS: tuple[str, ...] = (
    "winner", "winners", "champion", "champions", "championship", "championships",
    "title", "titles", "crown", "trophy", "medal", "gold", "score", "scoreline",
    "result", "results", "outcome", "victor", "victors", "victory", "final score",
    "standings", "final", "finals", "playoff", "playoffs", "grand final",
    "conference final", "semifinal", "quarterfinal", "bout", "match", "game",
)

#: The object of a result verb introduces the *event* being asked about. Used to
#: tell "who won the championship" (the event is the object) from "the Lakers won"
#: (the team is the subject and no event is named - the event is the context).
_EVENT_OBJECT_LEAD_RE = re.compile(
    r"\b(?:won|wins|win|beat|beats|defeated|defeats|crowned|crowns|took|takes|"
    r"taken|claimed|claims|secured|secures|captured|captures|clinched|clinches|"
    r"triumphed|prevailed|earned|snagged|is|was|are|were|for|at|in|of)\b\s+"
    r"(?:(?:the|a|an|this|that|last|next|latest|most\s+recent|previous)\s+)?",
    re.IGNORECASE,
)

#: Deictic temporal nouns: a bare "last night", "this year", "yesterday" is a
#: *temporal reference with no stated event* ("What happened last night?"). The
#: temporal expression is then the subject, and the request is a current-events
#: lookup about that window.
_DEICTIC_TIME_NOUNS: tuple[str, ...] = (
    "last night", "tonight", "today", "yesterday", "this morning", "this afternoon",
    "this evening", "this week", "last week", "this month", "last month",
        "this year", "last year", "three days ago", "two days ago", "a week ago",
        "days ago", "weeks ago", "months ago", "hours ago", "overnight", "so far",
        "right now", "recently", "consecutive year", "consecutive years", "in a row",
        "back to back", "back-to-back", "in history", "of all time",
    )

#: Temporal words that must never be the *subject* of a lookup: they name when,
#: not what. "I need to know who won yesterday" has its event supplied by context
#: (or by no one), and "yesterday" is certainly not the thing to search for.
_NEVER_SUBJECT_WORDS: frozenset[str] = frozenset(
    {
        "yesterday", "today", "tonight", "tomorrow", "now", "recently",
        "latest", "newest", "current", "currently", "last", "next", "this",
        "last night", "right now", "so far", "recent", "ago",
    }
)

#: Outcome *metaphors* that name a competition's prize rather than the competition
#: itself. "Who took the crown at the Baku Masters" asks about the Baku Masters, so
#: the metaphor is stripped and the actual event is what remains. This is the
#: lexical residue of the result relation, not a list of competitions.
_RESULT_METAPHOR_NOUNS: tuple[str, ...] = (
    "crown", "title", "trophy", "belt", "gold", "medal", "prize", "cup",
    "laurels", "honours", "honors", "top spot", "top prize", "number one spot",
)

#: Outcome *role* nouns: they name who holds the result ("the champion", "the
#: winner") rather than a competition. A bare one names no event, so it is dropped
#: when it is not qualified by an event.
_OUTCOME_ROLE_NOUNS: frozenset[str] = frozenset(
    {
        "champion", "champions", "winner", "winners", "victor", "victors",
        "champ", "champs",
    }
)

#: Nouns that ARE an event/competition - something a result can be an outcome of.
#: This is the test that separates "the championship" (an event, so it is the
#: subject) from "the title" (a prize, so it is not). It is a *semantic class*, not
#: an enumeration of competitions: it names the kinds of thing that can have a
#: winner, so an unfamiliar competition still matches by the frame it appears in.
_EVENT_INSTANCE_NOUNS: frozenset[str] = frozenset(
    {
        "championship", "championships", "final", "finals", "semifinal",
        "semifinals", "quarterfinal", "quarterfinals", "playoff", "playoffs",
        "tournament", "tournaments", "cup", "cups", "league", "leagues",
        "match", "matches", "game", "games", "race", "races", "series",
        "competition", "competitions", "contest", "contests", "event", "events",
        "season", "seasons", "round", "rounds", "tour", "tours", "bout", "bouts",
        "derby", "open", "masters", "olympics", "world cup", "super bowl",
        "superbowl", "grand prix", "marathon", "triathlon", "meet", "regatta",
    }
)


def _names_an_event(noun_phrase: str) -> bool:
    """True when a noun phrase names an event a result can be an outcome of."""

    lowered = noun_phrase.casefold()
    words = set(re.findall(r"[a-z0-9']+", lowered))
    if words & _EVENT_INSTANCE_NOUNS:
        return True
    # An extended event name ("the 2026 Baku Masters", "the Norwegian chess
    # league") carries its own specificity: a residual content word after the
    # determiner is enough to say the user named *something*, not a bare prize.
    return False


#: Temporal *scopes* the user may state that change how far back evidence must
#: reach ("two in a row", "of all time"). They are recognised here only so the
#: request is treated as current information about a real event; the authoritative
#: resolution of *how far back* lives in :mod:`reasoning.temporal_resolution`,
#: which owns every temporal decision.
_TEMPORAL_SCOPE_MARKERS: tuple[str, ...] = (
    "in a row", "consecutive year", "consecutive years", "back to back",
    "back-to-back", "in history", "of all time",
)


#: Markers that ask for the *outcome of an event that just happened* without naming
#: the event at all ("What happened?", "Did they win?"). These are current-events
#: questions: without a retrieved report, any answer would be invented.
_EVENT_OUTCOME_NOUNS: tuple[str, ...] = (
    "happened", "happening", "going on", "went down", "the news", "the latest",
    "the update", "the updates", "what's new", "whats new",
)

#: Grammar words that end a *subject* noun phrase. Used by the subject extractor to
#: stop at the verb phrase rather than swallowing the rest of the instruction
#: ("the newest iPhone" not "the newest iPhone please order it for me").
_NP_BOUNDARY_RE = re.compile(
    r"\s+(?:is|are|was|were|has|have|had|does|do|did|can|could|would|should|"
    r"will|won|wins|plays|played|take|took|become|became|going|goes|come|comes|"
    r"please|and|then|for|to|into|with|about|from)\b.*$",
    re.IGNORECASE,
)

#: Markers that make a request about the local machine rather than the world.
#: The screen/window vocabulary belongs here for the same reason the disk/CPU/ram
#: vocabulary does: "what is on my screen" is a question about *this machine* and
#: can never be answered by a public web search, so it must select the local
#: observation capability instead of a retrieval source.
_LOCAL_MARKERS: tuple[str, ...] = (
    "my computer", "my pc", "my machine", "my laptop", "my system",
    "this computer", "this pc", "this machine", "my disk", "my cpu", "my ram",
    "my files", "my documents", "my downloads", "my folder", "my desktop",
    "my screen", "the screen", "this screen", "on screen", "on my screen",
    "my monitor", "my window", "my windows", "open windows", "my browser",
)

#: Sizing/appearance words that describe the *requested output*, not the subject.
_OUTPUT_SHAPES: tuple[tuple[str, str], ...] = (
    ("list of", "list"), ("key points", "key_points"), ("summary", "summary"),
    ("table", "table"), ("ranking", "ranking"), ("comparison", "comparison"),
    ("step by step", "steps"), ("short", "concise"), ("brief", "concise"),
    ("concise", "concise"), ("detailed", "detailed"), ("in depth", "detailed"),
)

#: Pronouns/anaphora that make the subject a referent rather than a new subject.
_ANAPHORA: frozenset[str] = frozenset(
    {"it", "that", "this", "these", "those", "them", "they", "he", "she", "him",
     "her", "his", "hers", "their", "theirs", "one", "ones", "the same"}
)


def _contains_any(text: str, terms: tuple[str, ...]) -> str:
    """Return the first term present as a whole-word phrase, else ""."""

    for term in terms:
        if " " in term:
            if term in text:
                return term
        elif re.search(rf"\b{re.escape(term)}\b", text):
            return term
    return ""


@dataclass
class StructuralReading:
    """The grammatical shape of a request, before any capability planning."""

    comparative: bool = False
    superlative: bool = False
    criterion: str = ""
    criterion_proxy: str = ""
    criterion_subjective: bool = False
    candidate_set: str = ""
    freshness: str = "any"
    intrinsically_current: bool = False
    local: bool = False
    anaphoric: bool = False
    requested_output: str = ""
    question_form: bool = False
    imperative_form: bool = False
    #: True when the question asks for the factual outcome of a completed event
    #: (e.g. "Who won X?", "What was the score?"). Such questions depend on the
    #: current state of the world and must be treated as freshness="current"
    #: so the evidence layer does not silently answer them from model memory.
    final_event_result: bool = False
    #: True when the request asks for the outcome of an event by naming the result
    #: relation in any of its surface forms ("won", "took the title", "was crowned",
    #: "ended up winning", "who came out on top"). This is the *structural* signal;
    #: unlike :attr:`final_event_result` it does not require a question form, so an
    #: indirect request ("I need to know who won yesterday") is recognised too.
    event_result: bool = False
    #: The surface form of the result relation that fired (a diagnostic marker).
    event_result_term: str = ""
    #: True when the request names a *deictic time window* ("last night", "this
    #: year") without naming an event. The window itself is the subject, so the
    #: request is a current-events lookup rather than a lookup about an entity.
    deictic_time_question: bool = False
    #: True when the request asks for the factual outcome of an event but names no
    #: event and no subject ("What happened?", "Did they win?"). Nothing can be
    #: retrieved for it until the subject is established, so it must not be
    #: answered from model memory as if it were general knowledge.
    unanchored_event_question: bool = False
    #: True when the request is an *indirect* information request ("Can you check
    #: what Bitcoin is at?", "I need to know who won yesterday", "Find out which
    #: phone is newest"). The frame hands Atlas a question, so it is an
    #: information request and not a conversational message.
    indirect_request: bool = False
    #: True when the request asks for the current *value* of something (a price, a
    #: rate, a count, a version): "what is Bitcoin trading at", "how many
    #: subscribers does X have". The answer is a quantity that must be looked up,
    #: which is a different information need from "who won X".
    value_seeking: bool = False
    #: True when the request asks Atlas to *explain* something ("why did X win", "how
    #: does Y work", "what is the reason for Z"). The information need is reasoning
    #: about a subject Atlas is expected to know, which is different from a request
    #: to find information about an entity ("who is X"). The two must not be
    #: conflated: explaining is answered from knowledge, while asking *about* an
    #: entity may warrant external evidence.
    explanatory: bool = False
    #: True when the request asks for the *meaning of a word or term* ("What does
    #: \"current\" mean in physics?"). For a definition the temporal vocabulary in
    #: the question is the subject being defined, not a freshness requirement.
    definition: bool = False
    #: The term the user asked to have defined, when it could be isolated.
    definition_term: str = ""
    markers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparative": self.comparative,
            "superlative": self.superlative,
            "criterion": self.criterion,
            "criterion_proxy": self.criterion_proxy,
            "criterion_subjective": self.criterion_subjective,
            "candidate_set": self.candidate_set,
            "freshness": self.freshness,
            "intrinsically_current": self.intrinsically_current,
            "local": self.local,
            "anaphoric": self.anaphoric,
            "requested_output": self.requested_output,
            "question_form": self.question_form,
            "imperative_form": self.imperative_form,
            "final_event_result": self.final_event_result,
            "event_result": self.event_result,
            "event_result_term": self.event_result_term,
            "deictic_time_question": self.deictic_time_question,
            "unanchored_event_question": self.unanchored_event_question,
            "indirect_request": self.indirect_request,
            "value_seeking": self.value_seeking,
            "explanatory": self.explanatory,
            "definition": self.definition,
            "definition_term": self.definition_term,
            "markers": list(self.markers),
        }


#: A leading verb that makes a request an imperative (an action request).
_IMPERATIVE_LEAD_RE = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+|will you\s+|now\s+)*"
    r"(?:open|launch|start|run|create|write|make|generate|compose|produce|draft|"
    r"put|place|save|store|export|copy|move|rename|delete|remove|type|insert|"
    r"add|send|set|turn|convert|translate|summarize|summarise|shorten|expand|"
    r"rewrite|edit|format|organize|organise|sort|schedule|book|install)\b",
    re.IGNORECASE,
)

#: Quantity frames: the request asks *how much / how many* of something. The answer
#: is a number that must be looked up, which is a different information need from
#: "who won X".
_QUANTITY_FRAME_RE = re.compile(r"\b(?:how\s+many|how\s+much|what(?:'s|\s+is)\s+the\s+number)\b", re.IGNORECASE)

#: Value vocabulary: nouns whose *current value* is the thing asked for. This is a
#: lexical class the interpreter and query builder already share, not a topic list
#: of vendors or tickers - it names the kind of quantity, so any subject can carry
#: it ("the price of gold", "Bitcoin price").
_VALUE_NOUNS: tuple[str, ...] = (
    "price", "cost", "rate", "exchange rate", "interest rate", "value", "worth",
    "market cap", "valuation", "fee", "salary", "net worth", "balance",
    "version", "release date", "population", "subscribers", "subscriber count",
    "views", "followers", "score", "temperature", "stock price", "share price",
)

#: A value noun that is the *head of a named thing* rather than the property being
#: asked about ("the latest Python release", "the new version of Windows") is part
#: of the subject, so the request is not a value lookup. The test is whether a
#: content word precedes the value noun in the same noun phrase *and* the request
#: is not a question about a quantity: "what is the price of X" is a value request
#: even though "price" is preceded by "the".
_VALUE_NOUN_HEAD_RE = re.compile(
    r"\b[a-z][a-z0-9.'\-]+\s+(?:"
    + "|".join(
        re.escape(noun)
        for noun in ("release", "release date", "version", "price", "cost", "rate", "value")
    )
    + r")\b",
    re.IGNORECASE,
)


def _subject_contains_content(lowered: str) -> bool:
    """True when the value noun is part of a named thing, not the property asked for.

    "What is Bitcoin trading at?" asks for a property of Bitcoin; "the latest
    Python release" names a thing whose *head* is a value noun. The difference is
    whether a content word qualifies the value noun in the same phrase.
    """

    return bool(_VALUE_NOUN_HEAD_RE.search(lowered))

#: Verbs/phrases that ask for a current value without naming the noun ("what is
#: Bitcoin *trading at*", "what is X *going for*", "what is it *at* now"). The
#: verb *is* the value relation, so any subject can carry it.
_VALUE_VERB_RE = re.compile(
    r"\b(?:trading\s+at|traded\s+at|going\s+for|going\s+at|sell(?:ing)?\s+for|"
    r"cost(?:ing|s)?|priced\s+at|valued\s+at|worth|standing\s+at|currently\s+at|"
    r"\w+\s+is\s+at|\w+\s+are\s+at)\b",
    re.IGNORECASE,
)


def _is_quantity_subject(lowered: str) -> bool:
    """True when the answer to the request is a *value* rather than an entity.

    The quantity frame ("how many"), the value vocabulary ("price", "version"),
    and the value verbs ("trading at", "going for") are the three ways a value
    request is phrased. They are the *semantic class* of the answer, so they apply
    to any subject ("the price of gold", "Bitcoin price") without naming one.
    """

    return bool(
        _QUANTITY_FRAME_RE.search(lowered)
        or _contains_any(lowered, _VALUE_NOUNS)
        or _VALUE_VERB_RE.search(lowered)
    )


#: An interrogative opening.
_QUESTION_RE = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+|tell me\s+|do you know\s+)*"
    r"(?:what|who|whom|whose|when|where|why|how|which|is|are|was|were|does|do|did|"
    r"can|could|should|would|will|has|have|had)\b",
    re.IGNORECASE,
)


def analyze_structure(text: str) -> StructuralReading:
    """Return the grammatical structure of ``text`` (no capability decisions)."""

    reading = StructuralReading()
    lowered = " " + " ".join((text or "").casefold().split()) + " "
    stripped = lowered.strip()

    reading.question_form = bool(_QUESTION_RE.search(stripped)) or "?" in stripped
    reading.imperative_form = bool(_IMPERATIVE_LEAD_RE.search(stripped))
    if reading.question_form:
        reading.markers.append("question_form")
    if reading.imperative_form:
        reading.markers.append("imperative_form")

    # -- definition frame ------------------------------------------------------
    # When the user asks what a *word* means ("What does \"current\" mean in
    # physics?", "Define release date"), the temporal vocabulary in the question is
    # the *subject* being defined, not a freshness requirement. Reading it as a
    # freshness signal is the classic metalinguistic false positive, so the frame is
    # detected here - before freshness - and the temporal reading is suppressed for
    # it. The detection is grammatical: an interrogative asking for a meaning, with
    # the term in the subject position.
    definition = re.match(
        r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
        r"(?:"
        # "what does X mean (in Y)?" / "what does X refer to?"
        r"(?:what|which)\s+(?:does|do)\s+(?P<term1>.+?)\s+"
        r"(?:mean|refer\s+to|signify)\b"
        # "what is the meaning of X?"
        r"|(?:what|which)\s+(?:is|are)\s+the\s+meaning\s+of\s+(?P<term2>.+?)\s*\??$"
        # "define X" / "explain what X means" / "what is X" as a bare definition
        r"|(?:define|explain\s+what)(?:\s+(?:the\s+)?(?:term|word))?\s+(?P<term3>.+?)"
        r"(?:\s+means?)?\s*\??$"
        r")",
        stripped,
        re.IGNORECASE,
    )
    if definition:
        reading.definition = True
        reading.markers.append("definition_frame")
        # Suppress only the intrinsic-current reading: a definition is stable
        # knowledge whatever noun it happens to name.
        reading.definition_term = next(
            (group for group in definition.groups() if group),
            "",
        ).strip()

    # -- comparison / ranking --------------------------------------------------
    # An ordinal "#1"/"number one"/"no. 1" is a ranking claim whose marker is not
    # a word boundary match; check it before the word-based superlatives.
    ordinal_rank = re.search(r"(?:#\s*1\b|\bnumber\s+one\b|\bno\.?\s*1\b|\branked\s+first\b)", lowered)
    superlative = _contains_any(lowered, _SUPERLATIVES)
    if ordinal_rank and not superlative:
        superlative = "top"
    comparative_marker = _contains_any(lowered, _COMPARATIVES)
    # A periphrastic comparison ("better", "worse") is the *morphology* of
    # comparison, not a keyword: it is the same relation as "more X than" and is
    # read the same way so an unseen phrasing needs no new rule. A *dimensional*
    # comparative ("smaller", "newer") is a scale modifier unless the request is a
    # choice over candidates, which is what separates "Which laptop is smaller?"
    # from "Would that work with a smaller GPU?".
    graded = _graded_comparative(lowered)
    dimensional = _dimensional_comparative(lowered)
    if superlative:
        reading.superlative = True
        reading.comparative = True
        reading.markers.append(f"superlative:{superlative}")
    if comparative_marker:
        reading.comparative = True
        reading.markers.append(f"comparative:{comparative_marker.strip()}")
    if graded:
        reading.comparative = True
        reading.markers.append(f"graded_comparative:{graded}")
    if dimensional and _is_choice_comparison(lowered):
        reading.comparative = True
        reading.markers.append(f"choice_comparative:{dimensional}")

    # -- criterion -------------------------------------------------------------
    reading.criterion = _extract_criterion(lowered, superlative)
    if reading.criterion:
        reading.criterion_subjective = reading.criterion in _SUBJECTIVE_CRITERIA
        reading.criterion_proxy = _CRITERION_PROXIES.get(reading.criterion, "")
        if reading.criterion_subjective and not reading.criterion_proxy:
            reading.markers.append("subjective_criterion_without_proxy")
    if reading.comparative and not reading.candidate_set:
        reading.candidate_set = _extract_candidate_set(stripped, reading.criterion)

    # -- freshness -------------------------------------------------------------
    # The event-result relation is a strong, structural freshness signal, so it is
    # tested first: "Who took the title this year?" and "Who ended up winning?"
    # name no superlative and no intrinsically-current noun, yet their truth
    # depends on what actually happened. Reading it here means the freshness
    # decision does not depend on the *wording* of the result relation.
    event_result, event_term = _event_result_relation(lowered, question_form=reading.question_form)
    intrinsic = "" if reading.definition else _contains_any(lowered, _INTRINSICALLY_CURRENT)
    if event_result:
        reading.freshness = "current"
        reading.markers.append(f"event_result_is_current:{event_term}")
    elif intrinsic:
        reading.intrinsically_current = True
        reading.freshness = "current"
        reading.markers.append(f"intrinsically_current:{intrinsic}")
    elif re.search(r"\b(?:how\s+many|how\s+much)\b", lowered):
        reading.intrinsically_current = True
        reading.freshness = "current"
        reading.markers.append("quantity_question")
    elif re.search(r"\b(?:what\s+happened\s+(?:with|to|around)|happened\s+(?:with|to))\b", lowered):
        reading.intrinsically_current = True
        reading.freshness = "current"
        reading.markers.append("event_question")
    # -- value seeking ---------------------------------------------------------
    # A request for the current *value* of something is a distinct information need
    # from a request for an *entity* ("who won"). It is one decision, made here
    # from the quantity frame, the value vocabulary, and the value verbs - rather
    # than from a separate regex per phrasing - and it is what makes the query
    # builder ask for a value instead of for an outcome.
    #
    # A *question form is required*: "give me the latest news" and "search for the
    # latest Python release" name a time-bound topic, but they do not ask for a
    # value - treating them as value requests changed their queries for the worse.
    reading.value_seeking = bool(
        reading.question_form
        and (
            _QUANTITY_FRAME_RE.search(lowered)
            or _VALUE_VERB_RE.search(lowered)
            # A value *noun* names the quantity asked for wherever it sits: "the
            # price of a barrel of Brent crude" is a value request, and a preceding
            # content word ("Brent") qualifies the commodity, not the property, so
            # this branch is not gated on the noun being head-final.
            or _contains_any(lowered, _VALUE_NOUNS)
        )
    )
    if reading.value_seeking:
        reading.freshness = "current"
        reading.markers.append("value_seeking")
    elif reading.superlative:
        # A "who is the most X" question is a statement about the world as it is
        # now: the leader can change, so it is a current question by structure.
        reading.freshness = "current"
        reading.markers.append("superlative_is_current")
    elif reading.definition:
        # A definition is stable knowledge whatever noun it names ("what does
        # release mean", "define interest rate"): the vocabulary is the subject.
        reading.freshness = "stable"
        reading.markers.append("definition_is_stable")
    elif _contains_any(lowered, _STABLE_QUESTION_HEADS):
        reading.freshness = "stable"
        reading.markers.append("stable_definition")
    elif reading.freshness != "current":
        reading.freshness = "any"

    # -- locality --------------------------------------------------------------
    local_marker = _contains_any(lowered, _LOCAL_MARKERS)
    if local_marker:
        reading.local = True
        reading.markers.append(f"local:{local_marker}")

    # -- anaphora --------------------------------------------------------------
    words = set(re.findall(r"[a-z']+", lowered))
    if words & _ANAPHORA and reading.question_form or (words & _ANAPHORA and reading.imperative_form):
        reading.anaphoric = True
        reading.markers.append("anaphoric_subject")

    # -- final event result ----------------------------------------------------
    # A question that directly asks for the factual outcome of a completed
    # event ("who won X?", "what was the score?", "what happened in X?") asks
    # for information whose truth depends on what actually happened, not on
    # general knowledge. It is flagged so the freshness layer treats it as
    # current and the evidence layer does not silently answer it from model
    # memory. Explanatory leads (why, how, when) are excluded: "Why did X win?"
    # is an explanation, not a request for the event result.
    if reading.question_form and not re.search(
        r"^(?:why|how|when|where|explain|describe|tell\s+me)\b", lowered
    ):
        if re.search(
            r"\b(?:who|which|what)\s+(?:was\s+the\s+)?(?:winner|winners|score|result|champion|championship)\b"
            r"|\b(?:who|which)\s+won\b"
            r"|\b(?:who|which)\s+\w+\s+won\b"
            r"|\bwhat\s+happened\s+(?:in|with|to|around|during)\b",
            lowered,
        ):
            reading.final_event_result = True
            reading.markers.append("final_event_result")

    # -- event-result relation (structural, paraphrase-independent) -------------
    # The generic form of "who won X", "who took the title", "who was crowned
    # champion", "who ended up winning". What makes a request an *event-result*
    # request is that it names a result relation whose object is an event; the
    # wording of the relation is surface variation, so it is read from vocabulary
    # (a result verb, or a result noun next to a possessive/of-frame) rather than
    # from one enumeration of team names or competitions. The relation is resolved
    # once during freshness above and reused here.
    reading.event_result = event_result
    reading.event_result_term = event_term
    if reading.event_result:
        reading.final_event_result = True
        reading.markers.append(f"event_result:{reading.event_result_term}")

    # -- deictic time window ("last night", "this year") ----------------------
    # A window named without an event is itself the subject: "What happened last
    # night?" asks for a report covering that window, not for general knowledge.
    deictic = _contains_any(lowered, _DEICTIC_TIME_NOUNS)
    if deictic:
        reading.deictic_time_question = True
        if reading.freshness != "current":
            # An explicitly named recent window (this year, last night, yesterday,
            # three days ago) is a current-information reference even when the
            # request names no event and no intrinsically-current noun.
            reading.freshness = "current"
        reading.markers.append(f"deictic_time:{deictic}")

    # -- unanchored event question -------------------------------------------
    # A current-events question with no event and no subject of its own ("What
    # happened?", "Did they win last night?"). It cannot be answered from model
    # memory: it needs a retrieved report about a subject that must first be
    # established (from the time window, or from the conversation).
    if reading.question_form and not _names_concrete_subject(lowered, event_result=reading.event_result):
        asks_outcome = (
            reading.event_result or _contains_any(lowered, _EVENT_OUTCOME_NOUNS)
            # A request for a factual result (a score, a winner) with no event and
            # no entity is unanchored even when it names no result *verb*.
            or bool(re.search(r"\b(?:result|results|score|scoreline|standings)\b", lowered))
        )
        if asks_outcome:
            reading.unanchored_event_question = True
            reading.freshness = "current"
            reading.markers.append("unanchored_event_question")
            reading.markers.append("unanchored_is_current")

    # -- indirect information request ------------------------------------------
    # "Can you check what Bitcoin is at?", "Find out which phone is newest",
    # "I need to know who won yesterday". The frame is not an action request: it
    # hands Atlas an information question. It is marked so the evidence layer
    # treats it as external information rather than as a *conversational* message
    # that never asked to look anything up.
    reading.indirect_request = _is_indirect_information_request(stripped)
    if reading.indirect_request:
        reading.markers.append("indirect_request")
        if not _contains_any(lowered, _STABLE_QUESTION_HEADS):
            reading.freshness = "current"

    # -- explanation vs lookup -------------------------------------------------
    # "Why did X win?" and "How does Y work?" ask Atlas to *reason about* a known
    # subject; "Who is X?" and "Tell me about X" ask Atlas to *tell them about* an
    # entity. The two need different evidence treatment, so the distinction is read
    # from the question's lead and its object, never from the subject's name.
    reading.explanatory = bool(
        reading.question_form
        and re.match(r"^\s*(?:why|how|explain|describe)\b", stripped)
        and not re.match(r"^\s*(?:how\s+many|how\s+much)\b", stripped)
    )
    if reading.explanatory:
        reading.markers.append("explanatory_question")

    # -- requested output shape ------------------------------------------------
    for phrase, shape in _OUTPUT_SHAPES:
        if phrase in lowered:
            reading.requested_output = shape
            break

    return reading


#: A result relation reads as a *noun of outcome* either bare after a question
#: frame ("who is the champion") or as the object of a possessive/of-clause
#: ("the winner of X", "X's champion").
_RESULT_NOUN_RE = re.compile(
    r"\b(?:the|who|which|what|whose|a|an|\w+'s)\s+(?:"
    r"(?:final|grand|national|world|league|conference)\s+)?"
    r"(?P<noun>" + "|".join(re.escape(noun) for noun in _EVENT_RESULT_NOUNS) + r")\b",
    re.IGNORECASE,
)

#: A result verb whose object is an event. The object may be separated from the
#: verb by a determiner and an arbitrary (multi-word) event name, which is why the
#: test is on the verb only: the *presence* of a result verb with an object is the
#: structural fact, and the event name is captured later as the subject.
_RESULT_VERB_RE = re.compile(
    r"\b(?P<verb>" + "|".join(
        re.escape(verb) for verb in sorted(_EVENT_RESULT_VERBS, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)

#: Frames that name an event as the *subject* rather than asking about one
#: ("The Lakers won last night"). A named actor followed by a result verb is a
#: statement about a completed event, which is still a current fact.
_RESULT_STATEMENT_RE = re.compile(
    r"^\s*(?:did|does|has|have|had)?\s*(?:the|a|an)?\s*"
    r"[A-Za-z][A-Za-z0-9 .&'-]{1,40}?\s+"
    r"(?:" + "|".join(re.escape(verb) for verb in _EVENT_RESULT_VERBS) + r")\b",
    re.IGNORECASE,
)


def _event_result_relation(lowered: str, *, question_form: bool) -> tuple[bool, str]:
    """Return (is_event_result, surface_term) for a request's result relation.

    Two shapes qualify, and they are the same relation:

    * a **result noun** naming the outcome - "who took *the title*", "who is
      *the champion*"; the noun phrase is a definite description of the outcome;
    * a **result verb** with an object - "who *won* X", "who *claimed* the crown".

    A bare statement about an actor ("the Lakers won") also encodes an event
    result, because its truth depends on what happened.
    """

    # An *explanatory* question about a result ("Why did X win?", "How did X take
    # the title?") asks for a reason, not for the outcome. Its truth does not
    # depend on a fresh fact about who won, so it is not an event-result request.
    # This is a grammatical rule about the question's lead, not a topic filter:
    # "Why did X win" and "Who won X" are different questions about one event.
    # It is checked before the result-noun and result-verb paths so the *reason*
    # reading wins for the whole freshness/evidence decision.
    if question_form and re.match(r"^\s*(?:why|how)\b", lowered):
        return False, ""
    noun = _RESULT_NOUN_RE.search(lowered)
    if noun:
        return True, noun.group("noun").casefold()
    # "the title went to X" / "the crown belongs to X": the outcome is the subject
    # and the winner is the object, so the result noun is the wrong anchor - the
    # *object* is. Recognising the frame keeps the paraphrase on the same path.
    subject_frame = _RESULT_SUBJECT_FRAME_RE.search(lowered)
    if subject_frame:
        return True, subject_frame.group(0).split()[0].casefold()
    verb = _RESULT_VERB_RE.search(lowered)
    if verb:
        return True, verb.group("verb").casefold()
    # Copular shapes: "who became champion", "who is the champion of X",
    # "who was named champion". The subject complement is the result noun, which
    # :data:`_RESULT_NOUN_RE` already covers when it is definite; this catches the
    # bare copular form ("became champion" has no article).
    if question_form and re.search(
        r"\b(?:became|become|becomes|was|were|is|are|named|declared|named)\s+"
        r"(?:(?:the|a|an)\s+)?(?:" + "|".join(
            re.escape(result_noun) for result_noun in _EVENT_RESULT_NOUNS
        ) + r")\b",
        lowered,
    ):
        match = re.search(
            r"\b(?:" + "|".join(re.escape(n) for n in _EVENT_RESULT_NOUNS) + r")\b",
            lowered,
        )
        return True, match.group(0).casefold() if match else "champion"
    if question_form and re.search(r"\bended\s+up\b.*\bwin", lowered):
        return True, "ended up winning"
    return False, ""


def _names_concrete_subject(lowered: str, *, event_result: bool = False) -> bool:
    """True when the request names a concrete subject the answer could be about.

    Used to separate an *unanchored* current-events question ("What happened?",
    "Did they win?") from a request about a real entity ("Did the Lakers win last
    night?") or about a named event ("Who won the championship?"). Only structural
    evidence counts: a capitalised proper-noun-like run, a content noun that is not
    a pronoun/determiner, or a non-stopword of length >= 4 that is not itself
    temporal or event vocabulary.

    When the request *is* an event-result request, the result's own object names
    the anchor (the championship, the final, the Baku Masters), so the result noun
    must not itself be discarded as "mere result vocabulary".
    """

    vocabulary = list(_DEICTIC_TIME_NOUNS) + list(_EVENT_OUTCOME_NOUNS)
    if not event_result:
        # The result noun is only an anchor for an event-result request; for any
        # other request it is ordinary content and stays in the token stream.
        vocabulary.extend(_EVENT_RESULT_NOUNS)
    stripped = re.sub(
        r"\b(?:" + "|".join(re.escape(term) for term in vocabulary) + r")\b",
        " ",
        lowered,
    )
    tokens = [
        token for token in re.findall(r"[a-z0-9']+", stripped)
        if token not in _EVENT_SUBJECT_STOPWORDS and len(token) >= 4
    ]
    return bool(tokens)


def _is_indirect_information_request(lowered: str) -> bool:
    """True when the request indirectly asks Atlas to find a fact out.

    The test is compositional, not a phrase match: there must be a *request frame*
    that hands Atlas a task ("can you check", "find out", "I need to know") AND the
    thing handed over must be a *question* ("what Bitcoin is at", "which phone is
    newest", "who won yesterday"). Both halves are required, which is what keeps
    "Can you write a poem?" and "Tell me about recursion." conversational while
    making "Can you check what Bitcoin is at?" an information request.
    """

    lead = _INDIRECT_LEAD_RE.match(lowered)
    if not lead:
        return False
    return bool(_EMBEDDED_QUESTION_RE.match(lead.group("clause") or ""))


#: Function words, pronouns, and result vocabulary that do not name a subject.
_EVENT_SUBJECT_STOPWORDS: frozenset[str] = frozenset(
    {
        "who", "whom", "whose", "what", "which", "when", "where", "why", "how",
        "did", "does", "was", "were", "are", "the", "that", "this", "these",
        "those", "they", "them", "their", "there", "then", "than", "have",
        "has", "had", "will", "would", "could", "should", "took", "take",
        "taken", "won", "wins", "winner", "winners", "champion", "champions",
        "championship", "title", "crown", "trophy", "medal", "score", "result",
        "victory", "victor", "game", "match", "final", "finals", "last", "latest",
        "newest", "most", "recent", "current", "currently", "today", "tonight",
        "yesterday", "night", "year", "week", "month", "know", "check", "find",
        "out", "need", "tell", "please", "about", "from", "with", "into", "going",
        "went", "down", "right", "over", "again", "been", "just", "only",
        "became", "become", "becomes", "crowned", "emerged", "ended", "claimed",
        "took", "claimed", "defeated", "beat", "beats", "wins", "triumphed",
    }
)


#: Subject frames that are *indirect requests*: the user asks Atlas to find out /
#: check / verify something instead of asking directly. The frame is stripped so
#: the subject is the thing asked about, and the request is understood as a
#: request for external information ("Can you check what Bitcoin is at?" is the
#: indirect form of "What is Bitcoin trading at?").
#:
#: The vocabulary is narrow on purpose. A bare "tell me about X" or "can you write
#: a poem" is NOT an indirect information request - the user is asking Atlas to
#: explain or produce something, not to go and find a fact. Only frames that hand
#: Atlas an *embedded question* qualify, and the embedded clause must actually
#: begin with an interrogative word. That is what makes
#: "Can you check what Bitcoin is at?" an information request while
#: "Can you improve that?" stays a conversational one.
_INDIRECT_LEAD_RE = re.compile(
    r"^\s*(?:please\s+|hey\s+atlas[,\s]+)*"
    r"(?:"
    r"(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"(?:check|verify|confirm|see|find\s+out|look\s+up|tell\s+me|let\s+me\s+know)\s+"
    r"|(?:find\s+out|check|verify|confirm|see|look\s+up)\s+"
    r"|i\s+(?:need|want|would\s+like)\s+to\s+know\s+"
    r"|i'd\s+like\s+to\s+know\s+"
    r"|let\s+me\s+know\s+"
    # "Tell me who took the championship" hands Atlas an embedded *question*; the
    # frame is a request to answer it, so it is an indirect information request.
    r"|tell\s+me\s+"
    r"|(?:do\s+you\s+)?(?:know|happen\s+to\s+know)\s+"
    r")"
    r"(?P<clause>\S.*)$",
    re.IGNORECASE,
)

#: The embedded question must itself be a question ("what Bitcoin is at", "which
#: phone is newest", "who won yesterday"). This is what separates an indirect
#: *information* request from an indirect *instruction* ("can you write a poem").
#: The interrogative may also sit inside the clause ("which phone is newest"), so
#: the check is for a question opener anywhere in the first few words.
_EMBEDDED_QUESTION_RE = re.compile(
    r"^\s*(?:(?:what|which|who|whose|when|where|whether|if|how\s+much|how\s+many|"
    r"what'?s|who'?s|which'?s)\b|"
    r"(?:does|do|did|is|are|was|were|can|could|will|would)\s+\S+\s+"
    r"(?:win|wins|won|go|goes|cost|costs|work|stand|stands|matter|matters))",
    re.IGNORECASE,
)

#: The frame word aliases used to detect an indirect request without re-parsing.
_INDIRECT_SUBJECT_FRAME_RE = _INDIRECT_LEAD_RE

#: A bare "what/which X is ..." after an indirect lead still names the subject in
#: object position ("which phone is newest"), so it is captured rather than left
#: attached to the interrogative word.
_INDIRECT_OBJECT_RE = re.compile(
    r"^\s*(?:what|which|who|whose)\s+(?P<subject>[A-Za-z0-9][A-Za-z0-9 .'\-]*?)\s+"
    r"(?:is|are|was|were|has|have|does|do|did|won|wins|goes|going|costs|cost)\b",
    re.IGNORECASE,
)


#: An *explanatory actor* frame ("Why did the 2024 Lakers win?", "How did Arsenal
#: take the title?"): the actor between the auxiliary and the result verb is the
#: subject of the question, even though the question asks for a reason. The reason
#: is answered *about* that actor, so the actor must be extracted.
_EXPLANATORY_ACTOR_RE = re.compile(
    r"^\s*(?:why|how)\s+did\s+(?P<subject>[A-Za-z0-9][A-Za-z0-9 .&'-]{1,50}?)\s+"
    r"(?:" + "|".join(re.escape(verb) for verb in _EVENT_RESULT_VERBS) + r")\b",
    re.IGNORECASE,
)


#: Grammatical *frame prefixes* that introduce the thing being asked about. The
#: regex captures nothing: :func:`extract_subject` matches the frame and takes
#: everything after it as the subject. These are syntactic constructions (a
#: copula, a connector, a retrieval verb), not intent keywords: the same frame
#: carries an arbitrary subject, so an unseen subject needs no new rule.
#: Order matters: a more specific frame ("what do you know about X") must be tried
#: before a shorter one ("what is X") so it is not shadowed.
_SUBJECT_FRAME_RE = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+|would you\s+|will you\s+|hey\s+atlas[,\s]+)*"
    r"(?:"
    r"what\s+do\s+you\s+know\s+about\s+"
    r"|(?:can|could|would)\s+you\s+tell\s+me\s+(?:about|more\s+about)\s+"
    r"|tell\s+me\s+(?:about|more\s+about)\s+"
    r"|(?:give|show)\s+me\s+(?:some\s+)?(?:information|info|details|facts|background)\s+(?:about|on|regarding)\s+"
    r"|i\s+want\s+(?:some\s+)?(?:information|info|details)\s+(?:about|on|regarding)\s+"
    r"|(?:search|find|look\s+up|look\s+for|google|research|fetch|pull\s+up)\s+(?:for\s+|about\s+|up\s+|on\s+)?"
    # A plain request for an object ("Give me the newest iPhone", "Show me the
    # latest Pixel") names its subject in the object position after the verb.
    r"|(?:give|show|get|send|bring)\s+me\s+(?:the\s+|a\s+|an\s+)?"
    r"|(?:who|what|which|whose)\s+is\s+#\s*1\b\s*(?:in|on|at|for)?\s*"
    r"|(?:who|what|which|whose)\s+is\s+(?:number\s+one|no\.?\s*1)\b\s*(?:in|on|at|for)?\s*"
    # Event-result frames come BEFORE the generic copula frame: "Who won the
    # championship" contains no copula, but "Who became champion" and "Who was
    # crowned champion" do, and the generic "who is/are" frame would otherwise
    # swallow them and capture the wrong span. The *object* of a result verb is
    # the event being asked about, so the verb is what identifies the frame; the
    # event itself is arbitrary, which is what makes an unfamiliar competition
    # need no new rule. The determiner is optional so the captured subject is a
    # noun phrase ("championship") rather than the word "the".
    r"|(?:who|what|which|whose)\s+(?:(?:is|was|were|are|has|have)\s+)?"
    r"(?:(?:has|have|had)\s+)?"
    r"(?:won|wins|win|beat|beats|defeated|defeats|claimed|claims|captured|"
    r"captures|secured|secures|clinched|clinches|took|takes|taken|crowned|"
    r"crowns|survived|prevailed|earned|snagged|became|become|becomes)\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"|(?:the|a|an)\s+"
    r"(?:title|crown|trophy|championship|championships|medal|gold|victory|cup|belt|"
    r"prize|final|finals)\s+(?:went|goes|go|belongs|belong)\s+(?:to|with)\s+"
    r"|(?:what|which|who)\s+happened\s+(?:in|during|at|to|with|around)\s+"
    r"|(?:did|does|do)\s+(?P<subject2>[A-Za-z][A-Za-z0-9 .&'-]{1,50}?)\s+"
    r"(?:win|wins|won|beat|beats|lose|loses|lost|play|plays|played)\b"
    r"|(?:who|what|which|whose)\s+(?:is|are|was|were)\s+(?:the\s+)?"
    r"|(?:which|what)\s+[a-z][a-z0-9 '-]*?\s+(?:has|have|gets|with)\s+(?:the\s+)?"
    r"|(?:who|what|which|whose)\s+has\s+(?:the\s+)?"
    r")"
    r"(?P<subject>.+?)$",
    re.IGNORECASE,
)

#: Trailing clauses that belong to the *instruction*, not the subject.
_SUBJECT_TRAILING_RE = re.compile(
    r"\s+(?:and|then|please|in\s+notepad|into\s+notepad|to\s+a\s+file|in\s+a\s+file|"
    r"for\s+me|right\s+now|now)\b.*$",
    re.IGNORECASE,
)

#: Predicates that trail a subject and ask about a *property* of it rather than
#: naming it: "MrBeast **known for**", "MrBeast **famous for**". Stripped so the
#: subject stays the entity the user is asking about.
_SUBJECT_TRAILING_PREDICATE_RE = re.compile(
    r"\s+(?:known|famous|noted|recognized|celebrated)\s+(?:for|as)\b.*$",
    re.IGNORECASE,
)

#: A predicative adjective clause left on the subject by embedding ("which phone is
#: newest" -> "phone is newest"). The adjective is the property asked about, not
#: part of the entity, so the copula+adjective tail is stripped. The adjective may
#: be a comparative/superlative form ("newest", "better") or a plain gradable one.
_SUBJECT_TRAILING_PREDICATIVE_RE = re.compile(
    r"\s+(?:is|are|was|were|will\s+be|becomes?|became)\s+"
    r"(?:the\s+)?"
    r"(?:[a-z]+est|better|worse|best|worst|newer|older|bigger|smaller|"
    r"faster|slower|cheaper|more\s+[a-z]+|most\s+[a-z]+|[a-z]+er)\b.*$",
    re.IGNORECASE,
)

#: A comparative/ranking property clause: "... has the most subscribers",
#: "... with the largest audience". The *candidate set* is what precedes it; the
#: property is the criterion, which is captured separately.
_SUBJECT_PROPERTY_CLAUSE_RE = re.compile(
    r"\s+(?:has|have|with|gets|got|earning|earns|owns|controls|dominates)\s+(?:the\s+)?"
    r"(?:most|biggest|largest|highest|greatest|top|best|smallest|lowest|fewest|"
    r"number\s+one|#\s*1)\b.*$",
    re.IGNORECASE,
)

#: A ranking frame leaves a bare ordinal marker, a criterion adjective
#: ("most famous", "biggest"), or a prepositional location ("in Minecraft on
#: YouTube") stuck to the front of what is really the subject. All are stripped
#: so the subject is the *set* the ranking ranges over.
_SUBJECT_LEAD_NOISE_RE = re.compile(
    r"^(?:#\s*1\b|number\s+one\b|no\.?\s*1\b|ranked\s+first\b|"
    r"most|biggest|largest|highest|greatest|top|best|smallest|lowest|fewest|"
    r"least|richest|wealthiest|strongest|fastest|oldest|youngest|newest|"
    r"latest|leading|dominant)\s*(?:in|on|at|for|of)?\s*",
    re.IGNORECASE,
)


#: A ranking frame of the form "which <set> has/have the <property>". The *set*
#: before the verb is the subject, not the property after it, so this frame is
#: matched separately and its set captured.
_SUBJECT_WHICH_HAS_RE = re.compile(
    r"^\s*(?:which|what)\s+(?P<subject>[a-z][a-z0-9 '-]*?)\s+"
    r"(?:has|have|gets|with)\s+(?:the\s+)?",
    re.IGNORECASE,
)

#: A comparison/ranking frame of the form "which/what <set> is <graded adjective>"
#: ("Which laptop is better?", "What phone is best?"). The set before the copula
#: is the subject - the candidates being ranked - so it is captured here rather
#: than left attached to the comparative adjective.
_SUBJECT_RANKING_IS_RE = re.compile(
    r"^\s*(?:which|what)\s+(?:kind\s+of\s+|type\s+of\s+)?"
    r"(?P<subject>[a-z][a-z0-9 '-]*?)\s+"
    r"(?:is|are|would\s+be|was|were)\s+(?:the\s+)?"
    r"(?:better|best|worse|worst|greatest|best\s+value|more\s+[a-z]+|most\s+[a-z]+|"
    r"faster|slowest|fastest|cheapest|largest|smallest|newest|oldest|[a-z]+est)\b",
    re.IGNORECASE,
)

#: Quantitative factual questions such as "How many subscribers does MrBeast have?"
#: ask for a current value attached to a subject, not an instruction. The subject
#: is the entity before the verb phrase that owns the property.
_QUANTITY_SUBJECT_RE = re.compile(
    r"^\s*(?:how\s+many|how\s+much)\s+(?:[a-z0-9][a-z0-9 _-]*?)\s+"
    r"(?:does|do)\s+(?P<subject>[A-Za-z][A-Za-z0-9 ._-]*?)\s+"
    r"(?:have|has|own|owns|get|gets|contain|contains)\b",
    re.IGNORECASE,
)

#: Event lookups such as "What happened with Apple recently?" must identify the
#: entity in the object position rather than treating the verb as the subject.
_EVENT_SUBJECT_RE = re.compile(
    r"^\s*(?:what|which|who)\s+happened\s+(?:with|to|around)\s+"
    r"(?P<subject>[A-Za-z][A-Za-z0-9 ._-]*?)(?:\s+recently|\?|$)",
    re.IGNORECASE,
)

#: Recommendation / value checks such as "Is the RTX 5090 worth buying?" are a
#: comparison over real-world product options, so the product instance is the
#: subject even when the clause is phrased as a value judgment.
_WORTH_BUYING_RE = re.compile(
    r"^\s*(?:can\s+you\s+tell\s+me\s+whether|tell\s+me\s+whether|whether|if|"
    r"is\s+it|would\s+it|should\s+it)\s+"
    r"(?P<subject>[A-Za-z0-9][A-Za-z0-9 ._-]*?)\s+(?:is|are|be)\s+worth\s+"
    r"(?:buying|getting|using)\b"
    r"|^\s*(?:is|are|would|should|will)\s+(?P<subject_alt>[A-Za-z0-9][A-Za-z0-9 ._-]*?)\s+"
    r"worth\s+(?:buying|getting|using)\b",
    re.IGNORECASE,
)


def extract_subject(text: str) -> str:
    """Extract the subject of a request from its grammatical frame.

    This is the structural fallback used when the interpreter's topic extractor
    did not fire (a bare "Who is X?" names no connector and no search verb, so
    the topic extractor legitimately leaves it alone - yet X is unambiguously the
    subject). The frame is syntactic: interrogative + copula, an "about"
    connective, or a retrieval verb. The captured span is trimmed of trailing
    instruction clauses, of a property/predicate clause, and of a leading article
    and ranking criterion.

    Nothing here chooses a capability; it only tells the semantic layer *what the
    request is about*.
    """

    value = (text or "").strip()
    if not value:
        return ""

    # Event / value / quantity frames are more precise than the generic
    # "what/which ... has ..." subject extraction and must be tried first.
    event = _EVENT_SUBJECT_RE.match(value)
    if event:
        return _clean_subject(event.group("subject"))

    # An explanatory actor question ("Why did the 2024 Lakers win?") asks about the
    # actor, so the actor is the subject even though the *answer* is a reason.
    explanatory = _EXPLANATORY_ACTOR_RE.match(value)
    if explanatory:
        return _clean_subject(explanatory.group("subject"))

    # An *indirect request* ("Can you check what Bitcoin is at?", "Find out which
    # phone is newest", "I need to know who won yesterday") carries its subject in
    # an embedded clause. The lead is stripped and the embedded question is parsed
    # with the ordinary rules, so the indirect form of a question reaches the same
    # reading as the direct form.
    indirect = _INDIRECT_LEAD_RE.match(value)
    if indirect:
        embedded_text = (indirect.group("clause") or "").strip()
        # The embedded clause may itself be quoted; the quotes are the user's
        # punctuation, not part of the subject.
        embedded_text = embedded_text.strip("\"'“”‘’")
        if _EMBEDDED_QUESTION_RE.match(embedded_text):
            inner = extract_subject(embedded_text)
            if inner:
                return inner
            embedded = _INDIRECT_OBJECT_RE.match(embedded_text)
            if embedded:
                cleaned = _clean_subject(embedded.group("subject"))
                if cleaned:
                    return cleaned

    quantity = _QUANTITY_SUBJECT_RE.match(value)
    if quantity:
        return _clean_subject(quantity.group("subject"))

    worth_buying = _WORTH_BUYING_RE.match(value)
    if worth_buying:
        subject = worth_buying.group("subject") or worth_buying.group("subject_alt")
        return _clean_subject(subject)

    # A "which <set> has the <property>" frame names the set before the verb.
    which_has = _SUBJECT_WHICH_HAS_RE.match(value)
    if which_has:
        return _clean_subject(which_has.group("subject"))

    # A "which <set> is <comparative>" frame names the candidates before the copula.
    ranking_is = _SUBJECT_RANKING_IS_RE.match(value)
    if ranking_is:
        return _clean_subject(ranking_is.group("subject"))

    match = _SUBJECT_FRAME_RE.match(value)
    if not match:
        return ""
    # A "did <actor> win" frame captures the *actor* in its own group, because the
    # trailing time clause ("last night") is not the subject. The named group wins
    # over the generic capture when both matched.
    captured = match.groupdict().get("subject2") or match.group("subject") or ""
    if not captured:
        return ""
    return _clean_subject(captured)


#: A determiner or possessive that precedes the *result noun* without naming an
#: event ("the title", "their crown"). It is stripped so the bare-noun guard can
#: see the noun and drop it, while a determiner that precedes an *event*
#: ("the championship") is stripped and the event is kept.
_LEADING_DETERMINER_RE = re.compile(
    r"^(?:the|a|an|this|that|these|those|his|her|their|its|our|your|my)\s+",
    re.IGNORECASE,
)


def _clean_subject(subject: str) -> str:
    """Trim trailing instruction clauses, a property clause, and lead noise."""

    subject = subject.strip()
    # Trailing sentence punctuation is stripped *first*, because the result-noun
    # guards below compare the whole span ("the title?" must reduce to "the title"
    # before a bare prize reference is recognised).
    subject = subject.strip(" .,;:!?'\"")
    # A research *place* left at the front ("web for the latest Python release")
    # is where the user asked Atlas to look, not the subject: it is dropped along
    # with the connector before any trailing-clause trimming, so the payload after
    # it survives as the subject.
    subject = re.sub(
        r"^(?:the\s+)?(?:web|internet|online)\s+(?:for|about|on)?\s*",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()
    subject = _SUBJECT_TRAILING_RE.sub("", subject).strip()
    subject = _SUBJECT_TRAILING_PREDICATIVE_RE.sub("", subject).strip()
    subject = _SUBJECT_TRAILING_PREDICATE_RE.sub("", subject).strip()
    # A named answer-role word is "who", not "what": "the reigning champion of the
    # Norwegian chess league" names the *event*, so the role noun and any determiner
    # in front of it are stripped. Only the role words are dropped - the event after
    # them stays ("championship" on its own is kept; "champion of X" loses only
    # "champion").
    subject = re.sub(
        r"^(?:the\s+|a\s+|an\s+)?(?:reigning|current|incumbent|defending|outgoing|"
        r"former|new|record)?\s*(?:champion|champions|winner|winners|victor|victors)"
        r"\s+(?:of|for|at|in)\s+(?:the\s+)?",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()
    # A result metaphor *of* an event ("champion of the Norwegian chess league",
    # "winner of the 2010 World Cup") names the event after the preposition, so the
    # event is the subject and the outcome word is dropped. This is checked before
    # the bare-metaphor guards below, which would otherwise discard the whole span.
    of_event = re.match(
        r"^(?:the\s+|a\s+|an\s+)?(?:" + "|".join(
            re.escape(noun) for noun in _RESULT_METAPHOR_NOUNS
        ) + r")(?:'s)?\s+(?:of|for|at|in)\s+(?:the\s+)?(.+)$",
        subject,
        flags=re.IGNORECASE,
    )
    if of_event:
        subject = of_event.group(1).strip()
    # A trailing relative time clause attached to an object ("the title this year",
    # "last night's game") is a temporal qualifier, not part of the thing looked
    # up. It is removed here and re-attached by the query builder from the resolved
    # period, so "this year" appears once, as a resolved year, not as user text.
    # The *presence* of the window is recorded first: a bare result noun that a
    # window qualifies ("the title this year") names the outcome of that window's
    # edition of the event, so it must not be dropped as a bare prize reference.
    window_qualified = bool(
        re.search(
            r"(?:this|last|next|that)\s+(?:year|month|week|season|night|evening|morning|afternoon)|"
            r"\d+\s+(?:days?|weeks?|months?|years?)\s+ago\b|"
            r"\b(?:yesterday|today|tonight)\b",
            subject,
            flags=re.IGNORECASE,
        )
    )
    subject = re.sub(
        r"\s+(?:this|last|next|that)\s+(?:year|month|week|season|night|evening|morning|afternoon)\b.*$",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()
    subject = re.sub(
        r"\s+\d+\s+(?:days?|weeks?|months?|years?|hours?)\s+ago\b.*$",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()
    subject = re.sub(
        r"\s+(?:yesterday|today|tonight|recently)\b.*$",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()
    # A result *metaphor* ("the title", "the crown") names the prize, not the
    # competition. A bare metaphor ("Who took the title?") carries no event at all
    # and is dropped entirely; a metaphor followed by its event ("the crown at the
    # Baku Masters") is stripped so the event remains. A metaphor a *time window*
    # qualified ("the title this year") is kept, because it names the outcome of
    # that window's edition - see ``window_qualified`` above.
    metaphor = "(?:" + "|".join(re.escape(noun) for noun in _RESULT_METAPHOR_NOUNS) + r")"
    subject = re.sub(
        r"^(?:" + metaphor + r")\s+(?:at|in|of|for|to|from)\s+",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()
    if not window_qualified and re.fullmatch(r"(?:" + metaphor + r")(?:'s)?", subject, flags=re.IGNORECASE):
        return ""
    # A *bare result noun with a determiner* ("the title") names a prize, not an
    # event. The determiner is stripped conditionally: how a request means a noun
    # depends on whether it names an event ("the championship") or just the
    # outcome role ("the title"), so a fixed keyword list of "metaphors" cannot
    # express it - the question is *what the noun can be an instance of*.
    stripped_bare = _LEADING_DETERMINER_RE.sub("", subject, count=1).strip()
    if stripped_bare and not _names_an_event(stripped_bare):
        if stripped_bare.casefold() in _OUTCOME_ROLE_NOUNS:
            return ""
        # Also drop a preposition that trailed the dropped noun ("the title of").
        if re.fullmatch(r"(?:of|for|at|in|to)\b.*", stripped_bare, flags=re.IGNORECASE):
            return ""
    # A definite result noun that still carries its *article* is a bare prize
    # reference ("Who took the title?" -> "the title"), which names no event. It is
    # handled here before the article is stripped, because after stripping the
    # article the word is indistinguishable from a real subject noun that happens
    # to be in the result vocabulary ("the championship" -> "championship", which
    # is the desired result noun and must be kept).
    if re.fullmatch(
        r"(?:the|a|an|his|her|their|its)\s+(?:" + metaphor + r")(?:'s)?",
        subject,
        flags=re.IGNORECASE,
    ):
        return ""
    # A result metaphor *of* an event ("champion of the Norwegian chess league",
    # "winner of the 2010 World Cup") is handled above, before the bare-metaphor
    # guards, because those guards would otherwise discard the whole span.
    subject = _SUBJECT_PROPERTY_CLAUSE_RE.sub("", subject).strip()
    # Strip a leading ranking marker/criterion repeatedly: "most famous Minecraft
    # YouTuber" -> "Minecraft YouTuber", "#1 in Minecraft on YouTube" ->
    # "Minecraft on YouTube".
    for _ in range(3):
        cleaned = _SUBJECT_LEAD_NOISE_RE.sub("", subject).strip()
        if cleaned == subject:
            break
        subject = cleaned
    subject = re.sub(r"^(?:the|a|an)\s+", "", subject, flags=re.IGNORECASE).strip()
    # A ranking adjective that followed a stripped superlative ("most famous X"
    # leaves "famous X") names no set, so it is stripped too. Kept as one small
    # set of *gradable adjectives*, not a phrase table.
    subject = re.sub(
        r"^(?:famous|popular|known|successful|profitable|valuable|expensive|"
        r"important|influential|powerful|talented|skilled|notable)\s+",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()
    subject = subject.strip(" .,;:!?'\"")
    # A *verb* left in the subject position means the frame captured a verb phrase
    # rather than a noun phrase ("crowned champion" from "was crowned champion").
    # The verb is the frame, so it is stripped and the noun that remains (if any)
    # is the subject.
    subject = re.sub(r"^(?:" + "|".join(
        re.escape(verb) for verb in sorted(_EVENT_RESULT_VERBS, key=len, reverse=True)
    ) + r")\s+", "", subject, flags=re.IGNORECASE).strip()
    # A request *verb* left in the object position ("Find out which phone is
    # newest" -> "find out which phone") is the frame, not the subject. It is
    # stripped so the entity ("phone") is what remains.
    subject = re.sub(
        r"^(?:find\s+out|look\s+up|check|verify|confirm|see|search\s+for|fetch|"
        r"tell\s+me|let\s+me\s+know)\s+(?:which|what|who|whether|if)?\s*",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()
    # A result verb trailing an actor ("the Lakers win", "Arsenal won") leaves the
    # *actor* as the subject, which is what the user asked about; the verb is the
    # relation, not part of the entity.
    subject = re.sub(
        r"\s+(?:" + "|".join(
            re.escape(verb) for verb in sorted(_EVENT_RESULT_VERBS, key=len, reverse=True)
        ) + r")\b.*$",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()
    # A leading year on an actor is a temporal qualifier ("2024 Lakers win" ->
    # "Lakers"); the year is re-attached by the query builder from the resolved
    # period, so it is not part of the entity.
    subject = re.sub(r"^(?:in\s+|of\s+)?(?:19|20)\d{2}\s+", "", subject, flags=re.IGNORECASE).strip()
    subject = re.sub(
        r"^(?:reigning|current|incumbent|defending|outgoing|former|new)\s+",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip()
    # A temporal word names *when*, never *what*. "who won yesterday" leaves
    # "yesterday" in the object position, but the event is the thing to look up and
    # it is supplied by context (or is missing), so a bare temporal word is not a
    # subject.
    if subject.casefold() in _NEVER_SUBJECT_WORDS:
        return ""
    # A bare pronoun is a reference, not a subject; leave it to context resolution.
    if not subject or subject.casefold() in _ANAPHORA:
        return ""
    # A ranking adjective left alone names no set ("the most famous" -> nothing).
    if subject.casefold() in _SUPERLATIVE_WORDS:
        return ""
    # Guard against absorbing an entire instruction sentence as the subject.
    if len(subject) > 80 or len(subject.split()) > 12:
        return ""
    return subject


def _extract_criterion(lowered: str, superlative: str) -> str:
    """Return the criterion a ranking is judged by, as the user phrased it.

    Two sources, in priority order:

    1. An explicit lead ("by followers", "in terms of size", "based on speed").
    2. The adjective the superlative qualifies ("the **most famous** Minecraft
       YouTuber" -> "most famous"; "the **biggest** channel" -> "biggest").
    """

    for lead in _CRITERION_LEADS:
        match = re.search(
            rf"\b{re.escape(lead)}\s+([a-z][a-z '\-]{{1,30}}?)(?:\s+(?:in|on|at|of|with)\b|[.?!,]|$)",
            lowered,
        )
        if match:
            candidate = match.group(1).strip(" .,!?")
            if candidate:
                return candidate

    if superlative:
        # "the most famous X" -> criterion "most famous"; "the biggest X" ->
        # "biggest". Capture the superlative plus a following adjective, but never
        # invent a "most X" prefix for a superlative that is already complete
        # ("most best" is not a criterion; "best" is).
        match = re.search(
            rf"\b(?:the\s+|a\s+|an\s+)?({re.escape(superlative)})(?:\s+([a-z]{{3,}}))?\b",
            lowered,
        )
        if match:
            head = match.group(1)
            qualifier = match.group(2) or ""
            if head == "most" and qualifier:
                # Found "most <word>": keep the compound only when <word> is a
                # plain adjective rather than another superlative morpheme.
                if qualifier in _SUPERLATIVE_WORDS:
                    return qualifier
                return f"most {qualifier}"
            return head
    return ""


def _extract_candidate_set(lowered: str, criterion: str) -> str:
    """Return the set a comparison ranges over ("Minecraft YouTube creators").

    The candidate set is the head noun phrase the superlative qualifies: in
    "the most famous Minecraft YouTuber" it is "Minecraft YouTuber", and the
    criterion adjective is stripped off the front. Nothing here calls the web;
    the candidate set is a *description* the retrieval layer turns into a query.
    """

    text = lowered
    # Drop a leading criterion phrase so the remaining phrase is the set.
    if criterion:
        text = re.sub(rf"\b{re.escape(criterion)}\b", " ", text, count=1)
    # Drop the frame words around a ranking question.
    text = re.sub(
        r"^\s*(?:who|which|what|whose|tell me which|tell me who)\s+(?:is|are|has|have|was|were)?\s*",
        " ",
        text,
    )
    text = re.sub(r"\b(?:the|a|an|is|are|has|have|currently|right now|of|in|on)\b", " ", text)
    words = [word for word in text.split() if word and word not in {"most", "top"}]
    candidate = " ".join(words).strip()
    return candidate[:80].strip()
