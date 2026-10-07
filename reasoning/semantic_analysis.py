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
_INTRINSICALLY_CURRENT: tuple[str, ...] = (
    "president", "prime minister", "ceo", "chancellor", "governor", "mayor",
    "leader", "king", "queen", "pope", "champion", "winner", "record holder",
    "version", "release", "price", "stock", "score", "weather", "temperature",
    "population", "ranking", "rank", "subscribers", "views", "followers",
    "net worth", "exchange rate", "interest rate", "schedule", "news",
)

#: Markers that make a request about the local machine rather than the world.
_LOCAL_MARKERS: tuple[str, ...] = (
    "my computer", "my pc", "my machine", "my laptop", "my system",
    "this computer", "this pc", "this machine", "my disk", "my cpu", "my ram",
    "my files", "my documents", "my downloads", "my folder", "my desktop",
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

    # -- comparison / ranking --------------------------------------------------
    # An ordinal "#1"/"number one"/"no. 1" is a ranking claim whose marker is not
    # a word boundary match; check it before the word-based superlatives.
    ordinal_rank = re.search(r"(?:#\s*1\b|\bnumber\s+one\b|\bno\.?\s*1\b|\branked\s+first\b)", lowered)
    superlative = _contains_any(lowered, _SUPERLATIVES)
    if ordinal_rank and not superlative:
        superlative = "top"
    comparative_marker = _contains_any(lowered, _COMPARATIVES)
    if superlative:
        reading.superlative = True
        reading.comparative = True
        reading.markers.append(f"superlative:{superlative}")
    if comparative_marker:
        reading.comparative = True
        reading.markers.append(f"comparative:{comparative_marker.strip()}")

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
    intrinsic = _contains_any(lowered, _INTRINSICALLY_CURRENT)
    if intrinsic:
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
    elif re.search(r"\b(?:worth\s+(?:buying|getting|using)|worthwhile)\b", lowered):
        reading.intrinsically_current = True
        reading.freshness = "current"
        reading.comparative = True
        reading.criterion = "value"
        reading.criterion_proxy = "price, specs, and alternatives"
        reading.markers.append("value_evaluation")
    elif reading.superlative:
        # A "who is the most X" question is a statement about the world as it is
        # now: the leader can change, so it is a current question by structure.
        reading.freshness = "current"
        reading.markers.append("superlative_is_current")
    elif _contains_any(lowered, _STABLE_QUESTION_HEADS):
        reading.freshness = "stable"
        reading.markers.append("stable_definition")
    else:
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

    # -- requested output shape ------------------------------------------------
    for phrase, shape in _OUTPUT_SHAPES:
        if phrase in lowered:
            reading.requested_output = shape
            break

    return reading


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
    r"|(?:who|what|which|whose)\s+is\s+#\s*1\b\s*(?:in|on|at|for)?\s*"
    r"|(?:who|what|which|whose)\s+is\s+(?:number\s+one|no\.?\s*1)\b\s*(?:in|on|at|for)?\s*"
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

    match = _SUBJECT_FRAME_RE.match(value)
    if not match or not match.group("subject"):
        return ""
    return _clean_subject(match.group("subject"))


def _clean_subject(subject: str) -> str:
    """Trim trailing instruction clauses, a property clause, and lead noise."""

    subject = subject.strip()
    subject = _SUBJECT_TRAILING_RE.sub("", subject).strip()
    subject = _SUBJECT_TRAILING_PREDICATE_RE.sub("", subject).strip()
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
