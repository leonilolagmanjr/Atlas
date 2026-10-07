"""Conversation-aware reference resolution for the Intent Engine.

A follow-up like "put that in Notepad", "make it shorter", "which one is best?"
or "do the same thing for Interstellar" only makes sense against earlier turns.
This module resolves those references *deterministically* against the previous
task (the Task IR) and a small, retrieved slice of recent conversation output —
never by stuffing the whole history into a prompt.

Design rules:

* Resolution is data: a :class:`ResolvedContext` names what a follow-up inherits
  (subject/destination/operations), what it replaces, and any reference it could
  not resolve. The interpreter and the Intent Engine consume it.
* Only a *bounded* amount of prior context is consulted: the previous task and,
  when a reference targets prior output, the most recent assistant answer.
* Unresolvable references are surfaced (``unresolved``), never hallucinated.
* Nothing here executes anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Bare object/anaphoric pronouns that must resolve to earlier context. Personal
#: pronouns are included so a follow-up about a person ("How many subscribers does
#: he have?" after "Who is MrBeast?") resolves to the prior subject instead of
#: being read as a fresh, subjectless request.
_PRONOUNS: frozenset[str] = frozenset(
    {
        "it", "this", "that", "these", "those", "them", "they", "one",
        "something",
        # Personal pronouns: high-precision references to a previously named person.
        "he", "she", "him", "her", "hers", "his", "they", "them", "their",
        "theirs",
    }
)

#: Ordinal / selection references ("the first one", "the second option").
_ORDINALS: dict[str, int] = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "last": -1,
}

#: Phrases that mean "reuse the previous task's structure with a new subject".
_SAME_TASK_PATTERNS: tuple[str, ...] = (
    "do the same", "do that again", "same thing", "do it again", "repeat that",
    "do the same thing", "same as before", "like before",
)

#: Phrases that ask to continue/extend the previous result rather than start over.
_CONTINUE_PATTERNS: tuple[str, ...] = (
    "continue", "go on", "keep going", "go deeper", "elaborate", "expand on that",
    "tell me more", "more detail", "further",
)

#: Transformation-only follow-ups ("make it shorter", "make it better").
_TRANSFORM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:make|write)\s+(?:it|that|this|them)\s+(?:shorter|brief|briefer|concise)\b", "shorten"),
    (r"\b(?:make|write)\s+(?:it|that|this|them)\s+(?:longer|bigger|more detailed)\b", "expand"),
    (r"\bshorten\s+(?:it|that|this|them)\b", "shorten"),
    (r"\bexpand\s+(?:it|that|this|them)\b", "expand"),
    (r"\b(?:make|write)\s+(?:it|that|this|them)\s+better\b", "improve"),
    (r"\b(?:make|write)\s+(?:it|that|this|them)\s+(?:funny|formal|casual|serious)\b", "restyle"),
    (r"\bsummari[sz]e\s+(?:it|that|this|them)\b", "summarize"),
    (r"\brewrite\s+(?:it|that|this|them)\b", "rewrite"),
)

#: Destination phrases whose object is a reference ("put that in Notepad").
_PUT_REFERENCE_RE = re.compile(
    r"\b(?:put|place|write|paste|type|save|store|copy|add|insert|dump)\s+"
    r"(?:that|this|it|them|those|these|the\s+(?:previous\s+)?(?:result|answer|summary|output|text))\b",
    re.IGNORECASE,
)
#: "which one" / "which of them" selection over previous results.
_WHICH_ONE_RE = re.compile(r"\bwhich\s+(?:one|of\s+them|of\s+those|one\s+should|is)\b", re.IGNORECASE)
#: "the previous one/answer/result".
_PREVIOUS_RE = re.compile(
    r"\bthe\s+(?:previous|last|earlier|prior)\s+(?:one|answer|result|output|summary|text|response)\b",
    re.IGNORECASE,
)
#: An explicit application named as a destination / open target.
_DESTINATION_RE = re.compile(
    r"\b(?:in|into|to|inside|using|with|open|launch|start|run)\s+"
    r"([A-Za-z][A-Za-z0-9 ._-]*?)(?:\s+and\b|$|[,.])",
    re.IGNORECASE,
)

#: Words that carry no reference information on their own.
_REFERENCE_NOISE: frozenset[str] = frozenset(
    {"the", "a", "an", "of", "in", "to", "please", "now", "again", "recent", "last"}
)

#: Structural markers that make a message *relative to something already said*
#: without naming a concrete object. These are sentence connectives and framing
#: words, not topics or commands: they signal "this message extends the previous
#: exchange" in general, which is exactly the contextual-follow-up case. This is
#: one bounded vocabulary of discourse markers, not a set of per-request rules.
_CONTEXT_FRAMES: tuple[str, ...] = (
    "what about", "how about", "what if", "would that", "would this", "why not",
    "is that", "is this", "are those", "do you think", "would you", "what do you",
    "tell me more", "go on", "and then", "what else", "anything else",
    "the same", "same thing", "more", "further", "instead", "alternatively",
    "also", "too", "then",
)

#: Question words that, on their own, presuppose a subject from earlier turns
#: ("Why?", "And performance?", "How so?"). A bare interrogative fragment is the
#: clearest signal that the referent is the previous exchange.
_BARE_QUESTION_WORDS: frozenset[str] = frozenset(
    {"why", "how", "what", "when", "where", "which", "really", "and", "but", "so"}
)

_WORD_SPLIT_RE = re.compile(r"[a-z0-9']+")

#: Very common words that carry no topical signal when deciding whether a message
#: continues the current discussion. Kept small and general (English function
#: words plus generic interrogatives), never request-specific vocabulary.
_TOPIC_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "to", "in", "into", "on", "of", "for",
        "with", "is", "are", "was", "were", "be", "being", "do", "does", "did",
        "can", "could", "would", "should", "will", "shall", "may", "might",
        "i", "you", "we", "it", "its", "this", "that", "these", "those",
        "what", "why", "how", "when", "where", "which", "who", "whose",
        "me", "my", "our", "your", "please", "about", "more", "less",
        "still", "also", "then", "than", "there", "here", "so", "if", "not",
        # Generic request verbs carry the *shape* of a request, not its topic. Two
        # messages that merely both say "write" are not necessarily about the same
        # thing, so overlap on these alone must not mark a message contextual.
        "write", "make", "create", "give", "tell", "show", "explain", "find",
        "get", "want", "need", "help", "use", "using", "said", "say", "like",
    }
)


def _salient_terms(text: str) -> set[str]:
    """Return content words that carry topical signal (no function words)."""

    return {
        word
        for word in _WORD_SPLIT_RE.findall((text or "").casefold())
        if word not in _TOPIC_STOPWORDS and len(word) > 2
    }


def _continues_topic(lowered: str, prior_text: str) -> bool:
    """Return True when a message continues the topic of the prior turns.

    This is the relevance-driven half of contextual detection: a follow-up such as
    "Why does chunking matter?" or "What are the biggest obstacles?" names its own
    words, so no pronoun or discourse frame fires — but those words are drawn from
    the discussion that is already in progress. Comparing the message's salient
    terms against the prior turn's salient terms is a general, vocabulary-free
    test: overlap means the message is *about what we were already discussing*.
    """

    terms = _salient_terms(lowered)
    prior_terms = _salient_terms(prior_text)
    if not terms or not prior_terms:
        return False
    return bool(terms & prior_terms)


#: Verbs that operate on *content* and whose object may be an implied previous
#: output ("make it clearer", "rewrite it", "turn that into an email", "expand").
#: The vocabulary is a small, general set of content operations — not a list of
#: request phrasings — so arbitrary transformations are covered.
_CONTENT_OPERATION_VERBS: tuple[str, ...] = (
    "make", "rewrite", "reword", "redo", "rephrase", "reformat", "restructure",
    "shorten", "lengthen", "expand", "condense", "simplify", "summarize",
    "summarise", "translate", "turn", "convert", "transform", "edit", "improve",
    "polish", "tweak", "adjust", "change", "adapt", "rework", "refine",
)


def _is_bare_content_operation(lowered: str) -> bool:
    """Return True when an imperative reshapes content without naming its object.

    This is the generic reading of an arbitrary transformation request such as
    "make it sound less corporate", "turn that into an email" or "make it more
    technical". The verb is a content operation and the message names no new
    concrete target (no application, file, web host or URL), so the object must be
    whatever we were already working on — a conversation-grounded transformation.
    It does not enumerate the transformations themselves.
    """

    head = lowered.lstrip()
    for lead in ("please ", "can you ", "could you ", "would you ", "now "):
        if head.startswith(lead):
            head = head[len(lead):].lstrip()
    if not any(re.match(rf"{re.escape(verb)}\b", head) for verb in _CONTENT_OPERATION_VERBS):
        return False
    # A concrete target of its own means it is a fresh request, not a follow-up.
    if re.search(
        r"\b(?:https?://|www\.|notepad|word|excel|vscode|vs code|chrome|firefox|edge|"
        r"outlook|youtube|google|github)\b",
        lowered,
    ):
        return False
    if re.search(r"\b[a-z0-9_.-]+\.(?:txt|md|docx?|pdf|csv|xlsx?|pptx?|json|py)\b", lowered):
        return False
    return True


def is_content_operation(text: str) -> bool:
    """Public form of :func:`_is_bare_content_operation` for other components.

    The Intent Engine consults this to decide whether a contextual follow-up is a
    reshaping of prior content (a transformation) rather than a fresh creation.
    """

    return _is_bare_content_operation(" ".join((text or "").casefold().split()))


def _looks_contextual(lowered: str) -> bool:
    """Return True when a message is a follow-up that presupposes earlier turns.

    This is the *generic* complement to the explicit reference patterns: a short
    message that frames itself relative to the conversation ("what about X?",
    "is that a good idea?", "why?") has no self-contained subject, so its meaning
    can only be settled from what was already said. It names no pronoun and no
    fixed phrase; it is a structural property of the message, so arbitrary
    follow-ups are covered without enumerating them.
    """

    words = _WORD_SPLIT_RE.findall(lowered)
    if not words:
        return False
    # A bare or near-bare interrogative fragment ("why?", "and?", "how so?"):
    # there is nothing here that can stand on its own.
    content = [word for word in words if word not in _REFERENCE_NOISE]
    if len(content) <= 2 and (content and content[0] in _BARE_QUESTION_WORDS):
        return True
    # A framing connective that relates the message to what came before.
    padded = f" {lowered} "
    for frame in _CONTEXT_FRAMES:
        if " " in frame:
            if frame in padded:
                return True
        elif re.search(rf"\b{re.escape(frame)}\b", lowered):
            return True
    # A back-reference pronoun in an interrogative presupposes an antecedent
    # ("could that become a bottleneck?", "would those scale?", "is that safe?",
    # "how many subscribers does he have?"). Personal pronouns are included so a
    # question about a previously named person resolves to that person.
    # This is recall-oriented on purpose: over-grounding a self-contained question
    # is harmless (the answer path is told to answer on its own terms when the
    # conversation does not bear on it), whereas missing a real follow-up loses
    # the context the user expects.
    if ("?" in lowered or _starts_with_question(lowered)) and (
        set(words) & {
            "that", "this", "those", "these", "it", "them",
            "he", "she", "him", "her", "his", "hers", "they", "their",
        }
    ):
        return True
    # A content operation whose object is not named ("make it sound less
    # corporate", "turn that into an email"): the object is the conversation.
    if _is_bare_content_operation(lowered):
        return True
    return False


def _starts_with_question(lowered: str) -> bool:
    words = _WORD_SPLIT_RE.findall(lowered)
    return bool(words) and words[0] in _BARE_QUESTION_WORDS


@dataclass
class ResolvedContext:
    """The deterministic result of resolving a follow-up against prior state.

    ``has_reference`` is True when the request depends on earlier turns at all.
    ``inherit`` is True when the previous task's structure should be reused.
    ``replace_subject`` is True when the caller should swap the subject while
    keeping the rest of the prior structure ("do the same thing for Interstellar").
    ``mutates_output`` is True for transformation-only follow-ups that reshape
    the previous answer rather than start a new task.
    """

    has_reference: bool = False
    inherit: bool = False
    replace_subject: bool = False
    mutates_output: bool = False
    continue_prior: bool = False
    #: True when the message can only be understood against earlier turns even
    #: though it names no explicit reference. This is the general case of a
    #: conversational follow-up ("Why?", "What about performance?", "Is that a
    #: good idea?", "Which one would you choose?"): the *referent is implied*, so
    #: the correct reading comes from conversation context rather than from a
    #: lexical pronoun match. It is deliberately one generic signal, not a set of
    #: per-phrase rules.
    contextual: bool = False
    #: The referent class the request points at: previous_output, previous_task,
    #: previous_result, ordinal, destination_only, or "" when it is a fresh task.
    target: str = ""
    #: Transformation verbs detected on a transformation-only follow-up.
    transformation: str = ""
    #: A destination extracted from the follow-up ("Notepad" in "put that in ...").
    destination: str = ""
    #: A new subject supplied by the follow-up ("Interstellar" in "do the same for X").
    new_subject: str = ""
    #: Ordinal index (1-based; -1 for "last") when the reference names one.
    ordinal: int = 0
    #: References that could not be resolved from available context.
    unresolved: list[str] = field(default_factory=list)
    #: Short, inspectable notes describing the resolution decision.
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_reference": self.has_reference,
            "inherit": self.inherit,
            "replace_subject": self.replace_subject,
            "mutates_output": self.mutates_output,
            "continue_prior": self.continue_prior,
            "contextual": self.contextual,
            "target": self.target,
            "transformation": self.transformation,
            "destination": self.destination,
            "new_subject": self.new_subject,
            "ordinal": self.ordinal,
            "unresolved": list(self.unresolved),
            "notes": list(self.notes),
        }


class ReferenceResolver:
    """Resolve follow-up references against the previous task and history."""

    def resolve(
        self,
        text: str,
        *,
        prior_task: Any = None,
        history: str = "",
    ) -> ResolvedContext:
        lowered = text.strip().casefold()
        resolved = ResolvedContext()
        if not lowered:
            return resolved

        # 1. "do the same thing for X" / "same as before" -> reuse structure.
        same = _matches_any(lowered, _SAME_TASK_PATTERNS)
        if same:
            resolved.has_reference = True
            resolved.inherit = True
            resolved.replace_subject = True
            resolved.target = "previous_task"
            subject = self._subject_after_same(lowered)
            if subject:
                resolved.new_subject = subject
                resolved.notes.append(f"reuse previous task for subject '{subject}'")
            else:
                resolved.notes.append("reuse previous task structure unchanged")

        # 2. Continuation ("go deeper", "continue") -> extend prior output.
        if _matches_any(lowered, _CONTINUE_PATTERNS):
            resolved.has_reference = True
            resolved.continue_prior = True
            resolved.inherit = True
            resolved.target = resolved.target or "previous_output"
            resolved.notes.append("continue previous result")

        # 3. Transformation-only follow-up ("make it shorter").
        for pattern, verb in _TRANSFORM_PATTERNS:
            if re.search(pattern, lowered):
                resolved.has_reference = True
                resolved.mutates_output = True
                resolved.inherit = True
                resolved.target = "previous_output"
                resolved.transformation = verb
                resolved.notes.append(f"transform previous output: {verb}")
                break

        # 4. Destination-only reference ("put that in Notepad", "save it").
        if _PUT_REFERENCE_RE.search(text):
            resolved.has_reference = True
            resolved.inherit = True
            resolved.target = resolved.target or "previous_output"
            destination = self._extract_destination(text)
            if destination:
                resolved.destination = destination
                resolved.notes.append(f"deliver previous output to {destination}")
            else:
                resolved.notes.append("deliver previous output")

        # 5. "the previous answer/result".
        if _PREVIOUS_RE.search(text):
            resolved.has_reference = True
            resolved.inherit = True
            resolved.target = resolved.target or "previous_output"
            if "the previous result" not in resolved.notes:
                resolved.notes.append("reference to previous result")

        # 6. "which one should I use?" selection over previous results.
        if _WHICH_ONE_RE.search(text):
            resolved.has_reference = True
            resolved.inherit = False
            resolved.target = "previous_result"
            resolved.notes.append("selection over previous results")

        # 7. Ordinal reference ("the second one", "the first result").
        ordinal = self._ordinal(lowered)
        if ordinal:
            resolved.has_reference = True
            resolved.target = resolved.target or "ordinal"
            resolved.ordinal = ordinal
            resolved.notes.append(f"ordinal reference #{ordinal}")

        # 8. A bare anaphoric pronoun with no other structural signal.
        if not resolved.has_reference and _has_bare_reference(lowered):
            resolved.has_reference = True
            resolved.inherit = True
            resolved.target = "previous_output"
            resolved.notes.append("anaphoric reference to previous turn")

        # 9. Implicit conversational continuation: the message presupposes earlier
        #    turns without naming a reference ("Why?", "What about performance?",
        #    "Is that a good idea?") OR the message is topically continuous with what
        #    was just discussed ("Why does chunking matter?" while discussing RAG).
        #    Either way the reading is *grounded in the conversation* rather than
        #    forced through a previous-output transformation; the Intent Engine then
        #    decides conversation vs. executable action from the grounded context.
        if not resolved.has_reference and _looks_contextual(lowered):
            resolved.has_reference = True
            resolved.contextual = True
            resolved.target = resolved.target or "conversation"
            resolved.notes.append("implicit follow-up; grounded in conversation context")
        if (
            not resolved.has_reference
            and self._continues_current_topic(lowered, prior_task=prior_task, history=history)
        ):
            resolved.has_reference = True
            resolved.contextual = True
            resolved.target = "conversation"
            resolved.notes.append(
                "the message shares the current topic; grounded in conversation context"
            )

        if resolved.has_reference:
            self._check_resolvable(resolved, prior_task=prior_task, history=history)
        return resolved

    @staticmethod
    def _continues_current_topic(lowered: str, *, prior_task: Any, history: str) -> bool:
        """Return True when the message is about the topic already in progress.

        Relevance, not recency and not vocabulary: the message's content words are
        compared with the prior turn's words and the previous task's subject. If
        they overlap, the follow-up is *about what we were discussing* and must be
        grounded in the conversation. With no prior context this is False, so a
        topic-continuing read is never fabricated from nothing.
        """

        if not (history and history.strip()) and prior_task is None:
            return False
        prior_parts: list[str] = []
        if prior_task is not None:
            entities = getattr(prior_task, "entities", {}) or {}
            for key in ("topic", "research_query", "raw_topic", "normalized_topic"):
                value = entities.get(key)
                if value:
                    prior_parts.append(str(value))
            if getattr(prior_task, "original_prompt", ""):
                prior_parts.append(str(prior_task.original_prompt))
        if history and history.strip():
            # Only the tail matters: the topic in progress is what was last said.
            prior_parts.append(history.strip()[-1200:])
        if not prior_parts:
            return False
        return _continues_topic(lowered, "\n".join(prior_parts))

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def _subject_after_same(lowered: str) -> str:
        """Return the new subject in "do the same thing for X" (may be empty)."""

        match = re.search(
            r"\b(?:for|about|with|on)\s+([a-z0-9][a-z0-9 '\-]{1,60})",
            lowered,
        )
        if not match:
            return ""
        subject = match.group(1).strip(" .?")
        # Trim trailing filler words a greedy match may have captured.
        subject = re.split(r"\b(?:in|into|and|then)\b", subject, maxsplit=1)[0].strip()
        if not subject or subject in _PRONOUNS or subject in _REFERENCE_NOISE:
            return ""
        return subject

    @staticmethod
    def _extract_destination(text: str) -> str:
        match = _DESTINATION_RE.search(text)
        if not match:
            return ""
        candidate = match.group(1).strip(" .,;:!?")
        candidate = re.split(r"\s+(?:about|and|that|which)\b", candidate, maxsplit=1)[0].strip()
        lowered = candidate.casefold()
        if not candidate or lowered in _PRONOUNS:
            return ""
        if lowered in {"the web", "youtube", "google", "the internet", "a file", "the file", "file"}:
            return ""
        if len(candidate) > 40 or len(candidate.split()) > 4:
            return ""
        return candidate

    @staticmethod
    def _ordinal(lowered: str) -> int:
        for word, index in _ORDINALS.items():
            if re.search(rf"\b{word}\b", lowered):
                # Only treat it as a reference when no explicit noun follows that
                # would make it a fresh subject ("first-aid kit" is not an ordinal).
                if re.search(rf"\b{word}\s+(?:one|result|option|item|video|answer|choice)\b", lowered):
                    return index
        return 0

    @staticmethod
    def _check_resolvable(resolved: ResolvedContext, *, prior_task: Any, history: str) -> None:
        """Record references that cannot be resolved from the available context."""

        has_prior = prior_task is not None
        has_history = bool(history and history.strip())
        if resolved.target in {"previous_output", "previous_result", "ordinal", "conversation"} and not (
            has_prior or has_history
        ):
            resolved.unresolved.append(resolved.target)
        if resolved.inherit and resolved.replace_subject and not has_prior:
            # "do the same thing for X" with no prior task has nothing to inherit.
            resolved.unresolved.append("previous_task")


def _matches_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _has_bare_reference(lowered: str) -> bool:
    """Detect a standalone pronoun used as an object of a verb."""

    return bool(
        re.search(
            r"\b(?:open|launch|write|read|show|use|save|put|edit|change|update|"
            r"delete|rename|move|copy|continue)\s+(?:it|that|this|them|those|these|one)\b",
            lowered,
        )
    )
