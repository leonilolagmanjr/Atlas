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

#: Bare object/anaphoric pronouns that must resolve to earlier context.
_PRONOUNS: frozenset[str] = frozenset(
    {"it", "this", "that", "these", "those", "them", "they", "one", "something"}
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

        if resolved.has_reference:
            self._check_resolvable(resolved, prior_task=prior_task, history=history)
        return resolved

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
        if resolved.target in {"previous_output", "previous_result", "ordinal"} and not (
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
