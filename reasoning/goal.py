"""Structured goal understanding: what the user actually asked for.

A request is not a single tool command. This module converts free text into a
:class:`TaskGoal`: an ordered set of :class:`Outcome` records (open, retrieve,
create, transform, deliver), the requested output, the destination, and an
explicit completion contract.

The module is deterministic and model-free. It exists so that:

* compound requests are decomposed into clauses ("find X **and** summarize it
  **and** write the summary to Notepad" is three outcomes, not one match);
* an entity never selects a tool on its own - the *verb* and the requested
  outcome do ("write a poem about YouTube" creates content about YouTube; it
  does not search YouTube);
* the retrieval goal follows the requested output ("summarize the book" needs
  information *about* the book, not the book itself);
* every task carries completion criteria that can be verified afterwards.

Nothing here executes anything. It only describes the goal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Vocabulary: verbs are categories of intent, not entities.
# ---------------------------------------------------------------------------

OPEN_VERBS = ("open", "launch", "start", "run")
RETRIEVE_VERBS = ("search", "find", "look", "fetch", "retrieve", "read", "get",
                  "show", "pull", "browse", "google", "research", "download")
CREATE_VERBS = ("create", "write", "make", "generate", "compose", "produce",
                "draft", "design", "invent")
DELIVER_VERBS = ("copy", "paste", "put", "save", "type", "insert", "add",
                 "store", "send", "dump", "place", "drop", "export")

#: Transform verbs -> canonical transformation.
TRANSFORM_VERBS: dict[str, str] = {
    "summarize": "summarize", "summarise": "summarize", "summary": "summarize",
    "abstract": "summarize", "condense": "summarize", "digest": "summarize",
    "explain": "explain", "describe": "explain", "overview": "explain",
    "outline": "outline",
    "compare": "compare", "contrast": "compare",
}
#: Transform nouns/phrases that name the requested output shape.
TRANSFORM_PHRASES: tuple[tuple[str, str], ...] = (
    ("key points", "key_points"), ("key takeaways", "key_points"),
    ("main points", "key_points"), ("bullet points", "key_points"),
    ("takeaways", "key_points"), ("highlights", "key_points"),
    ("summary", "summarize"), ("summaries", "summarize"),
    ("recap", "summarize"), ("overview", "explain"),
    ("explanation", "explain"), ("comparison", "compare"),
    ("outline", "outline"),
)

#: Words that refer to something produced earlier rather than naming a thing.
REFERENCES = ("it", "its", "that", "this", "them", "those", "there", "the same",
              "the result", "the results", "the content", "the text", "the script",
              "the answer", "the summary", "the key points", "the file")

# Conjunctions that start a *new* intent clause.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?:;|\.\s+|\n+)"
    r"|\s+(?:(?:and\s+)?(?:then|next|afterwards|after\s+that|finally|also)"
    r"|and|and\s+also|plus)\s+",
    re.IGNORECASE,
)
_LEADING_FILLER_RE = re.compile(
    r"^\s*(?:please\s+|kindly\s+|can\s+you\s+|could\s+you\s+|i\s+want\s+you\s+to\s+"
    r"|i'?d\s+like\s+you\s+to\s+|help\s+me\s+|now\s+)+",
    re.IGNORECASE,
)
#: "to Notepad" / "into Notepad" / "in Notepad" / "inside Notepad"
_DEST_PREP_RE = re.compile(
    r"\b(?:to|into|onto|inside|within|in)\s+"
    r"(?P<dest>[A-Za-z][A-Za-z0-9._+-]*(?:\s+[A-Za-z][A-Za-z0-9._+-]*){0,3})"
    r"(?:\s*[,.!?]|\s*$|\s+(?:and|then|about|that|which)\b)",
    re.IGNORECASE,
)
_FILENAME_RE = re.compile(r"\b(?:as|named|called)\s+(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*\.[A-Za-z0-9]{1,6})\b", re.IGNORECASE)
_TOPIC_RE = re.compile(
    r"\b(?:about|regarding|on\s+the\s+topic\s+of|concerning|related\s+to)\s+"
    r"(?P<topic>.+?)\s*$",
    re.IGNORECASE,
)
_INFORMATION_RE = re.compile(
    r"\b(?:information|info|facts|details|research|background)\b", re.IGNORECASE
)
_ARTIFACT_NOUNS: dict[str, str] = {
    "script": "script", "screenplay": "script", "transcript": "transcript",
    "lyrics": "lyrics", "song": "song", "code": "code", "source code": "code",
    "documentation": "documentation", "manual": "documentation",
}
_CONTENT_NOUNS: dict[str, str] = {
    "poem": "poem", "story": "story", "essay": "essay", "song": "song",
    "joke": "joke", "letter": "letter", "email": "email", "summary": "summary",
    "note": "note", "report": "report", "article": "article", "message": "message",
    "list": "list", "plan": "plan", "caption": "caption", "paragraph": "paragraph",
    "haiku": "haiku", "limerick": "limerick", "verse": "verse", "bio": "bio",
    "resume": "resume", "outline": "outline", "guide": "guide",
}
#: Words that are never a search target or a destination on their own.
_STOP_TARGET_WORDS = frozenset({
    "the", "a", "an", "some", "any", "me", "it", "that", "this", "them",
    "information", "info", "and", "or", "for", "about", "on", "in", "of", "to",
})

#: Completion criteria keys used by the completion contract.
CRITERION_CONTENT_GENERATED = "content_generated"
CRITERION_EVIDENCE_SUFFICIENT = "evidence_sufficient"
CRITERION_TRANSFORMATION_PRODUCED = "transformation_produced"
CRITERION_DELIVERED_TO_APPLICATION = "delivered_to_application"
CRITERION_APPLICATION_OPENED = "application_opened"
CRITERION_FILE_WRITTEN = "file_written"
CRITERION_DELIVERY_VERIFIED = "delivery_verified"


def _matches_verb(text: str, verbs: Iterable[str]) -> str | None:
    for verb in verbs:
        if re.search(rf"\b{re.escape(verb)}\w*\b", text):
            return verb
    return None


def _contains(text: str, phrases: Iterable[str]) -> str | None:
    for phrase in phrases:
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return phrase
    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """One thing the user wants done, derived from one clause."""

    action: str  # open | retrieve | create | transform | deliver
    verb: str = ""
    raw_clause: str = ""
    #: Literal object phrase ("12 rules for life", "a poem").
    object: str = ""
    #: Subject to look up or write about ("cars", "quantum computing").
    topic: str = ""
    #: Canonical content type when known ("poem", "script").
    content_type: str = ""
    #: Canonical transformation when the clause asks for one.
    transform: str = ""
    #: True when the object refers to what an earlier outcome produced.
    uses_previous_result: bool = False
    destination: str = ""
    destination_kind: str = ""  # application | file | ""
    is_artifact_request: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "verb": self.verb,
            "object": self.object,
            "topic": self.topic,
            "content_type": self.content_type,
            "transform": self.transform,
            "uses_previous_result": self.uses_previous_result,
            "destination": self.destination,
            "destination_kind": self.destination_kind,
            "is_artifact_request": self.is_artifact_request,
            "clause": self.raw_clause,
        }


@dataclass
class TaskGoal:
    """The structured goal a request represents, plus its completion contract."""

    original_request: str
    objective: str
    intent: str = "unknown"
    outcomes: list[Outcome] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    requested_output: str = "summary"
    transform: str = ""
    destination: str = ""
    destination_kind: str = ""
    constraints: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    must_be_artifact: bool = False
    needs_retrieval: bool = False
    compound: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)

    # -- helpers ----------------------------------------------------------------

    def outcome(self, action: str) -> Outcome | None:
        for item in self.outcomes:
            if item.action == action:
                return item
        return None

    def has(self, action: str) -> bool:
        return self.outcome(action) is not None

    def record(self, event: str, **fields: Any) -> None:
        entry: dict[str, Any] = {"event": event}
        entry.update({key: value for key, value in fields.items() if value not in (None, "", [], {})})
        self.trace.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_request": self.original_request,
            "objective": self.objective,
            "intent": self.intent,
            "requested_output": self.requested_output,
            "transform": self.transform or None,
            "destination": self.destination or None,
            "destination_kind": self.destination_kind or None,
            "must_be_artifact": self.must_be_artifact,
            "needs_retrieval": self.needs_retrieval,
            "compound": self.compound,
            "entities": dict(self.entities),
            "constraints": list(self.constraints),
            "required_capabilities": list(self.required_capabilities),
            "subtasks": list(self.subtasks),
            "completion_criteria": list(self.completion_criteria),
            "outcomes": [item.to_dict() for item in self.outcomes],
            "trace": list(self.trace),
        }


# ---------------------------------------------------------------------------
# Clause splitting
# ---------------------------------------------------------------------------


def _has_verb(clause: str) -> bool:
    return any(
        _matches_verb(clause, verbs) is not None
        for verbs in (OPEN_VERBS, RETRIEVE_VERBS, CREATE_VERBS, DELIVER_VERBS)
    ) or _matches_verb(clause, TRANSFORM_VERBS) is not None


def split_clauses(text: str) -> list[str]:
    """Split a request into intent clauses, never splitting object phrases.

    A boundary only starts a new clause when the next piece actually contains a
    verb, so "a poem about cars and trucks" stays one clause while
    "find X and summarize it" becomes two.
    """

    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return []
    pieces = [piece.strip(" ,") for piece in _CLAUSE_BOUNDARY_RE.split(normalized)]
    clauses: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if not clauses or _has_verb(piece):
            clauses.append(piece)
        else:
            # No verb: this is a continuation of the previous clause's object.
            clauses[-1] = f"{clauses[-1]} {piece}".strip()
    # Leading fillers ("Please open Notepad") belong to no clause semantics.
    return [_LEADING_FILLER_RE.sub("", clause).strip() for clause in clauses if clause.strip()]


# ---------------------------------------------------------------------------
# Per-clause parsing
# ---------------------------------------------------------------------------

#: Common Windows applications. Callers may extend this from the live registry.
KNOWN_APPLICATIONS: frozenset[str] = frozenset({
    "notepad", "wordpad", "calculator", "calc", "paint", "mspaint",
    "explorer", "file explorer", "task manager", "command prompt", "cmd",
    "terminal", "powershell", "windows terminal", "vscode", "vs code",
    "visual studio code", "chrome", "google chrome", "edge", "microsoft edge",
    "firefox", "word", "excel", "powerpoint", "outlook", "spotify", "discord",
    "steam", "vlc", "settings", "snipping tool",
})


def _known(lowered: str, known_applications: Iterable[str]) -> str:
    """Return the application name a destination phrase names, else ''."""

    candidate = lowered.strip().strip(".,!?").strip()
    if not candidate:
        return ""
    words = candidate.split()
    for size in range(min(3, len(words)), 0, -1):
        phrase = " ".join(words[:size])
        if phrase in known_applications or phrase in KNOWN_APPLICATIONS:
            return phrase
    return ""


def destination_from_clause(
    clause: str,
    known_applications: Iterable[str],
    *,
    verb: str | None = None,
) -> tuple[str, str]:
    """Return ``(destination, kind)`` where kind is application | file | ''."""

    filename = _FILENAME_RE.search(clause)
    if filename:
        return filename.group("name").strip(), "file"
    match = _DEST_PREP_RE.search(clause)
    if match:
        candidate = match.group("dest").strip().strip(".,!?")
        application = _known(candidate.casefold(), known_applications)
        if application:
            # Preserve the user's capitalization when it is a proper name.
            return " ".join(candidate.split()[: len(application.split())]), "application"
    if verb:
        # "Open Notepad" names its target with no preposition at all.
        tail = re.sub(rf"^\s*{re.escape(verb)}\w*\b", "", clause, count=1, flags=re.IGNORECASE)
        candidate = _clean_target(tail)
        application = _known(candidate.casefold(), known_applications)
        if application:
            return " ".join(candidate.split()[: len(application.split())]), "application"
    return "", ""


def _detect_transform(lowered: str) -> str:
    for phrase, canonical in TRANSFORM_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            return canonical
    verb = _matches_verb(lowered, TRANSFORM_VERBS)
    return TRANSFORM_VERBS.get(verb or "", "")


def _detect_content_type(lowered: str) -> tuple[str, str]:
    """Return ``(content_type, artifact_noun)`` for a clause."""

    for noun, canonical in _ARTIFACT_NOUNS.items():
        if re.search(rf"\b{re.escape(noun)}\b", lowered):
            return canonical, canonical
    for noun, canonical in _CONTENT_NOUNS.items():
        if re.search(rf"\b{re.escape(noun)}\b", lowered):
            return canonical, ""
    return "", ""


def _topic_from_clause(clause: str) -> str:
    match = _TOPIC_RE.search(clause)
    if not match:
        return ""
    return _clean_target(match.group("topic"))


_TARGET_LEAD_RE = re.compile(
    r"^\s*(?:for|me|the|a|an|some|any|information|info|facts|details|about|on|of|"
    r"web|internet|online|and|please)\s+",
    re.IGNORECASE,
)
_TRAILING_CLAUSE_RE = re.compile(
    r"\s+(?:and|then|,)\s+(?:then\s+)?(?:summar\w+|explain\w*|describe|copy|paste|write|save|"
    r"put|type|insert|add|store|send|open|launch|browse)\b.*$",
    re.IGNORECASE,
)


def _clean_target(value: str) -> str:
    """Trim articles, command words, and trailing instruction clauses."""

    text = re.sub(r"\s+", " ", (value or "").strip()).strip(" .,!?;:\"'")
    text = _TRAILING_CLAUSE_RE.sub("", text)
    previous = None
    while text and previous != text:
        previous = text
        text = _TARGET_LEAD_RE.sub("", text, count=1).strip()
    text = text.strip(" .,!?;:\"'")
    return "" if text.casefold() in _STOP_TARGET_WORDS else text


def _object_phrase(clause: str, verb: str | None, *, destination: str) -> str:
    """Return the phrase the clause acts on, with the destination removed."""

    text = clause
    if destination:
        # Drop the destination preposition phrase so it cannot pollute the object.
        text = re.sub(
            r"\b(?:to|into|onto|inside|within|in)\s+" + re.escape(destination) + r"(?:\b|\s|$).*$",
            "",
            text,
            flags=re.IGNORECASE,
        )
    if verb:
        match = re.search(rf"\b{re.escape(verb)}\w*\b(.*)$", text, re.IGNORECASE)
        if match:
            text = match.group(1)
    return _clean_target(text)


def _is_reference(value: str) -> bool:
    lowered = (value or "").casefold().strip()
    if not lowered:
        return True
    if lowered.startswith(("it ", "it'", "that ", "this ", "them", "those")):
        return True
    return any(re.search(rf"\b{re.escape(reference)}\b", lowered) for reference in REFERENCES)


#: Verb groups, ordered by the intent each expresses (used to find the main verb).
_VERB_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("open", OPEN_VERBS),
    ("retrieve", RETRIEVE_VERBS),
    ("create", CREATE_VERBS),
    ("deliver", DELIVER_VERBS),
    ("transform", tuple(TRANSFORM_VERBS)),
)


def main_verb(clause: str) -> tuple[str, str]:
    """Return ``(action, verb)`` for the *first* verb in the clause.

    The first verb is the clause's head: "find how to write a resume" is a
    retrieval whose object is "how to write a resume", not a write request.
    """

    best_action, best_verb, best_position = "", "", len(clause) + 1
    for action, verbs in _VERB_GROUPS:
        for verb in verbs:
            match = re.search(rf"\b{re.escape(verb)}\w*\b", clause, re.IGNORECASE)
            if match and match.start() < best_position:
                best_action, best_verb, best_position = action, verb, match.start()
    return best_action, best_verb


def parse_clause(
    clause: str,
    *,
    previous: list[Outcome] | None = None,
    known_applications: Iterable[str] = (),
) -> Outcome | None:
    """Parse one clause into an :class:`Outcome`, or ``None`` when it is not one."""

    outcomes = previous or []
    lowered = clause.casefold()
    head_action, head_verb = main_verb(clause)
    destination, destination_kind = destination_from_clause(
        clause, known_applications, verb=head_verb if head_action == "open" else None
    )
    topic = _topic_from_clause(clause)
    transform = _detect_transform(lowered)
    content_type, artifact_noun = _detect_content_type(lowered)
    open_verb = _matches_verb(lowered, OPEN_VERBS)
    retrieve_verb = _matches_verb(lowered, RETRIEVE_VERBS)
    create_verb = _matches_verb(lowered, CREATE_VERBS)
    deliver_verb = _matches_verb(lowered, DELIVER_VERBS)
    object_phrase = _object_phrase(clause, head_verb or None, destination=destination)
    produced_reuse = bool(outcomes) and (
        bool(transform)
        or any(re.search(rf"\b{re.escape(phrase)}\b", lowered) for phrase, _ in TRANSFORM_PHRASES)
        or bool(re.search(r"\bresults?\b", lowered))
    )
    reference_object = _is_reference(object_phrase) or produced_reuse
    already_transformed = any(item.transform for item in outcomes)

    if transform and not already_transformed:
        return Outcome(
            action="transform",
            verb=transform,
            raw_clause=clause,
            object=object_phrase,
            topic=topic,
            content_type=content_type,
            transform=transform,
            uses_previous_result=reference_object or bool(outcomes),
            destination=destination,
            destination_kind=destination_kind,
        )
    if (
        head_action == "open"
        and destination_kind == "application"
        and not _INFORMATION_RE.search(clause)
    ):
        return Outcome(
            action="open",
            verb=open_verb or head_verb or "open",
            raw_clause=clause,
            object=destination,
            destination=destination,
            destination_kind=destination_kind,
        )
    if (
        head_action == "retrieve"
        and not reference_object
        and (object_phrase or topic or content_type)
    ):
        return Outcome(
            action="retrieve",
            verb=retrieve_verb or head_verb or "search",
            raw_clause=clause,
            object=topic or object_phrase,
            topic=topic or object_phrase,
            content_type=content_type,
            destination=destination,
            destination_kind=destination_kind,
            is_artifact_request=(
                bool(artifact_noun) and not _INFORMATION_RE.search(clause) and not transform
            ),
        )
    if head_action == "create" and not reference_object and (content_type or topic):
        return Outcome(
            action="create",
            verb=create_verb or head_verb or "create",
            raw_clause=clause,
            object=object_phrase,
            topic=topic,
            content_type=content_type,
            destination=destination,
            destination_kind=destination_kind,
        )
    if head_action == "deliver" or destination_kind:
        return Outcome(
            action="deliver",
            verb=deliver_verb or head_verb or "write",
            raw_clause=clause,
            object=object_phrase,
            topic=topic,
            content_type=content_type,
            uses_previous_result=reference_object or bool(outcomes),
            destination=destination,
            destination_kind=destination_kind,
        )
    return None


# ---------------------------------------------------------------------------
# Goal assembly
# ---------------------------------------------------------------------------

#: Capability implied by each outcome, used to express planning expectations.
_CAPABILITY_BY_ACTION: dict[str, str] = {
    "open": "applications.launch_named",
    "retrieve": "web.research",
    "create": "content.generate",
    "transform": "content.generate",
    "deliver": "applications.write_text",
}

#: Human labels for the requested output shape.
_OUTPUT_LABEL: dict[str, str] = {
    "summarize": "summary",
    "key_points": "key points",
    "explain": "explanation",
    "outline": "outline",
    "compare": "comparison",
}


def _objective(goal: TaskGoal) -> str:
    parts: list[str] = []
    if goal.has("create"):
        created = goal.outcome("create")
        label = (created.content_type or "content") if created else "content"
        about = f" about {created.topic}" if created and created.topic else ""
        parts.append(f"produce a {label}{about}")
    if goal.has("retrieve"):
        target = goal.outcome("retrieve")
        subject = (target.topic or target.object) if target else "the requested subject"
        verb = (
            "obtain the requested content for"
            if goal.must_be_artifact
            else "find useful information about"
        )
        parts.append(f"{verb} {subject}".strip())
    if goal.has("transform"):
        label = _OUTPUT_LABEL.get(goal.transform, goal.transform or "transformed content")
        parts.append(f"generate a {label} from that material")
    if goal.destination:
        parts.append(f"deliver it to {goal.destination}")
    if goal.has("open") and not goal.destination:
        opened = goal.outcome("open")
        parts.append(f"open {(opened.object if opened else 'the application')}")
    if not parts:
        return "understand and answer the request"
    return "; ".join(parts) + "."


def _intent(goal: TaskGoal) -> str:
    if not goal.outcomes:
        return "answer"
    if goal.destination and goal.has("transform"):
        return "research_and_deliver" if goal.has("retrieve") else "transform_and_deliver"
    if goal.destination and goal.has("create"):
        return "create_and_deliver"
    if goal.has("transform"):
        return "transform"
    if goal.has("create"):
        return "create"
    if goal.has("retrieve"):
        return "research"
    if goal.has("deliver"):
        return "deliver"
    if goal.has("open"):
        return "open"
    return "unknown"


def _capabilities(goal: TaskGoal) -> list[str]:
    capabilities: list[str] = []
    for outcome in goal.outcomes:
        if outcome.action == "deliver":
            capability = (
                "filesystem.write"
                if outcome.destination_kind == "file"
                else "applications.write_text"
            )
        else:
            capability = _CAPABILITY_BY_ACTION.get(outcome.action, "")
        if capability and capability not in capabilities:
            capabilities.append(capability)
    if goal.destination_kind == "application" and "applications.write_text" not in capabilities:
        capabilities.append("applications.write_text")
    return capabilities


def _subtasks(goal: TaskGoal) -> list[str]:
    steps: list[str] = []
    if goal.has("retrieve"):
        target = goal.outcome("retrieve")
        subject = (target.topic or target.object) if target else "the requested subject"
        if goal.must_be_artifact:
            steps.append(f"Locate and read the requested content: {subject}.")
        else:
            steps.append(f"Search for useful source material about {subject}.")
            steps.append("Evaluate whether the retrieved material is sufficient.")
            steps.append("Refine the query and retrieve again if it is not.")
    if goal.has("create"):
        created = goal.outcome("create")
        label = (created.content_type or "content") if created else "content"
        steps.append(f"Generate the requested {label}.")
    if goal.has("transform"):
        label = _OUTPUT_LABEL.get(goal.transform, goal.transform)
        steps.append(f"Generate the requested {label} from the material.")
        if goal.destination:
            steps.append(f"Format the {label} for delivery.")
    if goal.has("open") and goal.destination_kind != "application":
        opened = goal.outcome("open")
        steps.append(f"Open {(opened.object if opened else 'the application')}.")
    if goal.destination:
        steps.append(f"Deliver the result to {goal.destination}.")
        steps.append("Verify that the result was actually written.")
    elif goal.has("transform") or goal.has("create"):
        steps.append("Return the produced content.")
    if not steps:
        steps.append("Answer the request from the best available sources.")
    return steps


def completion_criteria(goal: TaskGoal) -> list[str]:
    """Return the explicit criteria that must hold for the goal to be complete."""

    criteria: list[str] = []
    if goal.has("retrieve"):
        criteria.append(CRITERION_EVIDENCE_SUFFICIENT)
    if goal.has("create"):
        criteria.append(CRITERION_CONTENT_GENERATED)
    if goal.has("transform"):
        criteria.append(CRITERION_TRANSFORMATION_PRODUCED)
    if goal.has("open"):
        criteria.append(CRITERION_APPLICATION_OPENED)
    if goal.destination:
        criteria.append(
            CRITERION_FILE_WRITTEN
            if goal.destination_kind == "file"
            else CRITERION_DELIVERED_TO_APPLICATION
        )
        criteria.append(CRITERION_DELIVERY_VERIFIED)
    return criteria


def build_goal(
    text: str,
    *,
    known_applications: Iterable[str] = (),
    clause_limit: int = 12,
) -> TaskGoal:
    """Turn a request into a structured, decomposed :class:`TaskGoal`."""

    clauses = split_clauses(text)[:clause_limit]
    goal = TaskGoal(original_request=text, objective="")
    goal.record("TASK", request=text, clauses=len(clauses))
    for clause in clauses:
        outcome = parse_clause(
            clause, previous=goal.outcomes, known_applications=known_applications
        )
        if outcome is None:
            goal.record("CLAUSE", clause=clause, outcome="unrecognized")
            continue
        goal.outcomes.append(outcome)
        goal.record("OUTCOME", clause=clause, action=outcome.action, object=outcome.object)

    goal.compound = len(goal.outcomes) > 1
    goal.transform = next((item.transform for item in goal.outcomes if item.transform), "")
    destination = next((item for item in goal.outcomes if item.destination), None)
    if destination is not None:
        goal.destination = destination.destination
        goal.destination_kind = destination.destination_kind
    goal.must_be_artifact = any(
        item.is_artifact_request for item in goal.outcomes if item.action == "retrieve"
    )
    goal.needs_retrieval = goal.has("retrieve")
    if goal.must_be_artifact:
        goal.requested_output = "full_text"
    elif goal.has("create") and not goal.has("retrieve"):
        goal.requested_output = "content"
    else:
        goal.requested_output = "summary"
    goal.entities = _entities(goal)
    goal.required_capabilities = _capabilities(goal)
    goal.intent = _intent(goal)
    goal.objective = _objective(goal)
    goal.subtasks = _subtasks(goal)
    goal.completion_criteria = completion_criteria(goal)
    goal.record(
        "PLAN",
        intent=goal.intent,
        capabilities=goal.required_capabilities,
        criteria=goal.completion_criteria,
    )
    return goal


def _entities(goal: TaskGoal) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    if goal.destination:
        if goal.destination_kind == "file":
            entities["filename"] = goal.destination
        else:
            entities["application"] = goal.destination
            entities["destination"] = goal.destination
    if goal.destination_kind:
        entities["destination_kind"] = goal.destination_kind
    if goal.transform:
        entities["transform"] = goal.transform
    created = goal.outcome("create")
    if created is not None:
        if created.content_type:
            entities["content_type"] = created.content_type
        if created.topic:
            entities["topic"] = created.topic
    retrieved = goal.outcome("retrieve")
    if retrieved is not None:
        if retrieved.topic and not entities.get("topic"):
            entities["topic"] = retrieved.topic
        if retrieved.content_type and not entities.get("content_type"):
            entities["content_type"] = retrieved.content_type
    if goal.must_be_artifact:
        entities["must_be_artifact"] = True
    return entities


def needs_decomposition(goal: TaskGoal) -> bool:
    """Return True when the request must be planned from the goal, not one match."""

    if not goal.outcomes:
        return False
    if goal.compound or goal.transform:
        return True
    # A retrieval whose result is delivered needs the composed chain even when
    # it arrived as a single clause.
    return goal.has("retrieve") and goal.has("deliver")
