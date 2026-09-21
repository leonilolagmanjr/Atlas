"""Intent Engine 2.0 — context-aware goal and intent understanding.

Atlas already had a capable task interpreter (LLM + deterministic heuristics)
and a deterministic validation → planning → execution pipeline. The Intent
Engine is the *understanding layer* that sits on top of the interpreter's output
and makes explicit what a small local model tends to leave implicit:

    USER LANGUAGE -> INTENT -> DESIRED OUTCOME -> STRUCTURED TASK -> PLAN

Specifically it:

* distinguishes GOAL (why) from OPERATIONS (how) from DESIRED OUTCOME (end state),
* resolves follow-up references against conversation context (see
  :mod:`reasoning.reference_resolver`),
* forms a small set of interpretation hypotheses and picks one deterministically,
* performs a *second* critical pass that repairs the common mistakes a small
  model makes (topic-vs-destination, retrieval-vs-action, unnecessary web),
* turns the desired outcome into semantic capability *requirements* that Atlas —
  not the model — validates against the live capability registry,
* consumes structured, high-confidence correction lessons
  (:mod:`reasoning.correction_memory`).

It is deliberately one component with several deterministic passes, not a swarm
of agents. Simple requests take the fast path; only genuinely ambiguous or
multi-step requests pay for the extra passes. It never executes anything and it
never exposes chain-of-thought — only concise, structured interpretation notes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from models_task import (
    INTENT_OPERATIONS,
    OBJECT_TYPES,
    Task,
    TaskAction,
)
from reasoning.correction_memory import CorrectionMemory
from reasoning.reference_resolver import ReferenceResolver, ResolvedContext

logger = logging.getLogger(__name__)

#: Ambiguity below this interpretation confidence is worth surfacing.
CLARIFY_CONFIDENCE = 0.45

#: Verbs that mean "make information available" rather than "perform an action".
_QUESTION_LEADS: tuple[str, ...] = (
    "what", "who", "when", "where", "why", "how", "which", "whose", "explain",
    "define", "describe", "tell me",
)
_SEARCH_VERBS: tuple[str, ...] = (
    "search", "find", "look up", "look for", "google", "browse", "research",
    "get me", "show me", "pull up", "fetch",
)
_CREATE_VERBS: tuple[str, ...] = (
    "create", "write", "make", "generate", "compose", "produce", "draft",
)
_TRANSFORM_VERBS: tuple[str, ...] = (
    "summarize", "summarise", "condense", "shorten", "expand", "translate",
    "rewrite", "transform", "extract", "digest", "abstract", "simplify",
)
_OPEN_VERBS: tuple[str, ...] = ("open", "launch", "start", "run")
_COMPARE_VERBS: tuple[str, ...] = ("compare", "contrast", "difference", "differences", "versus", "vs")
_ORGANIZE_VERBS: tuple[str, ...] = ("organize", "organise", "sort", "move", "copy", "rename", "group")
_INSPECT_VERBS: tuple[str, ...] = ("inspect", "check", "diagnose", "analyze", "analyse", "monitor")
_COMMUNICATE_VERBS: tuple[str, ...] = ("email", "message", "send", "share", "notify")

#: Nouns that indicate a video object (often YouTube).
_VIDEO_TERMS: tuple[str, ...] = ("video", "videos", "channel", "clip", "movie", "film")
#: Nouns that indicate a document/file object.
_DOCUMENT_TERMS: tuple[str, ...] = (
    "document", "documents", "file", "files", "pdf", "report", "resume", "cv",
    "spreadsheet", "docx", "attachment", "notes",
)
#: Current-information markers that require live web verification.
_FRESHNESS_TERMS: tuple[str, ...] = (
    "latest", "newest", "current", "currently", "today", "tonight",
    "right now", "this week", "this month", "recent", "recently", "up to date",
    "up-to-date", "news", "trending", "breaking", "release date", "price",
    "stock", "weather", "score", "scores",
)
#: Application names Atlas recognises for a destination/launch.
_KNOWN_APPLICATIONS: frozenset[str] = frozenset(
    {
        "notepad", "calculator", "calc", "chrome", "firefox", "edge", "vscode",
        "vs code", "explorer", "file explorer", "word", "excel", "powerpoint",
        "outlook", "terminal", "cmd", "command prompt", "powershell", "paint",
        "spotify", "slack", "discord", "settings", "task manager",
    }
)
#: Object type vocabulary the engine can infer deterministically.
_OBJECT_HINTS: dict[str, str] = {
    "video": "video", "videos": "video", "channel": "website",
    "image": "image", "images": "image", "photo": "image", "picture": "image",
    "code": "code", "script": "code", "program": "code", "function": "code",
}
#: Filler words that carry no subject on their own. A topic made only of these
#: plus the object noun itself ("some videos", "the videos") is under-specified:
#: the user named a *kind* of thing but not *what* about it, so Atlas should not
#: silently commit to a research reading (spec section 17).
_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "some", "any", "me", "my", "our", "your",
        "good", "great", "best", "nice", "new", "few", "couple", "of",
        "please", "for", "about", "show", "get", "give", "find", "pull",
    }
)


def _matches_any(text: str, terms: Iterable[str]) -> bool:
    for term in terms:
        if " " in term:
            if term in text:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", text):
            return True
    return False


@dataclass
class IntentHypothesis:
    """One plausible reading of a request with a deterministic score."""

    goal: str
    object_type: str
    reason: str
    score: float = 0.0
    destination: str = ""
    needs_web: bool = False
    needs_files: bool = False
    needs_application: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "object_type": self.object_type,
            "reason": self.reason,
            "score": round(self.score, 3),
            "destination": self.destination,
            "needs_web": self.needs_web,
            "needs_files": self.needs_files,
            "needs_application": self.needs_application,
        }


@dataclass
class IntentReading:
    """The structured result the Intent Engine writes back onto a Task."""

    goal: str = "unknown"
    desired_outcome: str = ""
    object_type: str = "unknown"
    operations: list[str] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    needs_web: bool = False
    needs_files: bool = False
    needs_application: bool = False
    ambiguities: list[str] = field(default_factory=list)
    confidence: float = 0.0
    hypotheses: list[IntentHypothesis] = field(default_factory=list)
    interpretation_notes: list[str] = field(default_factory=list)
    context: ResolvedContext | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "desired_outcome": self.desired_outcome,
            "object_type": self.object_type,
            "operations": list(self.operations),
            "transformations": list(self.transformations),
            "references": list(self.references),
            "required_capabilities": list(self.required_capabilities),
            "needs_web": self.needs_web,
            "needs_files": self.needs_files,
            "needs_application": self.needs_application,
            "ambiguities": list(self.ambiguities),
            "confidence": round(self.confidence, 3),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "interpretation_notes": list(self.interpretation_notes),
            "context": self.context.to_dict() if self.context else None,
        }


class IntentEngine:
    """Derive goal, desired outcome, and capability requirements for a request.

    The engine consumes an already-interpreted :class:`Task` (from
    :class:`reasoning.task_interpreter.SemanticTaskInterpreter`) and augments it
    in place with the Intent Engine 2.0 fields. It never plans or executes, and
    it never calls the model: all reasoning here is deterministic so the behavior
    is reproducible, debuggable, and free of extra inference latency.
    """

    def __init__(
        self,
        *,
        capabilities: Any = None,
        correction_memory: CorrectionMemory | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._resolver = ReferenceResolver()
        self._memory = correction_memory if correction_memory is not None else CorrectionMemory()

    # -- public API --------------------------------------------------------------

    def understand(
        self,
        task: Task,
        *,
        prior_task: Task | None = None,
        history: str = "",
    ) -> Task:
        """Augment ``task`` with goal / desired outcome / capability requirements."""

        text = task.original_prompt or ""
        reading = self._read(task, prior_task=prior_task, history=history)
        task.apply_intent(reading)
        # Record the interpretation for observability; never chain-of-thought.
        task.context["intent_reading"] = reading.to_dict()
        # Confidence represents interpretation certainty, not the confidence of the
        # action the interpreter happened to build. A genuine ambiguity the engine
        # could not settle from context lowers it; otherwise take the stronger read.
        if reading.ambiguities:
            task.confidence = round(min(task.confidence, reading.confidence), 4)
        else:
            task.confidence = round(max(task.confidence, reading.confidence), 4)
        return task

    def analyze(
        self,
        text: str,
        *,
        task: Task | None = None,
        prior_task: Task | None = None,
        history: str = "",
    ) -> IntentReading:
        """Return the reading without mutating a task (for tests/inspection)."""

        stub = task or Task(original_prompt=text)
        if not stub.original_prompt:
            stub.original_prompt = text
        return self._read(stub, prior_task=prior_task, history=history)

    # -- pass 1: understand ------------------------------------------------------

    def _read(self, task: Task, *, prior_task: Task | None, history: str) -> IntentReading:
        text = (task.original_prompt or "").strip()
        lowered = text.casefold()
        reading = IntentReading()
        reading.context = self._resolver.resolve(text, prior_task=prior_task, history=history)
        reading.notes = reading.interpretation_notes  # alias

        actions = list(task.actions)
        entity_application = _entity_application(task)
        destination = entity_application or self._destination_from_text(text)

        # Hypotheses: a small set (1-3) of plausible readings, scored.
        reading.hypotheses = self._hypotheses(lowered, task, destination=destination)
        best = max(reading.hypotheses, key=lambda h: h.score) if reading.hypotheses else None

        # Derive goal / object from the deterministic reading of text + actions.
        goal, object_type = self._goal_and_object(lowered, task, actions, best)
        # A follow-up may inherit the previous task's goal with a new subject
        # ("do the same thing for Interstellar"): a small reply carries no goal of
        # its own, so the prior task supplies it and the subject is replaced.
        if (
            reading.context is not None
            and reading.context.replace_subject
            and prior_task is not None
            and goal in {"answer", "converse", "unknown"}
        ):
            goal = prior_task.goal or goal
            object_type = prior_task.object_type if prior_task.object_type != "unknown" else object_type
            reading.interpretation_notes.append(f"inherited prior goal '{goal}' for new subject")
        reading.goal = goal
        reading.object_type = object_type
        reading.operations = self._operations(goal, task, actions, lowered, reading.context)
        reading.transformations = self._transformations(lowered, task, reading.context)
        reading.references = self._references(reading.context)

        # Needs flags: what the DESIRED OUTCOME actually requires.
        self._resolve_needs(reading, task, actions, lowered, destination)

        # Desired outcome phrased as an end state, not an operation.
        reading.desired_outcome = self._desired_outcome(task, reading, destination=destination)

        # Capability analysis: semantic requirements -> concrete capabilities.
        reading.required_capabilities = self._required_capabilities(reading, task, actions)

        # Pass 2: critically validate the interpretation and repair it. This is
        # the deterministic equivalent of a second model call, without one.
        self._critique(reading, task, actions, destination)

        reading.confidence = self._confidence(reading, task, actions, best)

        # Correction-memory lessons may override concrete fields.
        self._apply_lessons(reading, task, text)

        # Ambiguity detection (only when context cannot resolve it).
        reading.ambiguities = self._ambiguities(task, reading, actions, destination)

        reading.interpretation_notes = list(dict.fromkeys(reading.interpretation_notes))
        return reading

    # -- hypotheses --------------------------------------------------------------

    def _hypotheses(self, lowered: str, task: Task, *, destination: str) -> list[IntentHypothesis]:
        hypotheses: list[IntentHypothesis] = []
        is_question = task.task_type == "informational" or _starts_with(lowered, _QUESTION_LEADS)
        has_search = _matches_any(lowered, _SEARCH_VERBS)
        has_create = _matches_any(lowered, _CREATE_VERBS)
        has_transform = _matches_any(lowered, _TRANSFORM_VERBS)
        has_open = _matches_any(lowered, _OPEN_VERBS)
        has_delivery = bool(destination) or bool(task.entities.get("filename"))

        if is_question and not has_search and not has_create:
            hypotheses.append(
                IntentHypothesis("answer", self._object_hint(lowered) or "information",
                                 "informational question", score=0.75)
            )
        if has_search:
            needs_web = self._needs_web(lowered, task)
            hypotheses.append(
                IntentHypothesis(
                    "research" if needs_web else "find",
                    self._object_hint(lowered) or "information",
                    "explicit search verb", score=0.8,
                    needs_web=needs_web,
                )
            )
        if has_create:
            score = 0.85 if has_delivery else 0.7
            hypotheses.append(
                IntentHypothesis(
                    "create_and_deliver" if has_delivery else "create",
                    self._object_hint(lowered) or "content",
                    "explicit create verb", score=score,
                    destination=destination, needs_application=bool(destination),
                )
            )
        if has_transform and not has_create:
            hypotheses.append(
                IntentHypothesis(
                    "summarize" if "summar" in lowered else "transform",
                    "information", "explicit transformation verb", score=0.75,
                )
            )
        if has_open and destination:
            hypotheses.append(
                IntentHypothesis("execute", "application", "open/launch verb",
                                 score=0.85, destination=destination, needs_application=True)
            )
        if not hypotheses:
            hypotheses.append(
                IntentHypothesis("answer", "information", "default conversational reading", score=0.4)
            )
        # Keep the set small (1-3), highest first.
        hypotheses.sort(key=lambda h: h.score, reverse=True)
        return hypotheses[:3]

    # -- goal / object -----------------------------------------------------------

    def _goal_and_object(
        self,
        lowered: str,
        task: Task,
        actions: list[TaskAction],
        best: IntentHypothesis | None,
    ) -> tuple[str, str]:
        caps = {a.capability for a in actions}
        # A task that both gathers and delivers is a "…_and_deliver" goal.
        has_delivery = bool(caps & {"applications.write_text", "filesystem.write"})
        has_search = bool(caps & {"web.search", "web.research", "web.fetch"})
        has_generate = "content.generate" in caps
        # A transformation (summarize/condense/…) is the user's actual goal even
        # though the plan runs content.generate on the retrieved/read source.
        wants_transform = _matches_any(lowered, _TRANSFORM_VERBS) or bool(
            task.entities.get("transform")
        )
        obj_hint = self._object_hint(lowered)
        transform_goal = "summarize" if "summar" in lowered else "transform"

        if has_search and wants_transform:
            return (
                ("research_and_deliver" if has_delivery else "research"),
                obj_hint or "information",
            )
        if wants_transform and has_generate and not has_search:
            return (
                ("create_and_deliver" if has_delivery else transform_goal),
                obj_hint or "document",
            )
        if has_search and has_delivery:
            return "research_and_deliver", obj_hint or "information"
        if has_generate and has_delivery:
            return "create_and_deliver", obj_hint or "content"
        if has_search:
            return ("research" if self._needs_web(lowered, task) else "find",
                    self._object_hint(lowered) or "information")
        if has_generate:
            return "create", self._object_hint(lowered) or "content"
        if "applications.launch_named" in caps and len(actions) <= 1:
            return "execute", "application"
        if caps & {"filesystem.create_folder"}:
            return "organize", "file"
        if caps & {"filesystem.move", "filesystem.copy"}:
            return "organize", "file"
        if caps & {"filesystem.read", "filesystem.search", "filesystem.search_content"}:
            return "find", "document"

        # No actions: semantic reading of the text.
        if _matches_any(lowered, _COMPARE_VERBS):
            return "compare", self._object_hint(lowered) or "information"
        if _matches_any(lowered, _ORGANIZE_VERBS):
            return "organize", self._object_hint(lowered) or "file"
        if _matches_any(lowered, _INSPECT_VERBS):
            return "inspect", "system"
        if _matches_any(lowered, _COMMUNICATE_VERBS):
            return "communicate", "information"
        if _matches_any(lowered, _TRANSFORM_VERBS):
            return ("summarize" if "summar" in lowered else "transform",
                    self._object_hint(lowered) or "information")
        if task.task_type == "informational" or _starts_with(lowered, _QUESTION_LEADS):
            return "answer", self._object_hint(lowered) or "information"
        if best is not None:
            return best.goal, best.object_type
        return "converse", "conversation"

    @staticmethod
    def _object_hint(lowered: str) -> str:
        for term, canonical in _OBJECT_HINTS.items():
            if re.search(rf"\b{term}\b", lowered):
                return canonical
        if _matches_any(lowered, _VIDEO_TERMS):
            return "video"
        if _matches_any(lowered, _DOCUMENT_TERMS):
            return "document"
        for candidate in OBJECT_TYPES:
            if candidate in {"unknown", "content"}:
                continue
            if re.search(rf"\b{candidate}s?\b", lowered):
                return candidate
        return ""

    # -- operations / transformations / references -------------------------------

    def _operations(
        self,
        goal: str,
        task: Task,
        actions: list[TaskAction],
        lowered: str,
        context: ResolvedContext | None,
    ) -> list[str]:
        operations: list[str] = []
        for action in actions:
            cap = action.capability
            if cap.startswith("web."):
                operations.append("search")
            elif cap == "content.generate":
                operations.append("generate")
            elif cap == "content.format":
                operations.append("format")
            elif cap == "applications.write_text":
                operations.append("write")
            elif cap == "applications.launch_named":
                operations.append("open")
            elif cap == "filesystem.search":
                operations.append("search")
            elif cap == "filesystem.read":
                operations.append("read")
            elif cap == "filesystem.write":
                operations.append("write")
            elif cap.startswith("filesystem."):
                operations.append("organize")
            elif cap.startswith("system."):
                operations.append("verify")
        # A transformation-only follow-up is a transform operation.
        if context is not None and context.mutates_output:
            operations = ["transform"]
        # Reading-only goals still perform an "answer" operation.
        if not operations and goal == "answer":
            operations = ["answer"]
        return _dedupe([op for op in operations if op in INTENT_OPERATIONS])

    def _transformations(
        self, lowered: str, task: Task, context: ResolvedContext | None
    ) -> list[str]:
        transforms: list[str] = []
        if context is not None and context.transformation:
            transforms.append(context.transformation)
        if re.search(r"\bshort\w*\b|\bbrief\w*\b|\bconcise\b", lowered):
            transforms.append("shorten")
        if re.search(r"\blong\w*\b|\bdetailed\b|\bin-?depth\b", lowered):
            transforms.append("expand")
        if re.search(r"\bsummari[sz]e\w*\b|\bsummary\b", lowered):
            transforms.append("summarize")
        if re.search(r"\btranslate\b", lowered):
            transforms.append("translate")
        if re.search(r"\bformat\b|\bformatting\b", lowered):
            transforms.append("format")
        return _dedupe(transforms)

    @staticmethod
    def _references(context: ResolvedContext | None) -> list[str]:
        if context is None or not context.has_reference:
            return []
        refs: list[str] = []
        if context.target:
            refs.append(context.target)
        if context.new_subject:
            refs.append(f"subject:{context.new_subject}")
        if context.ordinal:
            refs.append(f"ordinal:{context.ordinal}")
        return refs

    # -- needs -------------------------------------------------------------------

    def _resolve_needs(
        self,
        reading: IntentReading,
        task: Task,
        actions: list[TaskAction],
        lowered: str,
        destination: str,
    ) -> None:
        caps = {a.capability for a in actions}
        reading.needs_web = bool(caps & {"web.search", "web.research", "web.fetch"})
        reading.needs_files = any(c.startswith("filesystem.") for c in caps) or bool(
            task.entities.get("file_subject") or task.entities.get("filename")
        )
        reading.needs_application = bool(destination) or "applications.write_text" in caps
        # If no actions were planned, fall back to the semantic reading.
        if not actions:
            if reading.goal in {"research", "find", "research_and_deliver"}:
                reading.needs_web = self._needs_web(lowered, task)
            if reading.goal in {"inspect", "organize"}:
                reading.needs_files = True
        # A pure transformation of previous output needs no new retrieval.
        if reading.context is not None and reading.context.mutates_output:
            reading.needs_web = False

    def _needs_web(self, lowered: str, task: Task) -> bool:
        if task.current_information_required:
            return True
        if _matches_any(lowered, _FRESHNESS_TERMS):
            return True
        # A named web platform or an explicit "search the web" always needs web.
        if re.search(r"\b(?:youtube|google|the web|the internet|online)\b", lowered):
            return True
        if bool(task.entities.get("site")):
            return True
        # Co-occurring search verb + freshness/entity terms.
        if _matches_any(lowered, _SEARCH_VERBS) and _matches_any(lowered, _FRESHNESS_TERMS):
            return True
        return False

    # -- desired outcome ---------------------------------------------------------

    def _desired_outcome(self, task: Task, reading: IntentReading, *, destination: str) -> str:
        if reading.context is not None and reading.context.mutates_output:
            verb = reading.context.transformation or "transform"
            return f"The user's previous output has been {verb}ed as requested."
        if reading.context is not None and reading.context.replace_subject:
            subject = reading.context.new_subject or "the new subject"
            return f"The previous task has been repeated for {subject}."

        goal = reading.goal
        obj = reading.object_type
        topic = self._subject(task)
        dest = destination or task.entities.get("filename") or ""

        if goal == "research_and_deliver":
            return _outcome("Useful key points from research exist in", dest)
        if goal == "create_and_deliver":
            if reading.transformations and "summarize" in reading.transformations:
                thing = "summary"
            else:
                thing = task.entities.get("content_type") or "content"
            return _outcome(f"A useful {thing}" + (f" about {topic}" if topic else "") + " has been written into", dest)
        if goal == "research":
            return f"The user has up-to-date, sourced information" + (f" about {topic}" if topic else "")
        if goal == "find":
            return f"The user has the {obj}" + (f" about {topic}" if topic else "") + " they asked for"
        if goal in {"create", "summarize", "transform"}:
            thing = task.entities.get("content_type") or obj or "content"
            return f"A useful {thing}" + (f" about {topic}" if topic else "") + " has been produced"
        if goal == "execute":
            return f"{dest or 'The requested application'} has been opened and is ready"
        if goal == "organize":
            return f"The requested files have been organized" + (f" in {dest}" if dest else "")
        if goal == "inspect":
            return "The requested system state has been inspected and reported"
        if goal == "compare":
            return f"The requested options" + (f" about {topic}" if topic else "") + " have been compared"
        if goal == "communicate":
            return f"The requested message has been composed" + (f" in {dest}" if dest else "")
        if goal == "answer":
            return f"The user's question" + (f" about {topic}" if topic else "") + " has been answered"
        return "The user's request has been satisfied"

    #: Boilerplate prefixes that can leak into a topic and make an outcome read
    #: awkwardly ("about latest information about rtx gpus"). Stripped for the
    #: human-readable desired outcome only; the entities themselves are untouched.
    _SUBJECT_NOISE = re.compile(
        r"^(?:the\s+)?(?:latest|newest|recent|current|up[- ]to[- ]date|today'?s?|this\s+(?:week|month))?\s*"
        r"(?:information|info|news|updates?|details?|facts?)?\s*"
        r"(?:about|on|regarding|concerning|of|for)?\s*",
        re.IGNORECASE,
    )

    @classmethod
    def _subject(cls, task: Task) -> str:
        entities = task.entities
        raw = str(
            entities.get("topic")
            or entities.get("file_subject")
            or entities.get("content_type")
            or ""
        ).strip()
        cleaned = cls._SUBJECT_NOISE.sub("", raw).strip(" .,")
        return cleaned or raw

    # -- capability analysis -----------------------------------------------------

    def _required_capabilities(
        self, reading: IntentReading, task: Task, actions: list[TaskAction]
    ) -> list[str]:
        """Map the desired outcome to semantic capability requirements.

        When the interpreter already planned concrete actions, those capabilities
        are the requirements (Atlas authored them deterministically). Otherwise
        the desired outcome implies a small, standard set that the validator will
        confirm against the registry.
        """

        if actions:
            return _dedupe([a.capability for a in actions])

        required: list[str] = []
        goal = reading.goal
        if reading.needs_web:
            required.append("web.research")
        if goal in {"research", "find"} and not reading.needs_web:
            required.append("web.search")
        if goal in {"create", "summarize", "transform", "create_and_deliver"} or reading.transformations:
            required.append("content.generate")
        if reading.needs_application:
            required.append("applications.write_text")
        if reading.needs_files:
            required.append("filesystem.search")
        if goal == "inspect":
            required.append("system.info")
        return _dedupe(required)

    def validate_capabilities(self, task: Task) -> dict[str, Any]:
        """Confirm the required capabilities actually exist in the registry.

        Returns a small report; unknown requirements are reported (never executed).
        This is the deterministic "Atlas decides whether a tool exists" boundary.
        """

        required = list(task.required_capabilities)
        if self._capabilities is None:
            return {"required": required, "available": required, "missing": []}
        exists = getattr(self._capabilities, "exists", None)
        if exists is None:
            return {"required": required, "available": required, "missing": []}
        available = [name for name in required if exists(name)]
        missing = [name for name in required if not exists(name)]
        return {"required": required, "available": available, "missing": missing}

    # -- confidence / ambiguity --------------------------------------------------

    def _confidence(
        self,
        reading: IntentReading,
        task: Task,
        actions: list[TaskAction],
        best: IntentHypothesis | None,
    ) -> float:
        base = task.confidence or 0.5
        if best is not None:
            base = max(base, best.score)
        # More competing high-scoring hypotheses -> less certain.
        if len(reading.hypotheses) > 1:
            top = reading.hypotheses[0].score
            runner_up = reading.hypotheses[1].score
            if top - runner_up < 0.1:
                base -= 0.1
        if reading.context is not None and reading.context.unresolved:
            base -= 0.2
        if reading.goal == "unknown":
            base -= 0.2
        # A named destination and a concrete subject raise certainty.
        if task.entities.get("application") or task.entities.get("filename"):
            base = max(base, 0.8)
        return max(0.0, min(1.0, base))

    def _ambiguities(
        self,
        task: Task,
        reading: IntentReading,
        actions: list[TaskAction],
        destination: str,
    ) -> list[str]:
        ambiguities: list[str] = []
        context = reading.context
        # An unresolved reference the engine could not settle.
        if context is not None and context.unresolved:
            ambiguities.extend(f"unresolved reference: {ref}" for ref in context.unresolved)
        # A delivery verb with no resolvable destination.
        if (
            context is not None
            and context.target == "previous_output"
            and not destination
            and not task.entities.get("filename")
            and reading.goal in {"create_and_deliver", "research_and_deliver"}
        ):
            ambiguities.append("destination unspecified")
        # A search with no concrete target.
        topic = str(task.entities.get("topic") or task.entities.get("query") or "").strip()
        if reading.goal in {"research", "find"} and not (
            topic or task.entities.get("content_type")
        ):
            ambiguities.append("search target unspecified")
        # A search target that is only filler + the object noun ("some videos",
        # "the videos") names a kind of thing but not what about it, so it is
        # under-specified even though a topic string is present (spec section 17).
        if reading.goal in {"research", "find"} and self._is_generic_topic(topic):
            ambiguities.append("search target unspecified")
            reading.confidence = min(reading.confidence, 0.45)
        # Ambiguity only survives when context could not resolve it.
        if reading.confidence >= CLARIFY_CONFIDENCE:
            ambiguities = [a for a in ambiguities if "destination unspecified" not in a]
        return _dedupe(ambiguities)

    @staticmethod
    def _is_generic_topic(topic: str) -> bool:
        # Returns True when a topic names only a kind of thing, nothing specific.
        return _generic_topic(topic)


    # -- pass 2: critique / repair -----------------------------------------------

    def _critique(
        self,
        reading: IntentReading,
        task: Task,
        actions: list[TaskAction],
        destination: str,
    ) -> None:
        """Critically validate the proposed interpretation and repair it.

        This is the deterministic second pass: it checks the specific mistakes a
        small model makes (topic-vs-destination, retrieval-vs-action, spurious
        web, lost reference, transformation mistaken for a new task) and repairs
        the reading in place. It records a concise reason per repair; it never
        emits chain-of-thought.
        """

        topic = str(task.entities.get("topic") or "").strip().casefold()
        dest_lower = (destination or "").strip().casefold()

        # 1. Topic confused with destination: the "in Notepad" app is not a topic.
        if topic and dest_lower and topic == dest_lower:
            reading.object_type = "content"
            reading.interpretation_notes.append("repaired: destination is not the topic")

        # 2. Information retrieval mistaken for an action (or vice versa):
        #    a pure question with a delivery target must be an action; an action
        #    with no mutating capability is informational.
        mutating = {
            a.capability for a in actions
            if a.capability in {"applications.write_text", "applications.launch_named"}
            or a.capability.startswith(("filesystem.write", "filesystem.move", "filesystem.copy", "computer."))
        }
        if reading.goal == "answer" and mutating:
            reading.goal = "execute"
            reading.interpretation_notes.append("repaired: goal was action, not answer")
        if reading.goal in {"execute", "organize"} and not mutating and not actions:
            reading.interpretation_notes.append("critique: no mutating capability planned")

        # 3. Spurious web: never search the web for creative/transformative content
        #    unless the request explicitly needs current information.
        if reading.needs_web and not self._explicitly_needs_web(task):
            if reading.goal in {"create", "create_and_deliver", "summarize", "transform"}:
                reading.needs_web = False
                reading.required_capabilities = [
                    c for c in reading.required_capabilities if not c.startswith("web.")
                ]
                reading.interpretation_notes.append("repaired: creative task does not need web")

        # 4. Unnecessary application: only delivery goals open an app.
        if reading.needs_application and reading.goal in {"answer", "find", "research"}:
            if not task.entities.get("application"):
                reading.needs_application = False
                reading.interpretation_notes.append("repaired: app not required by this goal")

        # 5. A transformation of the previous output must not become a new task.
        if reading.context is not None and reading.context.mutates_output:
            if reading.goal not in {"transform", "summarize", "modify"}:
                reading.goal = "transform"
                reading.interpretation_notes.append("repaired: transformation, not a new task")
            reading.needs_web = False
            reading.needs_files = False
        # 6. A lost reference: anaphoric request with a prior task must keep it.
        if reading.context is not None and reading.context.inherit and reading.context.unresolved:
            reading.interpretation_notes.append("critique: reference could not be resolved")

        # 7. Delivery explicitly requested but goal is information-only.
        delivery_requested = bool(destination) or bool(task.entities.get("filename"))
        if delivery_requested and reading.goal in {"answer", "find", "research", "create", "summarize", "transform"}:
            if reading.goal.endswith("_and_deliver") is False:
                base = reading.goal
                if base in {"research", "find"}:
                    reading.goal = "research_and_deliver"
                elif base in {"create", "summarize", "transform"}:
                    reading.goal = "create_and_deliver"
                reading.needs_application = True
                reading.interpretation_notes.append("repaired: delivery was explicitly requested")

    @staticmethod
    def _explicitly_needs_web(task: Task) -> bool:
        lowered = (task.original_prompt or "").casefold()
        if task.current_information_required:
            return True
        if re.search(r"\b(?:youtube|google|the web|the internet|online)\b", lowered):
            return True
        if _matches_any(lowered, _FRESHNESS_TERMS):
            return True
        if task.entities.get("site"):
            return True
        # An explicit search on factual subject matter (not creative content).
        return bool(task.entities.get("topic")) and _matches_any(lowered, _SEARCH_VERBS)

    # -- correction memory -------------------------------------------------------

    def _apply_lessons(self, reading: IntentReading, task: Task, text: str) -> None:
        for lesson in self._memory.match(text):
            for key, value in lesson.fields.items():
                if hasattr(reading, key):
                    setattr(reading, key, value)
                elif key in task.entities:
                    task.entities[key] = value
            if lesson.correct_behavior:
                reading.interpretation_notes.append(
                    "lesson: " + "; ".join(lesson.correct_behavior)
                )

    def _destination_from_text(self, text: str) -> str:
        lowered = text.casefold()
        for app in _KNOWN_APPLICATIONS:
            if re.search(rf"\b{re.escape(app)}\b", lowered):
                return app
        return ""


def _entity_application(task: Task) -> str:
    value = task.entities.get("application") or task.entities.get("destination")
    return str(value).strip() if value else ""


def _starts_with(lowered: str, leads: Iterable[str]) -> bool:
    stripped = lowered.lstrip()
    return any(stripped.startswith(lead) for lead in leads)


def _outcome(prefix: str, destination: str) -> str:
    dest = destination or "the destination"
    return f"{prefix} {dest}."


def _generic_topic(topic: str) -> bool:
    # True when a topic names only a kind of thing (e.g. 'some videos').
    words = [w for w in re.findall(r"[a-z0-9']+", topic.casefold()) if w]
    if not words:
        return True
    return all(word in _FILLER_WORDS or word in _OBJECT_HINTS for word in words)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
