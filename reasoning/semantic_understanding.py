"""Semantic understanding: natural language -> open-ended :class:`SemanticRequest`.

This is the stage the spec places **before** any capability routing:

    USER MESSAGE
      -> SEMANTIC UNDERSTANDING      <-- this module
      -> GOAL REPRESENTATION
      -> REQUIREMENTS / REASONING
      -> CAPABILITY SELECTION
      -> PLANNING -> EXECUTION -> EVIDENCE VALIDATION -> RESPONSE

It is LLM-first with a deterministic *structural* fallback:

* The model is asked to produce a :class:`SemanticRequest` reading (goal,
  subject, operation, evidence requirement, ...). Its structured output is
  recovered defensively; a malformed answer is a non-reading, never a crash.
* The deterministic fallback reasons about the request's **shape** using
  :mod:`reasoning.semantic_analysis` and the semantic topic/subject that the
  existing interpreter already extracted. It never matches an intent keyword
  list, so an unseen phrasing is understood on its structure.
* When the model is available, its reading is *reconciled* with the
  deterministic reading: the model proposes meaning, the deterministic pass
  supplies concrete lexical facts it is sure of (a resolved subject, a
  destination application).

Architectural rule enforced here: this module may not import the capability
registry, the planner, or the executor. Understanding must complete before any
capability decision is taken.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from models_task import Task
from reasoning.semantic_analysis import (
    StructuralReading,
    analyze_structure,
    extract_subject,
)
from reasoning.semantic_request import (
    AMBIGUITY_HIGH,
    AMBIGUITY_LOW,
    AMBIGUITY_MEDIUM,
    EVIDENCE_PREFERRED,
    EVIDENCE_REQUIRED,
    EVIDENCE_UNNECESSARY,
    FRESHNESS_ANY,
    FRESHNESS_CURRENT,
    FRESHNESS_STABLE,
    OP_ACT,
    OP_COMPARE,
    OP_CREATE,
    OP_EXPLAIN,
    OP_IDENTIFY,
    OP_RANK,
    OP_RETRIEVE,
    OP_TRANSFORM,
    OP_UNKNOWN,
    SemanticRequest,
)

logger = logging.getLogger(__name__)


class SemanticUnderstanding:
    """Produce an open-ended :class:`SemanticRequest` for one user message.

    ``ask`` is the optional model caller. When it is ``None`` the deterministic
    structural reading is used alone, so the layer is fully testable offline and
    never spends inference it does not need.
    """

    def __init__(self, *, ask: Callable[..., str] | None = None) -> None:
        self._ask = ask

    # -- public API -------------------------------------------------------------

    def understand(
        self,
        text: str,
        *,
        task: Task | None = None,
        history: str = "",
        prior_task: Task | None = None,
    ) -> SemanticRequest:
        """Return the semantic reading for ``text``."""

        prompt = (text or "").strip()
        if not prompt:
            return SemanticRequest(goal="", confidence=0.0, notes=["empty request"])

        structure = analyze_structure(prompt)
        deterministic = self._deterministic(prompt, task=task, structure=structure,
                                            prior_task=prior_task)

        if self._ask is None or task is None:
            return deterministic

        model_reading = self._model_reading(prompt, task=task, history=history,
                                            prior_task=prior_task)
        if model_reading is None:
            return deterministic
        return self._reconcile(model_reading, deterministic, structure=structure)

    # -- LLM path ---------------------------------------------------------------

    def _model_reading(
        self,
        text: str,
        *,
        task: Task,
        history: str,
        prior_task: Task | None,
    ) -> SemanticRequest | None:
        try:
            from reasoning.json_llm import safe_reasoning_call
            from reasoning.prompts import (
                SEMANTIC_UNDERSTANDING_SYSTEM,
                semantic_understanding_user_prompt,
            )

            data = safe_reasoning_call(
                system_prompt=SEMANTIC_UNDERSTANDING_SYSTEM,
                user_prompt=semantic_understanding_user_prompt(
                    request=text,
                    history=history,
                    draft=task.to_dict(),
                    prior_task=prior_task.to_dict() if prior_task is not None else None,
                ),
                ask=self._ask,
            )
        except Exception:  # pragma: no cover - a model failure is a non-reading
            logger.exception("Semantic understanding model call failed")
            return None
        if not isinstance(data, dict):
            return None
        reading = SemanticRequest.from_mapping(data)
        if not reading.goal and not reading.subject:
            return None
        reading.model_assisted = True
        return reading

    # -- deterministic structural path ------------------------------------------

    def _deterministic(
        self,
        text: str,
        *,
        task: Task | None,
        structure: StructuralReading,
        prior_task: Task | None,
    ) -> SemanticRequest:
        reading = SemanticRequest()
        subject = _subject_of(task, text)
        reading.subject = subject
        reading.entities = _entities_of(task)
        reading.constraints = list(getattr(task, "constraints", []) or [])
        reading.requested_output = structure.requested_output
        reading.local = structure.local
        reading.comparative = structure.comparative
        reading.criterion = structure.criterion
        reading.criterion_proxy = structure.criterion_proxy
        reading.candidate_set = structure.candidate_set

        # Priority 1: detect current/latest/final-state semantics before routing.
        lower = (text or "").casefold()
        reading.latest_request = any(marker in lower for marker in (
            "latest", "most recent", "newest", "right now", "currently", "at the moment"
        ))
        reading.most_recent_request = reading.latest_request or "most recent" in lower
        reading.current_knowledge = (
            reading.freshness_requirement == FRESHNESS_CURRENT
            or structure.intrinsically_current
            or structure.freshness == FRESHNESS_CURRENT
            or bool(reading.latest_request)
        )
        reading.ranking = bool(structure.comparative or structure.superlative)
        reading.dynamic_quantity = bool(
            reading.current_knowledge or reading.ranking or any(
                marker in lower for marker in ("subscribers", "views", "rank", "leader", "winner", "president", "version")
            )
        )
        reading.stable_knowledge = not (
            reading.current_knowledge or reading.ranking or reading.dynamic_quantity
        ) and reading.operation in {OP_EXPLAIN, OP_IDENTIFY}
        reading.subjective_criterion = bool(
            structure.criterion_subjective or any(
                term in (reading.criterion or "").casefold() for term in (
                    "best", "most famous", "most popular", "greatest", "coolest", "smartest"
                )
            )
        )
        reading.objective_criterion = bool(reading.criterion and not reading.subjective_criterion)
        reading.final_event_result = bool(
            "winner" in lower or "won" in lower or "champion" in lower or "final score" in lower
        ) or reading.operation == OP_RANK

        # Contextual grounding: the existing Intent Engine already resolved
        # references against the conversation; reuse its reading rather than
        # re-deriving anaphora with new rules.
        intent_context = _intent_context(task)
        reading.context_dependencies = list(getattr(task, "context_references", []) or [])
        reading.contextual = bool(reading.context_dependencies) or bool(
            intent_context.get("contextual")
        ) or bool(intent_context.get("has_reference"))
        reading.subject_is_referent = structure.anaphoric or reading.contextual

        # -- operation ----------------------------------------------------------
        reading.operation = _operation_for(task, structure, subject=subject)

        # -- goal (plain language) ---------------------------------------------
        reading.goal = _goal_text(reading, structure)

        # -- freshness & evidence ----------------------------------------------
        reading.freshness_requirement = _freshness_for(task, structure)
        reading.evidence_requirement, evidence_notes = _evidence_for(reading, structure, task)

        # -- ambiguity ----------------------------------------------------------
        reading.ambiguity = _ambiguity_for(reading, structure, task)

        # -- confidence ---------------------------------------------------------
        reading.confidence = _confidence_for(reading, structure, task)

        reading.notes.extend(structure.markers)
        reading.notes.extend(evidence_notes)
        if reading.subject_is_referent:
            reading.notes.append("subject resolves from conversation context")
        return reading

    # -- reconciliation ---------------------------------------------------------

    @staticmethod
    def _reconcile(
        model: SemanticRequest,
        deterministic: SemanticRequest,
        *,
        structure: StructuralReading,
    ) -> SemanticRequest:
        """Keep the model's meaning, backfill the facts the structure proves.

        A structural fact (a superlative makes the request comparative; an
        intrinsically current noun makes it current) is authoritative because it
        is read from the literal request, exactly as the interpreter trusts its
        own lexical facts over a model guess.
        """

        merged = SemanticRequest(**{**deterministic.to_dict(), **model.to_dict()})
        merged.model_assisted = True

        # Structural facts win where the model is silent or contradicts them.
        if structure.comparative:
            merged.comparative = True
            if not merged.criterion:
                merged.criterion = structure.criterion
            if not merged.criterion_proxy:
                merged.criterion_proxy = structure.criterion_proxy
            if not merged.candidate_set:
                merged.candidate_set = structure.candidate_set
            if merged.operation in {OP_EXPLAIN, OP_RETRIEVE, OP_UNKNOWN}:
                merged.operation = OP_RANK if structure.superlative else OP_COMPARE
        if structure.intrinsically_current or structure.superlative:
            merged.freshness_requirement = FRESHNESS_CURRENT
        if structure.local:
            merged.local = True

        # A concrete subject from the deterministic pass survives a model omission.
        if not merged.subject:
            merged.subject = deterministic.subject
        if not merged.entities:
            merged.entities = deterministic.entities
        if not merged.constraints:
            merged.constraints = deterministic.constraints

        # Evidence requirement is re-derived from the merged reading, so a model's
        # optimistic "unnecessary" cannot survive a proven current/ranking request.
        requirement, notes = _evidence_for(merged, structure, None)
        merged.evidence_requirement = requirement
        merged.notes = list(dict.fromkeys([*deterministic.notes, *model.notes, *notes]))
        if merged.confidence <= 0.0:
            merged.confidence = deterministic.confidence
        return merged


# ---------------------------------------------------------------------------
# Deterministic derivation helpers
# ---------------------------------------------------------------------------


def _intent_context(task: Task | None) -> dict:
    """Return the Intent Engine's resolved conversation context, if any.

    The Intent Engine already ran the reference resolver and recorded its
    ``ResolvedContext``. Reading it here reuses that resolution instead of
    re-deriving anaphora, and it is the signal that makes a follow-up such as
    "how many subscribers does he have?" grounded in the conversation.
    """

    if task is None:
        return {}
    reading = task.context.get("intent_reading")
    if not isinstance(reading, dict):
        return {}
    context = reading.get("context")
    return context if isinstance(context, dict) else {}


def _subject_of(task: Task | None, text: str) -> str:
    """Return the best available subject for the request.

    Preference order matters for correctness:

    1. A *structurally extracted* subject (the grammatical frame of the request).
       This is what makes five differently worded requests about the same thing
       converge: the interpreter's topic extractor only fires on an "about X"
       connector or a search verb, and when it *does* fire on an instruction like
       "what do you know about MrBeast" it captures the whole sentence, so it is
       not trustworthy on its own.
    2. The interpreter's resolved topic / research query, when the structural
       extraction found nothing (e.g. a creation request whose topic came from a
       "list of X" clause rather than a copula frame).

    The raw prompt is never used as the subject, because it carries instruction
    language.
    """

    structural = extract_subject(text)
    if structural:
        return structural
    if task is not None:
        entities = task.entities or {}
        for key in ("topic", "research_query", "normalized_topic", "raw_topic"):
            value = str(entities.get(key) or "").strip()
            if value and not _looks_like_instruction(value):
                return value
    return ""


#: Frame words that betray an instruction sentence mistakenly stored as a topic.
_INSTRUCTION_MARKERS: tuple[str, ...] = (
    "what do you know", "tell me", "can you", "could you", "give me",
    "show me", "i want", "search for", "look up", "find out",
)


def _looks_like_instruction(value: str) -> bool:
    lowered = value.casefold()
    return any(marker in lowered for marker in _INSTRUCTION_MARKERS)


def _entities_of(task: Task | None) -> list[Any]:
    if task is None:
        return []
    entities: list[Any] = []
    for key, value in (task.entities or {}).items():
        if key in {"topic", "research_query", "normalized_topic", "raw_topic",
                   "topic_reading", "content_type", "transform"}:
            continue
        if value in (None, "", [], {}):
            continue
        entities.append({"kind": key, "value": value})
    return entities


def _operation_for(task: Task | None, structure: StructuralReading, *, subject: str) -> str:
    """Return the requested operation, derived from structure and capabilities."""

    capabilities = {action.capability for action in (task.actions if task else [])}

    if structure.comparative:
        return OP_RANK if structure.superlative else OP_COMPARE

    # A transformation of existing content.
    transformations = list(getattr(task, "transformations", []) or [])
    if "transform" in capabilities or transformations:
        return OP_TRANSFORM

    # A delivery/creation goal: the plan mutates local state or writes an app.
    if capabilities & {
        "applications.write_text", "filesystem.write", "applications.launch_named",
        "filesystem.move", "filesystem.copy", "filesystem.create_folder",
    }:
        return OP_CREATE if "content.generate" in capabilities else OP_ACT
    if capabilities & {"filesystem.list", "filesystem.read", "filesystem.search",
                       "filesystem.search_content", "system.info"}:
        return OP_RETRIEVE
    if any(capability.startswith("computer.") for capability in capabilities):
        return OP_ACT
    if "content.generate" in capabilities:
        return OP_CREATE

    goal = str(getattr(task, "goal", "") or "")
    if goal in {"compare"}:
        return OP_COMPARE
    if goal in {"transform", "summarize", "modify"}:
        return OP_TRANSFORM
    if goal in {"research", "find", "research_and_deliver"}:
        return OP_RETRIEVE
    if goal in {"execute", "organize", "create", "create_and_deliver", "communicate"}:
        return OP_ACT
    if goal == "answer":
        return OP_EXPLAIN
    if goal == "converse":
        return OP_RETRIEVE if structure.question_form else OP_UNKNOWN

    if structure.question_form:
        return OP_EXPLAIN
    if structure.imperative_form:
        return OP_ACT
    if subject:
        return OP_IDENTIFY
    return OP_UNKNOWN


def _goal_text(reading: SemanticRequest, structure: StructuralReading) -> str:
    """Describe the user's goal in plain, open-ended language (not an enum)."""

    subject = reading.subject or "the request"
    if reading.comparative:
        direction = "ranking" if structure.superlative else "comparison"
        criterion = f" by {reading.criterion}" if reading.criterion else ""
        return f"identify the leading candidate in a {direction}{criterion} over {subject}"
    if reading.operation == OP_TRANSFORM:
        return f"transform the previously produced content about {subject}"
    if reading.operation == OP_CREATE:
        return f"create and deliver content about {subject}"
    if reading.operation == OP_ACT:
        return f"perform a computer action regarding {subject}"
    if reading.operation == OP_RETRIEVE:
        return f"obtain {subject} for the user"
    if reading.operation == OP_EXPLAIN:
        return f"explain {subject}" if reading.subject else "explain the requested subject"
    if reading.operation == OP_IDENTIFY:
        return f"identify {subject}"
    return f"satisfy the user's request about {subject}" if reading.subject else "satisfy the user's request"


def _freshness_for(task: Task | None, structure: StructuralReading) -> str:
    """Return current | stable | any from structure and the interpreter's flags."""

    # Priority 1: latest/most-recent timing should be treated as current evidence
    # requests even when they are worded as a superlative, not as a subjective
    # criterion.
    if structure.freshness == FRESHNESS_CURRENT or structure.intrinsically_current or structure.superlative:
        return FRESHNESS_CURRENT
    if task is not None and getattr(task, "current_information_required", False):
        return FRESHNESS_CURRENT
    if structure.local:
        return FRESHNESS_CURRENT
    if structure.freshness == FRESHNESS_STABLE:
        return FRESHNESS_STABLE
    return FRESHNESS_ANY


def _evidence_for(
    reading: SemanticRequest,
    structure: StructuralReading,
    task: Task | None,
) -> tuple[str, list[str]]:
    """Decide whether external evidence is required, preferred, or unnecessary.

    Semantic reasoning, not a keyword ruleset. The factors considered are the
    ones the spec names: is the information current; is the request about a real
    external entity; is it a ranking/comparison; is the answer quantitative;
    could it have changed recently; is it subjective; does answering require
    comparison; would a plausible generation be a fabrication?
    """

    notes: list[str] = []

    # Priority 1: a ranking/comparison or current/latest fact is not answerable
    # from model memory alone; the evidence gate must block it.
    if reading.comparative or reading.ranking:
        notes.append("ranking/comparison requires evidence to be trustworthy")
        return EVIDENCE_REQUIRED, notes

    if reading.latest_request or reading.most_recent_request or reading.current_knowledge:
        notes.append("the answer depends on current information")
        return EVIDENCE_REQUIRED, notes

    if reading.final_event_result and reading.freshness_requirement == FRESHNESS_CURRENT:
        notes.append("final event result requires fresh evidence")
        return EVIDENCE_REQUIRED, notes

    # 2. Information that changes over time cannot be answered from memory.
    if reading.freshness_requirement == FRESHNESS_CURRENT:
        notes.append("the answer depends on current information")
        return EVIDENCE_REQUIRED, notes

    # 3. A local-machine request is answered from observation, never the web.
    if reading.local:
        notes.append("the request is about this machine; answered from observation")
        return EVIDENCE_UNNECESSARY, notes

    # 4. A transformation/creation of *given* content needs no new evidence.
    if reading.operation in {OP_TRANSFORM, OP_CREATE} and not _needs_new_facts(task):
        notes.append("the request reshapes content Atlas already has")
        return EVIDENCE_UNNECESSARY, notes

    # 5. An action request is about doing, not about knowing.
    if reading.operation == OP_ACT:
        notes.append("the request performs an action; no external knowledge required")
        return EVIDENCE_UNNECESSARY, notes

    # 6. Stable factual knowledge the model holds reliably.
    if reading.freshness_requirement == FRESHNESS_STABLE:
        notes.append("stable factual knowledge; no external evidence necessary")
        return EVIDENCE_UNNECESSARY, notes

    # 7. An information request about a real external entity. The answer *may*
    #    depend on current/public information, so evidence materially improves
    #    reliability even though it is not strictly required.
    if reading.subject and reading.operation in {OP_EXPLAIN, OP_IDENTIFY, OP_RETRIEVE}:
        notes.append("external entity; public information may improve reliability")
        return EVIDENCE_PREFERRED, notes

    # 8. An explicit evidence demand the user made (the interpreter's own flag
    #    or an explicit source entity) always wins.
    if task is not None and (getattr(task, "needs_web", False) or task.entities.get("site")):
        notes.append("the user asked for external information")
        return EVIDENCE_REQUIRED, notes

    return EVIDENCE_UNNECESSARY, notes


def _needs_new_facts(task: Task | None) -> bool:
    """True when a create/transform goal still needs facts from outside."""

    if task is None:
        return False
    return bool(getattr(task, "needs_web", False)) or bool(
        getattr(task, "current_information_required", False)
    )


def _ambiguity_for(
    reading: SemanticRequest,
    structure: StructuralReading,
    task: Task | None,
) -> str:
    """Return low | medium | high from genuine, consequential ambiguity.

    A *subjective criterion with no objective proxy* is the canonical medium
    ambiguity: "who is the best Minecraft server" has no measurable answer, so
    Atlas must reason about it rather than invent one. An unresolved subject with
    a request that depends on it is high.
    """

    unresolved = list(getattr(task, "ambiguities", []) or []) if task else []
    if any("unresolved reference" in item for item in unresolved):
        return AMBIGUITY_HIGH
    if structure.criterion_subjective and not structure.criterion_proxy:
        return AMBIGUITY_MEDIUM
    if reading.comparative and not reading.subject and not reading.candidate_set:
        return AMBIGUITY_MEDIUM
    if unresolved:
        return AMBIGUITY_MEDIUM
    if reading.subject_is_referent and not reading.context_dependencies:
        return AMBIGUITY_HIGH
    return AMBIGUITY_LOW


def _confidence_for(
    reading: SemanticRequest,
    structure: StructuralReading,
    task: Task | None,
) -> float:
    base = float(getattr(task, "confidence", 0.0) or 0.0)
    if reading.subject:
        base = max(base, 0.7)
    if reading.comparative and reading.criterion:
        base = max(base, 0.75)
    if not reading.subject and reading.operation in {OP_EXPLAIN, OP_IDENTIFY}:
        base = min(base, 0.5)
    if reading.ambiguity == AMBIGUITY_HIGH:
        base = min(base, 0.4)
    elif reading.ambiguity == AMBIGUITY_MEDIUM:
        base = min(base, 0.7)
    return max(0.0, min(1.0, base))
