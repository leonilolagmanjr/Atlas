"""Structured task intermediate representation (the Task IR).

This module defines the canonical, composable representation of *what the user
wants*, as produced by the semantic interpreter, validated by the task validator,
and consumed by the dynamic planner.

Design rules:

* The LLM proposes meaning as a :class:`Task` made of :class:`TaskAction`
  entries. It never proposes shell text, code, or executable commands.
* Every field is validated defensively against untrusted model output: unknown
  keys are ignored, wrong types are coerced or dropped, and malformed entries
  are discarded rather than raising.
* Nothing here executes anything. Executors only ever act on validated
  capabilities resolved from the tool registry.

This module deliberately lives beside ``models.py`` (rather than inside it) so
the large, stable execution-context models stay untouched and this additive IR
can evolve without churn in unrelated dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

#: Canonical task types. Coarse and extensible; the goal/actions carry the
#: detail. "informational" preserves the existing retrieval/memory workflows.
TASK_TYPES: frozenset[str] = frozenset(
    {
        "computer_action",
        "informational",
        "search",
        "content_creation",
        "conversation",
        "multi_step",
        "unknown",
    }
)

#: Canonical risk levels mirroring tools.base.RiskLevel values.
RISK_LEVELS: tuple[str, ...] = ("read_only", "low_risk", "medium_risk", "high_risk", "critical")
REQUEST_TYPES = frozenset({"unknown", "question", "self_query", "memory_query", "action", "hybrid", "clarification"})
SOURCE_TYPES = frozenset({"self", "conversation", "memory", "knowledge", "model", "files", "web", "computer", "system"})
RESPONSE_MODES = frozenset({"unknown", "direct_answer", "grounded_answer", "web_research", "file_lookup", "system_diagnosis", "self_description", "memory_recall", "action_report", "limitation", "clarification"})
REASONING_FLAGS = (
    "requires_web", "requires_files", "requires_memory", "requires_knowledge",
    "requires_computer", "requires_system", "requires_self_introspection",
    "requires_model_knowledge", "current_information_required", "requires_verification",
)

#: A parameter reference such as ``$generated_text`` that consumes the output of
#: an earlier action instead of a literal value.
REFERENCE_PREFIX = "$"


class RiskLevel(str, Enum):
    """Risk vocabulary for a task action."""

    READ_ONLY = "read_only"
    LOW = "low_risk"
    MEDIUM = "medium_risk"
    HIGH = "high_risk"
    CRITICAL = "critical"


def is_reference(value: Any) -> bool:
    """Return True when ``value`` names an earlier action's produced output."""

    return isinstance(value, str) and value.startswith(REFERENCE_PREFIX) and len(value) > 1


@dataclass
class TaskAction:
    """One intended capability invocation within a task.

    An action describes *intent*, not execution: it names a capability and its
    semantic parameters. It is converted into an executable plan step only after
    validation confirms the capability exists and the parameters are valid.
    """

    action_id: str
    capability: str
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    #: Name under which this action's result is published for later actions,
    #: e.g. ``generated_text`` consumed as ``$generated_text``.
    produces: str | None = None
    expected_output: str | None = None
    risk_level: str = RiskLevel.READ_ONLY.value
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "capability": self.capability,
            "parameters": dict(self.parameters),
            "description": self.description,
            "depends_on": list(self.depends_on),
            "produces": self.produces,
            "expected_output": self.expected_output,
            "risk_level": self.risk_level,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class Task:
    """A validated, structured representation of a user request.

    This is the single boundary object between the LLM interpreter and Atlas's
    deterministic planning/execution layer.
    """

    goal: str = "unknown"
    task_type: str = "unknown"
    original_prompt: str = ""
    actions: list[TaskAction] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    confidence: float = 0.0
    requires_confirmation: bool = False
    execution_required: bool = True
    needs_clarification: bool = False
    clarification_question: str | None = None
    #: How the task was produced: "llm", "deterministic", or "hybrid".
    source: str = "deterministic"
    task_id: str = field(default_factory=lambda: str(uuid4()))

    # -- reasoning decision (written back onto the IR by the reasoning engine) ----
    #: question | self_query | memory_query | action | hybrid | clarification
    request_type: str = "unknown"
    #: Ordered source names the reasoning engine selected (self, knowledge, web...).
    sources: list[str] = field(default_factory=list)
    #: How the response should be produced (direct_answer, web_research, ...).
    response_mode: str = "unknown"
    #: Human-readable explanation of the routing decision (inspectable).
    reason: str = ""
    requires_web: bool = False
    requires_files: bool = False
    requires_memory: bool = False
    requires_knowledge: bool = False
    requires_computer: bool = False
    requires_system: bool = False
    requires_self_introspection: bool = False
    requires_model_knowledge: bool = False
    #: True when the answer depends on information that changes over time.
    current_information_required: bool = False
    #: True when the executed result should be verified where practical.
    requires_verification: bool = True

    def apply_decision(self, decision: Any) -> "Task":
        """Write a reasoning decision's fields back onto this task.

        The reasoning decision extends the IR instead of competing with it, so
        the planner and executor consume a single object. Attribute access is
        used deliberately: ``models_task`` stays independent of the reasoning
        package (no import cycle).
        """

        def value(name: str, default: Any) -> Any:
            found = getattr(decision, name, default)
            return found.value if hasattr(found, "value") else found

        self.request_type = _enum(value("request_type", self.request_type), REQUEST_TYPES, self.request_type)
        self.response_mode = _enum(value("response_mode", self.response_mode), RESPONSE_MODES, self.response_mode)
        self.reason = _text(value("reason", self.reason)) or ""
        self.sources = _sources(value("sources", self.sources))
        for name in REASONING_FLAGS:
            setattr(self, name, _boolean(value(name, getattr(self, name)), getattr(self, name)))
        self.goal = str(value("goal", self.goal) or self.goal)
        return self

    def with_actions(self, actions: list[TaskAction]) -> "Task":
        self.actions = actions
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "goal": self.goal,
            "original_prompt": self.original_prompt,
            "actions": [action.to_dict() for action in self.actions],
            "entities": dict(self.entities),
            "constraints": list(self.constraints),
            "context": dict(self.context),
            "dependencies": list(self.dependencies),
            "confidence": self.confidence,
            "requires_confirmation": self.requires_confirmation,
            "execution_required": self.execution_required,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "source": self.source,
            "request_type": self.request_type,
            "sources": list(self.sources),
            "response_mode": self.response_mode,
            "reason": self.reason,
            "requires_web": self.requires_web,
            "requires_files": self.requires_files,
            "requires_memory": self.requires_memory,
            "requires_knowledge": self.requires_knowledge,
            "requires_computer": self.requires_computer,
            "requires_system": self.requires_system,
            "requires_self_introspection": self.requires_self_introspection,
            "requires_model_knowledge": self.requires_model_knowledge,
            "current_information_required": self.current_information_required,
            "requires_verification": self.requires_verification,
        }

    def references(self) -> set[str]:
        """Return every ``$name`` reference used by this task's actions."""

        found: set[str] = set()
        for action in self.actions:
            for value in _walk_values(action.parameters):
                if is_reference(value):
                    found.add(value[len(REFERENCE_PREFIX):])
        return found

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, prompt: str, source: str = "llm") -> "Task":
        """Build a task from untrusted model output without ever raising.

        Unknown keys are ignored, wrong types are coerced or dropped, ids are
        generated when absent, and malformed actions are skipped.
        """

        if not isinstance(data, dict):
            return cls(original_prompt=prompt, source="deterministic", confidence=0.0)

        raw_actions = data.get("actions")
        actions: list[TaskAction] = []
        if isinstance(raw_actions, list):
            for index, raw in enumerate(raw_actions, start=1):
                action = _action_from_mapping(raw, index=index)
                if action is not None:
                    actions.append(action)

        return cls(
            goal=_text(data.get("goal")) or "unknown",
            task_type=_enum(data.get("task_type"), TASK_TYPES, "unknown"),
            original_prompt=prompt,
            actions=actions,
            entities=_dict(data.get("entities")),
            constraints=_string_list(data.get("constraints")),
            context=_dict(data.get("context")),
            dependencies=_string_list(data.get("dependencies")),
            confidence=_confidence(data.get("confidence")),
            requires_confirmation=_boolean(data.get("requires_confirmation")),
            execution_required=_boolean(data.get("execution_required"), True),
            needs_clarification=_boolean(data.get("needs_clarification")),
            clarification_question=_text(data.get("clarification_question")),
            source=source,
            request_type=_enum(data.get("request_type"), REQUEST_TYPES, "unknown"),
            sources=_sources(data.get("sources")),
            response_mode=_enum(data.get("response_mode"), RESPONSE_MODES, "unknown"),
            reason=_text(data.get("reason")) or "",
            **{name: _boolean(data.get(name), name == "requires_verification") for name in REASONING_FLAGS},
        )


def _action_from_mapping(raw: Any, *, index: int) -> TaskAction | None:
    if not isinstance(raw, dict):
        return None
    capability = _text(raw.get("capability"))
    if not capability:
        return None
    parameters = raw.get("parameters")
    if not isinstance(parameters, dict):
        # Some models emit "arguments" (the planner schema) instead of
        # "parameters"; accept it rather than dropping the action.
        parameters = raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {}
    return TaskAction(
        action_id=_text(raw.get("action_id")) or f"action_{index}",
        capability=capability,
        parameters=_clean_parameters(parameters),
        description=_text(raw.get("description")) or "",
        depends_on=_string_list(raw.get("depends_on")),
        produces=_text(raw.get("produces")),
        expected_output=_text(raw.get("expected_output")),
        risk_level=_enum(raw.get("risk_level"), set(RISK_LEVELS), RiskLevel.READ_ONLY.value),
        requires_confirmation=_boolean(raw.get("requires_confirmation")),
    )


def _clean_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in parameters.items():
        name = str(key).strip()
        if not name:
            continue
        cleaned[name] = _clean_value(value)
    return cleaned


def _clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_value(item) for item in value]
    return value


def _walk_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        found: list[Any] = []
        for item in value.values():
            found.extend(_walk_values(item))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for item in value:
            found.extend(_walk_values(item))
        return found
    return [value]


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        if text and text.casefold() not in {"null", "none", "n/a", "unknown", ""}:
            return text
    return None


def _boolean(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _sources(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    sources = [_enum(item.value if isinstance(item, Enum) else item, SOURCE_TYPES, "") for item in value]
    return list(dict.fromkeys(source for source in sources if source))


def _enum(value: Any, allowed: set[str] | frozenset[str], default: str) -> str:
    text = _text(value)
    if text is None:
        return default
    lowered = text.casefold()
    return lowered if lowered in allowed else default


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
