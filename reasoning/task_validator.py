"""Deterministic validation of structured tasks before planning.

The validator sits between the semantic interpreter and the planner:

    LLM -> Task -> VALIDATOR -> Planner -> Executor

It never trusts the model. Every action is checked against the capability
registry for existence, required parameters, parameter types, dependency
integrity, and risk. Invalid tasks are reported (not executed), and the
interpreter can fall back to a deterministic task when validation fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models_task import Task, TaskAction, is_reference
from tools.capabilities import Capability, CapabilityRegistry

#: Parameter names whose values are free-form text and therefore accept any
#: scalar. These are exempt from strict type checks.
_TEXT_PARAMS: frozenset[str] = frozenset(
    {"text", "content", "query", "topic", "application", "executable", "path",
     "source", "destination", "url", "instructions", "style", "tone", "content_type"}
)


@dataclass
class TaskValidationResult:
    """Outcome of validating a task."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Actions that survived validation (unknown/invalid actions removed).
    actions: list[TaskAction] = field(default_factory=list)
    requires_confirmation: bool = False
    risk_level: str = "read_only"
    #: True when the task is structurally fine but needs the user to clarify.
    needs_clarification: bool = False
    clarification_question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "requires_confirmation": self.requires_confirmation,
            "risk_level": self.risk_level,
            "needs_clarification": self.needs_clarification,
        }


class TaskValidator:
    """Validate a :class:`Task` against the live capability registry."""

    #: Ordered so the "highest" risk wins when summarizing a whole task.
    _RISK_ORDER = ("read_only", "low_risk", "medium_risk", "high_risk", "critical")

    def __init__(self, capabilities: CapabilityRegistry) -> None:
        self._capabilities = capabilities

    def validate(self, task: Task) -> TaskValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        actions: list[TaskAction] = []
        requires_confirmation = False
        highest_risk = "read_only"

        if not isinstance(task, Task):
            return TaskValidationResult(valid=False, errors=["task is not a Task"])

        seen_ids: set[str] = set()
        for action in task.actions:
            problem = self._validate_action(action, seen_ids=seen_ids, errors=errors, warnings=warnings)
            if problem is None:
                actions.append(action)
                capability = self._capabilities.get(action.capability)
                if capability is not None:
                    if capability.requires_confirmation or action.requires_confirmation:
                        requires_confirmation = True
                    highest_risk = self._higher_risk(highest_risk, capability.risk_level)

        # Dependency integrity: every depends_on must reference a kept action.
        kept_ids = {action.action_id for action in actions}
        for action in actions:
            for dependency in action.depends_on:
                if dependency not in kept_ids:
                    errors.append(
                        f"action '{action.action_id}' depends on unknown action '{dependency}'"
                    )

        # Reference integrity: every $name must be produced by an earlier action.
        produced: set[str] = set()
        for action in actions:
            dependency_error = self._check_references(action, produced)
            if dependency_error:
                errors.append(dependency_error)
            candidate = action.produces
            if not candidate and action.capability == "content.generate":
                candidate = "generated_text"
            if candidate:
                produced.add(str(candidate))

        # A task that requires execution with no valid actions cannot proceed,
        # unless it is explicitly an informational task handled by retrieval.
        if task.execution_required and not actions and task.task_type not in {"informational", "conversation", "unknown"}:
            errors.append("task requires execution but has no valid actions")

        if not actions:
            # Nothing executable: treat as informational unless the model asked
            # for clarification. This preserves knowledge-query workflows.
            task.task_type = task.task_type if task.task_type != "unknown" else "informational"

        valid = not errors
        needs_clarification = task.needs_clarification and not actions
        return TaskValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            actions=actions,
            requires_confirmation=requires_confirmation,
            risk_level=highest_risk,
            needs_clarification=needs_clarification,
            clarification_question=task.clarification_question,
        )

    # -- internals ---------------------------------------------------------------

    def _validate_action(
        self,
        action: TaskAction,
        *,
        seen_ids: set[str],
        errors: list[str],
        warnings: list[str],
    ) -> str | None:
        """Return None when the action is valid, else an error string."""

        if not isinstance(action, TaskAction):
            errors.append("action is not a TaskAction")
            return "invalid"
        if action.action_id in seen_ids:
            errors.append(f"duplicate action id '{action.action_id}'")
            return "duplicate"
        seen_ids.add(action.action_id)

        capability = self._capabilities.get(action.capability)
        if capability is None:
            errors.append(f"unknown capability '{action.capability}'")
            return "unknown-capability"

        missing = [
            name
            for name in capability.required_parameters
            if self._is_missing(action.parameters.get(name))
        ]
        if missing:
            errors.append(
                f"action '{action.action_id}' ({capability.name}) is missing required "
                f"parameter(s): {', '.join(missing)}"
            )
            return "missing-param"

        type_error = self._check_types(action, capability)
        if type_error:
            errors.append(type_error)
            return "bad-type"

        unknown = [
            name
            for name in action.parameters
            if capability.parameters and name not in capability.parameters
        ]
        if unknown:
            warnings.append(
                f"action '{action.action_id}' passed unadvertised parameter(s): "
                f"{', '.join(sorted(unknown))}"
            )

        if capability.risk_level in {"high_risk", "critical"}:
            warnings.append(
                f"action '{action.action_id}' targets a {capability.risk_level} capability"
            )
        return None

    def _check_types(self, action: TaskAction, capability: Capability) -> str | None:
        for name, value in action.parameters.items():
            spec = capability.parameters.get(name)
            if not isinstance(spec, dict):
                continue
            expected = str(spec.get("type", "string"))
            if is_reference(value):
                # A reference is resolved at execution time; its type is checked
                # against the producing action instead.
                continue
            if self._type_ok(expected, value, name):
                continue
            return (
                f"action '{action.action_id}' parameter '{name}' must be {expected}, "
                f"got {type(value).__name__}"
            )
        return None

    @staticmethod
    def _type_ok(expected: str, value: Any, name: str) -> bool:
        if value is None:
            return True
        if expected == "string":
            if isinstance(value, (list, tuple, dict)):
                return False
            return True
        if expected == "integer":
            if isinstance(value, bool):
                return False
            if isinstance(value, int):
                return True
            return isinstance(value, str) and value.strip().isdigit()
        if expected == "boolean":
            return isinstance(value, bool) or (
                isinstance(value, str) and value.strip().casefold() in {"true", "false"}
            )
        if expected == "array":
            return isinstance(value, (list, tuple))
        if expected == "object":
            return isinstance(value, dict)
        return True

    def _check_references(self, action: TaskAction, produced: set[str]) -> str | None:
        for value in _walk(action.parameters):
            if not is_reference(value):
                continue
            name = value[1:]
            if name not in produced:
                return (
                    f"action '{action.action_id}' references '{value}' but no earlier "
                    "action produces it"
                )
        for dependency in action.depends_on:
            if dependency not in produced and dependency not in {
                a for a in produced
            }:
                # depends_on uses action ids; checked separately below.
                pass
        return None

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        return False

    def _higher_risk(self, current: str, candidate: str) -> str:
        try:
            current_index = self._RISK_ORDER.index(current)
        except ValueError:
            current_index = 0
        try:
            candidate_index = self._RISK_ORDER.index(candidate)
        except ValueError:
            candidate_index = 0
        return self._RISK_ORDER[max(current_index, candidate_index)]


def _walk(value: Any) -> list[Any]:
    if isinstance(value, dict):
        found: list[Any] = []
        for item in value.values():
            found.extend(_walk(item))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for item in value:
            found.extend(_walk(item))
        return found
    return [value]
