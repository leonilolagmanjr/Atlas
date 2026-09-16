"""Dynamic planning from a structured :class:`Task`.

The task planner consumes a *validated* task and produces the smallest execution
plan that accomplishes it. It is deterministic glue: ordering, dependency
resolution, variable references, and confirmation flags are all computed here,
not asked of the model. The LLM's job ended at interpretation.

Rules:

* Actions are ordered so every dependency runs first (stable topological order).
* ``depends_on`` and ``$reference`` parameters are honoured literally.
* Confirmation flags are re-derived from the capability registry, so the model
  can never downgrade a consequential capability.
* Informational tasks (no actions) delegate to the existing retrieval plan,
  preserving all knowledge/memory workflows.
"""

from __future__ import annotations

from typing import Callable

from models import ExecutionPlan, ExecutionStep, PlannerDecision
from models_task import Task, TaskAction
from tools.capabilities import CapabilityRegistry


class TaskPlanner:
    """Turn a validated task into a deterministic execution plan."""

    def __init__(self, *, capabilities: CapabilityRegistry | None = None) -> None:
        self._capabilities = capabilities or CapabilityRegistry(None)

    def create_plan(
        self,
        task: Task,
        *,
        user_question: str,
        legacy_planner: Callable[..., PlannerDecision] | None = None,
    ) -> PlannerDecision:
        if not task.actions:
            # Informational / conversational: reuse the existing retrieval plan.
            if legacy_planner is not None:
                decision = legacy_planner(user_question, intent="UNKNOWN")
                decision.metadata["planned_by"] = "task_informational"
                return decision
            return PlannerDecision(
                plan=ExecutionPlan(user_question=user_question, steps=[]),
                strategy="task_informational",
                confidence=task.confidence,
            )

        ordered = self._order(task.actions)
        steps: list[ExecutionStep] = []
        for index, action in enumerate(ordered, start=1):
            capability = self._capabilities.get(action.capability)
            parameters = self._prepare_parameters(action)
            metadata = {
                "tool": action.capability,
                "parameters": parameters,
                "capability": capability.description if capability else action.capability,
                "planned_by": "task",
            }
            if action.produces:
                metadata["produces"] = action.produces
            if capability is not None:
                metadata["risk_level"] = capability.risk_level
            steps.append(
                ExecutionStep(
                    id=action.action_id or f"step_{index}",
                    name=action.description or action.capability,
                    action="invoke_tool",
                    description=action.description or "",
                    metadata=metadata,
                )
            )

        strategy = self._strategy(task)
        return PlannerDecision(
            plan=ExecutionPlan(user_question=user_question, steps=steps),
            strategy=strategy,
            confidence=task.confidence,
            metadata={
                "step_count": len(steps),
                "task_type": task.task_type,
                "goal": task.goal,
            },
        )

    # -- helpers -----------------------------------------------------------------

    def _order(self, actions: list[TaskAction]) -> list[TaskAction]:
        """Stable topological order honouring ``depends_on`` and references."""

        by_id = {action.action_id: action for action in actions}
        produced_by = {
            str(action.produces): action.action_id
            for action in actions
            if action.produces
        }
        # content.generate always publishes "generated_text" by convention.
        for action in actions:
            if action.capability == "content.generate":
                produced_by.setdefault("generated_text", action.action_id)

        ordered: list[TaskAction] = []
        placed: set[str] = set()

        def visit(action: TaskAction, stack: set[str]) -> None:
            if action.action_id in placed:
                return
            if action.action_id in stack:
                # A dependency cycle: fall back to declaration order for this
                # action rather than looping forever.
                return
            stack.add(action.action_id)
            # Explicit dependencies first, then implicit reference dependencies.
            for producer in self._implicit_producers(action, produced_by):
                producer_action = by_id.get(producer)
                if producer_action is not None and producer_action.action_id != action.action_id:
                    visit(producer_action, stack)
            for dependency in action.depends_on:
                dependency_action = by_id.get(dependency)
                if dependency_action is not None:
                    visit(dependency_action, stack)
            stack.discard(action.action_id)
            if action.action_id not in placed:
                placed.add(action.action_id)
                ordered.append(action)

        for action in actions:
            visit(action, set())
        return ordered

    def _implicit_producers(self, action: TaskAction, produced_by: dict[str, str]) -> list[str]:
        producers: list[str] = []
        for value in _walk(action.parameters):
            if isinstance(value, str) and value.startswith("$") and len(value) > 1:
                producer = produced_by.get(value[1:])
                if producer:
                    producers.append(producer)
        return producers

    def _prepare_parameters(self, action: TaskAction) -> dict:
        """Drop empty optional params; keep references and literals intact."""

        prepared: dict = {}
        for key, value in action.parameters.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            prepared[key] = value
        return prepared

    def _strategy(self, task: Task) -> str:
        if len(task.actions) > 1:
            return "task_multi_step"
        return f"task_{task.goal or task.task_type}"


def _walk(value):
    if isinstance(value, dict):
        found = []
        for item in value.values():
            found.extend(_walk(item))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for item in value:
            found.extend(_walk(item))
        return found
    return [value]
