"""Atlas Brain orchestration layer."""

from __future__ import annotations

import logging
import time

from config import DEBUG_PIPELINE, ENABLE_LLM_INTERPRETATION
from executor import Executor, UNKNOWN_RESPONSE
from llm import ask

from models import ExecutionContext, PlanStatus, StructuredIntent, TaskStatus
from planner import Planner
from reasoning.diagnostics import PipelineTrace
from reasoning.interpreter import SemanticInterpreter, classify_category
from reasoning.recovery import RecoveryManager
from tools.capabilities import render_capability_catalog
from vector_store import VectorStore

from memory.memory_manager import MemoryManager
from tools.router import ToolRouter

logger = logging.getLogger(__name__)


class Brain:
    """Create request context, plan execution, and return the final response."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        system_prompt: str,
        retrieval_template: str,
        planner: Planner | None = None,
        executor: Executor | None = None,
        memory_manager: MemoryManager | None = None,
        tool_router: ToolRouter | None = None,
        interpreter: SemanticInterpreter | None = None,
        recovery: RecoveryManager | None = None,
        llm_ask: object | None = None,
    ) -> None:
        self._memory_manager = memory_manager
        self._ask = llm_ask or ask
        self._planner = planner or Planner(registry=tool_router, ask=self._ask)
        self._pending_contexts: dict[str, ExecutionContext] = {}
        self._last_context: ExecutionContext | None = None
        # Conversation-scoped task state for follow-ups ("make it about cars").
        self._active_intent: StructuredIntent | None = None
        registry = getattr(tool_router, "_registry", None)
        catalog = render_capability_catalog(registry)
        self._interpreter = interpreter or SemanticInterpreter(
            ask=self._ask if ENABLE_LLM_INTERPRETATION else None,
            capability_catalog=catalog,
        )
        self._recovery = recovery or RecoveryManager(ask=self._ask)
        self._executor = executor or Executor(
            vector_store=vector_store,
            system_prompt=system_prompt,
            retrieval_template=retrieval_template,
            memory_manager=memory_manager,
            tool_router=tool_router,
            recovery=self._recovery,
        )


    def process(self, user_input: str) -> str:
        """Process a user request through Planner and Executor."""

        started_at = time.perf_counter()
        context = ExecutionContext(user_input=user_input, normalized_input=user_input.strip())
        self._last_context = context
        trace = PipelineTrace(user_input=user_input)

        logger.info("Brain received request")

        try:
            # 1. Semantic interpretation (LLM-first, deterministic fallback).
            history = ""
            if self._memory_manager is not None:
                history = self._memory_manager.build_conversation_history_for_prompt()
            structured = self._interpreter.interpret(
                context.normalized_input or user_input,
                context=self._active_intent,
                history=history,
            )
            context.structured_intent = structured
            context.intent_category = classify_category(structured, user_input)
            context.metadata["structured_intent"] = structured.to_dict()
            trace.record_intent(structured)

            # Ambiguous request: ask instead of hallucinating a target.
            if structured.needs_clarification and structured.clarification_question:
                context.final_response = structured.clarification_question
                context.status = TaskStatus.UNCERTAIN
                self._active_intent = structured
                trace.record_response(context.final_response)
                return self._complete(context, started_at, trace)

            # 2. Deterministic planning from the structured intent.
            planner_decision = self._planner.create_plan(
                context.normalized_input or user_input,
                intent=context.intent,
                structured=structured,
            )
            # For genuinely compositional intents the rule table cannot express,
            # fall back to a catalog-constrained LLM plan.
            if planner_decision.strategy in {"deterministic_unknown", "deterministic_content_generation"}:
                llm_plan = self._planner.plan_with_llm(
                    context.normalized_input or user_input, structured, history=history
                )
                if llm_plan is not None:
                    planner_decision = llm_plan
            context.intent = structured.intent
            context.execution_plan = planner_decision.plan
            context.metadata["planner_decision"] = planner_decision
            trace.record_plan(planner_decision)

            # 3. Validate the plan before executing anything.
            validation = _validate_plan(planner_decision.plan)
            context.metadata["plan_validation"] = validation
            if not validation["valid"]:
                context.final_response = (
                    "I could not build a safe plan for that request: "
                    + "; ".join(validation["errors"])
                )
                context.status = TaskStatus.FAILED
                trace.record_response(context.final_response)
                return self._complete(context, started_at, trace)

            # 4. Deterministic execution (with bounded recovery inside).
            self._executor.execute(planner_decision.plan, context)
            if context.status == TaskStatus.WAITING_FOR_CONFIRMATION:
                self._pending_contexts[context.task_id] = context
            if context.status == TaskStatus.COMPLETED:
                # Only remember a completed task as reusable context.
                self._active_intent = structured
            trace.record_execution(context)
            return self._complete(context, started_at, trace)

        except Exception:
            logger.exception("Brain execution failed")
            context.final_response = UNKNOWN_RESPONSE
            return self._complete(context, started_at, trace)

    def _complete(
        self,
        context: ExecutionContext,
        started_at: float,
        trace: PipelineTrace | None = None,
    ) -> str:
        context.execution_time = time.perf_counter() - started_at
        if context.execution_plan is not None:
            context.execution_plan.metadata["brain_execution_time"] = context.execution_time
        logger.info(
            "Brain completed request: execution_time=%.4fs selected_tool=%s plan_status=%s",
            context.execution_time,
            context.selected_tool,
            context.execution_plan.status.value if context.execution_plan is not None else "none",
        )
        if trace is not None:
            trace.record_response(context.final_response)
            if DEBUG_PIPELINE:
                trace.log()
        return context.final_response or UNKNOWN_RESPONSE

    @property
    def last_context(self) -> ExecutionContext | None:
        """Expose the latest task snapshot to an API adapter."""

        return self._last_context

    def approve_pending(self, task_id: str | None = None) -> str:
        """Approve and resume a confirmation-paused plan."""

        context = self._pending_context(task_id)
        if context is None or context.execution_plan is None:
            return "There is no pending action to approve."
        tool_names = {
            str(step.metadata.get("tool"))
            for step in context.execution_plan.steps
            if step.action == "invoke_tool" and step.metadata.get("tool")
        }
        context.metadata["approved_tools"] = tool_names
        self._executor.execute(context.execution_plan, context)
        if context.status != TaskStatus.WAITING_FOR_CONFIRMATION:
            self._pending_contexts.pop(context.task_id, None)
        return context.final_response or "The pending action completed."

    def deny_pending(self, task_id: str | None = None) -> str:
        """Cancel a confirmation-paused plan."""

        context = self._pending_context(task_id)
        if context is None:
            return "There is no pending action to deny."
        context.status = TaskStatus.CANCELLED
        if context.execution_plan is not None:
            context.execution_plan.status = PlanStatus.SKIPPED
        context.final_response = "Action cancelled."
        self._pending_contexts.pop(context.task_id, None)
        return context.final_response

    def _pending_context(self, task_id: str | None) -> ExecutionContext | None:
        if task_id:
            return self._pending_contexts.get(task_id)
        if len(self._pending_contexts) == 1:
            return next(iter(self._pending_contexts.values()))
        return None
def _validate_plan(plan) -> dict:
    """Validate a plan's shape before execution.

    Checks: every tool step names a known capability, has a dict parameter
    block, and every non-tool step uses a registered action. Malformed plans
    are rejected here rather than crashing mid-execution.
    """

    from tools.capabilities import PLANNABLE_CAPABILITIES

    errors: list[str] = []
    known_actions = {
        "retrieve_knowledge",
        "generate_response",
        "merge_evidence",
        "invoke_tool",
        "finalize_content",
    }
    for step in plan.steps:
        if step.action not in known_actions:
            errors.append(f"step '{step.id}' uses unknown action '{step.action}'")
            continue
        if step.action != "invoke_tool":
            continue
        tool = str(step.metadata.get("tool") or "").strip()
        if not tool:
            errors.append(f"step '{step.id}' is missing a tool name")
            continue
        # Inspection/powershell tools are valid legacy capabilities even though
        # they are not offered to the LLM for free-text planning.
        if tool not in PLANNABLE_CAPABILITIES and not tool.startswith(("powershell.", "processes.")):
            errors.append(f"step '{step.id}' references unavailable capability '{tool}'")
        if not isinstance(step.metadata.get("parameters", {}), dict):
            errors.append(f"step '{step.id}' parameters must be a mapping")
    return {"valid": not errors, "errors": errors}
