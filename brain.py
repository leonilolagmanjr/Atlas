"""Atlas Brain orchestration layer."""

from __future__ import annotations

import logging
import time

from functools import partial

from config import (
    DATABASE_FOLDER,
    COLLECTION_NAME,
    DEBUG_PIPELINE,
    EMBEDDING_MODEL_NAME,
    ENABLE_GENERAL_QUESTION_FALLBACK,
    ENABLE_LLM_INTERPRETATION,
    ENABLE_REASONING_ENGINE,
    EXECUTION_MODE,
    KNOWLEDGE_FOLDER,
    OLLAMA_MODEL,
)
from executor import Executor, UNKNOWN_RESPONSE
from knowledge_search import retrieve as retrieve_knowledge
from llm import ask
from models import ExecutionContext, PlanStatus, StructuredIntent, TaskStatus
from models_task import Task
from planner import Planner
from reasoning.diagnostics import PipelineTrace
from reasoning.answer_generator import AnswerGenerator
from reasoning.interpreter import SemanticInterpreter, classify_category
from reasoning.reasoning_engine import ReasoningEngine
from reasoning.recovery import RecoveryManager
from reasoning.self_introspection import SelfIntrospection
from reasoning.task_interpreter import SemanticTaskInterpreter
from reasoning.task_planner import TaskPlanner
from reasoning.task_validator import TaskValidator
from tools.capabilities import CapabilityRegistry, render_capability_catalog
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
        reasoning_engine: ReasoningEngine | None = None,
    ) -> None:
        self._memory_manager = memory_manager
        self._ask = llm_ask or ask
        self._planner = planner or Planner(registry=tool_router, ask=self._ask)
        self._pending_contexts: dict[str, ExecutionContext] = {}
        self._last_context: ExecutionContext | None = None
        # Conversation-scoped task state for follow-ups ("make it about cars").
        self._active_intent: StructuredIntent | None = None
        self._active_task: Task | None = None
        registry = getattr(tool_router, "_registry", None)
        catalog = render_capability_catalog(registry)
        # Capability catalog is the single source of truth shared by the
        # interpreter, validator, and task planner. It is built once at startup
        # and cached, so no per-request registry rendering happens.
        self._capabilities = CapabilityRegistry(registry)
        self._task_interpreter = SemanticTaskInterpreter(
            ask=self._ask if ENABLE_LLM_INTERPRETATION else None,
            capabilities=self._capabilities,
            capability_catalog=catalog,
        )
        self._task_validator = TaskValidator(self._capabilities)
        self._task_planner = TaskPlanner(capabilities=self._capabilities)
        # Legacy interpretation is retained as a deterministic fallback signal.
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
        self._self_introspection = SelfIntrospection(
            self._capabilities,
            model_name=OLLAMA_MODEL,
            knowledge_folder=KNOWLEDGE_FOLDER,
            database_folder=DATABASE_FOLDER,
            collection_name=COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL_NAME,
            execution_mode=EXECUTION_MODE,
        )
        self._answer_generator = AnswerGenerator(ask=self._ask)
        self._reasoning_engine = reasoning_engine or ReasoningEngine(
            self_introspection=self._self_introspection,
            answer_generator=self._answer_generator,
            tool_runner=(
                tool_router.execute
                if tool_router is not None
                else (lambda *_args, **_kwargs: None)
            ),
            capabilities=self._capabilities,
            retrieve=partial(retrieve_knowledge, vector_store=vector_store),
        )


    def process(self, user_input: str) -> str:
        """Process a user request through Planner and Executor."""

        started_at = time.perf_counter()
        context = ExecutionContext(user_input=user_input, normalized_input=user_input.strip())
        self._last_context = context
        trace = PipelineTrace(user_input=user_input)

        logger.info("Brain received request")

        try:
            history = ""
            if self._memory_manager is not None:
                history = self._memory_manager.build_conversation_history_for_prompt()

            # 1. Semantic interpretation: natural language -> structured Task.
            task = self._task_interpreter.interpret(
                context.normalized_input or user_input,
                context=self._active_task,
                history=history,
            )
            # 2. Deterministic validation against the capability registry.
            task_validation = self._task_validator.validate(task)
            context.metadata["task"] = task.to_dict()
            context.metadata["task_validation"] = task_validation.to_dict()
            trace.record_task(task.to_dict())
            trace.record_validation(task_validation.to_dict())

            # Keep a legacy StructuredIntent view so knowledge/compare/summarize
            # workflows and diagnostics stay intact without a second interpreter.
            structured = self._structured_view(task)
            context.structured_intent = structured
            context.intent_category = classify_category(structured, user_input)
            context.metadata["structured_intent"] = structured.to_dict()
            trace.record_intent(structured)
            general_fallback_disabled = (
                not ENABLE_GENERAL_QUESTION_FALLBACK
                and task.request_type == "question"
                and not task.actions
                and set(task.sources) <= {"model", "knowledge"}
                and not task.current_information_required
            )
            if ENABLE_REASONING_ENGINE and task_validation.valid and not general_fallback_disabled:
                answer = self._reasoning_engine.handle_request(
                    question=context.normalized_input or user_input,
                    task=task,
                    prior_task=self._active_task,
                    history=history,
                )
                context.metadata["task"] = task.to_dict()
                context.metadata["reasoning"] = task.context.get("reasoning", [])
                context.metadata["reasoning_summary"] = task.context.get("reasoning_summary", "")
                trace.record_task(task.to_dict())
                if answer is not None:
                    context.metadata["reasoning"] = (answer.metadata or {}).get(
                        "reasoning", []
                    )
                    context.metadata["reasoning_summary"] = (answer.metadata or {}).get(
                        "reasoning_summary", ""
                    )
                    context.metadata["reasoning_answer"] = answer.to_dict()
                    # Preserve the tool the reasoning engine actually ran so the
                    # execution snapshot stays observable on this early-answer path.
                    selected_tool = (answer.metadata or {}).get("selected_tool")
                    if selected_tool:
                        context.selected_tool = selected_tool
                    context.final_response = answer.text
                    context.status = TaskStatus.UNCERTAIN if task.needs_clarification or task.response_mode == "clarification" else TaskStatus.COMPLETED
                    if self._memory_manager is not None:
                        self._memory_manager.append_message(role="user", content=context.user_input)
                        if context.final_response:
                            self._memory_manager.append_message(role="assistant", content=context.final_response)
                    self._active_task = task
                    self._active_intent = structured
                    trace.record_response(context.final_response)
                    return self._complete(context, started_at, trace)

            # Ambiguous / clarification: ask instead of hallucinating a target.
            if task_validation.needs_clarification or (
                task.needs_clarification and task.clarification_question
            ):
                question = (
                    task_validation.clarification_question
                    or task.clarification_question
                    or "Could you clarify what you'd like Atlas to do?"
                )
                context.final_response = question
                context.status = TaskStatus.UNCERTAIN
                self._active_task = task
                self._active_intent = structured
                trace.record_response(context.final_response)
                return self._complete(context, started_at, trace)

            # 3. Plan deterministically from the validated task.
            task.actions = task_validation.actions
            interruption = self._interruption_message(task, task_validation)
            if interruption is not None:
                context.final_response = interruption
                context.status = TaskStatus.FAILED
                trace.record_response(context.final_response)
                return self._complete(context, started_at, trace)

            planner_decision = self._task_planner.create_plan(
                task,
                user_question=context.normalized_input or user_input,
                legacy_planner=self._planner.create_plan,
            )
            context.intent = structured.intent
            context.execution_plan = planner_decision.plan
            context.metadata["planner_decision"] = planner_decision
            context.metadata["task_requires_confirmation"] = task_validation.requires_confirmation
            trace.record_plan(planner_decision)

            # 4. Validate the concrete plan shape before executing anything.
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

            # Store task object in context for executor to access
            context.metadata["task_object"] = task

            # 5. Deterministic execution (with verification + recovery inside).
            self._executor.execute(planner_decision.plan, context)
            if context.status == TaskStatus.WAITING_FOR_CONFIRMATION:
                self._pending_contexts[context.task_id] = context
            if context.status == TaskStatus.COMPLETED:
                # Only remember a completed task as reusable context.
                self._active_task = task
                self._active_intent = structured
            trace.record_execution(context)
            return self._complete(context, started_at, trace)

        except Exception as exc:
            logger.exception("Brain execution failed")
            context.status = TaskStatus.FAILED
            context.errors.append(f"Brain execution failed ({type(exc).__name__})")
            if context.execution_plan is not None:
                context.execution_plan.status = PlanStatus.FAILED
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
    def _structured_view(self, task: Task) -> StructuredIntent:
        """Adapt a Task into the legacy StructuredIntent for shared consumers."""

        from reasoning.task_interpreter import task_to_intent_shim
        return StructuredIntent.from_mapping(task_to_intent_shim(task), source=task.source)

    @staticmethod
    def _interruption_message(task: Task, validation) -> str | None:
        """Return a user-facing failure when a validated task cannot proceed."""

        if not validation.valid:
            return (
                "I understood the request but could not build a valid task: "
                + "; ".join(validation.errors)
            )
        return None
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
