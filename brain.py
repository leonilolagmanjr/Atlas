"""Atlas Brain orchestration layer."""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Callable

from config import (
    CHROMA_FOLDER,
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
from computer.runtime import build_computer_observer
from experience.service import ExperienceService
from knowledge_search import retrieve as retrieve_knowledge
from llm import ask
from models import ExecutionContext, PlanStatus, StructuredIntent, TaskStatus
from models_task import Task, TaskAction
from planner import Planner
from reasoning.diagnostics import PipelineTrace
from reasoning.answer_generator import AnswerGenerator
from reasoning.intent_engine import IntentEngine
from reasoning.interpreter import classify_category
from reasoning.reasoning_engine import ReasoningEngine
from reasoning.recovery import RecoveryManager
from reasoning.semantic_reasoning import SemanticDecision, SemanticReasoning
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
        recovery: RecoveryManager | None = None,
        llm_ask: object | None = None,
        reasoning_engine: ReasoningEngine | None = None,
        experience_service: ExperienceService | None = None,
    ) -> None:
        self._memory_manager = memory_manager
        self._ask = llm_ask or ask
        self._planner = planner or Planner(registry=tool_router, ask=self._ask)
        self._pending_contexts: dict[str, ExecutionContext] = {}
        self._last_context: ExecutionContext | None = None
        #: Cooperative cancellation check for the in-flight request, supplied by
        #: the Conversation Runtime so a stopped turn really stops.
        self._cancel: Callable[[], bool] | None = None
        # Conversation-scoped task state for follow-ups ("make it about cars").
        self._active_intent: StructuredIntent | None = None
        self._active_task: Task | None = None
        #: The most recent semantic decision (understanding + evidence +
        #: capability selection), exposed for diagnostics and the answerability
        #: gate. Never used to execute anything directly.
        self._semantic_decision: SemanticDecision | None = None
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
        # Intent Engine 2.0: context-aware goal/intent understanding layer. It is
        # deterministic and runs after interpretation, before validation, so the
        # planner consumes a semantically complete Task IR.
        self._intent_engine = IntentEngine(capabilities=self._capabilities)
        # Semantic reasoning layer: open-ended understanding -> goal -> evidence
        # requirements -> capability selection. It runs after interpretation (it
        # needs the interpreter's resolved subject and references) and *before*
        # capability routing/validation, so the whole pipeline consumes one
        # semantic reading instead of re-deriving an intent from keywords. It
        # never plans or executes; the deterministic validator and executor keep
        # their authority over permissions, schemas, and state transitions.
        self._semantic_reasoning = SemanticReasoning(
            ask=self._ask if ENABLE_LLM_INTERPRETATION else None,
            capabilities=self._capabilities,
        )
        self._task_validator = TaskValidator(self._capabilities)
        self._task_planner = TaskPlanner(capabilities=self._capabilities)
        self._recovery = recovery or RecoveryManager(ask=self._ask)
        # Experience memory: the persistent feedback -> experience -> retrieval
        # loop. It is a separate memory from factual RAG and never overrides an
        # explicit instruction; it only supplies supporting context to planning.
        self._experience = experience_service or ExperienceService()
        self._last_experience_context = None
        self._executor = executor or Executor(
            vector_store=vector_store,
            system_prompt=system_prompt,
            retrieval_template=retrieval_template,
            memory_manager=memory_manager,
            tool_router=tool_router,
            recovery=self._recovery,
            observer=build_computer_observer(),
        )
        self._self_introspection = SelfIntrospection(
            self._capabilities,
            model_name=OLLAMA_MODEL,
            knowledge_folder=KNOWLEDGE_FOLDER,
            database_folder=CHROMA_FOLDER,
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


    def process(
        self,
        user_input: str,
        cancel: Callable[[], bool] | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        conversation_context: str = "",
    ) -> str:
        """Process a user request through Planner and Executor.

        ``cancel`` is an optional cooperative cancellation check. It is
        consulted after interpretation, after planning, and by the executor
        before every step, so a request that is stopped does not go on planning
        and running the rest of its plan. It is cooperative: a call already in
        flight (a model, file, or network call) is not preempted mid-call.

        ``on_event`` receives ``(event_type, payload)`` for real execution
        activity as it happens, so a streaming client does not have to wait for
        the whole task to learn what Atlas is doing.

        ``conversation_context`` is the bounded, relevance-selected conversation
        grounding assembled by the Conversation Runtime (recent turns, summary,
        recalled earlier turns, task/tool state). When supplied it *informs intent
        interpretation before execution is chosen*: the interpreter, the Intent
        Engine, the router and the reasoning engine all read the same grounded
        history, so a follow-up like "What about performance?" is understood in
        the context of what was just discussed instead of being read from zero. It
        is optional, so an injected legacy Brain double keeps working.
        """

        started_at = time.perf_counter()
        context = ExecutionContext(user_input=user_input, normalized_input=user_input.strip())
        self._last_context = context
        trace = PipelineTrace(user_input=user_input)
        self._cancel = cancel

        logger.info("Brain received request")

        try:
            if self._cancelled(cancel):
                return self._cancelled_response(context, started_at, trace)
            history = ""
            if self._memory_manager is not None:
                history = self._memory_manager.build_conversation_history_for_prompt()
            # Prefer the grounded conversation bundle when the runtime supplied
            # one: it is relevance-selected and bounded, whereas the flat history
            # is only the recent window. It is the *same* history the interpreter,
            # Intent Engine, router and reasoning engine all read here.
            grounded = (conversation_context or "").strip()
            if grounded:
                history = grounded if not history.strip() else (
                    f"{history}\n\n{grounded}"
                )
                context.metadata["conversation_context_used"] = True

            # 1. Semantic interpretation: natural language -> structured Task.
            task = self._task_interpreter.interpret(
                context.normalized_input or user_input,
                context=self._active_task,
                history=history,
            )
            # 1b. Intent Engine 2.0: derive goal, desired outcome, references,
            # and required capabilities, then validate the requirements against
            # the live registry (the model proposes meaning; Atlas checks tools).
            task = self._intent_engine.understand(
                task, prior_task=self._active_task, history=history
            )
            self._bind_previous_output(task)
            context.metadata["intent_reading"] = task.context.get("intent_reading", {})
            context.metadata["capability_check"] = self._intent_engine.validate_capabilities(task)
            trace.record_intent_reading(task.context.get("intent_reading", {}))
            # 1c. Semantic reasoning: understand what the user is trying to
            # accomplish (open-ended, not one of N intents), decide whether the
            # request needs external evidence, and select the capabilities that
            # would satisfy it - all before any capability routing or validation.
            semantic = self._semantic_reasoning.reason(
                context.normalized_input or user_input,
                task=task,
                prior_task=self._active_task,
                history=history,
            )
            self._semantic_decision = semantic
            context.metadata["semantic_reading"] = semantic.to_dict()
            context.metadata["semantic_trace"] = semantic.trace()
            trace.record_semantic(semantic.trace())
            if self._cancelled(cancel):
                return self._cancelled_response(context, started_at, trace)
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
            # The reasoning engine is the ONLY path that enforces evidence
            # discipline: it routes to WEB, gathers evidence, and evaluates
            # whether the evidence actually supports the claim. Both the general
            # model fallback and the legacy planner path may generate a plausible
            # answer from model memory — exactly what Atlas must never do for
            # dynamic facts.
            #
            # general_fallback_disabled is True when the request:
            #   * is a question,
            #   * has no actions,
            #   * is locally known ("model, knowledge" sources only), AND
            #   * does NOT require external evidence (required OR preferred).
            # Once semantic reasoning has marked evidence as "required" because
            # the answer depends on current facts, the general fallback is
            # disabled and we go through the reasoning engine.
            general_fallback_disabled = (
                not ENABLE_GENERAL_QUESTION_FALLBACK
                and task.request_type == "question"
                and not task.actions
                and set(task.sources) <= {"model", "knowledge"}
                and not task.current_information_required
                and task.evidence_requirement in {"unnecessary", "preferred"}
            )
            if (
                ENABLE_REASONING_ENGINE
                and task_validation.valid
                and not task_validation.needs_clarification
                and not general_fallback_disabled
            ):
                answer = self._reasoning_engine.handle_request(
                    question=context.normalized_input or user_input,
                    task=task,
                    prior_task=self._active_task,
                    history=history,
                    cancelled=cancel,
                    on_event=on_event,
                )
                context.metadata["task"] = task.to_dict()
                context.metadata["reasoning"] = task.context.get("reasoning", [])
                context.metadata["reasoning_summary"] = task.context.get("reasoning_summary", "")
                trace.record_task(task.to_dict())
                if answer is not None:
                    if self._cancelled(cancel):
                        return self._cancelled_response(context, started_at, trace)
                    context.metadata["reasoning"] = (answer.metadata or {}).get(
                        "reasoning", []
                    )
                    context.metadata["reasoning_summary"] = (answer.metadata or {}).get(
                        "reasoning_summary", ""
                    )
                    context.metadata["reasoning_answer"] = answer.to_dict()
                    # Extract web sources from evidence for display in UI
                    evidence = (answer.metadata or {}).get("evidence", {})
                    items = evidence.get("items", []) if isinstance(evidence, dict) else []
                    web_sources = []
                    web_results = []
                    seen = set()
                    for item in items:
                        if isinstance(item, dict):
                            url = item.get("source_identifier", "")
                            # Only include web sources with URLs, deduplicate
                            if url and url.startswith("http") and url not in seen:
                                seen.add(url)
                                web_sources.append(url)
                                # Also store full web result data for thumbnails
                                metadata = item.get("metadata", {})
                                web_results.append({
                                    "title": metadata.get("title", "") or item.get("source_identifier", ""),
                                    "url": url,
                                    "snippet": (item.get("content", "") or "").split("\n")[0][:300],
                                    "source": "YouTube" if "youtube.com" in url else "Web",
                                    "thumbnail_url": metadata.get("thumbnail_url"),
                                })
                    context.web_sources = web_sources
                    context.metadata["web_results"] = web_results
                    # Record the read-only tools the reasoning engine actually
                    # ran. The early-answer path does not go through the
                    # executor, so without this a conversation would show no
                    # activity for a turn that really searched or read files.
                    for observation in (answer.metadata or {}).get("tool_calls", []) or []:
                        if not isinstance(observation, dict) or not observation.get("tool"):
                            continue
                        success = bool(observation.get("success"))
                        context.tool_calls.append(
                            {
                                "tool": observation.get("tool"),
                                "status": "completed" if success else "failed",
                                "success": success,
                                "parameters": {},
                                "output": None,
                                "error": None,
                                "source": "reasoning",
                            }
                        )
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

            # 3b. Experience memory: retrieve supporting context from comparable
            # past tasks for sufficiently complex requests. This NEVER overrides
            # the current request — it is context the reasoning model may use. It
            # is also skipped for trivial single-step requests, so a simple
            # action does not pay for retrieval.
            experience_context = self._retrieve_experience_context(task, user_input)
            if experience_context.items:
                context.metadata["experience_context"] = experience_context.to_dict()
                logger.info(
                    "Experience context supplied: %d item(s) for goal=%s",
                    len(experience_context.items),
                    task.goal,
                )

            planner_decision = self._task_planner.create_plan(
                task,
                user_question=context.normalized_input or user_input,
                legacy_planner=self._planner.create_plan,
                experience_context=experience_context.text,
            )
            context.intent = structured.intent
            context.execution_plan = planner_decision.plan
            context.metadata["planner_decision"] = planner_decision
            context.metadata["task_requires_confirmation"] = task_validation.requires_confirmation
            trace.record_plan(planner_decision)

            if self._cancelled(cancel):
                return self._cancelled_response(context, started_at, trace)

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

            # Initialize working state for observation tracking
            from models import WorkingState
            context.working_state = WorkingState(
                task_id=context.task_id,
                objective=task.goal,
                current_plan=context.normalized_input or user_input,
                pending_steps=[step.id for step in planner_decision.plan.steps] if planner_decision.plan else [],
            )

            # 5. Deterministic execution (with verification + recovery inside).
            self._executor.execute(planner_decision.plan, context, cancel=cancel, on_event=on_event)
            if context.status == TaskStatus.WAITING_FOR_CONFIRMATION:
                self._pending_contexts[context.task_id] = context
            if context.status == TaskStatus.COMPLETED:
                # Only remember a completed task as reusable context.
                self._active_task = task
                self._active_intent = structured
            trace.record_execution(context)
            # 5b. Experience memory: record a compact, unevaluated experience for
            # a meaningful completed task so the user's Success/Failed answer has
            # something durable to attach to. Casual conversation is skipped, and
            # a cancelled turn records nothing: it teaches nothing about success.
            if context.status != TaskStatus.CANCELLED:
                self._record_experience(context, experience_context)
            return self._complete(context, started_at, trace)

        except Exception as exc:
            if self._cancelled(cancel):
                logger.info("Brain request cancelled during execution")
                return self._cancelled_response(context, started_at, trace)
            logger.exception("Brain execution failed")
            context.status = TaskStatus.FAILED
            context.errors.append(f"Brain execution failed ({type(exc).__name__})")
            if context.execution_plan is not None:
                context.execution_plan.status = PlanStatus.FAILED
            context.final_response = UNKNOWN_RESPONSE
            return self._complete(context, started_at, trace)

    @staticmethod
    def _cancelled(cancel: Callable[[], bool] | None) -> bool:
        return bool(cancel is not None and cancel())

    def _cancelled_response(
        self,
        context: ExecutionContext,
        started_at: float,
        trace: PipelineTrace | None = None,
    ) -> str:
        """Record a real cancellation and stop the request there.

        Nothing further is interpreted, planned, or executed, and the context
        says ``CANCELLED`` rather than pretending to have failed.
        """

        context.status = TaskStatus.CANCELLED
        context.metadata["cancelled"] = True
        if context.execution_plan is not None:
            context.execution_plan.status = PlanStatus.SKIPPED
        context.final_response = "Cancelled before Atlas finished that."
        logger.info("Brain request cancelled before execution finished")
        return self._complete(context, started_at, trace)

    def _complete(
        self,
        context: ExecutionContext,
        started_at: float,
        trace: PipelineTrace | None = None,
    ) -> str:
        # The cancellation check belongs to the request that supplied it. Clear
        # it here so a later approval/resume cannot inherit a stale token.
        self._cancel = None
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

    def _bind_previous_output(self, task: Task) -> None:
        """Bind a resolved prior-output reference from the persisted conversation."""

        if "previous_output" not in task.context_references:
            return
        intent_reading = task.context.get("intent_reading") or {}
        resolution = intent_reading.get("context") or {}
        if "previous_output" in resolution.get("unresolved", []):
            return
        previous_output = self._latest_assistant_output()
        if not previous_output:
            task.actions = []
            task.needs_clarification = True
            task.clarification_question = (
                "I can carry that out, but I couldn't determine what the previous "
                "output refers to. What would you like me to use?"
            )
            return
        task.actions = [
            _bind_previous_output(action, previous_output)
            for action in task.actions
        ]
        if not task.actions:
            task.needs_clarification = True
            task.clarification_question = (
                "I found the previous output, but couldn't determine how to use it "
                "for this request. What would you like me to do with it?"
            )

    def _latest_assistant_output(self) -> str:
        if self._memory_manager is None:
            return ""
        for message in reversed(self._memory_manager.get_recent_messages()):
            if message.role != "assistant":
                continue
            for result in reversed(message.tool_results):
                tool = str(result.get("tool") or "")
                output = result.get("output")
                if tool in {
                    "web.research", "web.fetch", "filesystem.read",
                    "content.generate", "content.format",
                } and isinstance(output, dict):
                    for key in ("text", "content"):
                        value = output.get(key)
                        if isinstance(value, str) and value.strip():
                            return value
            for call in reversed(message.tool_calls):
                if call.get("tool") != "applications.write_text":
                    continue
                parameters = call.get("parameters") or {}
                value = parameters.get("text")
                if isinstance(value, str) and value.strip():
                    return value
            content = str(message.content or "").strip()
            if content:
                return content
        return ""

    # -- experience memory -------------------------------------------------------

    def _retrieve_experience_context(self, task: Task, user_input: str):
        """Retrieve bounded experience context for a non-trivial request.

        Experience retrieval is skipped when the request is a single simple step
        with no delivery/composition, so a trivial action does not pay for a
        lookup it cannot benefit from. The retrieval itself lives in
        :class:`experience.service.ExperienceService` and is bounded; it never
        scans the whole store and never calls a model.
        """

        from experience.memory import ExperienceContext

        if not self._experience.enabled:
            return ExperienceContext()
        if not self._is_experience_eligible(task):
            return ExperienceContext()
        context = self._experience.retrieve_for_planning(user_input, task=task)
        self._last_experience_context = context
        return context

    @staticmethod
    def _is_experience_eligible(task: Task) -> bool:
        """True when a request is complex enough for experience to help.

        A single read-only lookup gains nothing from history; a multi-step,
        delivery, research, or hybrid task is exactly where a past workflow (or a
        past failure) is useful.
        """

        actions = list(task.actions)
        if len(actions) > 1:
            return True
        if task.goal in {"research_and_deliver", "create_and_deliver", "organize"}:
            return True
        if task.needs_application or task.request_type == "hybrid":
            return True
        for action in actions:
            if action.capability.startswith(("applications.", "computer.", "filesystem.write")):
                return True
            if action.capability in {"web.research", "content.format"}:
                return True
        return False

    def _record_experience(self, context: ExecutionContext, experience_context) -> None:
        """Persist a compact experience for a meaningful completed task.

        Recording is deliberately cheap and never raises into the request; it
        only stores an *unevaluated* candidate, so the user's later answer is
        what turns it into a learning signal.
        """

        try:
            retrieved_ids = (
                self._experience.retrieved_ids(experience_context)
                if experience_context is not None
                else []
            )
            conversation_id = ""
            if self._memory_manager is not None:
                conversation_id = self._memory_manager.get_active_session_id() or ""
            outcome = self._experience.record_task_outcome(
                context,
                record_id=context.metadata.get("record_id", "") or "",
                conversation_id=conversation_id,
                retrieved_experience_ids=retrieved_ids,
            )
            context.metadata["experience_recorded"] = outcome.recorded
            context.metadata["experience_id"] = outcome.experience_id
            context.metadata["feedback_eligible"] = outcome.meaningful
            if outcome.recorded:
                logger.info(
                    "Experience recorded: id=%s outcome=%s meaningful=%s",
                    outcome.experience_id,
                    outcome.outcome,
                    outcome.meaningful,
                )
        except Exception:  # noqa: BLE001 - the loop must never break a request
            logger.exception("Recording the experience failed")

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
        from executor import approval_signature
        tool_names = {
            str(step.metadata.get("tool"))
            for step in context.execution_plan.steps
            if step.action == "invoke_tool" and step.metadata.get("tool")
        }
        context.metadata["approved_tools"] = tool_names
        # Bind the grant to the exact arguments the user approved, so a recovery
        # that rewrites a consequential step's arguments cannot proceed on the
        # old approval.
        signatures = set()
        for step in context.execution_plan.steps:
            if step.action != "invoke_tool":
                continue
            tool = str(step.metadata.get("tool") or "")
            parameters = step.metadata.get("parameters") or {}
            if tool and isinstance(parameters, dict):
                signatures.add(approval_signature(tool, parameters))
        context.metadata["approved_signatures"] = signatures
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


def _bind_previous_output(action: TaskAction, output: str) -> TaskAction:
    action.parameters = _replace_output_reference(action.parameters, output)
    return action


def _replace_output_reference(value, output: str):
    if value == "$previous_output":
        return output
    if isinstance(value, dict):
        return {
            key: _replace_output_reference(item, output)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_output_reference(item, output) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_output_reference(item, output) for item in value)
    return value


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
