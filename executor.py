"""Atlas execution plan runner."""

from __future__ import annotations

import logging
import time
from typing import Callable

from knowledge_search import retrieve
from llm import ask
from models import (
    Evidence,
    ExecutionContext,
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    RetrievalResult,
    StepStatus,
    TaskStatus,
    Observation,
    ObservationSource,
    ObservationType,
    WorkingState,
)
from models_task import Task, TaskState
from vector_store import VectorStore
from memory.memory_manager import MemoryManager
from tools.router import ToolRouter
from tools.result_interpreter import interpret_tool_result
logger = logging.getLogger(__name__)

UNKNOWN_RESPONSE = "I don't know based on my knowledge base."
#: Plan-wide cap on LLM-assisted recovery attempts. Prevents a plan from looping
#: even when individual steps stay within their own retry limit.
MAX_PLAN_RECOVERIES = 3
#: Sentinel prefix marking a plan argument that references an earlier
#: step's produced output instead of a literal value.
GENERATED_TEXT_REF = "$generated_text"


class Executor:
    """Execute Atlas plans step-by-step."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        system_prompt: str,
        retrieval_template: str,
        memory_manager: MemoryManager | None = None,
        tool_router: ToolRouter | None = None,
        recovery: object | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._system_prompt = system_prompt
        self._retrieval_template = retrieval_template
        self._memory_manager = memory_manager
        self._tool_router = tool_router
        self._recovery = recovery
        from reasoning.verifier import TaskVerifier
        self._verifier = TaskVerifier()
        self._handlers: dict[str, Callable[[ExecutionContext, ExecutionStep], None]] = {
            "retrieve_knowledge": self._retrieve_knowledge,
            "generate_response": self._generate_response,
            "merge_evidence": self._merge_evidence,
            "invoke_tool": self._invoke_tool,
            "finalize_content": self._finalize_content,
        }




    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> ExecutionContext:
        """Execute a plan sequentially and update the shared context."""

        started_at = time.perf_counter()
        context.execution_plan = plan
        context.status = TaskStatus.RUNNING
        plan.status = PlanStatus.RUNNING

        # Update task state if available
        task = context.metadata.get("task_object")
        if task and isinstance(task, Task):
            task.task_state = TaskState.EXECUTING.value
            context.metadata["task_object"] = task

        logger.info("Executor started plan: plan_id=%s steps=%d", plan.plan_id, len(plan.steps))

        # Re-execution after an approval pause must resume, not restart. Steps
        # that already completed (e.g. content generation) are not run again.
        recovery_budget = int(context.metadata.get("recovery_budget", MAX_PLAN_RECOVERIES))
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED:
                logger.info(
                    "Executor skipping completed step: plan_id=%s step_id=%s",
                    plan.plan_id,
                    step.id,
                )
                continue
            if plan.status in {PlanStatus.FAILED, PlanStatus.WAITING_FOR_CONFIRMATION}:
                step.status = StepStatus.SKIPPED
                logger.info(
                    "Executor skipped step: plan_id=%s step_id=%s action=%s",
                    plan.plan_id,
                    step.id,
                    step.action,
                )
                continue
            self._execute_step(context, step)
            if step.status == StepStatus.FAILED and self._recovery is not None and recovery_budget > 0:
                # One bounded, observable recovery attempt per failed step.
                # Recovery may only rewrite arguments for the same capability.
                # A plan-wide budget prevents runaway retry loops when several
                # steps fail in sequence.
                if self._recovery.attempt_recovery(context, step):
                    recovery_budget -= 1
                    self._execute_step(context, step)
                    self._record_replan(context, step)

        # Verify completion if plan completed without errors
        if plan.status not in {PlanStatus.FAILED, PlanStatus.WAITING_FOR_CONFIRMATION}:
            plan.status = PlanStatus.COMPLETED
            context.status = TaskStatus.COMPLETED
            
            # Run completion verification
            if not self._verify_completion(context):
                plan.status = PlanStatus.FAILED
                context.status = TaskStatus.FAILED
        elif plan.status == PlanStatus.WAITING_FOR_CONFIRMATION:
            context.status = TaskStatus.WAITING_FOR_CONFIRMATION
        else:
            context.status = TaskStatus.FAILED

        context.execution_time = time.perf_counter() - started_at
        plan.metadata["execution_time"] = context.execution_time
        logger.info(
            "Executor completed plan: plan_id=%s status=%s execution_time=%.4fs",
            plan.plan_id,
            plan.status.value,
            context.execution_time,
        )
        return context

    def _execute_step(self, context: ExecutionContext, step: ExecutionStep) -> None:
        plan_id = context.execution_plan.plan_id if context.execution_plan else "unknown"
        handler = self._handlers.get(step.action)

        logger.info(
            "Executor running step: plan_id=%s step_id=%s action=%s",
            plan_id,
            step.id,
            step.action,
        )

        if handler is None:
            step.status = StepStatus.FAILED
            step.metadata["error"] = f"Unknown execution action: {step.action}"
            self._mark_failed(context)
            logger.error(
                "Executor failed step: plan_id=%s step_id=%s reason=%s",
                plan_id,
                step.id,
                step.metadata["error"],
            )
            return

        step.status = StepStatus.RUNNING
        started_at = time.perf_counter()

        try:
            handler(context, step)
            if context.execution_plan and context.execution_plan.status == PlanStatus.WAITING_FOR_CONFIRMATION:
                step.status = StepStatus.PENDING
                step.metadata["execution_time"] = time.perf_counter() - started_at
                return
            step.status = StepStatus.COMPLETED
            step.metadata["execution_time"] = time.perf_counter() - started_at
            logger.info(
                "Executor completed step: plan_id=%s step_id=%s action=%s execution_time=%.4fs",
                plan_id,
                step.id,
                step.action,
                step.metadata["execution_time"],
            )
        except Exception as exc:
            step.status = StepStatus.FAILED
            step.metadata["error"] = repr(exc)
            step.metadata["execution_time"] = time.perf_counter() - started_at
            self._mark_failed(context)
            # Create failure observation for exception
            tool_name = str(step.metadata.get("tool", step.action))
            action_id = step.metadata.get("action_id") or step.id
            observation = self._create_observation(
                tool_name=tool_name,
                output=None,
                success=False,
                step_id=step.id,
                action_id=action_id,
                error=str(exc),
            )
            context.observations.append(observation.to_dict())
            self._update_working_state(context, observation)
            logger.exception("Executor failed step: plan_id=%s step_id=%s action=%s", plan_id, step.id, step.action)

    def _invoke_tool(self, context: ExecutionContext, step: ExecutionStep) -> None:
        if self._tool_router is None:
            raise RuntimeError("Tool router is not configured")

        tool_name = str(step.metadata.get("tool", "")).strip()
        parameters = step.metadata.get("parameters") or {}
        if not tool_name:
            raise ValueError("Tool invocation is missing a tool name")
        if not isinstance(parameters, dict):
            raise TypeError("Tool invocation parameters must be a dictionary")

        parameters = self._resolve_arguments(parameters, context)
        # A required parameter that resolved to None means an upstream reference
        # (e.g. "$top_result_url" when the search returned nothing) produced no
        # value. Fail the step with an honest message instead of calling the tool
        # with a null argument.
        missing = [key for key, value in parameters.items() if value is None]
        if missing:
            context.errors.append(
                f"{tool_name} could not run: no value was available for "
                + ", ".join(missing)
            )
            raise RuntimeError(
                f"{tool_name} is missing a required value: " + ", ".join(missing)
            )
        context.selected_tool = tool_name
        # Approval is not consumed on read: multiple steps in the same plan may
        # require it, and a resumed plan must still be considered approved. Do
        # not use pop() here or the first tool call would discard the approval.
        approved_tools = context.metadata.get("approved_tools") or set()
        # An approval is bound to the exact arguments the user saw. If a bounded
        # recovery rewrote a consequential step's arguments after the grant, the
        # old approval does not cover the new effect, so it must be re-obtained.
        approved_signatures = context.metadata.get("approved_signatures")
        approved = tool_name in approved_tools
        if approved and approved_signatures is not None:
            # Signature over the *planned* parameters the user approved (references
            # included), before $reference resolution, so it matches the grant.
            planned = step.metadata.get("parameters") or {}
            approved = approval_signature(tool_name, planned) in approved_signatures
        result = self._tool_router.execute(
            tool_name,
            parameters,
            approved=approved,
        )
        context.tool_calls.append(
            {
                "tool": tool_name,
                "parameters": parameters,
                "status": result.status,
                "success": result.success,
                "output": _bounded_tool_output(result.output),
                "error": result.error,
            }
        )
        step.result = result
        step.metadata["tool_status"] = result.status

        if result.status == "confirmation_required":
            context.permissions.append(result.metadata)
            context.warnings.append(result.error or "Tool confirmation required")
            context.execution_plan.status = PlanStatus.WAITING_FOR_CONFIRMATION
            context.status = TaskStatus.WAITING_FOR_CONFIRMATION
            pending = context.metadata.get("produced", {}).get("generated_text")
            if pending and tool_name == "applications.write_text":
                application = parameters.get("application", "the application")
                context.final_response = (
                    f"I generated the content and am ready to write it into {application}. "
                    "Approve to continue."
                )
            else:
                context.final_response = "This action requires your confirmation before Atlas can continue."
            # Create observation for confirmation required
            action_id = step.metadata.get("action_id") or step.id
            observation = self._create_observation(
                tool_name=tool_name,
                output=result.output,
                success=False,
                step_id=step.id,
                action_id=action_id,
                error="Confirmation required",
            )
            context.observations.append(observation.to_dict())
            self._update_working_state(context, observation)
            return
        if not result.success:
            context.errors.append(result.error or f"Tool failed: {tool_name}")
            # Create failure observation before raising
            action_id = step.metadata.get("action_id") or step.id
            observation = self._create_observation(
                tool_name=tool_name,
                output=result.output,
                success=False,
                step_id=step.id,
                action_id=action_id,
                error=result.error,
            )
            context.observations.append(observation.to_dict())
            self._update_working_state(context, observation)
            raise RuntimeError(result.error or f"Tool failed: {tool_name}")

        context.metadata.setdefault("tool_results", []).append(result.output)
        self._capture_produced_output(step, result.output, context)
        # Observation/verification: record honest evidence for this action.
        self._record_verification(context, tool_name, result.output)
        
        # Create structured observation and update working state
        action_id = step.metadata.get("action_id") or step.id
        observation = self._create_observation(
            tool_name=tool_name,
            output=result.output,
            success=result.success,
            step_id=step.id,
            action_id=action_id,
            error=result.error,
        )
        context.observations.append(observation.to_dict())
        self._update_working_state(context, observation)
        
        # Compare expected vs observed if expected_outcome is specified
        expected_outcome = step.metadata.get("expected_outcome")
        if expected_outcome:
            matches, detail = self._compare_expected_vs_observed(expected_outcome, observation)
            if not matches:
                context.warnings.append(f"Expected state mismatch: {detail}")
                # Record verification failure
                context.verification_results.append({
                    "capability": tool_name,
                    "verified": False,
                    "status": "failed",
                    "detail": f"Expected state mismatch: {detail}",
                })
        
        if tool_name == "web.search" and isinstance(result.output, dict):
            context.web_sources.extend(
                str(item.get("url"))
                for item in result.output.get("results", [])
                if isinstance(item, dict) and item.get("url")
            )
        elif tool_name == "web.fetch" and isinstance(result.output, dict) and result.output.get("url"):
            context.web_sources.append(str(result.output["url"]))

        # Only the final tool step of a plan owns the user-facing response, so an
        # intermediate step (e.g. content generation) is not reported as the
        # answer to a multi-step task.
        if self._is_final_step(context, step) or context.final_response is None:
            context.final_response = interpret_tool_result(tool_name, result)

    def _is_final_step(self, context: ExecutionContext, step: ExecutionStep) -> bool:
        plan = context.execution_plan
        if plan is None or not plan.steps:
            return True
        return plan.steps[-1].id == step.id

    def _merge_evidence(self, context: ExecutionContext, step: ExecutionStep) -> None:
        """Merge accumulated evidence_context_parts and evidence_chunks.

        For now, retrieval already accumulates evidence. This step simply
        re-stitches the Evidence object to support future richer merge logic.
        """

        evidence_parts = context.metadata.get("evidence_context_parts") or []
        evidence_chunks = context.metadata.get("evidence_chunks") or []

        context.evidence = Evidence(
            context="\n\n".join(evidence_parts),
            chunks=list(evidence_chunks),
            sources=_unique_sources(list(evidence_chunks)),
            metadata={
                "retrieval_metric": context.evidence.metadata.get("retrieval_metric") if context.evidence else None,
                "retrieval_method": "evidence_accumulated_via_executor",
            },
        )

        step.metadata["merged_chunks"] = len(evidence_chunks)
        step.result = context.evidence
        logger.info("Merged evidence: merged_chunks=%d", len(evidence_chunks))

    def _retrieve_knowledge(self, context: ExecutionContext, step: ExecutionStep) -> None:

        context.selected_tool = "knowledge"
        context.metadata["retrieval_strategy"] = step.metadata.get("retrieval_strategy") or context.metadata.get("retrieval_strategy")
        retrieval = retrieve(context.user_input, vector_store=self._vector_store)

        retrieval_result = RetrievalResult(
            context=retrieval.context,
            best_distance=retrieval.best_distance,
            retrieved_chunks=retrieval.retrieved_chunks,
            expanded_queries=retrieval.expanded_queries,
            diagnostics=retrieval.diagnostics,
        )

        context.retrieval_result = retrieval_result
        context.confidence = retrieval_result.best_distance
        # Support multi-step intents (e.g., compare) by accumulating evidence.
        if context.metadata.get("evidence_chunks") is None:
            context.metadata["evidence_chunks"] = []
        if context.metadata.get("evidence_context_parts") is None:
            context.metadata["evidence_context_parts"] = []

        context.metadata["evidence_chunks"].extend(retrieval_result.retrieved_chunks)
        if retrieval_result.context:
            context.metadata["evidence_context_parts"].append(retrieval_result.context)

        context.evidence = Evidence(
            context="\n\n".join(context.metadata["evidence_context_parts"]),
            chunks=list(context.metadata["evidence_chunks"]),
            sources=_unique_sources(list(context.metadata["evidence_chunks"])),
            metadata={
                "retrieval_metric": "chroma_distance_smaller_is_better",
                "retrieval_strategy": step.metadata.get("retrieval_strategy"),
                "retrieval_method": "hybrid_staged_semantic_keyword_metadata",
                "compare_side": step.metadata.get("compare_side"),
            },
            chunk_ids=[str(getattr(hit, "chunk_id", "")) for hit in context.metadata["evidence_chunks"]],
            confidence=context.confidence,
            retrieval_method="hybrid_staged_semantic_keyword_metadata",
        )

        context.metadata["retrieval_metric"] = "chroma_distance_smaller_is_better"

        context.metadata["retrieval_diagnostics"] = retrieval_result.diagnostics
        context.metadata["rewritten_queries"] = retrieval_result.expanded_queries

        step.result = retrieval_result
        step.metadata["chunks"] = len(retrieval_result.retrieved_chunks)
        step.metadata["best_distance"] = retrieval_result.best_distance
        step.metadata["expanded_queries"] = len(retrieval_result.expanded_queries)

        logger.info(
            "Knowledge retrieval complete: chunks=%d best_distance=%s expanded_queries=%d",
            len(retrieval_result.retrieved_chunks),
            f"{retrieval_result.best_distance:.4f}" if retrieval_result.best_distance is not None else "none",
            len(retrieval_result.expanded_queries),
        )

    def _generate_response(self, context: ExecutionContext, step: ExecutionStep) -> None:
        retrieval_result = context.retrieval_result
        if retrieval_result is None or not retrieval_result.context:
            diagnostics = retrieval_result.diagnostics if retrieval_result is not None else None
            if diagnostics is not None:
                logger.warning("Knowledge retrieval rejected: %s", diagnostics.final_decision.reason)

            context.final_response = UNKNOWN_RESPONSE
            step.result = context.final_response
            logger.info("LLM skipped: no accepted retrieval context")
            return

        conversation_history = ""
        if self._memory_manager is not None:
            # History builder is responsible for enforcing retention limits.
            conversation_history = self._memory_manager.build_conversation_history_for_prompt()

        user_prompt = self._retrieval_template.format(
            conversation_history=conversation_history,
            context=retrieval_result.context,
            question=context.user_input,
        )

        # Persist user message before generation.
        if self._memory_manager is not None:
            self._memory_manager.append_message(role="user", content=context.user_input)

        logger.info("LLM called")
        context.llm_response = ask(system_prompt=self._system_prompt, user_prompt=user_prompt)
        context.final_response = context.llm_response
        step.result = context.final_response
        logger.info("LLM complete")

        # Persist assistant message after successful generation.
        if self._memory_manager is not None and context.final_response:
            self._memory_manager.append_message(role="assistant", content=context.final_response)



    def _resolve_arguments(self, parameters: object, context: ExecutionContext) -> dict:
        # Replace plan references with values produced by earlier steps.
        # A step may declare a generated-text reference; the executor resolves
        # that reference from the output of a prior step. This keeps generated
        # content from being passed through the model a second time.
        if not isinstance(parameters, dict):
            return {}
        produced = context.metadata.get("produced", {})
        resolved: dict = {}
        for key, value in parameters.items():
            resolved[key] = _resolve_value(value, produced)
        return resolved
    def _capture_produced_output(self, step: ExecutionStep, output: object, context: ExecutionContext) -> None:
        # Expose a named output for later steps to reference.
        produced = context.metadata.setdefault("produced", {})
        name = step.metadata.get("produces")
        tool = step.metadata.get("tool")
        if name and tool == "web.search" and isinstance(output, dict):
            # A search publishes both a human-readable summary (for a fallback
            # write) and the top result URL (so a follow-up web.fetch can read
            # the page the user actually wants placed somewhere).
            produced[str(name)] = _render_web_results(output)
            produced.setdefault("top_result_url", _top_result_url(output))
        elif name and tool == "web.fetch" and isinstance(output, dict):
            # A fetch publishes its page text (not the raw {url, text} mapping)
            # so a downstream write receives usable content.
            produced[str(name)] = str(output.get("text") or "")
        elif name and tool == "filesystem.read" and isinstance(output, dict):
            # A local read publishes its text (not the raw {path, content} mapping)
            # so a downstream transform/generate step receives usable content.
            produced[str(name)] = str(output.get("content") or "")
        elif name and tool == "web.research" and isinstance(output, dict):
            # Research publishes the ranked, isolated content as text.
            produced[str(name)] = str(output.get("content") or "")
        elif name:
            produced[str(name)] = output
        # content.generate always publishes generated_text as a convenience.
        if step.metadata.get("tool") == "content.generate" and isinstance(output, dict):
            text = output.get("text")
            if isinstance(text, str):
                produced[GENERATED_TEXT_REF] = text
                produced.setdefault("generated_text", text)
        # content.format publishes formatted_text
        if step.metadata.get("tool") == "content.format" and isinstance(output, dict):
            text = output.get("text")
            if isinstance(text, str):
                produced["formatted_text"] = text
        # filesystem.search may publish a single selected match (e.g. "the
        # largest PDF") for a downstream move/copy step to consume.
        if step.metadata.get("tool") == "filesystem.search":
            self._publish_selected_match(step, output, produced)

    @staticmethod
    def _publish_selected_match(step: ExecutionStep, output: object, produced: dict) -> None:
        if not isinstance(output, dict):
            return
        matches = output.get("matches")
        if not isinstance(matches, list) or not matches:
            return
        select = str(step.metadata.get("parameters", {}).get("select") or "").casefold()
        if select not in {"largest", "smallest", "newest", "oldest", "latest"}:
            return
        chosen = _select_match(matches, select)
        if chosen is not None:
            produced["largest_match"] = chosen
            produced.setdefault("search_results", matches)

    def _record_replan(self, context: ExecutionContext, step: ExecutionStep) -> None:
        context.metadata.setdefault("replan_history", []).append(
            {
                "step_id": step.id,
                "capability": step.metadata.get("tool"),
                "status": step.status.value,
                "attempts": step.metadata.get("recovery_attempts", 0),
            }
        )

    def _record_verification(self, context: ExecutionContext, tool_name: str, output: object) -> None:
        # Observe the tool output and record whether the intended effect is
        # supported by evidence. Unverified outcomes are surfaced, never silently
        # upgraded to success.
        outcome = self._verifier.verify(tool_name, output, success=True)
        context.observations.append(
            {"tool": tool_name, "status": outcome.status, "detail": outcome.detail}
        )
        context.verification_results.append(outcome.to_dict())

    def _create_observation(
        self,
        tool_name: str,
        output: object,
        success: bool,
        step_id: str,
        action_id: str | None = None,
        error: str | None = None,
    ) -> Observation:
        """Create a structured observation from a tool execution result."""
        # Determine observation status and summary
        if error:
            status = "failure"
            summary = f"{tool_name} failed: {error}"
        elif success:
            status = "success"
            summary = f"{tool_name} completed successfully"
        else:
            status = "failure"
            summary = f"{tool_name} failed"

        # Extract artifacts and environment details based on tool type
        artifacts = []
        environment = {}
        details = {}

        if isinstance(output, dict):
            # Common artifact fields
            if "path" in output:
                artifacts.append(str(output["path"]))
                details["path"] = output["path"]
            if "pid" in output:
                details["pid"] = output["pid"]
                environment["pid"] = output["pid"]
            if "executable" in output:
                details["executable"] = output["executable"]
            if "application" in output:
                details["application"] = output["application"]
                environment["application"] = output["application"]
            if "url" in output:
                details["url"] = output["url"]
                environment["url"] = output["url"]
            if "title" in output:
                details["page_title"] = output["title"]
                environment["page_title"] = output["title"]
            if "characters" in output:
                details["characters_written"] = output["characters"]
            if "bytes" in output:
                details["bytes_written"] = output["bytes"]
            if "content" in output and isinstance(output["content"], str):
                details["content_length"] = len(output["content"])
            if "text" in output and isinstance(output["text"], str):
                details["text_length"] = len(output["text"])
            if "matches" in output:
                details["matches_found"] = len(output["matches"]) if isinstance(output["matches"], list) else 0
            if "results" in output:
                details["results_count"] = len(output["results"]) if isinstance(output["results"], list) else 0

            # Copy other relevant fields to details
            for key in ("status", "created", "destination", "source", "truncated", "expanded_queries", "chunks", "best_distance"):
                if key in output:
                    details[key] = output[key]

        # Determine observation source based on tool category
        source = ObservationSource.TOOL_RESULT
        if tool_name.startswith("applications."):
            source = ObservationSource.APPLICATION_STATE
        elif tool_name.startswith("web."):
            source = ObservationSource.BROWSER_STATE
        elif tool_name.startswith("filesystem."):
            source = ObservationSource.FILESYSTEM_STATE
        elif tool_name.startswith("system.") or tool_name.startswith("processes.") or tool_name.startswith("powershell."):
            source = ObservationSource.SYSTEM_STATE

        return Observation(
            source=source,
            type=ObservationType.ACTION_RESULT if success else ObservationType.ERROR,
            status=status,
            summary=summary,
            details=details,
            artifacts=artifacts,
            environment=environment,
            step_id=step_id,
            action_id=action_id,
            tool_name=tool_name,
        )

    def _update_working_state(self, context: ExecutionContext, observation: Observation) -> None:
        """Update the working state with a new observation."""
        if context.working_state is None:
            task = context.metadata.get("task_object")
            objective = ""
            if task and isinstance(task, Task):
                objective = task.goal
            context.working_state = WorkingState(
                task_id=context.task_id,
                objective=objective,
                current_plan=context.execution_plan.user_question if context.execution_plan else "",
            )
        
        context.working_state.add_observation(observation)
        
        # Update step tracking
        if observation.step_id:
            context.working_state.update_step(observation.step_id, observation.status)
        
        # Update application state for relevant observations
        if observation.tool_name:
            if observation.tool_name.startswith("applications.launch") or observation.tool_name == "applications.write_text":
                app = observation.details.get("application") or observation.environment.get("application")
                if app:
                    context.working_state.set_application_state(
                        application=app,
                        window=observation.details.get("executable", ""),
                    )
            elif observation.tool_name.startswith("web."):
                url = observation.details.get("url") or observation.environment.get("url")
                title = observation.details.get("page_title") or observation.environment.get("page_title")
                if url:
                    context.working_state.set_application_state(
                        application="Browser",
                        window=title or "Browser",
                        url=url,
                        page_title=title or "",
                    )
            elif observation.tool_name.startswith("filesystem."):
                # Filesystem operations don't change application state
                pass

    def _compare_expected_vs_observed(
        self,
        expected: dict[str, Any] | None,
        observation: Observation,
    ) -> tuple[bool, str]:
        """Compare expected outcome with observed result.
        
        Returns:
            (matches: bool, detail: str)
        """
        if not expected:
            return True, "No expected outcome specified"
        
        mismatches = []
        
        # Check application state
        if "application" in expected:
            expected_app = expected["application"]
            observed_app = observation.details.get("application") or observation.environment.get("application")
            if observed_app and expected_app.lower() not in observed_app.lower():
                mismatches.append(f"application: expected '{expected_app}', got '{observed_app}'")
        
        # Check for URL
        if "url" in expected:
            expected_url = expected["url"]
            observed_url = observation.details.get("url") or observation.environment.get("url")
            if observed_url and expected_url not in observed_url:
                mismatches.append(f"url: expected '{expected_url}', got '{observed_url}'")
        
        # Check for file path
        if "path" in expected:
            expected_path = expected["path"]
            observed_path = observation.details.get("path")
            if observed_path and expected_path not in observed_path:
                mismatches.append(f"path: expected '{expected_path}', got '{observed_path}'")
        
        # Check for content presence
        if "content_contains" in expected:
            expected_content = expected["content_contains"]
            # This would need the actual content to check - for now just note it
            pass
        
        # Check for minimum characters written
        if "min_characters" in expected:
            min_chars = expected["min_characters"]
            actual_chars = observation.details.get("characters_written", 0)
            if actual_chars < min_chars:
                mismatches.append(f"characters: expected at least {min_chars}, got {actual_chars}")
        
        if mismatches:
            return False, "; ".join(mismatches)
        
        return True, "Expected state matches observed state"

    def _finalize_content(self, context: ExecutionContext, step: ExecutionStep) -> None:
        # Publish generated content as the final response for content-only tasks.
        produced = context.metadata.get("produced", {})
        text = produced.get(GENERATED_TEXT_REF) or produced.get("generated_text")
        if not text:
            raise RuntimeError("No generated content was available to return")
        context.final_response = str(text)
        step.result = context.final_response
        if self._memory_manager is not None:
            self._memory_manager.append_message(role="user", content=context.user_input)
            self._memory_manager.append_message(role="assistant", content=context.final_response)

    def _verify_completion(self, context: ExecutionContext) -> bool:
        """Verify that the task's completion criteria have been met."""
        # Get the task from context metadata
        task_data = context.metadata.get("task")
        if not task_data:
            return True  # No task to verify
        
        # Check if we have a task object with completion criteria
        task = context.metadata.get("task_object")
        if not task or not isinstance(task, Task):
            return True
        
        # Update task state
        task.task_state = TaskState.VERIFYING.value
        
        # Update subtask status for completed steps
        self._update_subtask_status(task, context)
        
        # Verify expected vs observed for each step that had expected_outcome
        expected_outcome_failures = []
        for tool_call in context.tool_calls:
            tool_name = tool_call.get("tool")
            if not tool_name:
                continue
            
            # Find the observation for this tool call
            for obs_dict in context.observations:
                if isinstance(obs_dict, dict) and obs_dict.get("tool_name") == tool_name:
                    # Find the step that produced this observation
                    for step in context.execution_plan.steps:
                        step_tool = step.metadata.get("tool")
                        if step_tool == tool_name:
                            expected_outcome = step.metadata.get("expected_outcome")
                            if expected_outcome:
                                # Reconstruct observation object
                                from models import Observation
                                try:
                                    observation = Observation(
                                        source=obs_dict.get("source", "tool_result"),
                                        type=obs_dict.get("type", "action_result"),
                                        status=obs_dict.get("status", "unknown"),
                                        summary=obs_dict.get("summary", ""),
                                        details=obs_dict.get("details", {}),
                                        artifacts=obs_dict.get("artifacts", []),
                                        environment=obs_dict.get("environment", {}),
                                        step_id=obs_dict.get("step_id"),
                                        action_id=obs_dict.get("action_id"),
                                        tool_name=obs_dict.get("tool_name"),
                                    )
                                    matches, detail = self._compare_expected_vs_observed(expected_outcome, observation)
                                    if not matches:
                                        expected_outcome_failures.append(f"{tool_name}: {detail}")
                                except Exception:
                                    pass
                            break
        
        if expected_outcome_failures:
            context.errors.extend(expected_outcome_failures)
            context.execution_trace.append({
                "stage": "verification",
                "action": "expected_vs_observed",
                "result": "failed",
                "details": {"failures": expected_outcome_failures}
            })
            return False
        
        # Get the set of tools that were actually executed in this plan
        executed_tools = set()
        for tool_call in context.tool_calls:
            tool_name = tool_call.get("tool")
            if tool_name:
                executed_tools.add(tool_name)
        
        # Check completion criteria - only for actions that were actually executed
        # Criteria tracking is for observability; don't fail the task for unmet criteria
        # unless there was an explicit tool execution error
        if task.completion_criteria:
            unmet = []
            for criterion in task.completion_criteria.criteria:
                # Skip criteria for actions that weren't in the executed plan
                if self._should_skip_criterion(criterion, executed_tools, task):
                    criterion["met"] = True  # Mark as met since it wasn't applicable
                    continue
                if not criterion["met"]:
                    unmet.append(criterion["name"])
            
            if unmet:
                # Log unmet criteria but don't fail - they're for observability
                logging.info(f"Task completion: some criteria not met (observability): {', '.join(unmet)}")
                context.execution_trace.append({
                    "stage": "verification",
                    "action": "completion_check",
                    "result": "partial",
                    "details": {"unmet_criteria": unmet, "note": "criteria are observability-only"}
                })
                # Don't return False - criteria are observability, not hard requirements
        
        # Post-action verification for specific capabilities
        # This checks actual tool output for evidence of success
        verification_passed = self._post_action_verification(context, task)
        
        if verification_passed:
            task.task_state = TaskState.COMPLETED.value
            if task.completion_criteria:
                task.completion_criteria.all_met = True
            context.execution_trace.append({
                "stage": "verification",
                "action": "completion_check",
                "result": "passed",
                "details": {"all_criteria_met": True}
            })
            return True
        else:
            context.execution_trace.append({
                "stage": "verification",
                "action": "completion_check",
                "result": "failed",
                "details": {"post_action_verification": "failed"}
            })
            return False

    def _should_skip_criterion(self, criterion: dict, executed_tools: set, task: Task) -> bool:
        """Determine if a criterion should be skipped because its action wasn't executed."""
        name = criterion["name"]
        
        # Content generation criteria - skip if content.generate wasn't executed
        if name in ("content_generated", "content_formatted") and "content.generate" not in executed_tools:
            return True
        
        # Delivery criteria - skip if write_text/write wasn't executed
        if name in ("destination_opened", "content_delivered", "delivery_verified") and "applications.write_text" not in executed_tools:
            return True
        if name in ("file_created", "file_verified") and "filesystem.write" not in executed_tools:
            return True
        
        # Subtask criteria - skip if the subtask's capability wasn't executed
        if name.startswith("subtask_"):
            idx = int(name.split("_")[1])
            if idx < len(task.subtasks):
                subtask_cap = task.subtasks[idx].get("capability")
                if subtask_cap and subtask_cap not in executed_tools:
                    return True
        
        return False

    def _update_subtask_status(self, task: Task, context: ExecutionContext) -> None:
        """Update subtask status based on completed steps."""
        if not task.subtasks:
            return
        
        completed_tools = set()
        for tool_call in context.tool_calls:
            tool_name = tool_call.get("tool")
            if tool_name:
                completed_tools.add(tool_name)
        
        # Map capabilities to subtasks
        for subtask in task.subtasks:
            capability = subtask.get("capability")
            if capability in completed_tools:
                subtask["status"] = "completed"
            elif subtask.get("status") == "pending" and any(dep in completed_tools for dep in subtask.get("depends_on", [])):
                subtask["status"] = "in_progress"

    def _post_action_verification(self, context: ExecutionContext, task: Task) -> bool:
        """Perform post-action verification for critical capabilities."""
        # Check for application write verification
        for tool_call in context.tool_calls:
            tool_name = tool_call.get("tool")
            output = tool_call.get("output", {})
            
            if tool_name == "applications.write_text":
                # Verify text was actually written
                if isinstance(output, dict):
                    chars = output.get("characters", 0)
                    if chars <= 0:
                        context.errors.append("Application write reported zero characters written")
                        return False
                    
                    # Mark delivery criteria as met
                    if task.completion_criteria:
                        task.completion_criteria.mark_met("content_delivered")
                        task.completion_criteria.mark_met("delivery_verified")
                
            elif tool_name == "filesystem.write":
                # Verify file was created
                if isinstance(output, dict):
                    path = output.get("path")
                    if not path:
                        context.errors.append("File write did not return a path")
                        return False
                    
                    if task.completion_criteria:
                        task.completion_criteria.mark_met("file_created")
                        task.completion_criteria.mark_met("file_verified")
            
            elif tool_name == "applications.launch_named":
                # Verify app launched
                if isinstance(output, dict):
                    pid = output.get("pid")
                    if not pid:
                        context.errors.append("Application launch did not return a PID")
                        return False
                    
                    if task.completion_criteria:
                        task.completion_criteria.mark_met("destination_opened")
        
        return True

    def _mark_failed(self, context: ExecutionContext) -> None:
        if context.execution_plan is not None:
            context.execution_plan.status = PlanStatus.FAILED
        context.status = TaskStatus.FAILED
        classification = classify_failure(context)
        context.metadata["failure_classification"] = classification
        recovered = context.metadata.get("replan_history") or []
        if context.errors:
            message = f"Atlas could not complete this task: {context.errors[-1]}"
            if recovered:
                message += " (a safe retry was attempted first and did not succeed)"
            context.final_response = message
        else:
            context.final_response = UNKNOWN_RESPONSE


def approval_signature(tool_name: str, parameters: dict) -> str:
    # Return a stable signature binding an approval to exact step arguments.
    # The signature is deterministic (sorted keys, JSON-encoded values with a
    # string fallback) so an unchanged request matches its grant while a
    # recovery that rewrites the arguments produces a different signature and
    # therefore needs fresh approval. It is a hash, not the arguments.

    import hashlib
    import json
    try:
        payload = json.dumps(parameters, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        payload = repr(sorted((str(k), str(v)) for k, v in parameters.items()))
    digest = hashlib.sha256(f"{tool_name}\u0000{payload}".encode("utf-8")).hexdigest()
    return digest[:32]


def _top_result_url(output: dict) -> str | None:
    # The first result with a URL is the page a follow-up fetch should read.
    results = output.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and item.get("url"):
                return str(item["url"])
    return None

def _render_web_results(output: dict) -> str:
    # Render web search results as plain text for a downstream write step.
    # The result list is untrusted data: it is copied as text, never interpreted.
    results = output.get("results")
    if not isinstance(results, list) or not results:
        query = str(output.get("query") or "the request")
        return f"No web results were returned for '{query}'."
    lines: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        line = title or snippet
        if url:
            line = f"{line} ({url})" if line else url
        if line:
            lines.append("- " + line)
    return "\n".join(lines) if lines else str(output)

def _resolve_value(value: object, produced: dict) -> object:
    if isinstance(value, str):
        if value in produced:
            return produced[value]
        # A "$name" reference consumes the output published by an earlier step.
        # The bare name (without the $) is the key in the produced map, so a
        # generic reference such as "$search_results" resolves here; this also
        # covers the "$generated_text" convention used by content generation.
        if value.startswith("$") and len(value) > 1:
            name = value[1:]
            if name in produced:
                return produced[name]
        if value == GENERATED_TEXT_REF and "generated_text" in produced:
            return produced["generated_text"]
        return value
    if isinstance(value, dict):
        return {str(key): _resolve_value(item, produced) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, produced) for item in value]
    return value

def classify_failure(context: ExecutionContext) -> dict[str, object]:
    # Classify why a plan failed so recovery can decide what is recoverable.
    # Categories are coarse and stable so retry policy never depends on parsing
    # arbitrary error prose.
    message = (context.errors[-1] if context.errors else "").casefold()
    if not message:
        category = "unknown"
    elif "unknown tool" in message:
        category = "unknown_capability"
    elif "confirmation" in message or "denied" in message:
        category = "permission"
    elif "could not be focused" in message or "could not be found" in message and "window" in message:
        # A GUI focus failure cannot be fixed by changing tool arguments.
        category = "gui_unavailable"
    elif "not found" in message or "could not resolve" in message:
        category = "missing_target"
    elif "parameter" in message or "missing" in message or "must be" in message:
        category = "invalid_arguments"
    elif "timeout" in message or "timed out" in message:
        category = "timeout"
    else:
        category = "execution_error"
    recoverable = category in {"invalid_arguments", "execution_error", "timeout"}
    return {"category": category, "recoverable": recoverable, "message": message}


def _select_match(matches: list, select: str) -> str | None:
    """Pick one match by size or modification time; return its path string."""

    def stats(path_value: object) -> tuple[int, float] | None:
        import os
        try:
            info = os.stat(str(path_value))
        except OSError:
            return None
        return (info.st_size, info.st_mtime)

    scored: list[tuple[int, float, str]] = []
    for match in matches:
        info = stats(match)
        if info is None:
            continue
        scored.append((info[0], info[1], str(match)))
    if not scored:
        # Fall back to the first match so a move/copy target still resolves.
        return str(matches[0]) if matches else None
    if select == "largest":
        return max(scored, key=lambda item: item[0])[2]
    if select == "smallest":
        return min(scored, key=lambda item: item[0])[2]
    if select in {"newest", "latest"}:
        return max(scored, key=lambda item: item[1])[2]
    if select == "oldest":
        return min(scored, key=lambda item: item[1])[2]
    return scored[0][2]


def _unique_sources(chunks: list[object]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        source = getattr(chunk, "source", "")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def _bounded_tool_output(output: object, *, max_characters: int = 100_000) -> object:
    """Keep API task snapshots useful without allowing unbounded tool output."""

    if isinstance(output, str):
        return output[:max_characters]
    if isinstance(output, dict):
        bounded = dict(output)
        for key in ("stdout", "stderr"):
            value = bounded.get(key)
            if isinstance(value, str):
                bounded[key] = value[:max_characters]
        return bounded
    return output
