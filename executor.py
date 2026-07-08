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
)
from vector_store import VectorStore

logger = logging.getLogger(__name__)

UNKNOWN_RESPONSE = "I don't know based on my knowledge base."


class Executor:
    """Execute Atlas plans step-by-step."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        system_prompt: str,
        retrieval_template: str,
    ) -> None:
        self._vector_store = vector_store
        self._system_prompt = system_prompt
        self._retrieval_template = retrieval_template
        self._handlers: dict[str, Callable[[ExecutionContext, ExecutionStep], None]] = {
            "retrieve_knowledge": self._retrieve_knowledge,
            "generate_response": self._generate_response,
        }

    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> ExecutionContext:
        """Execute a plan sequentially and update the shared context."""

        started_at = time.perf_counter()
        context.execution_plan = plan
        plan.status = PlanStatus.RUNNING

        logger.info("Executor started plan: plan_id=%s steps=%d", plan.plan_id, len(plan.steps))

        for step in plan.steps:
            if plan.status == PlanStatus.FAILED:
                step.status = StepStatus.SKIPPED
                logger.info(
                    "Executor skipped step: plan_id=%s step_id=%s action=%s",
                    plan.plan_id,
                    step.id,
                    step.action,
                )
                continue

            self._execute_step(context, step)

        if plan.status != PlanStatus.FAILED:
            plan.status = PlanStatus.COMPLETED

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
            logger.exception("Executor failed step: plan_id=%s step_id=%s action=%s", plan_id, step.id, step.action)

    def _retrieve_knowledge(self, context: ExecutionContext, step: ExecutionStep) -> None:
        context.selected_tool = "knowledge"
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
        context.evidence = Evidence(
            context=retrieval_result.context,
            chunks=retrieval_result.retrieved_chunks,
            sources=_unique_sources(retrieval_result.retrieved_chunks),
            metadata={"retrieval_metric": "chroma_distance_smaller_is_better"},
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

        user_prompt = self._retrieval_template.format(
            context=retrieval_result.context,
            question=context.user_input,
        )

        logger.info("LLM called")
        context.llm_response = ask(system_prompt=self._system_prompt, user_prompt=user_prompt)
        context.final_response = context.llm_response
        step.result = context.final_response
        logger.info("LLM complete")

    def _mark_failed(self, context: ExecutionContext) -> None:
        if context.execution_plan is not None:
            context.execution_plan.status = PlanStatus.FAILED
        context.final_response = UNKNOWN_RESPONSE


def _unique_sources(chunks: list[object]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        source = getattr(chunk, "source", "")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)
    return sources
