"""Atlas Brain orchestration layer."""

from __future__ import annotations

import logging
import time

from executor import Executor, UNKNOWN_RESPONSE
from intent_classifier import IntentClassifier

from models import ExecutionContext
from planner import Planner
from vector_store import VectorStore

from memory.memory_manager import MemoryManager

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
    ) -> None:
        self._planner = planner or Planner()
        self._executor = executor or Executor(
            vector_store=vector_store,
            system_prompt=system_prompt,
            retrieval_template=retrieval_template,
            memory_manager=memory_manager,
        )


    def process(self, user_input: str) -> str:
        """Process a user request through Planner and Executor."""

        started_at = time.perf_counter()
        context = ExecutionContext(user_input=user_input, normalized_input=user_input.strip())

        logger.info("Brain received request")

        try:
            intent_result = IntentClassifier().classify(context.normalized_input or user_input)
            context.intent = intent_result.intent
            context.metadata["intent_confidence"] = intent_result.confidence
            context.metadata["intent_rationale"] = intent_result.rationale
            context.metadata["intent_signals"] = intent_result.signals

            logger.info(
                "Brain intent classified intent=%s confidence=%.2f rationale=%s",
                context.intent,
                intent_result.confidence,
                intent_result.rationale,
            )

            planner_decision = self._planner.create_plan(context.normalized_input or user_input, intent=context.intent)

            context.execution_plan = planner_decision.plan
            context.metadata["planner_decision"] = planner_decision
            self._executor.execute(planner_decision.plan, context)
            return self._complete(context, started_at)

        except Exception:
            logger.exception("Brain execution failed")
            context.final_response = UNKNOWN_RESPONSE
            return self._complete(context, started_at)

    def _complete(self, context: ExecutionContext, started_at: float) -> str:
        context.execution_time = time.perf_counter() - started_at
        if context.execution_plan is not None:
            context.execution_plan.metadata["brain_execution_time"] = context.execution_time

        logger.info(
            "Brain completed request: execution_time=%.4fs selected_tool=%s plan_status=%s",
            context.execution_time,
            context.selected_tool,
            context.execution_plan.status.value if context.execution_plan is not None else "none",
        )
        return context.final_response or UNKNOWN_RESPONSE
