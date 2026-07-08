"""Deterministic Atlas request planner."""

from __future__ import annotations

import logging

from models import ExecutionPlan, ExecutionStep, PlannerDecision

logger = logging.getLogger(__name__)


class Planner:
    """Create execution plans without using the LLM."""

    def create_plan(self, user_question: str) -> PlannerDecision:
        """Return the initial deterministic plan for every request."""

        steps = [
            ExecutionStep(
                id="retrieve_knowledge",
                name="Retrieve Knowledge",
                action="retrieve_knowledge",
                description="Retrieve supporting knowledge from the local knowledge base.",
            ),
            ExecutionStep(
                id="generate_response",
                name="Generate Response",
                action="generate_response",
                description="Generate an answer from retrieved evidence.",
            ),
        ]
        plan = ExecutionPlan(user_question=user_question, steps=steps)
        decision = PlannerDecision(
            plan=plan,
            strategy="deterministic_two_step_rag",
            metadata={"step_count": len(steps)},
        )

        logger.info(
            "Planner created plan: plan_id=%s steps=%d strategy=%s",
            plan.plan_id,
            len(plan.steps),
            decision.strategy,
        )
        return decision

