"""Deterministic Atlas request planner."""

from __future__ import annotations

import logging

from models import ExecutionPlan, ExecutionStep, PlannerDecision

logger = logging.getLogger(__name__)


class Planner:
    """Create execution plans without using the LLM."""

    def create_plan(self, user_question: str, *, intent: str | None = None) -> PlannerDecision:
        """Create a deterministic execution plan based on classified intent."""

        intent = intent or "UNKNOWN"

        # Plan templates are deterministic rule outputs.
        # Even before retrieval strategies diverge, the *plan shape* differs.
        if intent == "COMPARE":
            steps = [
                ExecutionStep(
                    id="retrieve_compare_left",
                    name="Retrieve Compare (Left)",
                    action="retrieve_knowledge",
                    description="Retrieve supporting knowledge for the first part of a comparison.",
                    metadata={"compare_side": "left"},
                ),
                ExecutionStep(
                    id="retrieve_compare_right",
                    name="Retrieve Compare (Right)",
                    action="retrieve_knowledge",
                    description="Retrieve supporting knowledge for the second part of a comparison.",
                    metadata={"compare_side": "right"},
                ),
                ExecutionStep(
                    id="merge_evidence",
                    name="Merge Evidence",
                    action="merge_evidence",
                    description="Merge evidence from both retrievals into a unified context.",
                ),
                ExecutionStep(
                    id="generate_response",
                    name="Generate Response",
                    action="generate_response",
                    description="Generate a comparison answer from retrieved evidence.",
                ),
            ]
            strategy = "deterministic_compare"

        elif intent == "COUNT":
            steps = [
                ExecutionStep(
                    id="retrieve_count",
                    name="Retrieve Relevant Section",
                    action="retrieve_knowledge",
                    description="Retrieve relevant knowledge to count entities mentioned in documents.",
                    metadata={"retrieval_strategy": "expand_section_like"},
                ),
                ExecutionStep(
                    id="generate_response",
                    name="Generate Response",
                    action="generate_response",
                    description="Generate a count answer from retrieved evidence.",
                ),
            ]
            strategy = "deterministic_count"

        elif intent == "SUMMARIZE":
            steps = [
                ExecutionStep(
                    id="retrieve_summary",
                    name="Retrieve Chapter",
                    action="retrieve_knowledge",
                    description="Retrieve complete context for summarization.",
                    metadata={"retrieval_strategy": "collect_complete_context"},
                ),
                ExecutionStep(
                    id="generate_response",
                    name="Generate Response",
                    action="generate_response",
                    description="Generate a summary from retrieved evidence.",
                ),
            ]
            strategy = "deterministic_summarize"

        elif intent in {"LIST", "FACT", "PERSON", "EXPLAIN", "DEFINITION", "DATE", "LOCATION", "PROCEDURE"}:
            steps = [
                ExecutionStep(
                    id="retrieve_knowledge",
                    name="Retrieve Knowledge",
                    action="retrieve_knowledge",
                    description="Retrieve supporting knowledge from the local knowledge base.",
                    metadata={"retrieval_strategy": intent.lower()},
                ),
                ExecutionStep(
                    id="generate_response",
                    name="Generate Response",
                    action="generate_response",
                    description="Generate an answer from retrieved evidence.",
                ),
            ]
            strategy = f"deterministic_{intent.lower()}"

        else:
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
            strategy = "deterministic_unknown"

        plan = ExecutionPlan(user_question=user_question, steps=steps)
        decision = PlannerDecision(
            plan=plan,
            strategy=strategy,
            metadata={"step_count": len(steps), "intent": intent},
        )


        logger.info(
            "Planner created plan: plan_id=%s steps=%d strategy=%s",
            plan.plan_id,
            len(plan.steps),
            decision.strategy,
        )
        return decision

