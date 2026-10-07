"""Structured per-request diagnostics for the Atlas pipeline.

When ``DEBUG_PIPELINE`` is enabled, every stage of a request is recorded and
emitted as a single readable block so a failure can be explained end-to-end:

    [INTENT] ... [PLAN] ... [VALIDATION] ... [EXECUTION] ... [RESULT]

Sensitive values are redacted before anything is logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import REDACT_KEYS
from models import ExecutionContext, PlannerDecision, StructuredIntent

logger = logging.getLogger(__name__)

_REDACTED = "***redacted***"


def redact(value: Any) -> Any:
    """Recursively redact configured sensitive keys from a structure."""

    if isinstance(value, dict):
        return {
            str(key): (_REDACTED if str(key).casefold() in REDACT_KEYS else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


@dataclass
class PipelineTrace:
    """Accumulate the stages of a single request for debugging.

    Stages mirror the live pipeline:
    USER -> TASK -> VALIDATION -> PLAN -> EXECUTION -> VERIFICATION -> RESPONSE.
    """

    user_input: str
    intent: dict[str, Any] = field(default_factory=dict)
    intent_reading: dict[str, Any] = field(default_factory=dict)
    #: The open-ended semantic reading + evidence requirement + selected
    #: capabilities. This is the stage that replaces rigid intent classification.
    semantic: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    plan: list[dict[str, Any]] = field(default_factory=list)
    execution: list[dict[str, Any]] = field(default_factory=list)
    verification: list[dict[str, Any]] = field(default_factory=list)
    response: str | None = None

    def record_task(self, task: dict[str, Any]) -> None:
        self.task = task
    def record_semantic(self, trace: dict[str, Any]) -> None:
        """Record the semantic reasoning trace (structured, never chain-of-thought)."""

        self.semantic = trace
    def record_intent_reading(self, reading: dict[str, Any]) -> None:
        """Record the Intent Engine's structured reading (never chain-of-thought)."""

        self.intent_reading = reading
    def record_validation(self, validation: dict[str, Any]) -> None:
        self.validation = validation
    def record_intent(self, structured: StructuredIntent) -> None:
        self.intent = {
            "intent": structured.intent,
            "action": structured.action,
            "target": structured.target,
            "content_type": structured.content_type,
            "topic": structured.topic,
            "query": structured.query,
            "destination": structured.destination,
            "tone": structured.tone,
            "length": structured.length,
            "sort": structured.sort,
            "confidence": structured.confidence,
            "source": structured.source,
        }

    def record_plan(self, decision: PlannerDecision) -> None:
        self.plan = [
            {
                "id": step.id,
                "action": step.action,
                "tool": step.metadata.get("tool"),
                "parameters": step.metadata.get("parameters"),
            }
            for step in decision.plan.steps
        ]

    def record_execution(self, context: ExecutionContext) -> None:
        self.execution = [
            {
                "tool": call.get("tool"),
                "parameters": call.get("parameters"),
                "status": call.get("status"),
                "success": call.get("success"),
                "error": call.get("error"),
            }
            for call in context.tool_calls
        ]
        self.verification = list(context.verification_results)

    def record_response(self, response: str | None) -> None:
        self.response = response

    def log(self) -> None:
        """Emit the whole trace as one structured, redacted block."""

        lines = ["Atlas pipeline trace", f"USER INPUT: {self.user_input}"]
        if self.intent_reading:
            lines.append("INTENT READING: " + _dump(self.intent_reading))
        if self.semantic:
            # The semantic reading is the primary diagnosis: it says whether Atlas
            # understood the goal, what evidence it judged necessary, and which
            # capabilities it selected - so a failure can be attributed to
            # understanding, capability choice, retrieval, or generation.
            lines.append("SEMANTIC READING: " + _dump(self.semantic))
        if self.task:
            lines.append("TASK: " + _dump(self.task))
            # A compact, human-readable topic decomposition makes an
            # interpretation problem distinguishable from an execution problem.
            reading = self.task.get("topic_reading") or {}
            if reading:
                lines.append(
                    "TOPIC DECOMPOSITION: "
                    + _dump({
                        "raw_topic": reading.get("raw_topic"),
                        "connector": reading.get("connector"),
                        "stripped_head": reading.get("stripped_head"),
                        "content_noun": reading.get("content_noun"),
                        "subject": reading.get("subject"),
                        "normalized": reading.get("normalized"),
                        "research_query": self.task.get("research_query"),
                    })
                )
        if self.validation:
            lines.append("VALIDATION: " + _dump(self.validation))
        if self.intent:
            lines.append("INTENT: " + _dump(self.intent))
        if self.plan:
            lines.append("PLAN:")
            for index, step in enumerate(self.plan, start=1):
                lines.append(f"  {index}. {_dump(step)}")
        if self.execution:
            lines.append("EXECUTION:")
            for index, call in enumerate(self.execution, start=1):
                lines.append(f"  {index}. {_dump(call)}")
        if self.verification:
            lines.append("VERIFICATION:")
            for index, item in enumerate(self.verification, start=1):
                lines.append(f"  {index}. {_dump(item)}")
        if self.response is not None:
            lines.append(f"RESULT: {self.response}")
        logger.info("\n".join(lines))


def _dump(value: Any) -> str:
    import json

    return json.dumps(redact(value), ensure_ascii=False, default=str)
