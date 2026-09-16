"""Atlas request planner.

The planner turns a :class:`StructuredIntent` into the *smallest valid*
:class:`ExecutionPlan`. It is deterministic by default and only consults the
local language model for compositional requests. When it does consult the LLM,
the model may only select from the capability catalog; it never emits commands.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from config import ENABLE_LLM_INTERPRETATION
from models import ExecutionPlan, ExecutionStep, PlannerDecision, StructuredIntent
from reasoning.json_llm import safe_reasoning_call
from tools.capabilities import PLANNABLE_CAPABILITIES, render_capability_catalog
from tools.powershell_planner import PowerShellTaskPlanner

logger = logging.getLogger(__name__)

EXECUTABLE_RE = re.compile(r"(?:[A-Za-z]:[\\/][^\n]*?\.exe|[^\s]+\.exe)", re.IGNORECASE)

#: Intents that must generate text before writing it to a destination.
_TEXT_CONTENT_INTENTS = {"write_content", "creative_generation"}
#: Remote site names that web.search handles itself (no explicit site filter).
_DEFAULT_SITES = {"the web", "the internet", "google"}


class Planner:
    """Create execution plans from structured intents."""

    def __init__(
        self,
        *,
        registry: object | None = None,
        ask: Callable[..., str] | None = None,
    ) -> None:
        self._registry = registry
        self._ask = ask

    def create_plan(
        self,
        user_question: str,
        *,
        intent: str | None = None,
        structured: StructuredIntent | None = None,
    ) -> PlannerDecision:
        """Create an execution plan.

        Backward compatible: callers that only have a coarse ``intent`` string
        still work exactly as before.
        """

        if structured is not None:
            return self._plan_structured(user_question, structured)
        return self._plan_legacy(user_question, intent or "UNKNOWN")

    # -- structured planning -----------------------------------------------------

    def _plan_structured(self, user_question: str, structured: StructuredIntent) -> PlannerDecision:
        intent = structured.intent
        if intent == "search":
            return self._plan_search(user_question, structured)
        if intent in _TEXT_CONTENT_INTENTS:
            return self._plan_content(user_question, structured)
        if intent == "open_application":
            return self._plan_launch(user_question, structured)
        if intent in {"system_operation", "file_operation"}:
            legacy = self._plan_legacy(user_question, "COMPUTER")
            if legacy.strategy != "deterministic_unknown":
                return legacy
        return self._plan_legacy(user_question, "UNKNOWN")

    def _plan_search(self, user_question: str, structured: StructuredIntent) -> PlannerDecision:
        query = structured.query or structured.topic or user_question
        parameters: dict[str, object] = {"query": query, "max_results": 8}
        if structured.target and structured.target not in _DEFAULT_SITES:
            parameters["site"] = structured.target
        if structured.sort:
            parameters["sort"] = structured.sort
        if structured.filters:
            parameters["filters"] = list(structured.filters)
        if structured.content_type:
            parameters["content_type"] = structured.content_type
        plan = ExecutionPlan(
            user_question=user_question,
            steps=[
                ExecutionStep(
                    id="search_public_web",
                    name="Search Public Web",
                    action="invoke_tool",
                    description="Search current public web sources and return attributed links.",
                    metadata={
                        "tool": "web.search",
                        "parameters": parameters,
                        "capability": "current web research",
                    },
                )
            ],
        )
        return PlannerDecision(
            plan=plan,
            strategy="deterministic_web_search",
            confidence=structured.confidence,
            metadata={"step_count": 1, "intent": structured.intent},
        )

    def _plan_content(self, user_question: str, structured: StructuredIntent) -> PlannerDecision:
        """Plan content generation, optionally writing the result to an app."""

        destination = structured.destination
        steps: list[ExecutionStep] = [
            ExecutionStep(
                id="generate_content",
                name="Generate Requested Content",
                action="invoke_tool",
                description="Generate the requested content from its modifiers.",
                metadata={
                    "tool": "content.generate",
                    "parameters": {
                        "content_type": structured.content_type or "text",
                        "topic": structured.topic,
                        "tone": structured.tone,
                        "style": structured.style,
                        "length": structured.length,
                    },
                    "capability": "content generation",
                    "produces": "generated",
                },
            )
        ]
        if destination:
            steps.append(
                ExecutionStep(
                    id="write_application_text",
                    name="Write Text In Application",
                    action="invoke_tool",
                    description="Open the requested application and enter the generated text.",
                    metadata={
                        "tool": "applications.write_text",
                        "parameters": {
                            "application": destination,
                            "text": "$generated_text",
                        },
                        "capability": "application text entry",
                    },
                )
            )
            strategy = "deterministic_application_text_entry"
        else:
            steps.append(
                ExecutionStep(
                    id="finalize_content",
                    name="Return Generated Content",
                    action="finalize_content",
                    description="Return the generated text as the final response.",
                    metadata={"source": "$generated_text"},
                )
            )
            strategy = "deterministic_content_generation"
        plan = ExecutionPlan(user_question=user_question, steps=steps)
        return PlannerDecision(
            plan=plan,
            strategy=strategy,
            confidence=structured.confidence,
            metadata={"step_count": len(steps), "intent": structured.intent},
        )

    def _plan_launch(self, user_question: str, structured: StructuredIntent) -> PlannerDecision:
        executable_match = EXECUTABLE_RE.search(user_question)
        executable = executable_match.group(0).strip('"\'') if executable_match else ""
        if executable:
            steps = [
                ExecutionStep(
                    id="launch_application",
                    name="Launch Application",
                    action="invoke_tool",
                    description="Launch the explicitly selected executable after permission approval.",
                    metadata={
                        "tool": "applications.launch",
                        "parameters": {"executable": executable},
                    },
                )
            ]
            strategy = "deterministic_application_launch"
        else:
            application = structured.destination or structured.target or _named_application(user_question)
            steps = [
                ExecutionStep(
                    id="launch_named_application",
                    name="Launch Named Application",
                    action="invoke_tool",
                    description="Resolve the requested application to a trusted executable and launch it.",
                    metadata={
                        "tool": "applications.launch_named",
                        "parameters": {"application": application},
                    },
                )
            ]
            strategy = "deterministic_named_application_launch"
        plan = ExecutionPlan(user_question=user_question, steps=steps)
        return PlannerDecision(
            plan=plan,
            strategy=strategy,
            confidence=structured.confidence,
            metadata={"step_count": len(steps), "intent": structured.intent},
        )

    def plan_with_llm(
        self,
        user_question: str,
        structured: StructuredIntent,
        *,
        history: str = "",
    ) -> Optional[PlannerDecision]:
        """Ask the LLM for a plan, validated against the capability catalog.

        Returns ``None`` when the model is unavailable or produces something
        that is not a valid, catalog-only plan. Used for genuinely novel
        compositional requests that the rule table cannot already express.
        """

        if not ENABLE_LLM_INTERPRETATION or self._ask is None:
            return None
        from reasoning.prompts import PLANNER_SYSTEM, planner_user_prompt

        catalog = render_capability_catalog(
            self._registry if hasattr(self._registry, "list_metadata") else None
        )
        data = safe_reasoning_call(
            system_prompt=PLANNER_SYSTEM,
            user_prompt=planner_user_prompt(
                intent_json=json.dumps(structured.to_dict(), ensure_ascii=False),
                capabilities=catalog,
                history=history,
            ),
            ask=self._ask,
        )
        if not data:
            return None
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return None
        steps: list[ExecutionStep] = []
        for index, raw in enumerate(raw_steps, start=1):
            if not isinstance(raw, dict):
                return None
            capability = str(raw.get("capability", "")).strip()
            if capability not in PLANNABLE_CAPABILITIES:
                logger.warning("LLM planner proposed disallowed capability: %s", capability)
                return None
            arguments = raw.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            steps.append(
                ExecutionStep(
                    id=f"llm_step_{index}",
                    name=str(raw.get("description") or capability),
                    action="invoke_tool",
                    description=str(raw.get("description") or ""),
                    metadata={"tool": capability, "parameters": arguments, "planned_by": "llm"},
                )
            )
        plan = ExecutionPlan(user_question=user_question, steps=steps)
        return PlannerDecision(
            plan=plan,
            strategy="llm_assisted_plan",
            confidence=float(data.get("confidence") or 0.6),
            metadata={"step_count": len(steps), "intent": structured.intent},
        )

    # -- legacy deterministic planning (kept for compatibility) ------------------

    def _plan_legacy(self, user_question: str, intent: str) -> PlannerDecision:
        if intent == "COMPUTER":
            powershell_plan = PowerShellTaskPlanner().create(user_question)
            if powershell_plan is not None:
                steps = [
                    ExecutionStep(
                        id="discover_powershell_tool",
                        name="Discover PowerShell Tool",
                        action="invoke_tool",
                        description="Select a documented PowerShell capability for the requested inspection.",
                        metadata={
                            "tool": "powershell.execute",
                            "parameters": {"command": powershell_plan.command},
                            "capability": powershell_plan.capability,
                            "candidates": [candidate.tool for candidate in powershell_plan.candidates],
                        },
                    )
                ]
                plan = ExecutionPlan(user_question=user_question, steps=steps)
                return PlannerDecision(
                    plan=plan,
                    strategy="knowledge_backed_powershell_selection",
                    confidence=0.9,
                    metadata={"step_count": len(steps), "intent": intent},
                )

        if intent == "WEB_SEARCH":
            query = user_question
            if re.search(r"\b(latest|newest|recent)\s+video\b", user_question, re.IGNORECASE):
                query = user_question
            plan = ExecutionPlan(
                user_question=user_question,
                steps=[
                    ExecutionStep(
                        id="search_public_web",
                        name="Search Public Web",
                        action="invoke_tool",
                        description="Search current public web sources and return attributed links.",
                        metadata={
                            "tool": "web.search",
                            "parameters": {"query": query, "max_results": 8},
                            "capability": "current web research",
                        },
                    )
                ],
            )
            return PlannerDecision(
                plan=plan,
                strategy="read_only_web_search",
                confidence=0.86,
                metadata={"step_count": 1, "intent": intent},
            )

        # Plan templates are deterministic rule outputs.
        if intent == "APPLICATION":
            executable_match = EXECUTABLE_RE.search(user_question)
            executable = executable_match.group(0).strip('"\'') if executable_match else ""
            if not executable:
                application = _named_application(user_question)
                steps = [
                    ExecutionStep(
                        id="launch_named_application",
                        name="Launch Named Application",
                        action="invoke_tool",
                        description="Resolve the requested application to a trusted executable and launch it after approval.",
                        metadata={
                            "tool": "applications.launch_named",
                            "parameters": {"application": application},
                        },
                    )
                ]
                strategy = "deterministic_named_application_launch"
            else:
                steps = [
                    ExecutionStep(
                        id="launch_application",
                        name="Launch Application",
                        action="invoke_tool",
                        description="Launch the explicitly selected executable after permission approval.",
                        metadata={
                            "tool": "applications.launch",
                            "parameters": {"executable": executable},
                        },
                    )
                ]
                strategy = "deterministic_application_launch"

        elif intent == "WRITE_APPLICATION":
            application = _extract_write_application(user_question)
            text = _extract_requested_text(user_question)
            steps = [
                ExecutionStep(
                    id="write_application_text",
                    name="Write Text In Application",
                    action="invoke_tool",
                    description="Open the requested application and enter the planned text after approval.",
                    metadata={
                        "tool": "applications.write_text",
                        "parameters": {"application": application, "text": text},
                        "capability": "application text entry",
                    },
                )
            ]
            strategy = "deterministic_application_text_entry"

        elif intent == "COMPARE":
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


def _named_application(request: str) -> str:
    match = re.search(r"\b(?:open|launch|start)\s+(.+)$", request, re.IGNORECASE)
    return match.group(1).strip().strip('"\'') if match else ""


def _extract_write_application(request: str) -> str:
    match = re.search(r"\b(?:in|into)\s+([A-Za-z0-9 ._-]+?)(?:\s+(?:app|application))?(?:\s*$|\s+with\s+|\s*:\s*)", request, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Notepad"


def _extract_requested_text(request: str) -> str:
    quoted = re.search(r"[\"'](.+?)[\"']", request, re.DOTALL)
    if quoted:
        return quoted.group(1)
    return request.strip()
