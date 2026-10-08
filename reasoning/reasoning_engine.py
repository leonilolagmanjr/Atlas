from __future__ import annotations

import logging
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from config import (
    COMPUTER_ROOT, ENABLE_GENERAL_QUESTION_FALLBACK, FILESYSTEM_CONTENT_MAX_BYTES, FILESYSTEM_CONTENT_MAX_FILES,
    MAX_REASONING_ITERATIONS, MAX_REASONING_SOURCES, MAX_REASONING_TOOL_CALLS,
    REASONING_TIMEOUT_SECONDS, WEB_RESEARCH_MAX_PAGES, WEB_RESEARCH_MAX_RESULTS,
)
from knowledge_search import retrieve
from models_task import Task, EvidenceState, EvidenceSource
from reasoning.answer_generator import Answer, AnswerGenerator
from reasoning.evidence_gate import EvidenceGate
from reasoning.evidence_manager import EvidenceManager
from reasoning.query_router import QueryRouter, RoutingSignals
from reasoning.reasoning_models import (
    ReasoningDecision, ReasoningStage, ReasoningTrace,
    RequestType, ResponseMode, SourceType,
)
from reasoning.self_introspection import SelfIntrospection
from reasoning.source_selector import SourceSelector
from reasoning.temporal_resolution import TemporalResolver, TemporalResolution
from reasoning.topic_extraction import normalize_query
from web_task import reformulate_query, validate_content, detect_source_type, RetrievalTask
#: Read-only research capabilities the reasoning engine performs itself while
#: synthesizing a cited answer: web lookups and non-mutating file inspection.
#: A task whose actions are all drawn from this set is served as research here;
#: anything else (mutations, application control, content generation) is
#: delegated to the deterministic execution pipeline.
_RESEARCH_ONLY_ACTIONS = frozenset({
    "web.search", "web.fetch", "web.research",
    "filesystem.list", "filesystem.search", "filesystem.read",
    "filesystem.metadata", "filesystem.search_content",
})

logger = logging.getLogger(__name__)
ToolRunner = Callable[[str, Mapping[str, Any]], Any]
REASONING_DEADLINE_SECONDS = REASONING_TIMEOUT_SECONDS
_READ_TOOLS = frozenset({
    "filesystem.list", "filesystem.search", "filesystem.read", "filesystem.metadata",
    "filesystem.search_content", "web.search", "web.fetch", "web.research",
    "system.info", "processes.list",
    # Read-only UI perception tools are safe for the reasoning loop.
    "computer.windows", "computer.observe",
    "computer.vision_observe", "computer.find",
})


@dataclass
class _Run:
    deadline: float
    cancelled: Callable[[], bool]
    trace: ReasoningTrace
    tool_limit: int
    tool_calls: int = 0
    selected_tool: str | None = None
    notes: list[str] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    #: Optional listener for real progress, supplied by the conversation stream.
    on_event: Callable[[str, dict[str, Any]], None] | None = None

    def active(self) -> bool:
        if self.cancelled():
            self.notes.append("request cancelled")
            return False
        if time.monotonic() >= self.deadline:
            self.notes.append("reasoning deadline reached")
            return False
        return True

    def emit(self, event_type: str, **payload: Any) -> None:
        """Report real activity to an interested client, never failing the run."""

        if self.on_event is None:
            return
        try:
            self.on_event(event_type, payload)
        except Exception:  # noqa: BLE001 - progress reporting is not reasoning
            logger.exception("Reasoning event listener failed: %s", event_type)


_run: ContextVar[_Run | None] = ContextVar("atlas_reasoning_run", default=None)


class ReasoningEngine:
    def __init__(
        self, *, self_introspection: SelfIntrospection, answer_generator: AnswerGenerator,
        tool_runner: ToolRunner, capabilities: Any, retrieve: Callable[..., Any] = retrieve,
        max_iterations: int = MAX_REASONING_ITERATIONS,
        max_tool_calls: int = MAX_REASONING_TOOL_CALLS,
        max_sources: int = MAX_REASONING_SOURCES,
        web_max_results: int = WEB_RESEARCH_MAX_RESULTS,
        web_max_pages: int = WEB_RESEARCH_MAX_PAGES,
        content_max_files: int = FILESYSTEM_CONTENT_MAX_FILES,
        content_max_bytes: int = FILESYSTEM_CONTENT_MAX_BYTES,
        timeout_seconds: float = REASONING_DEADLINE_SECONDS,
        filesystem_root: Path = COMPUTER_ROOT,
        allow_general_fallback: bool = ENABLE_GENERAL_QUESTION_FALLBACK,
    ) -> None:
        self._introspection = self_introspection
        self._answers = answer_generator
        self._run_tool = tool_runner
        self._retrieve = retrieve
        self._capabilities = capabilities
        self._router = QueryRouter()
        self._selector = SourceSelector(capabilities)
        self._evidence_gate = EvidenceGate()
        self._max_iterations = max(1, int(max_iterations))
        self._max_tool_calls = max(0, int(max_tool_calls))
        self._max_sources = max(1, int(max_sources))
        self._web_max_results = max(1, int(web_max_results))
        self._web_max_pages = max(0, int(web_max_pages))
        self._content_max_files = max(1, int(content_max_files))
        self._content_max_bytes = max(1, int(content_max_bytes))
        self._timeout_seconds = max(0.0, float(timeout_seconds))
        self._filesystem_root = Path(filesystem_root)
        self._allow_general_fallback = bool(allow_general_fallback)

    def handle_request(
        self, *, question: str, task: Task, prior_task: Task | None = None,
        history: str = "", cancelled: Callable[[], bool] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> Answer | None:
        run = _Run(time.monotonic() + self._timeout_seconds, cancelled or (lambda: False),
                   ReasoningTrace(), self._max_tool_calls, on_event=on_event)
        token = _run.set(run)
        try:
            return self._handle(question, task, prior_task, history, run)
        finally:
            _run.reset(token)

    def _handle(self, question: str, task: Task, prior_task: Task | None,
                history: str, run: _Run) -> Answer | None:
        evidence = EvidenceManager()
        # Use task's evidence_state if available, otherwise create one
        evidence_state = getattr(task, 'evidence_state', None)
        if evidence_state is None:
            from models_task import EvidenceState
            evidence_state = EvidenceState(
                target=task.entities.get("topic") or "",
                goal="find_information",
                required_content_type=task.entities.get("content_type", "generic"),
                must_be_artifact=False,
            )
        # Store on task for answer generator
        task.evidence_state = evidence_state
        
        # Resolve temporal expressions once, before source gathering.  The
        # resolution is cheap, deterministic, and authoritative (system clock).
        # Downstream sources (web search, evidence evaluation) read it from the
        # task context so they all share the same temporal frame.
        resolver = TemporalResolver()
        temporal_resolution = resolver.resolve_from_structure(
            question,
            final_event_result=bool(getattr(task, "semantic_reading", {}).get("final_event_result")),
            freshness=getattr(task, "semantic_reading", {}).get("freshness", "any"),
            superlative=bool(getattr(task, "comparative", False)),
        )
        task.context["temporal_resolution"] = temporal_resolution.to_dict()
        run.trace.record(ReasoningStage.UNDERSTANDING, "temporal resolution",
                         temporal=temporal_resolution.to_dict())
        
        signals = self._router.route(question, task=task, prior_task=prior_task,
                                     history=history, self_introspection=self._introspection)
        plan = self._selector.select(signals, task_type=task.task_type)
        decision = self._decide(plan, signals, question)
        decision.actions = [action.capability for action in task.actions]
        decision.llm_used = task.source == "llm"
        decision.multi_step = len(task.actions) > 1 or len(plan.sources) > 1
        decision.requires_verification = bool(task.actions)
        run.trace.record(ReasoningStage.UNDERSTANDING, decision.request_type.value)
        run.trace.record(ReasoningStage.SOURCE_SELECTION, plan.reason,
                         sources=[source.value for source in plan.sources])
        task.apply_decision(decision)
        task.context["reasoning"] = run.trace.to_list()
        if not run.active():
            return self._finish(self._answers.limitation(decision, notes=run.notes), decision, evidence, run, task)
        if decision.needs_clarification:
            answer = self._answers.clarification(task.clarification_question or decision.clarification_question)
            return self._finish(answer, decision, evidence, run, task)
        if decision.requires_computer or self._must_delegate(task):
            run.trace.record(ReasoningStage.PLANNING, "delegating validated actions to the execution pipeline")
            task.context["reasoning"] = run.trace.to_list()
            task.context["reasoning_summary"] = run.trace.summary()
            return None
        if decision.response_mode is ResponseMode.SELF_DESCRIPTION:
            answer = self._answers.self_description(self._introspection.answer(signals.self_kind))
            return self._finish(answer, decision, evidence, run, task)

        mode = decision.response_mode
        for iteration, source in enumerate(plan.sources[:self._max_sources], 1):
            if iteration > self._max_iterations or not run.active():
                break
            run.trace.iteration = iteration
            if source is SourceType.MODEL:
                continue
            run.trace.record(ReasoningStage.RETRIEVING, "consulting " + source.value)
            gained = self._gather(source, question=question, task=task, history=history,
                                  evidence=evidence, signals=signals, evidence_state=evidence_state)
            run.trace.record(ReasoningStage.EVALUATING,
                             "evidence collected" if gained else "source supplied no usable evidence",
                             source=source.value, count=gained)
            candidate_mode = self._MODE_BY_SOURCE.get(source, mode)
            content_required = bool(source is SourceType.FILES and signals.file_intent
                                    and signals.file_intent.wants_read)
            if evidence.sufficient_for(candidate_mode, content_required=content_required):
                mode = candidate_mode
                break
            run.notes.append(source.value + ": no sufficient evidence")
            run.trace.record(ReasoningStage.PLANNING, "selecting next permitted source")

        # Deterministic evidence gate: after retrieval, check whether Atlas has
        # enough to answer without fabricating. This re-evaluates the semantic
        # layer's evidence decision against what was actually retrieved.
        gate_result = self._evidence_gate.evaluate(task=task, evidence_manager=evidence)
        if not gate_result.can_answer:
            answer = self._evidence_shortfall_answer(task, question, run)
            run.trace.record(ReasoningStage.EVALUATING,
                             "evidence gate blocked answer", status=gate_result.status)
            return self._finish(answer, decision, evidence, run, task)

        if not run.active():
            answer = self._answers.limitation(decision, notes=run.notes)
        elif mode is ResponseMode.MEMORY_RECALL:
            answer = self._answers.memory_recall(question, history)
        elif evidence.sufficient_for(mode, content_required=bool(
            mode is ResponseMode.FILE_LOOKUP and signals.file_intent and signals.file_intent.wants_read
        )) and mode is not ResponseMode.DIRECT_ANSWER:
            answer = self._answers.grounded(question, evidence, mode=mode, history=history, task=task)
        elif any(source in decision.sources for source in (
            SourceType.FILES, SourceType.SYSTEM, SourceType.MEMORY, SourceType.CONVERSATION,
        )):
            answer = self._answers.limitation(decision, notes=run.notes)
        elif gate_result.must_disclose and SourceType.MODEL in decision.sources:
            # When the evidence gate says "must disclose" (insufficient or
            # conflicting evidence), do NOT generate a model-driven answer even
            # with a caveat. When the request depends on current facts, a caveat
            # in the answer text is not a verification gate — the user asked for
            # a dynamic fact and we do not have one we can verify.
            if decision.current_information_required:
                answer = self._answers.limitation(
                    decision,
                    notes=list(run.notes) + [
                        "current information required but not verified"
                    ],
                )
                answer.metadata["answerability"] = gate_result.answerability.status
                return self._finish(answer, decision, evidence, run, task)
            note = "; ".join(run.notes)
            answer = self._answers.direct(question, history=history, fallback_note=note)
            answer.metadata["answerability"] = "must_disclose_with_caveat"
            answer.metadata["notes"] = run.notes
        elif (
            SourceType.MODEL in decision.sources
            and self._allow_general_fallback
            and not decision.current_information_required
        ):
            answer = self._answers.direct(question, history=history, fallback_note="; ".join(run.notes))
        else:
            answer = self._answers.limitation(decision, notes=run.notes)
        if not run.active():
            answer = self._answers.limitation(decision, notes=run.notes)
        return self._finish(answer, decision, evidence, run, task)

    _MODE_BY_SOURCE = {
        SourceType.SELF: ResponseMode.SELF_DESCRIPTION,
        SourceType.CONVERSATION: ResponseMode.MEMORY_RECALL,
        SourceType.MEMORY: ResponseMode.MEMORY_RECALL,
        SourceType.WEB: ResponseMode.WEB_RESEARCH,
        SourceType.FILES: ResponseMode.FILE_LOOKUP,
        SourceType.SYSTEM: ResponseMode.SYSTEM_DIAGNOSIS,
        SourceType.COMPUTER: ResponseMode.ACTION_REPORT,
        SourceType.KNOWLEDGE: ResponseMode.GROUNDED_ANSWER,
    }

    def _decide(self, plan: Any, signals: RoutingSignals, question: str) -> ReasoningDecision:
        sources = list(plan.sources)
        computer = SourceType.COMPUTER in sources
        clarification = not sources
        mode = self._MODE_BY_SOURCE.get(sources[0], ResponseMode.DIRECT_ANSWER) if sources else ResponseMode.CLARIFICATION
        request_type = RequestType.QUESTION
        if clarification:
            request_type = RequestType.CLARIFICATION
        elif computer:
            request_type = RequestType.HYBRID if len(sources) > 1 else RequestType.ACTION
        elif SourceType.SELF in sources:
            request_type = RequestType.SELF_QUERY
        elif SourceType.CONVERSATION in sources or SourceType.MEMORY in sources:
            request_type = RequestType.MEMORY_QUERY
        return ReasoningDecision(
            goal=question, request_type=request_type, sources=sources, source_plan=plan,
            requires_web=SourceType.WEB in sources, requires_files=SourceType.FILES in sources,
            requires_memory=SourceType.MEMORY in sources or SourceType.CONVERSATION in sources,
            requires_knowledge=SourceType.KNOWLEDGE in sources, requires_computer=computer,
            requires_system=SourceType.SYSTEM in sources, requires_self_introspection=SourceType.SELF in sources,
            requires_model_knowledge=SourceType.MODEL in sources,
            current_information_required=plan.current_information_required,
            requires_clarification=clarification, needs_clarification=clarification,
            clarification_question="Please specify the file, application, or topic you mean." if clarification else None,
            reason=plan.reason, response_mode=mode,
        )

    @staticmethod
    def _must_not_fabricate(task: Task) -> bool:
        """True when a generated answer would be a fabrication rather than an answer.

        A ranking/comparison over a criterion cannot be produced from plausible
        recall: the candidates and their ordering must be *retrieved*. Likewise a
        request the semantic layer judged to **require** external evidence must not
        fall back to model generation. A *preferred*-evidence request may fall back
        (with the limitation disclosed), because an honest grounded-in-model answer
        is still a legitimate answer to "who is X".
        """

        if bool(getattr(task, "comparative", False)):
            return True
        return str(getattr(task, "evidence_requirement", "unnecessary")) == "required"

    def _evidence_shortfall_answer(self, task: Task, question: str, run: _Run) -> Answer:
        """Answer honestly when required evidence could not be obtained.

        A subjective criterion with no objective proxy is surfaced as a question
        about the criterion; anything else states clearly that Atlas could not
        retrieve the evidence and therefore will not assert an answer.
        """

        reading = getattr(task, "semantic_reading", {}) or {}
        criterion = str(reading.get("criterion") or task.entities.get("criterion") or "")
        proxy = str(reading.get("criterion_proxy") or task.entities.get("criterion_proxy") or "")
        if criterion and not proxy and reading.get("subjective_criterion", False):
            answer = self._answers.clarification(
                f"'{criterion}' is subjective and there is no single measure for it. "
                "Which definition should I use - for example a specific metric, or the "
                "most recent ranking you trust?"
            )
            answer.metadata = dict(answer.metadata or {})
            answer.metadata["answerability"] = "ambiguous_subjective_criterion"
            return answer
        notes = list(run.notes) + [
            "required evidence was not obtained; refusing to state an unverified answer"
        ]
        answer = self._answers.clarification(
            "I could not retrieve the evidence needed to answer this reliably, and I do "
            "not want to guess - a plausible-sounding answer here would be fabricated. "
            "Please try again, or narrow which candidates or source you want me to use."
        )
        answer.metadata = dict(answer.metadata or {})
        answer.metadata["answerability"] = "insufficient_evidence"
        answer.metadata["notes"] = notes
        return answer

    @staticmethod
    def _must_delegate(task: Task) -> bool:
        # Delegate to the execution pipeline unless every action is a read-only
        # research step the reasoning engine performs itself. A pure web search
        # is answered here (synthesized, with citations); a web search
        # that also writes somewhere is a hybrid and carries non-research actions,
        # so it delegates. File reads/list/search delegate as actions too, because
        # the executor owns those observations.
        if not task.actions:
            return False
        return any(action.capability not in _RESEARCH_ONLY_ACTIONS for action in task.actions)

    def _gather(self, source: SourceType, *, question: str, task: Task, history: str,
                evidence: EvidenceManager, signals: RoutingSignals, evidence_state: EvidenceState) -> int:
        try:
            if source is SourceType.KNOWLEDGE:
                result = self._retrieve(question)
                value = result.get if isinstance(result, Mapping) else lambda key, default=None: getattr(result, key, default)
                diagnostics = value("diagnostics")
                final = getattr(diagnostics, "final_decision", None)
                if final is not None and not getattr(final, "accepted", False):
                    return 0
                chunks = value("retrieved_chunks", []) or []
                sources = value("sources", []) or []
                return evidence.add_knowledge(
                    context=value("context", "") or "", best_distance=value("best_distance"),
                    sources=sources or [str(hit.get("source", "") if isinstance(hit, Mapping)
                                           else getattr(hit, "source", "")) for hit in chunks],
                )
            if source is SourceType.WEB:
                # Only use iterative retrieval for tasks that explicitly need artifact retrieval
                # (web.research actions) or have must_be_artifact flag
                needs_iterative = False
                if task.actions:
                    for action in task.actions:
                        if action.capability == "web.research":
                            needs_iterative = True
                            break
                        if action.capability == "web.search" and action.parameters.get("must_be_artifact"):
                            needs_iterative = True
                            break
                if needs_iterative:
                    return self._iterative_web_retrieval(question, task, evidence, signals, evidence_state)
                else:
                    return self._gather_web(question, evidence, task=task, evidence_state=evidence_state)
            if source is SourceType.FILES:
                return self._gather_files(task, signals, evidence)
            if source is SourceType.SYSTEM:
                output = self._execute("system.info", {})
                return evidence.add_system_output("system.info", output) if output is not None else 0
            if source in (SourceType.CONVERSATION, SourceType.MEMORY):
                return evidence.add_conversation(history)
        except Exception:
            logger.exception("Reasoning source %s failed", source.value)
        return 0

    def _temporal_search_query(self, question: str, temporal_dict: dict[str, Any]) -> str:
        """Reformulate a search query using the resolved temporal context.

        The goal is to turn a natural-language question into a search-optimized
        query that includes the resolved time period.  For example:

            "Who won the most recent NBA Finals?"
            -> "2026 NBA Finals winner"
        """
        relation = str(temporal_dict.get("relation") or "").casefold()
        period = str(temporal_dict.get("resolved_period") or "").strip()
        completion = str(temporal_dict.get("completion_state") or "").casefold()
        expression = str(temporal_dict.get("expression") or "").strip()

        lowered = question.casefold().strip(" ?.")

        # 1. Strip leading question words and auxiliaries.
        core = re.sub(
            r"^(?:who|what|which|where|when|why|how|is|are|was|were|do|does|did|can|could|would|should)\s+(?:is\s+|are\s+|was\s+|were\s+|did\s+|do\s+|does\s+)?",
            "",
            lowered,
        ).strip()

        if not core:
            return question

        # 2. Calendar relations: the relation itself is the useful search term.
        if relation in {"today", "yesterday", "this_week", "this_month", "this_year",
                        "last_week", "last_month", "last_year", "last_night", "tonight",
                        "now", "as_of", "so_far"}:
            return relation.replace("_", " ")

        # 3. Remove the resolved temporal expression only for completed events,
        #    where the resolved period will replace it.  For superlatives like
        #    "latest iPhone", keep the expression because it is the query.
        if relation.endswith("_completed") or completion == "completed":
            if expression:
                expr_pat = re.escape(expression.casefold())
                core = re.sub(
                    r"\b(?:the\s+)?(?:" + expr_pat + r")(?:\s+(?:of|for|in|at|by|to)\s+\w+(?:\s+\w+){0,3})?\b",
                    "",
                    core,
                    flags=re.IGNORECASE,
                ).strip()

        # 4. Clean articles, possessives, and whitespace.
        core = re.sub(r"\s+", " ", core).strip()
        core = re.sub(r"\b(?:the|a|an)\s+", "", core, flags=re.IGNORECASE).strip()
        core = re.sub(r"'s\s+", " ", core, flags=re.IGNORECASE).strip()
        core = core.strip(" ?.")

        # 5. Event-result: reformulate as "[event] winner".
        if re.search(r"\b(?:won|win|winner|victory|championship|champion)\b", core, re.IGNORECASE):
            if re.search(r"\b(?:won|win)\b", core, re.IGNORECASE):
                # "won nba finals" -> "nba finals winner"
                event = re.sub(
                    r"^\b(?:won|win)\b\s*",
                    "",
                    core,
                    flags=re.IGNORECASE,
                ).strip()
            else:
                # "nba finals winner" -> "nba finals"
                event = re.sub(
                    r"\b(?:winner|victory|championship|champion)\b.*",
                    "",
                    core,
                    flags=re.IGNORECASE,
                ).strip()
            event = re.sub(r"\b(?:the|a|an)\s+", "", event, flags=re.IGNORECASE).strip()
            event = re.sub(r"\s+", " ", event).strip()
            if event:
                core = f"{event} winner"
            else:
                core = re.sub(r"\b(?:won|win)\b", "winner", core, flags=re.IGNORECASE)
            if period and period not in core and not re.search(r"\b20\d{2}\b", core):
                return f"{period} {core}"
            return core

        # 6. Superlative / current relations: keep the natural query.
        if relation in {"latest", "newest", "current", "currently", "most_recent"}:
            return core

        # 7. Generic fallback: prepend period if it adds information.
        if period and period not in core and not re.search(r"\b20\d{2}\b", core):
            return f"{period} {core}"
        return core

    def _gather_web(self, question: str, evidence: EvidenceManager, *, task: Task | None = None, evidence_state: EvidenceState | None = None) -> int:
        # Honour the interpreter's planned web.search parameters (site, cleaned
        # query, sort) when present: they carry more information than the raw
        # request text the engine would otherwise search for.
        parameters: dict[str, Any] = {"query": question, "max_results": self._web_max_results}
        if task is not None:
            for action in task.actions:
                if action.capability == "web.search":
                    parameters = dict(action.parameters)
                    parameters.setdefault("query", question)
                    parameters.setdefault("max_results", self._web_max_results)
                    break
        
        # If the task carries a temporal resolution, use it to formulate a
        # temporally-aware search query.  "Who won the most recent NBA Finals?"
        # becomes "2026 NBA Finals winner" - a much better search query.
        if task is not None and parameters.get("query") == question:
            temporal_dict = getattr(task, "context", {}).get("temporal_resolution", {}) or {}
            if temporal_dict.get("confidence", 0.0) > 0.5:
                # Prefer the interpreter's cleaned query when available; fall back
                # to normalizing the raw question so connector words do not leak
                # into the tool call.
                cleaned = (
                    getattr(task, "research_query", "") or
                    getattr(task, "normalized_topic", "") or
                    question
                )
                temporaled_query = self._temporal_search_query(cleaned, temporal_dict)
                if temporaled_query and temporaled_query != cleaned:
                    parameters = dict(parameters)
                    parameters["query"] = temporaled_query
                    if task is not None:
                        task.context.setdefault("temporal_resolution", {})["search_query"] = temporaled_query
        search = self._execute("web.search", parameters)
        if search is None:
            return 0
        gained = evidence.add_web_results(search)
        for item in search.get("results", [])[:self._web_max_pages]:
            if not isinstance(item, Mapping) or not item.get("url"):
                continue
            thumbnail_url = str(item.get("thumbnail_url") or "").strip()
            page = self._execute("web.fetch", {"url": str(item["url"])})
            if page is not None:
                gained += evidence.add_web_page(page, thumbnail_url=thumbnail_url)
                # Also add to evidence_state for synthesized answer
                if evidence_state is not None:
                    source_type = detect_source_type(page.get("url", ""), page.get("title", ""), page.get("text", ""))
                    search_title = str(item.get("title") or "").strip()
                    search_snippet = str(item.get("snippet") or "").strip()
                    fetched_title = str(page.get("title") or "").strip()
                    fetched_text = str(page.get("text") or "").strip()
                    
                    # For video content, the search result has the actual video info
                    # The fetched page is just YouTube boilerplate. Use search data.
                    is_video_source = source_type == "video"
                    
                    if is_video_source:
                        # Use search result title (cleaner) and snippet (has view counts, etc.)
                        title = search_title or fetched_title
                        # Build content from search snippet + any meaningful fetched content
                        content_parts = []
                        if search_snippet:
                            content_parts.append(search_snippet)
                        # Only add fetched text if it's not just boilerplate
                        if fetched_text and len(fetched_text) > 200 and "About Press Copyright" not in fetched_text[:200]:
                            content_parts.append(fetched_text)
                        content = "\n\n".join(content_parts) if content_parts else search_snippet
                    else:
                        # Non-video: use fetched content with search snippet as prefix
                        title = search_title if len(search_title) > len(fetched_title) else fetched_title
                        content = fetched_text
                        if search_snippet and search_snippet not in content:
                            content = f"{search_snippet}\n\n{content}"
                    
                    from models_task import EvidenceSource
                    source = EvidenceSource(
                        source_id=f"src_{len(evidence_state.sources)}",
                        url=page.get("url", ""),
                        title=title,
                        source_type=self._classify_source_type(source_type),
                        content_type=source_type,
                        content=content,
                        relevance_score=0.0,
                        quality_score=self._calculate_quality_score(source_type, page.get("url", "")),
                        completeness=self._calculate_completeness(content),
                        is_artifact=self._is_artifact_source(source_type, evidence_state),
                    )
                    source.relevance_score = self._calculate_relevance(source, evidence_state)
                    evidence_state.add_source(source)
        return gained

    def _gather_files(self, task: Task, signals: RoutingSignals, evidence: EvidenceManager) -> int:
        intent = signals.file_intent
        target = str(task.entities.get("filename") or task.entities.get("path") or "").strip()
        folder = intent.folder if intent else None
        path = folder or "."
        if folder and folder in {"Downloads", "Documents", "Desktop", "Pictures", "Music", "Videos"}:
            path = str(Path.home() / folder)
        if intent and intent.wants_list:
            output = self._execute("filesystem.list", {"path": path})
            return evidence.add_file_output("filesystem.list", output) if output is not None else 0
        if target and (Path(target).suffix or Path(target).is_absolute()):
            output = self._execute("filesystem.read", {"path": target, "max_bytes": self._content_max_bytes})
            if output is not None:
                return evidence.add_file_output("filesystem.read", output)
            return 0
        if intent is None or not (intent.subject or intent.extension or intent.content_query):
            return 0
        if intent.content_query and intent.wants_content_search:
            output = self._execute("filesystem.search_content", {
                "query": intent.content_query, "path": path, "pattern": intent.pattern,
                "max_files": self._content_max_files, "max_bytes": self._content_max_bytes,
                "max_results": 20,
            })
            return evidence.add_file_output("filesystem.search_content", output) if output is not None else 0
        subject = intent.subject
        pattern = intent.pattern
        if subject:
            pattern = "*" + subject.replace("*", "").replace("?", "") + "*"
            if intent.extension:
                pattern += "." + intent.extension
        output = self._execute("filesystem.search", {"pattern": pattern, "path": path, "max_results": 20})
        if output is None:
            return 0
        gained = evidence.add_file_output("filesystem.search", output)
        matches = output.get("matches", [])
        if intent.wants_read:
            if len(matches) != 1:
                run = _run.get()
                if run:
                    run.notes.append("specify one document to read; filename search did not identify a unique file")
                return gained
            match = matches[0]
            filename = match.get("path") if isinstance(match, Mapping) else match
            read = self._execute("filesystem.read", {"path": str(filename), "max_bytes": self._content_max_bytes})
            if read is not None:
                gained += evidence.add_file_output("filesystem.read", read)
        return gained

    def _iterative_web_retrieval(self, question: str, task: Task, evidence: EvidenceManager, signals: RoutingSignals) -> int:
        """Perform iterative web retrieval with query rewriting based on evidence evaluation."""
        if task.evidence_state is None:
            # Initialize evidence state if not present
            task.evidence_state = EvidenceState(
                target=task.entities.get("topic", question),
                goal="find_information",
                required_content_type=task.entities.get("content_type", "generic"),
                must_be_artifact=self._determine_must_be_artifact(task),
            )
        
        evidence_state = task.evidence_state
        max_attempts = evidence_state.max_retrieval_attempts
        gained_total = 0
                #: Bounded query-normalization recovery. When the first query returns
        #: nothing usable, it may be malformed (a connector leaked in, or the
        #: topic was mistyped). One normalization retry is attempted before the
        #: generic reformulation, and it is recorded so the feedback loop can
        #: learn that interpretation/query-normalization was the problem.
        normalization_retried = False

        for attempt in range(max_attempts):
            if not self._run_active():
                break

            evidence_state.retrieval_attempts = attempt + 1

            # Generate query for this attempt
            if attempt == 0:
                # Use the interpreter's planned query or the original question
                query = self._get_initial_query(task, question)
            else:
                query = None
                if not normalization_retried:
                    query = self._normalized_retry_query(task, evidence_state)
                    if query is not None:
                        normalization_retried = True
                        task.execution_trace.append({
                            "stage": "retrieval",
                            "action": "query_normalization_recovery",
                            "result": "retrying_with_normalized_query",
                            "details": {"query": query},
                        })
                if query is None:
                    # Reformulate query based on previous rejection
                    query = self._reformulate_query_for_attempt(evidence_state, attempt)
            
            evidence_state.last_query = query
            
            # Perform search and fetch
            gained = self._perform_web_search_and_fetch(query, evidence, evidence_state, task)
            gained_total += gained
            
            # Evaluate evidence
            if evidence_state.sufficient_for_output:
                logging.info(f"Web retrieval succeeded on attempt {attempt + 1}: sufficient evidence gathered")
                task.execution_trace.append({
                    "stage": "retrieval",
                    "action": "iterative_web_retrieval",
                    "result": "sufficient_evidence",
                    "details": {"attempt": attempt + 1, "query": query, "sources_found": len(evidence_state.relevant_sources)}
                })
                break
            
            # If not sufficient and we have more attempts, continue loop
            if attempt < max_attempts - 1:
                logging.info(f"Web retrieval attempt {attempt + 1} insufficient, will retry with reformulated query")
                task.execution_trace.append({
                    "stage": "retrieval",
                    "action": "iterative_web_retrieval",
                    "result": "insufficient_evidence",
                    "details": {"attempt": attempt + 1, "query": query, "confidence": evidence_state.confidence, "rejection_reason": evidence_state.last_rejection_reason}
                })
            else:
                logging.warning(f"Web retrieval exhausted after {max_attempts} attempts")
                task.execution_trace.append({
                    "stage": "retrieval",
                    "action": "iterative_web_retrieval",
                    "result": "max_attempts_reached",
                    "details": {"total_attempts": max_attempts, "final_confidence": evidence_state.confidence}
                })
        
        return gained_total

    def _run_active(self) -> bool:
        run = _run.get()
        return run is not None and run.active()

    def _determine_must_be_artifact(self, task: Task) -> bool:
        """Determine if the task requires the artifact itself vs information about it."""
        text = task.original_prompt.casefold()
        content_type = task.entities.get("content_type", "generic")
        
        # Explicit artifact phrases
        artifact_phrases = ("full script", "complete script", "entire script", "the script",
                           "full transcript", "complete transcript", "entire transcript", "the transcript",
                           "full lyrics", "the lyrics", "full text", "complete text", "whole text",
                           "the source code", "full code", "entire code", "the raw text")
        if any(phrase in text for phrase in artifact_phrases):
            return True
        # Info phrases mean they DON'T want the artifact
        info_phrases = ("information about", "info about", "information on", "learn about",
                       "tell me about", "explain", "what is", "who is", "overview of",
                       "background on", "facts about", "details about")
        if any(phrase in text for phrase in info_phrases):
            return False
        # Question openers ask about the subject
        if re.match(r'^(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|will\s+you\s+)*(?:what|who|when|where|why|how|which|whose|is|are|was|were|does|do|did)\b', text):
            return False
        # Document types without info phrases = artifact request
        if content_type in {"movie_script", "transcript", "lyrics", "code", "documentation", "list"}:
            return True
        return False

    def _get_initial_query(self, task: Task, question: str) -> str:
        """Get the initial search query from task actions or use the question."""
        for action in task.actions:
            if action.capability in ("web.search", "web.research") and "query" in action.parameters:
                return str(action.parameters["query"])
        return question

    def _normalized_retry_query(self, task: Task, evidence_state: EvidenceState) -> str | None:
        """Return a semantically normalized query for one bounded retry.

        A first web pass that returned nothing usable often means the query was
        malformed (a connector leaked in, or the topic carried framing words).
        This builds a cleaned candidate and returns it only when it actually
        differs from what was searched, so the retry is meaningful and cannot
        loop: the caller allows it once.

        Preference order: the task's own normalized query/topic (which the
        interpreter derived from the whole request) wins over a lighter cleaning
        of the failed query, because it reflects the full intent rather than the
        already-truncated string that failed.
        """

        current = (evidence_state.last_query or self._get_initial_query(task, "")).strip()
        if not current:
            return None
        current_key = current.casefold()

        # Prefer the interpreter's own normalized query/topic when it differs.
        for candidate in (
            task.research_query,
            task.normalized_topic,
            str(task.entities.get("topic") or ""),
        ):
            candidate = (candidate or "").strip()
            if not candidate or candidate.casefold() == current_key:
                continue
            cleaned_candidate, _ = normalize_query(candidate)
            if cleaned_candidate and cleaned_candidate.casefold() != current_key:
                return cleaned_candidate

        # Otherwise, clean the failed query itself (strips a leaked connector or
        # framing head noun). Returned only when it genuinely changed.
        cleaned, changed = normalize_query(current)
        if changed and cleaned and cleaned.casefold() != current_key:
            return cleaned
        return None

    def _reformulate_query_for_attempt(self, evidence_state: EvidenceState, attempt: int) -> str:
        """Reformulate query based on what was missing in previous attempts."""
        # Create a temporary RetrievalTask for reformulation
        retrieval_task = RetrievalTask(
            goal=evidence_state.goal,
            target=evidence_state.target,
            content_type=evidence_state.required_content_type,
            must_be_artifact=evidence_state.must_be_artifact,
        )
        
        reason = evidence_state.last_rejection_reason or "no suitable content"
        return reformulate_query(retrieval_task, reason=reason, attempt=attempt)

    def _perform_web_search_and_fetch(self, query: str, evidence: EvidenceManager, evidence_state: EvidenceState, task: Task) -> int:
        """Perform web search and fetch, then evaluate and classify sources."""
        parameters = {"query": query, "max_results": self._web_max_results}
        search = self._execute("web.search", parameters)
        if search is None:
            return 0
        
        # Add search results to evidence manager (for answer generation)
        gained = evidence.add_web_results(search)
        
        # Process each result page with task-aware evaluation
        results = search.get("results", [])[:self._web_max_pages]
        for item in results:
            if not isinstance(item, Mapping) or not item.get("url"):
                continue
            thumbnail_url = str(item.get("thumbnail_url") or "").strip()
            page = self._execute("web.fetch", {"url": str(item["url"])})
            if page is None:
                continue
            
            # Classify the source using fetched page
            source_type = detect_source_type(page.get("url", ""), page.get("title", ""), page.get("text", ""))
            
            # Use search result title/snippet for better relevance (they have actual content)
            # but fall back to fetched page if search result is sparse
            search_title = str(item.get("title") or "").strip()
            search_snippet = str(item.get("snippet") or "").strip()
            fetched_title = str(page.get("title") or "").strip()
            fetched_text = str(page.get("text") or "").strip()
            
            # Use search result title if meaningful, else fetched title
            title = search_title if len(search_title) > len(fetched_title) else fetched_title
            # Combine search snippet and fetched text for content
            content = fetched_text
            if search_snippet and search_snippet not in content:
                content = f"{search_snippet}\n\n{content}"
            
            # Create EvidenceSource with classification
            source = EvidenceSource(
                source_id=f"src_{len(evidence_state.sources)}",
                url=page.get("url", ""),
                title=title,
                source_type=self._classify_source_type(source_type),
                content_type=source_type,
                content=content,
                relevance_score=0.0,  # Will be calculated
                quality_score=self._calculate_quality_score(source_type, page.get("url", "")),
                completeness=self._calculate_completeness(content),
                is_artifact=self._is_artifact_source(source_type, evidence_state),
            )
            
            # Score relevance against task
            source.relevance_score = self._calculate_relevance(source, evidence_state)
            
            # Add to evidence state
            evidence_state.add_source(source)
            
            # Also add to evidence manager for answer generation
            evidence.add_web_page(page, thumbnail_url=thumbnail_url)
            gained += 1
            
            # Task-aware validation
            validation = validate_content(
                retrieval_task=RetrievalTask(
                    goal=evidence_state.goal,
                    target=evidence_state.target,
                    content_type=evidence_state.required_content_type,
                    must_be_artifact=evidence_state.must_be_artifact,
                ),
                content=source.content,
                detected_type=source.content_type,
                title=source.title,
                url=source.url,
            )
            
            if not validation.ok:
                evidence_state.last_rejection_reason = validation.reason
                logging.debug(f"Source {source.url} rejected: {validation.reason}")
        
        return gained

    def _classify_source_type(self, detected_type: str) -> str:
        """Map web_task detected type to source type class."""
        mapping = {
            "reference": "reference",
            "review": "reference",
            "news": "reference",
            "article": "secondary",
            "movie_script": "primary",
            "transcript": "primary",
            "lyrics": "primary",
            "code": "primary",
            "documentation": "primary",
            "forum": "discussion",
            "social": "social",
            "product": "retail",
            "video": "reference",
            "document_host": "retail",
            "listing": "search_result",
            "generic": "secondary",
        }
        return mapping.get(detected_type, "unknown")

    def _calculate_quality_score(self, detected_type: str, url: str) -> float:
        """Calculate source quality score based on type and domain."""
        host = url.split("/")[2] if "//" in url else ""
        if detected_type in {"reference"}:
            return 0.9
        if detected_type in {"primary", "movie_script", "transcript", "lyrics", "code", "documentation"}:
            return 0.85
        if detected_type in {"article", "news", "review"}:
            return 0.7
        if detected_type in {"video"}:
            return 0.7
        if detected_type in {"forum", "discussion"}:
            return 0.5
        if detected_type in {"product", "retail", "document_host", "listing"}:
            return 0.3
        if detected_type in {"social"}:
            return 0.2
        return 0.5

    def _calculate_completeness(self, text: str) -> float:
        """Calculate content completeness based on length and structure."""
        length = len(text.strip())
        if length >= 12000:
            return 1.0
        if length >= 4000:
            return 0.7
        if length >= 1200:
            return 0.4
        if length >= 400:
            return 0.2
        return 0.05 if length else 0.0

    def _is_artifact_source(self, detected_type: str, evidence_state: EvidenceState) -> bool:
        """Determine if this source IS the requested artifact."""
        if not evidence_state.must_be_artifact:
            return False
        artifact_types = {"movie_script", "transcript", "lyrics", "code", "documentation", "list"}
        return detected_type in artifact_types

    def _calculate_relevance(self, source: EvidenceSource, evidence_state: EvidenceState) -> float:
        """Calculate relevance score of a source to the task."""
        target = evidence_state.target.casefold()
        if not target:
            return 0.5
        
        # Filter out generic stopwords that don't indicate specific relevance
        stopwords = {"videos", "video", "search", "find", "look", "show", "get", "results", "result", 
                     "youtube", "google", "web", "internet", "online", "the", "for", "and", "or", "with",
                     "about"}
        
        # Check title, URL, and content for target terms
        target_terms = [t for t in target.split() if len(t) >= 3 and t not in stopwords]
        if not target_terms:
            return 0.5
        
        title_match = sum(1 for t in target_terms if t in source.title.casefold()) / len(target_terms)
        url_match = sum(1 for t in target_terms if t in source.url.casefold()) / len(target_terms)
        content_match = sum(1 for t in target_terms if t in source.content.casefold()[:6000]) / len(target_terms)
        
        return (title_match * 0.3 + url_match * 0.1 + content_match * 0.6)

    def _execute(self, name: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        run = _run.get()
        if run is None or not run.active():
            return None
        if name not in _READ_TOOLS or not self._capabilities.exists(name):
            run.notes.append(name + ": capability unavailable to reasoning")
            return None
        if run.tool_calls >= run.tool_limit:
            run.notes.append("reasoning tool-call limit reached")
            return None
        run.tool_calls += 1
        run.trace.record(ReasoningStage.RETRIEVING, name)
        run.selected_tool = name
        run.emit("tool_started", tool=name, parameters=sorted(str(key) for key in parameters))
        try:
            result = self._run_tool(name, parameters)
            success = bool(result is not None and getattr(result, "success", False))
            run.observations.append({"tool": name, "success": success})
            run.emit(
                "tool_completed",
                tool=name,
                status=str(getattr(result, "status", "") or ("completed" if success else "failed")),
                success=success,
            )
            if not success:
                run.notes.append(name + ": " + str(getattr(result, "error", "failed"))[:300])
                return None
            if not run.active():
                return None
            output = getattr(result, "output", None)
            return output if isinstance(output, Mapping) else None
        except Exception:
            logger.exception("Reasoning tool %s failed", name)
            run.notes.append(name + ": tool failed")
            return None

    def _finish(self, answer: Answer, decision: ReasoningDecision, evidence: EvidenceManager,
                run: _Run, task: Task) -> Answer:
        decision.response_mode = answer.mode
        decision.confidence_level = answer.confidence_level
        decision.confidence = 0.0
        task.apply_decision(decision)
        run.trace.record(ReasoningStage.ANSWERING, answer.mode.value)
        task.context["reasoning"] = run.trace.to_list()
        task.context["reasoning_summary"] = run.trace.summary()
        if run.selected_tool:
            task.context["selected_tool"] = run.selected_tool
        answer.metadata.update({
            "selected_tool": run.selected_tool,
            "reasoning": run.trace.to_list(), "reasoning_summary": run.trace.summary(),
            "decision": decision.to_dict(), "evidence": evidence.to_dict(),
            "confidence": answer.confidence_level.value, "tool_calls": run.observations,
            "tool_call_count": run.tool_calls,
        })
        for step in run.trace.steps:
            logger.info("reasoning stage=%s iteration=%s detail=%s", step.stage.value, step.iteration, step.detail)
        return answer
