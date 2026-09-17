from __future__ import annotations

import logging
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
from models_task import Task
from reasoning.answer_generator import Answer, AnswerGenerator
from reasoning.evidence_manager import EvidenceManager
from reasoning.query_router import QueryRouter, RoutingSignals
from reasoning.reasoning_models import (
    ConfidenceLevel, ReasoningDecision, ReasoningStage, ReasoningTrace,
    RequestType, ResponseMode, SourceType,
)
from reasoning.self_introspection import SelfIntrospection
from reasoning.source_selector import SourceSelector
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

    def active(self) -> bool:
        if self.cancelled():
            self.notes.append("request cancelled")
            return False
        if time.monotonic() >= self.deadline:
            self.notes.append("reasoning deadline reached")
            return False
        return True


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
    ) -> Answer | None:
        run = _Run(time.monotonic() + self._timeout_seconds, cancelled or (lambda: False),
                   ReasoningTrace(), self._max_tool_calls)
        token = _run.set(run)
        try:
            return self._handle(question, task, prior_task, history, run)
        finally:
            _run.reset(token)

    def _handle(self, question: str, task: Task, prior_task: Task | None,
                history: str, run: _Run) -> Answer | None:
        evidence = EvidenceManager()
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
                                  evidence=evidence, signals=signals)
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

        if not run.active():
            answer = self._answers.limitation(decision, notes=run.notes)
        elif mode is ResponseMode.MEMORY_RECALL:
            answer = self._answers.memory_recall(question, history)
        elif evidence.sufficient_for(mode, content_required=bool(
            mode is ResponseMode.FILE_LOOKUP and signals.file_intent and signals.file_intent.wants_read
        )) and mode is not ResponseMode.DIRECT_ANSWER:
            answer = self._answers.grounded(question, evidence, mode=mode, history=history)
        elif any(source in decision.sources for source in (
            SourceType.FILES, SourceType.SYSTEM, SourceType.MEMORY, SourceType.CONVERSATION,
        )):
            answer = self._answers.limitation(decision, notes=run.notes)
        elif SourceType.MODEL in decision.sources and self._allow_general_fallback:
            note = "; ".join(run.notes)
            if decision.current_information_required:
                note += "; live verification unavailable; do not claim a current answer"
            answer = self._answers.direct(question, history=history, fallback_note=note)
            if decision.current_information_required and answer.mode is ResponseMode.DIRECT_ANSWER:
                answer.text = "I could not verify current information; this may be out of date.\n\n" + answer.text
                answer.confidence_level = ConfidenceLevel.LOW
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
                evidence: EvidenceManager, signals: RoutingSignals) -> int:
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
                return self._gather_web(question, evidence, task=task)
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

    def _gather_web(self, question: str, evidence: EvidenceManager, *, task: Task | None = None) -> int:
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
        search = self._execute("web.search", parameters)
        if search is None:
            return 0
        gained = evidence.add_web_results(search)
        for item in search.get("results", [])[:self._web_max_pages]:
            if not isinstance(item, Mapping) or not item.get("url"):
                continue
            page = self._execute("web.fetch", {"url": str(item["url"])})
            if page is not None:
                gained += evidence.add_web_page(page)
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
        try:
            result = self._run_tool(name, parameters)
            success = bool(result is not None and getattr(result, "success", False))
            run.observations.append({"tool": name, "success": success})
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
