"""The evaluation harness: serve one case through the real Conversation Runtime.

This is the heart of the evaluation system. It does three things and nothing
else:

1. **Serves the case through the existing path.** It builds a
   :class:`memory.conversation_runtime.ConversationRuntime` around a real
   ``Brain`` (with the real tool registry, planner, validator and executor) or,
   for simulated cases, around a deterministic fake provider. It never calls the
   Planner, the Executor, or a tool directly: every case goes through
   ``runtime.handle_message`` exactly as a user message does.

2. **Records the real event stream.** It subscribes to the runtime's event sink
   and records every event (``tool_started`` / ``tool_completed`` /
   ``observation`` / ``verification`` / ``assistant_completed`` / ...) in order.
   The observed tool sequence is derived from *this*, not from the final answer.

3. **Decides each assertion deterministically.** The structured
   :class:`~evaluation.cases.Expectation` is compared against the recorded trace,
   turn kinds, and citation sets. No LLM judges anything.

Simulated vs live
-----------------
A *simulated* case injects a deterministic ``ask`` (a canned model boundary) and
a recording tool runner, so the routing decision is exercised without a model.
A *live* case uses the real provider and real tools, and is recorded separately
so simulated and live results are never mixed in the statistics.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from evaluation.cases import (
    ATTRIBUTION_EXTERNAL,
    ATTRIBUTION_HARNESS,
    EvalCase,
)
from evaluation.report import CaseResult, decide_attribution

#: A citation is any http(s) URL appearing in the final answer text.
_URL_RE = re.compile(r"https?://[^\s<>'\"\)\]]+")

#: Phrases that indicate a turn honestly reported a failure/shortfall.
_FAILURE_PHRASES = (
    "could not",
    "couldn't",
    "cannot",
    "can't",
    "unable",
    "failed",
    "did not",
    "didn't",
    "not able",
    "no results",
    "could not find",
    "not possible",
    "not available",
    "i don't have",
    "i do not have",
    "unavailable",
    "error",
    "no pending",
    "does not exist",
    "not installed",
)

#: Phrases that falsely claim success.
_SUCCESS_CLAIMS = (
    " done",
    "done.",
    "done!",
    "i've opened",
    "i have opened",
    "i wrote",
    "i've written",
    "i have written",
    "successfully",
    "has been opened",
    "has been written",
    "all set",
)


@dataclass
class TurnObservation:
    """What one served turn actually produced."""

    text: str = ""
    kind: str = ""
    execution_state: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    #: Full turn result dict, for the raw report.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tools(self) -> list[str]:
        """The ordered tool names observed in the event stream.

        Duplicates are preserved (a repeated tool is meaningful — it can indicate
        a loop or a re-search), so ``max_turns`` and ``forbid_repeat_of`` can see
        the repetition. Comparison against ``required_tools`` uses set semantics
        separately in the assertion layer.
        """

        names: list[str] = []
        for event in self.events:
            if event.get("type") != "tool_started":
                continue
            tool = str((event.get("data") or {}).get("tool") or "")
            if tool:
                names.append(tool)
        if names:
            return names
        # No live tool_started events arrived (an injected Brain, or an early
        # reasoning answer): fall back to the recorded tool calls so the trace
        # still reflects what the execution path recorded.
        return [str(call.get("tool") or "") for call in self.tool_calls if call.get("tool")]

    @property
    def classification_kind(self) -> str:
        """The kind the *runtime* decided, from the ``assistant_started`` event.

        This is the routing decision itself and is available even when the turn
        later ends in a clarification. The final ``kind`` is overwritten by the
        executor's outcome (``_kind_for_context``), so it cannot answer "which
        path was chosen" on its own.
        """

        for event in self.events:
            if event.get("type") != "assistant_started":
                continue
            kind = str((event.get("data") or {}).get("kind") or "")
            if kind:
                return kind
        return self.kind

    @property
    def delegated(self) -> bool:
        """True when the turn was handed to the agent execution path.

        The runtime emits a ``tool_started`` event the moment it delegates, so
        its presence is the architectural signal that delegation happened — even
        if the agent then ran no tool.
        """

        return any(event.get("type") == "tool_started" for event in self.events)


def _extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for url in _URL_RE.findall(text or ""):
        cleaned = url.rstrip(".,;:")
        if cleaned not in seen:
            seen.append(cleaned)
    return seen


class _EventRecorder:
    """Collect the runtime's real events in order."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: Any) -> None:
        try:
            payload = event.to_dict()
        except AttributeError:
            payload = {"type": getattr(event, "type", "?"), "data": getattr(event, "data", {})}
        self.events.append(payload)


class EvaluationHarness:
    """Serve cases through the real runtime and record what happened.

    The harness is constructed once per run with a *scenario*: a builder that
    supplies the runtime (memory store, brain, tool runner, model boundary) for
    simulated or live execution. Keeping the runtime construction behind a
    builder is what lets the exact same assertion logic run for both modes while
    never mixing their results.
    """

    def __init__(self, *, scenario: Any) -> None:
        self._scenario = scenario

    def run_case(self, case: EvalCase) -> CaseResult:
        """Serve one case and decide its outcome against the expectation."""

        started = time.perf_counter()
        result = CaseResult(
            case_id=case.id,
            category=case.category,
            description=case.description,
            outcome="FAIL",
            simulated=not case.live,
        )
        try:
            turns = self._serve(case)
        except Exception as exc:  # noqa: BLE001 - a harness crash is a harness error
            result.duration_seconds = time.perf_counter() - started
            result.outcome = "FAIL"
            result.failure_reason = f"harness raised {type(exc).__name__}: {exc}"
            result.attribution = ATTRIBUTION_HARNESS
            result.notes.append("The harness itself failed while serving the case.")
            return result

        result.actual_turn_tools = [turn.tools for turn in turns]
        result.actual_tools = _flatten_ordered(turns)
        observed = turns[-1] if turns else TurnObservation()
        # Record the *classification* kind (the routing decision) and the final
        # reported kind separately. The routing contract is decided by the former;
        # the latter is overwritten by the executor's outcome.
        result.actual_kind = observed.classification_kind
        result.final_kind = observed.kind
        result.actual_events = observed.events
        result.answer_text = observed.text
        result.answer_citations = _extract_citations(observed.text)
        result.turn_citations = list(observed.citations)
        result.retrieved_citations = _retrieved_citations(turns)
        result.dependency_status = self._scenario.dependency_status(case)

        self._assert_case(case, turns, result)
        result.duration_seconds = time.perf_counter() - started
        return result

    # -- serving ---------------------------------------------------------------

    def _serve(self, case: EvalCase) -> list[TurnObservation]:
        """Serve the setup turns then the case turn, recording each."""

        # Let the scenario apply any case-specific simulated tool overrides
        # (e.g. forcing a tool to fail for a honesty case) before it is served.
        setter = getattr(self._scenario, "set_case", None)
        if callable(setter):
            setter(case)
        conversation = self._scenario.open_conversation()
        observations: list[TurnObservation] = []
        for setup_message in case.conversation_setup:
            observations.append(conversation.send(setup_message))
        observations.append(conversation.send(case.user_message))
        return observations

    # -- assertions ------------------------------------------------------------

    def _assert_case(
        self, case: EvalCase, turns: list[TurnObservation], result: CaseResult
    ) -> None:
        expectation = case.expected_behavior
        observed = turns[-1] if turns else TurnObservation()
        observed_tools = observed.tools
        tool_set = set(observed_tools)
        failures: list[str] = []
        external_gap = False

        # required_tools: every named tool must have run at least once.
        missing = [tool for tool in expectation.required_tools if tool not in tool_set]
        if missing:
            failures.append(f"required tool(s) did not run: {', '.join(missing)}")

        # required_tools_any_of: each group must have at least one member run.
        # This is how "a retrieval happened" is asserted without pinning the
        # exact capability, when more than one is a legitimate choice.
        for group in expectation.required_tools_any_of:
            if not any(tool in tool_set for tool in group):
                failures.append(
                    "none of the acceptable tool(s) ran: " + ", ".join(group)
                )

        # forbidden_tools: none may have run.
        forbidden = [tool for tool in expectation.forbidden_tools if tool in tool_set]
        if forbidden:
            failures.append(f"forbidden tool(s) ran: {', '.join(forbidden)}")

        # tool_sequence: the named tools must appear in order.
        if expectation.tool_sequence and not _contains_subsequence(
            observed_tools, list(expectation.tool_sequence)
        ):
            failures.append(
                f"tool sequence {list(expectation.tool_sequence)} not found in {observed_tools}"
            )

        # forbid_repeat_of: a tool from a prior turn must not be repeated. This is
        # how a "do not re-search" continuity case is decided: the setup turn may
        # search, but the follow-up must not.
        if expectation.forbid_repeat_of:
            setup_tools = set()
            for turn in turns[:-1]:
                setup_tools.update(turn.tools)
            repeated = [tool for tool in expectation.forbid_repeat_of if tool in tool_set and tool in setup_tools]
            if repeated:
                failures.append(f"follow-up repeated prior tool(s): {', '.join(repeated)}")

        # expected_kind: the *routing decision* must be one of the acceptable
        # kinds. This reads the runtime's classification (from the
        # ``assistant_started`` event), not the executor's final kind, because
        # the contract under test is "which path was chosen".
        chosen_kind = observed.classification_kind
        if expectation.expected_kind and chosen_kind not in expectation.expected_kind:
            failures.append(
                f"classified kind {chosen_kind!r} not in {list(expectation.expected_kind)}"
            )

        # Delegation: the agent path must have been used. The runtime emits a
        # tool_started event the moment it delegates, so this is the architectural
        # signal — not the wording of the reply.
        if expectation.expect_delegation and not observed.delegated:
            failures.append(
                "expected delegation to the agent, but the turn was answered conversationally"
            )

        # Direct answer: the reasoning/answer path must have been used, not the
        # computer-action path.
        if expectation.expect_direct_answer and chosen_kind == "computer":
            failures.append("expected a direct answer, but the turn was delegated as a computer action")

        # Clarification.
        if expectation.expect_clarification and not _is_clarification(observed):
            failures.append("expected a clarification question, but none was returned")

        # Citations: the answer must be traceable to sources. In live mode the
        # authority is the answer text (a model wrote it). In simulated mode the
        # answer is canned and cannot carry citations, so the runtime's own
        # recorded citations for the turn are the authority — they come from the
        # evidence manager, which is the same data a real answer is grounded in.
        if expectation.expect_citations:
            available = result.answer_citations or (
                result.turn_citations if result.simulated else []
            )
            if not available:
                failures.append(
                    "expected traceable sources for a research answer, but none were"
                    " recorded in the answer or on the turn"
                )
            elif result.retrieved_citations and not _citations_trace_to_sources(
                available, result.retrieved_citations
            ):
                failures.append(
                    "answer cited URLs that do not correspond to retrieved sources: "
                    + ", ".join(available)
                )

        # Honesty: a failure must be reported, not a success claimed.
        if expectation.expect_failure_report:
            reported, claimed = _report_quality(observed.text)
            if claimed and not reported:
                failures.append(
                    "a success was claimed where a failure should have been reported"
                )
            elif not reported:
                failures.append("no honest failure/shortfall report was present in the answer")

        # Turn budget (loop detection).
        if expectation.max_turns is not None and len(observed_tools) > expectation.max_turns:
            failures.append(
                f"tool-call turns ({len(observed_tools)}) exceeded max_turns ({expectation.max_turns})"
            )

        # Conversation-store persistence: the tools that ran were recorded.
        if expectation.expect_conversation_tool_persisted:
            recorded = {str(call.get("tool") or "") for call in observed.tool_calls}
            expected_tools = set(expectation.required_tools)
            if expected_tools and not expected_tools.issubset(recorded):
                failures.append(
                    "expected tool calls to be persisted into the conversation store, but "
                    + ", ".join(sorted(expected_tools - recorded))
                    + " were not recorded"
                )

        if not failures:
            result.outcome = "PASS"
            return

        # If the only failing requirement needs a dependency that is missing,
        # this is an external gap, not an Atlas behavior failure.
        dependency_gap = self._scenario.missing_dependency_for(case)
        result.outcome = "FAIL"
        result.failure_reason = "; ".join(failures)
        result.notes.extend(failures)
        if dependency_gap:
            external_gap = True
            result.notes.append(f"dependency gap: {dependency_gap}")
        result.attribution = decide_attribution(
            atlas_mismatch=True,
            external_unavailable=external_gap,
        )

    def _serve_for_blocked(self, case: EvalCase, reason: str) -> CaseResult:
        return CaseResult(
            case_id=case.id,
            category=case.category,
            description=case.description,
            outcome="BLOCKED",
            simulated=not case.live,
            failure_reason=reason,
            attribution=ATTRIBUTION_EXTERNAL,
        )


# ---------------------------------------------------------------------------
# Pure assertion helpers (unit-testable on their own)
# ---------------------------------------------------------------------------


def _flatten_ordered(turns: list[TurnObservation]) -> list[str]:
    tools: list[str] = []
    for turn in turns:
        tools.extend(turn.tools)
    return tools


def _retrieved_citations(turns: list[TurnObservation]) -> list[str]:
    """URLs the tooling actually produced across every turn.

    These come from recorded tool results (``web.search``/``web.fetch`` outputs)
    and from citations the runtime attached to a turn — never from the answer
    text, so they are an independent check on what the answer claims.
    """

    urls: list[str] = []
    for turn in turns:
        for citation in turn.citations:
            if citation not in urls:
                urls.append(citation)
        for call in turn.tool_calls:
            text = repr(call) + repr(turn.raw.get("tool_results", ""))
            for url in _URL_RE.findall(text):
                cleaned = url.rstrip(".,;:'\"")
                if cleaned not in urls:
                    urls.append(cleaned)
    return urls


def _contains_subsequence(observed: list[str], required: list[str]) -> bool:
    if not required:
        return True
    iterator = iter(observed)
    return all(any(candidate == want for candidate in iterator) for want in required)


def _is_clarification(observation: TurnObservation) -> bool:
    """A clarification is a question asked instead of an action being taken.

    The runtime reports ``kind == "clarification"`` when the interpreter or
    validator asked rather than guessed. As a secondary, rule-based signal the
    answer text itself must be interrogative and must not claim an action ran.
    """

    if observation.kind == "clarification":
        return True
    text = (observation.text or "").strip()
    if not text:
        return False
    if "?" not in text:
        return False
    lowered = text.casefold()
    # A question that also claims an action happened is not a clarification.
    return not any(phrase in lowered for phrase in ("has been opened", "has been written", "i've opened", "i wrote"))


def _report_quality(text: str) -> tuple[bool, bool]:
    """Return ``(reported_failure, claimed_success)`` for an answer."""

    lowered = f" {(text or '').casefold()} "
    reported = any(phrase in lowered for phrase in _FAILURE_PHRASES)
    claimed = any(phrase in lowered for phrase in _SUCCESS_CLAIMS)
    return reported, claimed


def _citations_trace_to_sources(answer_citations: list[str], retrieved: list[str]) -> bool:
    """True when every answer citation corresponds to a retrieved source.

    Matching is on host+path (scheme, query strings and trailing slashes
    ignored), so a citation is accepted when it names a page the tools actually
    returned. A citation with no retrieved counterpart is a fabrication signal.
    """

    if not retrieved:
        return True  # nothing retrieved to compare against; presence was asserted
    normalized = {_normalize_url(url) for url in retrieved}
    for citation in answer_citations:
        if _normalize_url(citation) not in normalized:
            return False
    return True


def _normalize_url(url: str) -> str:
    value = (url or "").strip().rstrip("/")
    value = value.split("#", 1)[0]
    lower = value.casefold()
    if lower.startswith("https://"):
        return lower[len("https://"):]
    if lower.startswith("http://"):
        return lower[len("http://"):]
    return lower
