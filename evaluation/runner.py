"""Shared run state and the "serve one case" helper for the evaluation.

pytest loads ``conftest.py`` as a *plugin* while ``test_evaluation.py`` imports
it as a *module*; those can be two distinct module instances with distinct
globals, and pytest's import mode can likewise give ``evaluation.runner`` two
identities. Relying on an in-process list for the report therefore risks the
report being built from a *different* list than the one the cases were recorded
into (a FAILed case would silently vanish from the JSON report — which is
exactly the failure mode this module exists to avoid).

So results are appended to a **JSON-lines file** as they are produced, and the
report is built by reading that file back. File-backed state is immune to module
identity and makes the report recoverable even if the run crashes part-way.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from evaluation.cases import EvalCase
from evaluation.harness import EvaluationHarness
from evaluation.report import CaseResult
from evaluation.scenarios import RunEnvironment

#: Environment flag: include live cases (real model / web / Windows).
LIVE_ENV = "ATLAS_EVAL_LIVE"
#: Environment path: where the JSON report is written.
REPORT_ENV = "ATLAS_EVAL_REPORT"

#: Default report location (repository-relative, git-ignored).
DEFAULT_REPORT = (
    Path(__file__).resolve().parent.parent / "evaluation_reports" / "evaluation.json"
)

#: Environment path: the JSON-lines file results are appended to as they run.
RESULTS_ENV = "ATLAS_EVAL_RESULTS"
_DEFAULT_RESULTS = DEFAULT_REPORT.with_name("results.jsonl")


def live_enabled() -> bool:
    """True when live cases should be included."""

    return str(os.environ.get(LIVE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def report_path() -> Path:
    """Where the JSON report should be written."""

    return Path(os.environ.get(REPORT_ENV, str(DEFAULT_REPORT)))


def results_path() -> Path:
    """Where the per-case JSON-lines results are accumulated."""

    return Path(os.environ.get(RESULTS_ENV, str(_DEFAULT_RESULTS)))


def reset_results() -> None:
    """Start a fresh results file for this run."""

    target = results_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")


def record(result: CaseResult) -> None:
    """Append one case result to the run's results file."""

    target = results_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def load_results() -> list[CaseResult]:
    """Read back every recorded result.

    A malformed line is skipped rather than raising: a partially written run
    must still produce a report of what did complete.
    """

    target = results_path()
    if not target.exists():
        return []
    results: list[CaseResult] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        results.append(_result_from_dict(payload))
    return results


def _result_from_dict(payload: dict) -> CaseResult:
    """Rebuild a :class:`CaseResult` from its serialized form."""

    known = {
        "case_id",
        "category",
        "description",
        "outcome",
        "actual_kind",
        "final_kind",
        "failure_reason",
        "attribution",
        "answer_text",
        "simulated",
    }
    kwargs = {key: payload[key] for key in known if key in payload}
    result = CaseResult(**kwargs)
    result.actual_tools = list(payload.get("actual_tools") or [])
    result.actual_turn_tools = [list(turn) for turn in payload.get("actual_turn_tools") or []]
    result.actual_events = list(payload.get("actual_events") or [])
    result.answer_citations = list(payload.get("answer_citations") or [])
    result.turn_citations = list(payload.get("turn_citations") or [])
    result.retrieved_citations = list(payload.get("retrieved_citations") or [])
    result.dependency_status = dict(payload.get("dependency_status") or {})
    result.notes = list(payload.get("notes") or [])
    result.duration_seconds = float(payload.get("duration_seconds") or 0.0)
    return result


def run_case(environment: RunEnvironment, case: EvalCase) -> CaseResult:
    """Serve one case, recording a BLOCKED result when a dependency is missing.

    The result is recorded (to the results file) before the caller decides what
    to do with it, so a FAIL that pytest turns into an assertion failure is still
    present in the JSON report.
    """

    scenario = environment.scenario(live=case.live)
    blocked = scenario.blocked_for(case)
    if blocked:
        result = CaseResult(
            case_id=case.id,
            category=case.category,
            description=case.description,
            outcome="BLOCKED",
            simulated=not case.live,
            failure_reason=blocked,
            attribution="external_dependency",
        )
        record(result)
        return result
    result = EvaluationHarness(scenario=scenario).run_case(case)
    record(result)
    return result
