"""The evaluation runner: a pytest module, not a second test framework.

The runner drives every case in the catalog through the
:class:`evaluation.harness.EvaluationHarness`, classifies each result, and emits
two outputs:

* a **console summary** — one line per case (``PASS``/``FAIL``/``BLOCKED`` plus a
  failure reason), grouped by category; and
* a **JSON report** — every case's expectation, actual trace and attribution,
  written to ``evaluation_reports/evaluation.json`` (configurable via
  ``ATLAS_EVAL_REPORT``), consumable by CI.

How the pytest integration works
--------------------------------
Each case is one parametrized pytest test (``test_case``), so pytest's own
reporting, filtering and exit code are the source of truth — the evaluation does
not invent a parallel notion of pass/fail. A ``BLOCKED`` case is reported as a
*skip* with the dependency reason as the skip message: it is not counted as a
pass or a fail, exactly as the specification requires.

Run it with::

    python -m pytest evaluation/test_evaluation.py -v

Live cases (which need the real local model, the network, or Windows) are
excluded unless ``ATLAS_EVAL_LIVE=1`` is set, so the default run is offline and
deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.cases import EvalCase, all_cases
from evaluation.runner import live_enabled, reset_results, run_case
from evaluation.report import CaseResult, EvaluationReport
from evaluation.scenarios import RunEnvironment


def _select_cases() -> list[EvalCase]:
    cases = list(all_cases())
    if live_enabled():
        return cases
    return [case for case in cases if not case.live]


@pytest.mark.parametrize("case", _select_cases(), ids=lambda case: case.id)
def test_case(case: EvalCase, evaluation_environment: RunEnvironment) -> None:
    """Serve one case and assert its structured expectation."""

    result = run_case(evaluation_environment, case)

    if result.outcome == "BLOCKED":
        pytest.skip(f"BLOCKED: {result.failure_reason}")

    if result.outcome == "FAIL":
        pytest.fail(_format_failure(result), pytrace=False)


def _format_failure(result: CaseResult) -> str:
    lines = [
        f"[{result.attribution or 'unknown'}] {result.failure_reason}",
        f"  classified kind: {result.actual_kind}  final kind: {result.final_kind}",
        f"  actual tools: {result.actual_tools}",
    ]
    if result.actual_turn_tools and len(result.actual_turn_tools) > 1:
        lines.append(f"  per-turn tools: {result.actual_turn_tools}")
    lines.append(f"  answer: {result.answer_text[:400]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone entry point (also usable without pytest)
# ---------------------------------------------------------------------------


def run_all(*, live: bool = False, report_path: Path | str | None = None) -> EvaluationReport:
    """Run every case through the *same* harness and return the report.

    This is the programmatic entry point used by ``python -m evaluation``. It
    shares the exact verification the pytest module uses, so there is only one
    implementation of "how a case is decided".
    """

    from evaluation.report import EvaluationReport
    from evaluation.runner import load_results

    reset_results()
    environment = RunEnvironment.create(mode="live" if live else "simulated")
    try:
        for case in all_cases():
            if case.live and not live:
                continue
            # Delegate to the shared helper so the standalone path and the
            # pytest path decide a case in exactly the same way and record it to
            # the same results file.
            run_case(environment, case)
        report = EvaluationReport(
            results=load_results(),
            dependencies={
                name: probe.to_dict() for name, probe in environment.dependencies.items()
            },
            metadata={"live_enabled": live, "mode": "live" if live else "simulated"},
        )
        if report_path is not None:
            report.write_json(report_path)
        return report
    finally:
        environment.cleanup()


if __name__ == "__main__":  # pragma: no cover - manual convenience
    import json

    _report = run_all(live=live_enabled())
    print(_report.summary_text())
    print(json.dumps(_report.attribution_counts(), indent=2))

