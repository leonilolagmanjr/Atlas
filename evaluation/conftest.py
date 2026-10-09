"""pytest wiring for the behavioral evaluation.

The evaluation is a *pytest module*, not a second test framework: the runner is
``evaluation/test_evaluation.py`` and this file provides the two things pytest
needs to run it —

* a **session-scoped environment fixture** (the isolated temp store plus the
  dependency snapshot, created once and reused by every case), and
* a **session-finish report flush** that writes the JSON report and prints the
  console summary.

Keeping this here (rather than inside the test module) is what makes the hooks
actually fire: pytest only treats ``pytest_configure``/``pytest_unconfigure`` as
hooks in a plugin or ``conftest.py``.
"""

from __future__ import annotations

import pytest

from evaluation.report import EvaluationReport
from evaluation.runner import load_results, live_enabled, report_path, reset_results
from evaluation.scenarios import RunEnvironment


@pytest.fixture(scope="session", autouse=True)
def _fresh_results() -> None:
    """Start each session with an empty results file."""

    reset_results()


@pytest.fixture(scope="session")
def evaluation_environment() -> RunEnvironment:
    """The isolated environment every case is served in.

    It owns a temp conversation store (so evaluation never touches the user's
    real conversations) and the dependency snapshot (so a missing dependency is
    reported as ``BLOCKED`` rather than a false ``FAIL``).
    """

    environment = RunEnvironment.create(mode="live" if live_enabled() else "simulated")
    yield environment
    environment.cleanup()


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, config: pytest.Config) -> None:  # noqa: ARG001
    """Print the evaluation summary and write the CI-consumable JSON report.

    The results are read back from the run's results file rather than an
    in-process list, because pytest's plugin/module identity rules make a shared
    global unreliable across ``conftest.py`` and the test module.
    """

    results = load_results()
    if not results:
        return
    from evaluation.dependencies import probe_dependencies

    probes = probe_dependencies()
    report = EvaluationReport(
        results=results,
        dependencies={name: probe.to_dict() for name, probe in probes.items()},
        metadata={
            "live_enabled": live_enabled(),
            "mode": "live" if live_enabled() else "simulated",
        },
    )
    target = report_path()
    report.write_json(target)
    terminalreporter.write_line("")
    terminalreporter.write_line(report.summary_text())
    terminalreporter.write_line(f"\nJSON report: {target}")

