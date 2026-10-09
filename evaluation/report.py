"""Result and report structures for the evaluation system.

The report is the durable artifact. It contains, for every case: the structured
expectation, the *actual* execution trace, the outcome (PASS/FAIL/BLOCKED), and
— for a FAIL — a failure attribution. The JSON form is written so CI can consume
it, and the console form is written so a human can read one line per case.

Raw results are recorded without embellishment. A FAIL is never converted to
XFAIL because it is "a known issue"; the failure signal is preserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.cases import (
    ATTRIBUTION_ATLAS,
    ATTRIBUTION_EXTERNAL,
    ATTRIBUTION_HARNESS,
    ATTRIBUTION_UNKNOWN,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
)


@dataclass
class CaseResult:
    """The recorded outcome of one evaluated case."""

    case_id: str
    category: str
    description: str
    outcome: str  # PASS | FAIL | BLOCKED
    #: The conversation kind actually reported for the observed turn.
    actual_kind: str = ""
    #: The final kind after execution (may differ from the classification kind:
    #: a research turn that found nothing is reported as a clarification).
    final_kind: str = ""
    #: The ordered tools actually observed in the event stream.
    actual_tools: list[str] = field(default_factory=list)
    #: Per-turn tool traces for multi-turn cases: list of turns, each a list of tools.
    actual_turn_tools: list[list[str]] = field(default_factory=list)
    #: The full event log for the observed turn(s), for the raw report.
    actual_events: list[dict[str, Any]] = field(default_factory=list)
    #: Citations extracted from the final answer (not the trace).
    answer_citations: list[str] = field(default_factory=list)
    #: Citations the *runtime* attached to the turn (from the evidence manager's
    #: citable sources), independent of the answer text.
    turn_citations: list[str] = field(default_factory=list)
    #: Citations the *tooling* actually produced, for comparison.
    retrieved_citations: list[str] = field(default_factory=list)
    #: The final answer text (kept for the raw report; never the sole evidence).
    answer_text: str = ""
    #: Human-readable failure reason, empty on PASS.
    failure_reason: str = ""
    #: One of atlas_behavior / harness_error / external_dependency / unknown.
    attribution: str = ""
    #: Dependency states observed for this case.
    dependency_status: dict[str, str] = field(default_factory=dict)
    #: Notes captured during evaluation (per-assertion detail).
    notes: list[str] = field(default_factory=list)
    #: True when the case was served by a simulated (fake) provider.
    simulated: bool = True
    #: Whether this case's expectation was met (PASS), not met (FAIL), or
    #: could not be decided (BLOCKED).
    duration_seconds: float = 0.0

    @property
    def is_pass(self) -> bool:
        return self.outcome == "PASS"

    @property
    def is_blocked(self) -> bool:
        return self.outcome == "BLOCKED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "category_label": CATEGORY_LABELS.get(self.category, self.category),
            "description": self.description,
            "outcome": self.outcome,
            "simulated": self.simulated,
            "actual_kind": self.actual_kind,
            "final_kind": self.final_kind,
            "actual_tools": list(self.actual_tools),
            "actual_turn_tools": [list(turn) for turn in self.actual_turn_tools],
            "answer_citations": list(self.answer_citations),
            "turn_citations": list(self.turn_citations),
            "retrieved_citations": list(self.retrieved_citations),
            "answer_text": self.answer_text,
            "failure_reason": self.failure_reason,
            "attribution": self.attribution,
            "dependency_status": dict(self.dependency_status),
            "notes": list(self.notes),
            "duration_seconds": round(self.duration_seconds, 4),
            "actual_events": self.actual_events,
        }


def decide_attribution(
    *,
    atlas_mismatch: bool,
    harness_error: str = "",
    external_unavailable: bool = False,
) -> str:
    """Choose the single most honest attribution for a FAIL."""

    if external_unavailable:
        return ATTRIBUTION_EXTERNAL
    if harness_error:
        return ATTRIBUTION_HARNESS
    if atlas_mismatch:
        return ATTRIBUTION_ATLAS
    return ATTRIBUTION_UNKNOWN


@dataclass
class EvaluationReport:
    """The complete evaluation run: every case, grouped and attributed."""

    results: list[CaseResult] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    #: The dependency probe snapshot for the whole run.
    dependencies: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Free-form run metadata (mode, git revision if available, etc.).
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()
        if not self.finished_at:
            self.finished_at = self.started_at

    # -- counters ---------------------------------------------------------------

    @property
    def passed(self) -> list[CaseResult]:
        return [r for r in self.results if r.outcome == "PASS"]

    @property
    def failed(self) -> list[CaseResult]:
        return [r for r in self.results if r.outcome == "FAIL"]

    @property
    def blocked(self) -> list[CaseResult]:
        return [r for r in self.results if r.outcome == "BLOCKED"]

    def counts_by_category(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for category in CATEGORY_ORDER:
            counts[category] = {"PASS": 0, "FAIL": 0, "BLOCKED": 0, "total": 0}
        for result in self.results:
            bucket = counts.setdefault(
                result.category, {"PASS": 0, "FAIL": 0, "BLOCKED": 0, "total": 0}
            )
            bucket[result.outcome] = bucket.get(result.outcome, 0) + 1
            bucket["total"] += 1
        return counts

    def counts_by_mode(self) -> dict[str, dict[str, int]]:
        """PASS/FAIL/BLOCKED split into simulated vs live. Never mixed."""

        modes: dict[str, dict[str, int]] = {
            "simulated": {"PASS": 0, "FAIL": 0, "BLOCKED": 0, "total": 0},
            "live": {"PASS": 0, "FAIL": 0, "BLOCKED": 0, "total": 0},
        }
        for result in self.results:
            bucket = modes["simulated" if result.simulated else "live"]
            bucket[result.outcome] += 1
            bucket["total"] += 1
        return modes

    # -- output -----------------------------------------------------------------

    def summary_lines(self) -> list[str]:
        """One console-readable line per case, plus a header per category."""

        lines: list[str] = []
        for category in CATEGORY_ORDER:
            rows = [r for r in self.results if r.category == category]
            if not rows:
                continue
            lines.append(f"== {CATEGORY_LABELS.get(category, category)} ==")
            for result in rows:
                mode = "sim" if result.simulated else "live"
                suffix = ""
                if result.outcome == "FAIL" and result.failure_reason:
                    suffix = f" -- [{result.attribution or 'unknown'}] {result.failure_reason}"
                elif result.outcome == "BLOCKED":
                    suffix = f" -- {result.failure_reason or 'dependency unavailable'}"
                lines.append(
                    f"  {result.outcome:<7} {result.case_id:<48} ({mode}){suffix}"
                )
        return lines

    def summary_text(self) -> str:
        counts = self.counts_by_category()
        total = len(self.results)
        passed = len(self.passed)
        failed = len(self.failed)
        blocked = len(self.blocked)
        lines = list(self.summary_lines())
        lines.append("")
        lines.append(f"TOTAL: {total}  PASS: {passed}  FAIL: {failed}  BLOCKED: {blocked}")
        for category in CATEGORY_ORDER:
            row = counts.get(category)
            if not row or row["total"] == 0:
                continue
            lines.append(
                f"  {CATEGORY_LABELS.get(category, category)}: "
                f"PASS={row['PASS']} FAIL={row['FAIL']} BLOCKED={row['BLOCKED']}"
            )
        modes = self.counts_by_mode()
        for mode, row in modes.items():
            if row["total"]:
                lines.append(
                    f"  [{mode}] PASS={row['PASS']} FAIL={row['FAIL']} BLOCKED={row['BLOCKED']} "
                    f"(of {row['total']})"
                )
        lines.append("")
        lines.append(f"Attribution of FAILs: {self.attribution_counts()}")
        return "\n".join(lines)

    def attribution_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.failed:
            key = result.attribution or ATTRIBUTION_UNKNOWN
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": dict(self.metadata),
            "dependencies": dict(self.dependencies),
            "totals": {
                "cases": len(self.results),
                "pass": len(self.passed),
                "fail": len(self.failed),
                "blocked": len(self.blocked),
            },
            "counts_by_category": self.counts_by_category(),
            "counts_by_mode": self.counts_by_mode(),
            "attribution_counts": self.attribution_counts(),
            "results": [result.to_dict() for result in self.results],
        }

    def write_json(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target
