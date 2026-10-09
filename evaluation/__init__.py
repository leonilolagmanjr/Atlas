"""Behavioral evaluation system for Atlas.

This package is *not* a second test framework and *not* a second execution
path. It is a thin layer of structured behavioral cases served through the
existing Conversation Runtime, plus a runner that reads the runtime's real
event stream and asserts decidable expectations against it.

The evaluation exists because Atlas's existing unit suite covers *components*
(a tool's parameter validation, an interpreter's field mapping) but not
*behavioral contracts* ("the user says X, and Atlas actually selects tool Y").
The value of this package is filling exactly that gap.

Design constraints it must respect (see the README's Testing section):

* The runner never plans, selects tools, or bypasses a layer. Every case is
  served by :class:`memory.conversation_runtime.ConversationRuntime`, which
  delegates to the existing Brain exactly as a real message does.
* Verification reads the **event stream** (``tool_started`` /
  ``tool_completed`` / ``observation`` / ``verification`` / ``assistant_*``),
  not the final answer string. Which tools ran is an architectural fact; the
  wording of a reply is not.
* External dependencies are probed explicitly so "Atlas behaved incorrectly"
  is never conflated with "an external dependency is unavailable".
"""

from evaluation.cases import (
    CATEGORY_LABELS,
    DependencyStatus,
    EvalCase,
    Expectation,
    ExpectedBehavior,
    VerificationMethod,
    all_cases,
    cases_for_category,
)
from evaluation.report import CaseResult, EvaluationReport

__all__ = [
    "CATEGORY_LABELS",
    "CaseResult",
    "DependencyStatus",
    "EvalCase",
    "EvaluationReport",
    "Expectation",
    "ExpectedBehavior",
    "VerificationMethod",
    "all_cases",
    "cases_for_category",
]
