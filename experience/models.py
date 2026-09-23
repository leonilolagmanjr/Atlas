"""Experience models for Atlas's human-feedback learning loop.

An *experience* is a compact, structured record of one meaningful task: what the
user asked, how Atlas understood it, how it was planned and executed, whether it
worked, and — critically — what the user said about it afterwards.

Design rules (mirroring the rest of Atlas):

* This is NOT model training. The LLM stays the reasoning engine; experiences
  are supporting context that the planner may consult, never an authority that
  overrides an explicit user instruction.
* Records are deliberately compact. A task representation is stored, not a whole
  conversation, and nothing here executes anything.
* Every field is validated defensively when read back from disk, so a corrupt or
  hand-edited line degrades to "no experience" instead of raising.

The vocabulary is intentionally small and stable so retrieval and ranking can
key on it deterministically (see :mod:`experience.memory`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

#: What kind of experience this is. Kept distinct from knowledge/answer memory:
#: an experience answers "what worked or failed when performing a task?".
EXPERIENCE_TYPES: frozenset[str] = frozenset(
    {
        "successful_task",
        "failed_task",
        "correction",
        "workflow",
        "strategy",
    }
)

#: Lifecycle states. A single interaction is ``new``; repeated success promotes
#: it. Promotion never rewrites global behaviour on its own (spec section 16).
EXPERIENCE_STATES: tuple[str, ...] = ("new", "observed", "evaluated", "reliable")

#: Outcome vocabulary. Unknown/None is honest: the user has not answered yet.
OUTCOMES: frozenset[str] = frozenset({"success", "failure", "unknown"})

#: Canonical failure categories surfaced in the UI. They are coarse and stable
#: so pattern analysis never depends on parsing free text.
FAILURE_CATEGORIES: tuple[str, ...] = (
    "wrong_interpretation",
    "wrong_action",
    "incomplete_result",
    "did_not_follow_instruction",
    "wrong_information",
    "delivery_not_completed",
    "verification_failed",
    "other",
)

#: Human-readable labels for each failure category. The API returns the label so
#: the frontend never hardcodes prose.
FAILURE_CATEGORY_LABELS: dict[str, str] = {
    "wrong_interpretation": "Wrong interpretation",
    "wrong_action": "Wrong action",
    "incomplete_result": "Incomplete result",
    "did_not_follow_instruction": "Did not follow instruction",
    "wrong_information": "Wrong information",
    "delivery_not_completed": "Failed to deliver to requested application",
    "verification_failed": "Verification failed",
    "other": "Other",
}

#: Structured, checkable completion criteria. These are the deterministic signals
#: the existing verifier and executor already produce; keeping them together lets
#: the experience record say exactly *which* requirement was unmet.
COMPLETION_CRITERIA: tuple[str, ...] = (
    "intent_match",
    "required_information_present",
    "transformation_completed",
    "requested_application_used",
    "requested_destination_reached",
    "execution_completed",
    "delivery_verified",
    "final_result_valid",
)

#: A conservative secret scrubber. It is a best-effort filter, not a guarantee:
#: it removes obvious credential shapes before an experience is persisted so a
#: pasted token does not become durable memory (spec section 23).
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE), r"\1=[redacted]"),
    (re.compile(r"\b(api[_-]?key|apikey|token|secret|bearer)\s*[:=]\s*\S+", re.IGNORECASE), r"\1=[redacted]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"), "[redacted]"),
    (re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[redacted]"),
    (re.compile(r"\b\d{12,19}\b"), "[redacted-number]"),
)

#: Bound on any single stored text field, so one pathological request cannot blow
#: up the store. Task representations stay compact by design.
_MAX_TEXT = 2000
_MAX_LIST = 24
_MAX_LIST_ITEM = 240


def sanitize_text(value: Any, *, limit: int = _MAX_TEXT) -> str:
    """Return a bounded, credential-scrubbed string.

    Whitespace is normalized and one line is kept compact; the result is safe to
    persist and safe to embed. This is deliberately conservative — when in doubt
    it truncates rather than inventing content.
    """

    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    collapsed = " ".join(value.split())
    for pattern, replacement in _SECRET_PATTERNS:
        collapsed = pattern.sub(replacement, collapsed)
    return collapsed[:limit]


def sanitize_list(values: Any, *, limit: int = _MAX_LIST) -> list[str]:
    """Return a de-duplicated, bounded list of sanitized strings."""

    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: list[str] = []
    for item in values:
        text = sanitize_text(item, limit=_MAX_LIST_ITEM)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:limit]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for a record."""

    return datetime.now(timezone.utc).isoformat()


@dataclass
class CompletionChecks:
    """Structured, deterministic completion criteria for one task.

    ``None`` means "not applicable / not checkable" and is deliberately distinct
    from ``False`` ("checkable and it did not hold"). That distinction is what
    keeps an unverifiable UI effect from being reported as a verified success.
    """

    intent_match: bool | None = None
    required_information_present: bool | None = None
    transformation_completed: bool | None = None
    requested_application_used: bool | None = None
    requested_destination_reached: bool | None = None
    execution_completed: bool | None = None
    delivery_verified: bool | None = None
    final_result_valid: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in COMPLETION_CRITERIA}

    @classmethod
    def from_dict(cls, data: Any) -> "CompletionChecks":
        checks = cls()
        if not isinstance(data, dict):
            return checks
        for name in COMPLETION_CRITERIA:
            value = data.get(name)
            if isinstance(value, bool) or value is None:
                setattr(checks, name, value)
        return checks

    def unmet(self) -> list[str]:
        """Return criteria that are checkable and did not hold."""

        return [name for name in COMPLETION_CRITERIA if getattr(self, name) is False]

    def all_checkable_met(self) -> bool:
        """True when nothing checkable failed. Unchecked criteria are not a pass."""

        return not self.unmet()


@dataclass
class Experience:
    """One compact, durable record of a completed interaction.

    The field set is the union of the success and failure requirements from the
    specification; failure-only fields stay empty on a success record.
    """

    #: Stable identity. The user never types this; it is generated here.
    experience_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=utc_now)
    experience_type: str = "successful_task"

    # -- provenance ---------------------------------------------------------
    #: API task record id, i.e. the id the UI attaches feedback to.
    record_id: str = ""
    #: Brain execution task id.
    task_id: str = ""
    conversation_id: str = ""
    #: The model that produced the interpretation/plan, for model-agnostic audit.
    model_used: str = ""

    # -- what the user asked / what Atlas understood ------------------------
    original_user_request: str = ""
    interpreted_intent: str = ""
    desired_outcome: str = ""
    task_type: str = "unknown"
    goal: str = "unknown"
    request_type: str = "unknown"
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    #: "notepad", "file", or "" — the single most commonly lost requirement.
    requested_destination: str = ""
    requested_format: str = ""
    constraints: list[str] = field(default_factory=list)

    # -- how Atlas acted ----------------------------------------------------
    generated_plan: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    execution_steps: list[dict[str, Any]] = field(default_factory=list)
    verification_result: dict[str, Any] = field(default_factory=dict)
    final_result: str = ""
    #: Deterministic completion criteria evaluated after execution.
    completion_checks: CompletionChecks = field(default_factory=CompletionChecks)

    # -- what happened afterwards ------------------------------------------
    outcome: str = "unknown"
    feedback: str = "unknown"
    feedback_reason: str = ""
    failure_category: str = ""
    user_correction: str = ""
    failed_step: str = ""
    expected_behavior: str = ""
    actual_behavior: str = ""
    #: Free-text "response quality" feedback, deliberately optional.
    response_quality: str = ""

    # -- learning linkage ---------------------------------------------------
    state: str = "new"
    #: Experience ids that informed this run's planning context.
    retrieved_experience_ids: list[str] = field(default_factory=list)
    #: A strategy/lesson id if one was applied (reasoning-policy placeholder).
    reasoning_strategy_ids: list[str] = field(default_factory=list)
    #: The attempt* this record belongs to, so a retry forms a recoverable
    #: trajectory: attempt_1 failure -> attempt_2 success (spec section 13).
    attempt: int = 1
    #: The experience this record retried/recovered, when applicable.
    recovers_experience_id: str = ""
    #: Count of independent confirmations (same workflow succeeding again).
    confirmations: int = 0
    #: Which pattern-analysis group this belongs to, when one was derived.
    pattern_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "timestamp": self.timestamp,
            "experience_type": self.experience_type,
            "record_id": self.record_id,
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "model_used": self.model_used,
            "original_user_request": self.original_user_request,
            "interpreted_intent": self.interpreted_intent,
            "desired_outcome": self.desired_outcome,
            "task_type": self.task_type,
            "goal": self.goal,
            "request_type": self.request_type,
            "extracted_entities": dict(self.extracted_entities),
            "requested_destination": self.requested_destination,
            "requested_format": self.requested_format,
            "constraints": list(self.constraints),
            "generated_plan": list(self.generated_plan),
            "tools_used": list(self.tools_used),
            "execution_steps": [dict(step) for step in self.execution_steps],
            "verification_result": dict(self.verification_result),
            "final_result": self.final_result,
            "completion_checks": self.completion_checks.to_dict(),
            "outcome": self.outcome,
            "feedback": self.feedback,
            "feedback_reason": self.feedback_reason,
            "failure_category": self.failure_category,
            "user_correction": self.user_correction,
            "failed_step": self.failed_step,
            "expected_behavior": self.expected_behavior,
            "actual_behavior": self.actual_behavior,
            "response_quality": self.response_quality,
            "state": self.state,
            "retrieved_experience_ids": list(self.retrieved_experience_ids),
            "reasoning_strategy_ids": list(self.reasoning_strategy_ids),
            "attempt": self.attempt,
            "recovers_experience_id": self.recovers_experience_id,
            "confirmations": self.confirmations,
            "pattern_key": self.pattern_key,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Experience | None":
        """Rebuild a record from untrusted (on-disk) data without raising."""

        if not isinstance(data, dict):
            return None
        experience_id = sanitize_text(data.get("experience_id"), limit=64)
        if not experience_id:
            return None
        experience_type = str(data.get("experience_type") or "").strip()
        if experience_type not in EXPERIENCE_TYPES:
            experience_type = "successful_task"
        state = str(data.get("state") or "new").strip()
        if state not in EXPERIENCE_STATES:
            state = "new"
        outcome = str(data.get("outcome") or "unknown").strip()
        if outcome not in OUTCOMES:
            outcome = "unknown"
        category = str(data.get("failure_category") or "").strip()
        if category and category not in FAILURE_CATEGORIES:
            category = "other"

        def number(value: Any, default: int) -> int:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return default

        steps = data.get("execution_steps")
        clean_steps: list[dict[str, Any]] = []
        if isinstance(steps, list):
            for step in steps[:_MAX_LIST]:
                if not isinstance(step, dict):
                    continue
                clean_steps.append(
                    {
                        "capability": sanitize_text(step.get("capability"), limit=_MAX_LIST_ITEM),
                        "status": sanitize_text(step.get("status"), limit=64),
                        "detail": sanitize_text(step.get("detail"), limit=_MAX_LIST_ITEM),
                    }
                )
        verification = data.get("verification_result")
        entities = data.get("extracted_entities")
        return cls(
            experience_id=experience_id,
            timestamp=sanitize_text(data.get("timestamp"), limit=64) or utc_now(),
            experience_type=experience_type,
            record_id=sanitize_text(data.get("record_id"), limit=64),
            task_id=sanitize_text(data.get("task_id"), limit=64),
            conversation_id=sanitize_text(data.get("conversation_id"), limit=64),
            model_used=sanitize_text(data.get("model_used"), limit=64),
            original_user_request=sanitize_text(data.get("original_user_request")),
            interpreted_intent=sanitize_text(data.get("interpreted_intent"), limit=240),
            desired_outcome=sanitize_text(data.get("desired_outcome"), limit=500),
            task_type=sanitize_text(data.get("task_type"), limit=64) or "unknown",
            goal=sanitize_text(data.get("goal"), limit=64) or "unknown",
            request_type=sanitize_text(data.get("request_type"), limit=64) or "unknown",
            extracted_entities={
                sanitize_text(key, limit=64): sanitize_text(value, limit=_MAX_LIST_ITEM)
                for key, value in entities.items()
                if isinstance(key, str) and isinstance(value, (str, int, float, bool))
            } if isinstance(entities, dict) else {},
            requested_destination=sanitize_text(data.get("requested_destination"), limit=120),
            requested_format=sanitize_text(data.get("requested_format"), limit=64),
            constraints=sanitize_list(data.get("constraints")),
            generated_plan=sanitize_list(data.get("generated_plan")),
            tools_used=sanitize_list(data.get("tools_used")),
            execution_steps=clean_steps,
            verification_result=dict(verification) if isinstance(verification, dict) else {},
            final_result=sanitize_text(data.get("final_result")),
            completion_checks=CompletionChecks.from_dict(data.get("completion_checks")),
            outcome=outcome,
            feedback=sanitize_text(data.get("feedback"), limit=32) or "unknown",
            feedback_reason=sanitize_text(data.get("feedback_reason"), limit=240),
            failure_category=category,
            user_correction=sanitize_text(data.get("user_correction")),
            failed_step=sanitize_text(data.get("failed_step"), limit=240),
            expected_behavior=sanitize_text(data.get("expected_behavior"), limit=500),
            actual_behavior=sanitize_text(data.get("actual_behavior"), limit=500),
            response_quality=sanitize_text(data.get("response_quality"), limit=32),
            state=state,
            retrieved_experience_ids=sanitize_list(data.get("retrieved_experience_ids")),
            reasoning_strategy_ids=sanitize_list(data.get("reasoning_strategy_ids")),
            attempt=number(data.get("attempt"), 1) or 1,
            recovers_experience_id=sanitize_text(data.get("recovers_experience_id"), limit=64),
            confirmations=number(data.get("confirmations"), 0),
            pattern_key=sanitize_text(data.get("pattern_key"), limit=120),
        )

    # -- derived helpers ----------------------------------------------------

    def is_success(self) -> bool:
        return self.outcome == "success" and self.experience_type != "failed_task"

    def is_failure(self) -> bool:
        return self.outcome == "failure" or self.experience_type == "failed_task"

    def retrieval_text(self) -> str:
        """Return the text an embedding/lexical index should match against.

        Deliberately built from the request-shaped fields — never from the final
        result, which can be long and would swamp the signal.
        """

        parts = [
            self.original_user_request,
            self.interpreted_intent,
            self.desired_outcome,
            self.goal,
            self.task_type,
            self.requested_destination,
            " ".join(self.tools_used),
        ]
        return " ".join(part for part in parts if part).strip()

    def reliability(self) -> float:
        """A compact 0..1 trust score used when several experiences compete.

        Verified successes with confirmations and an explicit correction are the
        strongest signals; a single unverified interaction is weak. A failure is
        not "low reliability" — it is reliable evidence *about a failure*, which
        is why failures score on their category strength rather than being zeroed.
        """

        score = 0.4
        if self.state == "reliable":
            score += 0.3
        elif self.state == "evaluated":
            score += 0.15
        if self.outcome == "unknown":
            score -= 0.15
        score += min(0.2, 0.05 * self.confirmations)
        if self.completion_checks.delivery_verified is True:
            score += 0.1
        if self.user_correction:
            score += 0.05
        return max(0.0, min(1.0, score))
