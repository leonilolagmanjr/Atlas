"""Experience service: the small façade the rest of Atlas talks to.

This is the *only* module Brain and the API need to know about. It orchestrates
the store, the retrieval layer, the builder, and experience promotion, and it
keeps every operation bounded and failure-tolerant so the feedback loop can never
make a normal interaction slower or more fragile (spec section 24).

Responsibilities:

* ``record_task_outcome`` — after a meaningful task completes, write a compact
  **unevaluated** experience so feedback can later attach to it.
* ``apply_feedback``      — attach the user's explicit Success/Failed answer
  (plus optional reason and correction) to the right experience and persist it.
* ``retrieve_for_planning`` — return bounded, rendered experience context.
* ``analysis_report``     — periodic, cheap pattern analysis (never per message).

It never executes anything and never calls a model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from config import (
    ENABLE_EXPERIENCE_MEMORY,
    EXPERIENCE_ANALYSIS_THRESHOLD,
    EXPERIENCE_MIN_RELEVANCE,
    EXPERIENCE_RETRIEVAL_LIMIT,
    OLLAMA_MODEL,
)
from experience.builder import build_experience
from experience.memory import ExperienceContext, ExperienceEmbedder, ExperienceMemory
from experience.models import (
    Experience,
    FAILURE_CATEGORIES,
    FAILURE_CATEGORY_LABELS,
    sanitize_text,
)
from experience.store import ExperienceStore, FeedbackEvent

logger = logging.getLogger(__name__)

#: Capabilities that make a request "meaningful" for feedback purposes. A pure
#: conversational answer has no task outcome to evaluate, so it is not offered
#: feedback controls (spec section 3).
MEANINGFUL_CAPABILITY_PREFIXES: tuple[str, ...] = (
    "applications.",
    "filesystem.write",
    "filesystem.create_folder",
    "filesystem.move",
    "filesystem.copy",
    "computer.",
    "web.research",
    "powershell.execute",
    "content.format",
)

#: Number of confirmations of the same workflow after which an experience is
#: promoted to ``reliable``. A single interaction never becomes global policy
#: (spec sections 16 and 17).
_RELIABLE_CONFIRMATIONS = 2


@dataclass
class FeedbackRequest:
    """Normalized, validated user feedback."""

    record_id: str
    outcome: str  # "success" | "failure"
    reason: str = ""
    failure_category: str = ""
    correction: str = ""
    expected_behavior: str = ""
    response_quality: str = ""

    @classmethod
    def from_payload(cls, record_id: str, payload: dict[str, Any]) -> "FeedbackRequest":
        """Build a feedback request from untrusted API input.

        ``outcome`` is the only required field. A reason and a correction are
        always optional — the user is never forced to justify a click (spec
        section 4).
        """

        raw_outcome = str(payload.get("outcome") or "").strip().casefold()
        outcome = "success" if raw_outcome in {"success", "succeeded", "ok", "pass"} else (
            "failure" if raw_outcome in {"failure", "failed", "fail"} else ""
        )
        category = sanitize_text(payload.get("failure_category"), limit=64)
        if category and category not in FAILURE_CATEGORIES:
            category = "other"
        quality = str(payload.get("response_quality") or "").strip().casefold()
        if quality not in {"positive", "negative"}:
            quality = ""
        return cls(
            record_id=sanitize_text(record_id, limit=64),
            outcome=outcome,
            reason=sanitize_text(payload.get("reason"), limit=240),
            failure_category=category,
            correction=sanitize_text(payload.get("correction")),
            expected_behavior=sanitize_text(payload.get("expected_behavior"), limit=500),
            response_quality=quality,
        )


@dataclass
class FeedbackResult:
    """Outcome of applying feedback, returned to the API/UI."""

    ok: bool
    detail: str = ""
    experience_id: str = ""
    feedback: str = "unknown"
    failure_category: str = ""
    failure_category_label: str = ""
    user_correction: str = ""
    #: True when the feedback disagreed with Atlas's own verification. This is
    #: surfaced (not hidden) because a disagreement is the most informative
    #: signal the loop can receive.
    disagrees_with_verification: bool = False


@dataclass
class TaskOutcome:
    """Result of recording a completed task as an experience candidate."""

    recorded: bool = False
    experience_id: str = ""
    meaningful: bool = False
    outcome: str = "unknown"


class ExperienceService:
    """Own the experience loop: record, evaluate, retrieve, analyse."""

    def __init__(
        self,
        *,
        store: ExperienceStore | None = None,
        embedder: ExperienceEmbedder | None = None,
        enabled: bool = ENABLE_EXPERIENCE_MEMORY,
        retrieval_limit: int = EXPERIENCE_RETRIEVAL_LIMIT,
        min_relevance: float = EXPERIENCE_MIN_RELEVANCE,
        analysis_threshold: int = EXPERIENCE_ANALYSIS_THRESHOLD,
        model_used: str = OLLAMA_MODEL,
    ) -> None:
        self._enabled = bool(enabled)
        self._store = store if store is not None else (
            ExperienceStore() if self._enabled else None
        )
        self._memory = ExperienceMemory(
            store=self._store,
            embedder=embedder,
            limit=retrieval_limit,
            min_relevance=min_relevance,
        )
        self._analysis_threshold = max(1, int(analysis_threshold))
        self._model_used = model_used
        # Small in-process cache of record_id -> experience_id, so a feedback
        # click resolves the right experience without scanning the store.
        self._record_index: dict[str, str] = {}
        self._last_analysis_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled and self._store is not None

    @property
    def store(self) -> ExperienceStore | None:
        return self._store

    # -- feedback eligibility ------------------------------------------------

    def should_offer_feedback(self, context: Any) -> bool:
        """Decide whether a completed request merits Success/Failed controls.

        Feedback is offered for tool executions, research, application actions,
        multi-step tasks, generated deliverables, and tasks with explicit success
        criteria. Casual conversation is not offered feedback (spec section 3).
        """

        if not self.enabled or context is None:
            return False
        status = _status_value(getattr(context, "status", None))
        # Only a settled task can be evaluated; a paused or cancelled one cannot.
        if status not in {"completed", "failed"}:
            return False
        plan = getattr(context, "execution_plan", None)
        steps = getattr(plan, "steps", None) or []
        tool_calls = getattr(context, "tool_calls", None) or []
        if not steps and not tool_calls:
            # A direct answer has no task outcome. That is conversational.
            return False
        capabilities = {str(step.metadata.get("tool") or "") for step in steps if getattr(step, "metadata", None)}
        capabilities.update(
            str(call.get("tool") or "") for call in tool_calls if isinstance(call, dict)
        )
        # Multi-step work is always meaningful, even when each step is read-only.
        if len(steps) > 1:
            return True
        for capability in capabilities:
            if not capability:
                continue
            if any(capability.startswith(prefix) for prefix in MEANINGFUL_CAPABILITY_PREFIXES):
                return True
        # A single read-only lookup is a question answered with a plan; a
        # research task (multi-attempt retrieval) still counts.
        task = _task_object(context)
        if task is not None and getattr(task, "request_type", "") in {"action", "hybrid"}:
            return True
        return False

    # -- recording -----------------------------------------------------------

    def record_task_outcome(
        self,
        context: Any,
        *,
        record_id: str = "",
        conversation_id: str = "",
        retrieved_experience_ids: list[str] | None = None,
        model_used: str | None = None,
    ) -> TaskOutcome:
        """Persist an unevaluated experience candidate for a completed task.

        The initial ``outcome`` mirrors what Atlas itself determined, which is a
        *hypothesis* until the user answers. Feedback can then correct it — the
        record is promoted rather than duplicated.
        """

        if not self.enabled or context is None:
            return TaskOutcome()
        # A context with no request text is not a task. Refusing to record it
        # keeps the store free of empty, meaningless records.
        if not str(getattr(context, "user_input", "") or "").strip():
            return TaskOutcome()
        try:
            status = _status_value(getattr(context, "status", None))
            outcome = "success" if status == "completed" else (
                "failure" if status == "failed" else "unknown"
            )
            meaningful = self.should_offer_feedback(context)
            experience = build_experience(
                context,
                record_id=record_id,
                conversation_id=conversation_id,
                outcome=outcome,
                feedback="unknown",
                retrieved_experience_ids=retrieved_experience_ids or [],
                model_used=model_used or self._model_used,
            )
            # The candidate is ``observed``: Atlas saw what happened, the user
            # has not yet judged it.
            experience.state = "observed"
            assert self._store is not None
            if not self._store.add(experience):
                return TaskOutcome(recorded=False, meaningful=meaningful, outcome=outcome)
            if record_id:
                self._record_index[record_id] = experience.experience_id
            return TaskOutcome(
                recorded=True,
                experience_id=experience.experience_id,
                meaningful=meaningful,
                outcome=outcome,
            )
        except Exception:  # noqa: BLE001 - the loop must never break a request
            logger.exception("Could not record task outcome as an experience")
            return TaskOutcome()

    # -- feedback ------------------------------------------------------------

    def apply_feedback(self, request: FeedbackRequest) -> FeedbackResult:
        """Attach explicit user feedback to the experience for a task record.

        If the task was never recorded (for example the loop was disabled when it
        ran), feedback still produces a durable feedback event so the answer is
        never lost, and the result says so honestly rather than pretending an
        experience was updated.
        """

        if not self.enabled or self._store is None:
            return FeedbackResult(ok=False, detail="Experience memory is disabled.")
        if request.outcome not in {"success", "failure"}:
            return FeedbackResult(ok=False, detail="Feedback must be success or failure.")
        if not request.record_id:
            return FeedbackResult(ok=False, detail="Feedback must reference a task.")

        # Persist the click first: the answer survives even if promotion fails.
        self._store.record_feedback(
            FeedbackEvent(
                record_id=request.record_id,
                outcome=request.outcome,
                reason=request.reason,
                failure_category=request.failure_category,
                correction=request.correction,
                expected_behavior=request.expected_behavior,
                response_quality=request.response_quality,
            )
        )

        experience = self._experience_for_record(request.record_id)
        if experience is None:
            return FeedbackResult(
                ok=True,
                detail=(
                    "Feedback recorded. This task was not stored as an experience "
                    "(the experience loop was off for it), so nothing was updated."
                ),
                feedback=request.outcome,
                failure_category=request.failure_category,
                failure_category_label=FAILURE_CATEGORY_LABELS.get(request.failure_category, ""),
                user_correction=request.correction,
            )

        self._promote(experience, request)
        self._index(experience)
        return FeedbackResult(
            ok=True,
            detail="Feedback stored as experience.",
            experience_id=experience.experience_id,
            feedback=request.outcome,
            failure_category=experience.failure_category,
            failure_category_label=FAILURE_CATEGORY_LABELS.get(experience.failure_category, ""),
            user_correction=experience.user_correction,
            disagrees_with_verification=_disagrees(experience),
        )

    def _experience_for_record(self, record_id: str) -> Experience | None:
        if self._store is None:
            return None
        indexed = self._record_index.get(record_id)
        if indexed:
            found = self._store.find_by_id(indexed)
            if found is not None:
                return found
        candidates = self._store.find_for_record(record_id)
        if not candidates:
            return None
        # The most recent candidate for this record is the one being evaluated.
        chosen = candidates[-1]
        self._record_index[record_id] = chosen.experience_id
        return chosen

    def _promote(self, experience: Experience, request: FeedbackRequest) -> None:
        """Write the user's answer onto the experience and promote its state.

        Correction and confirmation counts are what turn a single interaction
        into a *reliable* experience, and only repeated success does that.
        """

        experience.feedback = request.outcome
        experience.feedback_reason = request.reason
        if request.response_quality:
            experience.response_quality = request.response_quality

        if request.outcome == "success":
            experience.outcome = "success"
            experience.failure_category = ""
            # Idempotent confirmation: answering twice never inflates the count.
            previous = self._store.latest_feedback(request.record_id) if self._store else None
            if previous is None or previous.outcome != "success":
                experience.confirmations = max(1, experience.confirmations + 1)
            else:
                experience.confirmations = max(1, experience.confirmations)
            if experience.user_correction:
                # A corrected-then-successful attempt is a retry trajectory.
                experience.experience_type = "workflow"
            else:
                experience.experience_type = "successful_task"
        else:
            experience.outcome = "failure"
            experience.failure_category = request.failure_category or experience.failure_category or "other"
            if request.correction:
                experience.user_correction = request.correction
            if request.expected_behavior:
                experience.expected_behavior = request.expected_behavior
            experience.experience_type = "correction" if experience.user_correction else "failed_task"
            using_correction = bool(experience.user_correction)
            if using_correction and experience.generated_plan:
                # A named correction is directly reusable, so it is a workflow
                # lesson even though the run itself failed.
                experience.pattern_key = experience.pattern_key or _pattern_key(experience)

        experience.state = "reliable" if (
            experience.outcome == "success"
            and experience.confirmations >= _RELIABLE_CONFIRMATIONS
        ) else "evaluated"

        if self._store is not None:
            self._store.replace(experience)

    def _index(self, experience: Experience) -> None:
        embedder = self._memory._embedder  # noqa: SLF001 - same package, single owner
        if embedder is None:
            return
        try:
            embedder.index(experience)
        except Exception:  # noqa: BLE001
            logger.debug("Experience indexing skipped", exc_info=True)

    # -- retrieval -----------------------------------------------------------

    def retrieve_for_planning(self, request: str, *, task: Any = None) -> ExperienceContext:
        """Return bounded experience context for a new, non-trivial request."""

        if not self.enabled or not request:
            return ExperienceContext()
        try:
            return self._memory.retrieve(request, task=task)
        except Exception:  # noqa: BLE001
            logger.exception("Experience retrieval failed")
            return ExperienceContext()

    def retrieved_ids(self, context: ExperienceContext) -> list[str]:
        return [item.experience.experience_id for item in context.items]

    # -- lifecycle / analysis ------------------------------------------------

    def promote_repeated_patterns(self) -> int:
        """Promote repeated successful workflows and failure patterns.

        This is deliberately conservative: it only marks an experience
        ``reliable`` once the *same* workflow has been confirmed more than once,
        so one interaction cannot rewrite global behaviour (spec section 16).
        Returns the number of promotions.
        """

        if not self.enabled or self._store is None:
            return 0
        groups: dict[str, list[Experience]] = {}
        for experience in self._store.experiences():
            if experience.outcome != "success":
                continue
            key = _pattern_key(experience)
            if key:
                groups.setdefault(key, []).append(experience)
        promoted = 0
        for key, members in groups.items():
            if len(members) < _RELIABLE_CONFIRMATIONS:
                continue
            for experience in members:
                if experience.state == "reliable":
                    continue
                experience.state = "reliable"
                experience.pattern_key = key
                experience.confirmations = max(experience.confirmations, len(members))
                if self._store.replace(experience):
                    promoted += 1
        return promoted

    def analysis_report(self) -> dict[str, Any]:
        """Produce candidate improvements from accumulated experiences.

        Never run per message: it is gated on the number of *newly evaluated*
        experiences since the last analysis (spec section 18). Its output is
        **proposed**, never applied — Atlas does not modify its own code or
        policy automatically (spec section 19).
        """

        if not self.enabled or self._store is None:
            return {"available": False, "reason": "experience memory disabled"}
        experiences = self._store.experiences()
        evaluated = [item for item in experiences if item.state in {"evaluated", "reliable"}]
        if self._last_analysis_count and len(evaluated) - self._last_analysis_count < self._analysis_threshold:
            return {
                "available": False,
                "reason": f"fewer than {self._analysis_threshold} new evaluations since last analysis",
                "evaluated": len(evaluated),
            }

        failures = [item for item in evaluated if item.is_failure()]
        successes = [item for item in evaluated if item.is_success()]
        categories: dict[str, int] = {}
        for item in failures:
            if item.failure_category:
                categories[item.failure_category] = categories.get(item.failure_category, 0) + 1
        destinations_lost = sum(
            1
            for item in failures
            if item.requested_destination
            and (
                item.completion_checks.requested_destination_reached is False
                or item.failure_category == "delivery_not_completed"
            )
        )
        corrected = [item for item in evaluated if item.user_correction]

        proposals: list[dict[str, Any]] = []
        for category, count in sorted(categories.items(), key=lambda pair: -pair[1]):
            if count < 2:
                continue
            proposals.append(
                {
                    "kind": "recurring_failure",
                    "evidence": count,
                    "statement": (
                        f"'{FAILURE_CATEGORY_LABELS.get(category, category)}' occurred "
                        f"{count} times. Consider surfacing this requirement earlier in planning."
                    ),
                }
            )
        if destinations_lost >= 2:
            proposals.append(
                {
                    "kind": "lost_requirement",
                    "evidence": destinations_lost,
                    "statement": (
                        "Destination requirements are frequently lost between intent "
                        "interpretation and execution. Treat a named destination as a hard "
                        "task constraint and verify delivery before reporting success."
                    ),
                }
            )
        for key, members in _group(successes).items():
            if len(members) >= _RELIABLE_CONFIRMATIONS:
                proposals.append(
                    {
                        "kind": "reusable_workflow",
                        "evidence": len(members),
                        "statement": (
                            f"Workflow '{key}' succeeded {len(members)} times and is a "
                            "candidate reusable strategy."
                        ),
                        "procedure": members[-1].generated_plan,
                    }
                )

        self._last_analysis_count = len(evaluated)
        return {
            "available": True,
            "evaluated": len(evaluated),
            "successes": len(successes),
            "failures": len(failures),
            "corrections": len(corrected),
            "failure_categories": categories,
            "proposals": proposals[:10],
            # Explicitly proposed, never applied: safe self-improvement boundary.
            "applied": False,
        }

    def status(self) -> dict[str, Any]:
        """Small, safe snapshot for the UI/API (counts only, no task content)."""

        if not self.enabled or self._store is None:
            return {"enabled": False}
        counts = self._store.counts()
        return {
            "enabled": True,
            "counts": counts,
            "failure_categories": list(FAILURE_CATEGORIES),
            "failure_category_labels": FAILURE_CATEGORY_LABELS,
        }


# -- helpers --------------------------------------------------------------------


def _status_value(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value or "").casefold()


def _task_object(context: Any) -> Any:
    metadata = getattr(context, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get("task_object")
    return None


def _pattern_key(experience: Experience) -> str:
    """A stable key naming the workflow shape, for grouping and promotion.

    It is built from the goal, the tool set, and the destination — the pieces
    that define "this kind of task" — never from the free-text subject.
    """

    goal = (experience.goal or experience.task_type or "unknown").strip().casefold()
    tools = ",".join(sorted(experience.tools_used))
    destination = (experience.requested_destination or "none").strip().casefold()
    if not goal and not tools:
        return ""
    return f"{goal}|{tools}|{destination}"


def _group(experiences: list[Experience]) -> dict[str, list[Experience]]:
    groups: dict[str, list[Experience]] = {}
    for experience in experiences:
        key = _pattern_key(experience)
        if key:
            groups.setdefault(key, []).append(experience)
    return groups


def _disagrees(experience: Experience) -> bool:
    """True when user feedback contradicts Atlas's own verification.

    A user saying "failed" about work Atlas verified, or "success" about work the
    verifier explicitly failed, is the most valuable signal in the loop: it tells
    us the verification contract itself needs attention.
    """

    if experience.outcome == "failure" and experience.completion_checks.delivery_verified is True:
        return True
    if experience.outcome == "success" and experience.completion_checks.execution_completed is False:
        return True
    return False
