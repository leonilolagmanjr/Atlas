"""Turn a completed task into a compact experience record.

This module is the bridge between the *existing* Atlas execution machinery and
experience memory. It reads what already exists — the Task IR, the execution
plan, the tool calls, the verifier's outcomes, the failure classification — and
derives:

* a compact, sanitized :class:`experience.models.Experience`,
* structured completion criteria evaluated from **deterministic evidence**, not
  from a model's claim.

The most important rule here (spec sections 14 and 15) is that an LLM saying
"Done." is not evidence. Completion criteria are only marked ``True`` when a
tool result or the existing verifier supports it; when there is no checkable
evidence the criterion stays ``None`` (unknown) rather than being upgraded.

Nothing in this module executes, plans, or calls a model.
"""

from __future__ import annotations

import logging
from typing import Any

from config import OLLAMA_MODEL
from experience.models import (
    CompletionChecks,
    Experience,
    FAILURE_CATEGORIES,
    sanitize_list,
    sanitize_text,
)

logger = logging.getLogger(__name__)

#: Capabilities that put content somewhere the user asked for.
_DELIVERY_CAPABILITIES: frozenset[str] = frozenset(
    {
        "applications.write_text",
        "filesystem.write",
        "computer.type",
        "computer.keypress",
    }
)

#: Which requirements a capability is expected to satisfy, so the criteria are
#: derived from what Atlas actually planned rather than from the request text.
_APPLICATION_CAPABILITIES: frozenset[str] = frozenset(
    {"applications.write_text", "applications.launch", "applications.launch_named"}
)


def build_experience(
    context: Any,
    *,
    record_id: str = "",
    conversation_id: str = "",
    outcome: str = "unknown",
    feedback: str = "unknown",
    feedback_reason: str = "",
    failure_category: str = "",
    user_correction: str = "",
    expected_behavior: str = "",
    response_quality: str = "",
    retrieved_experience_ids: list[str] | None = None,
    attempt: int = 1,
    recovers_experience_id: str = "",
    model_used: str | None = None,
) -> Experience:
    """Build one experience from a completed :class:`models.ExecutionContext`.

    ``outcome`` is the authoritative task outcome: it may come from the executor
    (``COMPLETED``/``FAILED``) or be corrected by explicit user feedback. The
    completion checks are always derived from execution evidence, never from the
    outcome argument, so user feedback and Atlas's own verification stay separate
    signals (that separation is what makes a disagreement visible).
    """

    task = _task_object(context)
    checks = evaluate_completion(context, task)
    experience_type = _experience_type(outcome, user_correction)
    resolved_outcome = outcome if outcome in {"success", "failure"} else "unknown"

    return Experience(
        record_id=sanitize_text(record_id, limit=64),
        task_id=sanitize_text(getattr(context, "task_id", ""), limit=64),
        conversation_id=sanitize_text(conversation_id, limit=64),
        model_used=sanitize_text(model_used if model_used is not None else OLLAMA_MODEL, limit=64),
        original_user_request=sanitize_text(getattr(context, "user_input", "")),
        interpreted_intent=_interpreted_intent(task),
        desired_outcome=sanitize_text(getattr(task, "desired_outcome", ""), limit=500),
        task_type=sanitize_text(getattr(task, "task_type", ""), limit=64) or "unknown",
        goal=sanitize_text(getattr(task, "goal", ""), limit=64) or "unknown",
        request_type=sanitize_text(getattr(task, "request_type", ""), limit=64) or "unknown",
        extracted_entities=_entities(task),
        requested_destination=_destination(task),
        requested_format=_format(task),
        constraints=sanitize_list(getattr(task, "constraints", [])),
        generated_plan=_plan_steps(context),
        tools_used=_tools_used(context, task),
        execution_steps=_execution_steps(context),
        verification_result=_verification_summary(context),
        final_result=sanitize_text(getattr(context, "final_response", "")),
        completion_checks=checks,
        outcome=resolved_outcome,
        feedback=sanitize_text(feedback, limit=32) or "unknown",
        feedback_reason=sanitize_text(feedback_reason, limit=240),
        failure_category=_failure_category(failure_category, context, checks, resolved_outcome),
        user_correction=sanitize_text(user_correction),
        failed_step=_failed_step(context),
        expected_behavior=sanitize_text(expected_behavior, limit=500) or _default_expected(task),
        actual_behavior=sanitize_text(_default_actual(context, checks), limit=500),
        response_quality=sanitize_text(response_quality, limit=32),
        experience_type=experience_type,
        retrieved_experience_ids=sanitize_list(retrieved_experience_ids or []),
        attempt=max(1, int(attempt or 1)),
        recovers_experience_id=sanitize_text(recovers_experience_id, limit=64),
    )


# -- deterministic completion evaluation ----------------------------------------


def evaluate_completion(context: Any, task: Any) -> CompletionChecks:
    """Derive structured completion criteria from real execution evidence.

    Each criterion is ``True``, ``False``, or ``None``:

    * ``True``  — a tool result or the verifier positively supports it,
    * ``False`` — a checkable requirement that did **not** hold,
    * ``None``  — not checkable in this environment (honest unknown).

    This is what stops "write this into Notepad" from being recorded as a
    verified success merely because the model said so.
    """

    checks = CompletionChecks()
    plan_steps = _plan_steps_raw(context)
    tool_calls = list(getattr(context, "tool_calls", []) or [])
    verifications = list(getattr(context, "verification_results", []) or [])
    status = _status_value(getattr(context, "status", None))

    expected_capabilities = {
        str(step.get("tool") or "") for step in plan_steps if step.get("tool")
    }
    delivered_capabilities = {
        str(call.get("tool") or "")
        for call in tool_calls
        if isinstance(call, dict) and call.get("success")
    }

    # --- intent_match: the interpretation produced a plan and did not ask for
    # clarification or an unsupported capability --------------------------------
    needs_clarification = bool(getattr(task, "needs_clarification", False))
    if needs_clarification:
        checks.intent_match = False
    elif expected_capabilities or plan_steps:
        checks.intent_match = True
    elif status in {"completed", "failed"}:
        # An answered conversational/informational request has no plan; the
        # interpretation still "matched" if Atlas produced a response.
        checks.intent_match = bool(getattr(context, "final_response", "")) or None
    if status == "failed" and not expected_capabilities:
        checks.intent_match = False

    # --- execution_completed ---------------------------------------------------
    if status == "completed":
        checks.execution_completed = True
    elif status in {"failed", "cancelled"}:
        checks.execution_completed = False

    # --- required_information_present -----------------------------------------
    # Checkable only when the plan actually gathered information.
    gathering = expected_capabilities & {"web.search", "web.fetch", "web.research", "filesystem.read", "filesystem.search", "filesystem.search_content"}
    if gathering:
        checks.required_information_present = bool(gathering & delivered_capabilities)

    # --- transformation_completed ---------------------------------------------
    if "content.format" in expected_capabilities:
        checks.transformation_completed = _format_verified(tool_calls)
    elif "content.generate" in expected_capabilities:
        checks.transformation_completed = _generation_succeeded(tool_calls)

    # --- requested_application_used / requested_destination_reached -----------
    application_expected = bool(expected_capabilities & _APPLICATION_CAPABILITIES)
    if application_expected:
        launched = bool(expected_capabilities & {"applications.launch", "applications.launch_named"})
        if launched:
            checks.requested_application_used = _launch_succeeded(tool_calls)
    if expected_capabilities & _DELIVERY_CAPABILITIES:
        checks.requested_destination_reached = _delivery_reached(tool_calls, verifications)

    # --- delivery_verified: only the verifier may set this --------------------
    # ``verified`` here means an independent/definitionally-strong signal. An
    # ``unverified`` outcome is left as None (unknown), never True.
    delivery_verifications = [
        item for item in verifications
        if isinstance(item, dict) and str(item.get("capability") or "") in _DELIVERY_CAPABILITIES
    ]
    if delivery_verifications:
        if any(item.get("verified") is True for item in delivery_verifications):
            checks.delivery_verified = True
        elif any(item.get("status") == "failed" for item in delivery_verifications):
            checks.delivery_verified = False
        # Otherwise it stays None: the tool sent the content but the effect
        # could not be confirmed. That is not a verified success.

    # --- final_result_valid ---------------------------------------------------
    if status == "completed":
        checks.final_result_valid = bool(getattr(context, "final_response", "").strip())
    elif status == "failed":
        checks.final_result_valid = False

    return checks


def _format_verified(tool_calls: list[Any]) -> bool | None:
    for call in tool_calls:
        if not isinstance(call, dict) or call.get("tool") != "content.format":
            continue
        output = call.get("output")
        if isinstance(output, dict):
            metadata = output.get("metadata")
            if isinstance(metadata, dict) and "verification_passed" in metadata:
                return bool(metadata.get("verification_passed"))
            if output.get("text"):
                return True
    return None


def _generation_succeeded(tool_calls: list[Any]) -> bool | None:
    for call in tool_calls:
        if not isinstance(call, dict) or call.get("tool") != "content.generate":
            continue
        output = call.get("output")
        if isinstance(output, dict) and isinstance(output.get("text"), str):
            return bool(output["text"].strip())
    return None


def _launch_succeeded(tool_calls: list[Any]) -> bool | None:
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        if not str(call.get("tool") or "").startswith("applications.launch"):
            continue
        output = call.get("output")
        if isinstance(output, dict):
            return bool(output.get("pid"))
    return None


def _delivery_reached(tool_calls: list[Any], verifications: list[Any]) -> bool | None:

    verified = any(
        isinstance(item, dict)
        and str(item.get("capability") or "") in _DELIVERY_CAPABILITIES
        and item.get("verified") is True
        for item in verifications
    )
    if verified:
        return True

    reached: bool | None = None
    for call in tool_calls:
        if not isinstance(call, dict) or not call.get("success"):
            continue
        tool = str(call.get("tool") or "")
        if tool not in _DELIVERY_CAPABILITIES:
            continue
        output = call.get("output")
        if tool == "applications.write_text" and isinstance(output, dict):
            if output.get("observed") is False or output.get("verification") == "failed":
                return False
            if output.get("characters") or output.get("observed") is True:
                reached = True
        elif tool == "filesystem.write" and isinstance(output, dict):
            reached = bool(output.get("path")) or reached
        else:
            reached = True
    return reached


# -- field extraction -----------------------------------------------------------


def _task_object(context: Any) -> Any:
    metadata = getattr(context, "metadata", None)
    if isinstance(metadata, dict):
        task = metadata.get("task_object")
        if task is not None:
            return task
    return None


def _status_value(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value or "").casefold()


def _interpreted_intent(task: Any) -> str:
    """Summarize how Atlas read the request: the goal plus its operations."""

    if task is None:
        return ""
    goal = str(getattr(task, "goal", "") or "")
    operations = [str(item) for item in (getattr(task, "operations", None) or [])]
    parts = [part for part in (goal, " + ".join(operations)) if part]
    return sanitize_text(": ".join(parts) if len(parts) > 1 else (parts[0] if parts else ""), limit=240)


def _entities(task: Any) -> dict[str, Any]:
    if task is None:
        return {}
    entities = getattr(task, "entities", None)
    if not isinstance(entities, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in entities.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (str, int, float, bool)):
            text = sanitize_text(value, limit=240)
            if text:
                cleaned[sanitize_text(key, limit=64)] = text
    return cleaned


def _destination(task: Any) -> str:
    if task is None:
        return ""
    entities = getattr(task, "entities", None)
    if not isinstance(entities, dict):
        return ""
    for key in ("application", "filename", "destination"):
        value = entities.get(key)
        if isinstance(value, str) and value.strip():
            return sanitize_text(value, limit=120)
    return ""


def _format(task: Any) -> str:
    if task is None:
        return ""
    entities = getattr(task, "entities", None)
    if isinstance(entities, dict):
        value = entities.get("content_type")
        if isinstance(value, str) and value.strip():
            return sanitize_text(value, limit=64)
    return ""


def _tools_used(context: Any, task: Any) -> list[str]:
    tools: list[str] = []
    for call in getattr(context, "tool_calls", []) or []:
        if isinstance(call, dict) and call.get("tool"):
            tools.append(str(call["tool"]))
    if not tools and task is not None:
        for action in getattr(task, "actions", []) or []:
            capability = getattr(action, "capability", None)
            if capability:
                tools.append(str(capability))
    return sanitize_list(tools)


def _plan_steps(context: Any) -> list[str]:
    return sanitize_list(
        [
            step.get("capability") or step.get("tool") or step.get("id") or ""
            for step in _plan_steps_raw(context)
        ]
    )


def _plan_steps_raw(context: Any) -> list[dict[str, Any]]:
    plan = getattr(context, "execution_plan", None)
    steps = getattr(plan, "steps", None)
    if not isinstance(steps, list):
        return []
    result: list[dict[str, Any]] = []
    for step in steps:
        metadata = getattr(step, "metadata", None) or {}
        result.append(
            {
                "id": str(getattr(step, "id", "") or ""),
                "tool": str(metadata.get("tool") or ""),
                "capability": str(metadata.get("tool") or metadata.get("capability") or ""),
                "status": _status_value(getattr(step, "status", "")),
            }
        )
    return result


def _execution_steps(context: Any) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for step in _plan_steps_raw(context):
        steps.append(
            {
                "capability": step["capability"],
                "status": step["status"],
                "detail": "",
            }
        )
    for call in getattr(context, "tool_calls", []) or []:
        if not isinstance(call, dict):
            continue
        status = str(call.get("status") or ("completed" if call.get("success") else "failed"))
        detail = sanitize_text(call.get("error") or "", limit=240)
        steps.append(
            {
                "capability": sanitize_text(call.get("tool"), limit=240),
                "status": sanitize_text(status, limit=64),
                "detail": detail,
            }
        )
    return steps[:24]


def _verification_summary(context: Any) -> dict[str, Any]:
    results = list(getattr(context, "verification_results", []) or [])
    counts: dict[str, int] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    classification = getattr(context, "metadata", {}).get("failure_classification")
    return {
        "verified": counts.get("verified", 0),
        "unverified": counts.get("unverified", 0),
        "failed": counts.get("failed", 0),
        "failure_category": str(classification.get("category") or "") if isinstance(classification, dict) else "",
    }


def _failed_step(context: Any) -> str:
    plan = getattr(context, "execution_plan", None)
    steps = getattr(plan, "steps", None)
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if _status_value(getattr(step, "status", "")) == "failed":
            metadata = getattr(step, "metadata", None) or {}
            return sanitize_text(metadata.get("tool") or getattr(step, "name", ""), limit=240)
    return ""


def _failure_category(
    provided: str,
    context: Any,
    checks: CompletionChecks,
    outcome: str,
) -> str:
    """Pick the failure category, preferring what the user actually said.

    When the user supplies a category it always wins, because the user observed
    the failure and Atlas only inferred it. Otherwise Atlas derives a category
    deterministically from the verification evidence.
    """

    if provided and provided in FAILURE_CATEGORIES:
        return provided
    if outcome != "failure":
        return ""
    classification = getattr(context, "metadata", {}).get("failure_classification")
    if isinstance(classification, dict):
        derived = str(classification.get("category") or "")
        mapping = {
            "ui_verification_failed": "verification_failed",
            "permission": "wrong_action",
            "unknown_capability": "wrong_action",
            "invalid_arguments": "wrong_action",
            "ui_target_missing": "delivery_not_completed",
            "missing_target": "wrong_interpretation",
        }
        if derived in mapping:
            return mapping[derived]

    if checks.delivery_verified is False or checks.requested_destination_reached is False:
        return "delivery_not_completed"
    if checks.intent_match is False:
        return "wrong_interpretation"
    if checks.final_result_valid is False:
        return "incomplete_result"
    return "other"


def _default_expected(task: Any) -> str:
    if task is None:
        return ""
    outcome = str(getattr(task, "desired_outcome", "") or "")
    if outcome:
        return sanitize_text(outcome, limit=500)
    goal = str(getattr(task, "goal", "") or "")
    return sanitize_text(goal, limit=500)


def _default_actual(context: Any, checks: CompletionChecks) -> str:
    """Describe what Atlas actually did, from its own recorded state."""

    status = _status_value(getattr(context, "status", None)) or "unknown"
    unmet = checks.unmet()
    if unmet:
        return f"task {status}; unmet requirement(s): {', '.join(unmet)}"
    return f"task {status}"


def _experience_type(outcome: str, user_correction: str) -> str:
    """Classify the record. A correction is its own, higher-value type."""

    if outcome == "failure":
        return "correction" if user_correction else "failed_task"
    if outcome == "success":
        return "successful_task"
    # No explicit outcome yet: the record exists (observed) but is unevaluated.
    return "workflow"
