"""Controlled, LLM-assisted recovery for failed plan steps.

Recovery is deliberately narrow:

* It can only rewrite *arguments* for the *same* capability.
* It never adds, removes, or reorders capabilities.
* It is bounded by ``MAX_RECOVERY_ATTEMPTS`` per step.
* It refuses to retry with identical arguments (no infinite loops).
* Permanent failures (missing app, denied permission) are never retried.
* Uses observations to understand what actually happened vs what was expected.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from config import ENABLE_LLM_INTERPRETATION, MAX_RECOVERY_ATTEMPTS
from models import ExecutionContext, ExecutionStep, Observation
from reasoning.json_llm import safe_reasoning_call
from tools.capabilities import PLANNABLE_CAPABILITIES

logger = logging.getLogger(__name__)


class RecoveryManager:
    """Analyze a failed step and, when safe, correct its arguments."""

    def __init__(self, *, ask: Callable[..., str] | None = None) -> None:
        self._ask = ask

    def attempt_recovery(self, context: ExecutionContext, step: ExecutionStep) -> bool:
        """Return True when ``step`` was corrected and should be retried."""

        classification = context.metadata.get("failure_classification") or {}
        step.metadata.setdefault("recovery_attempts", 0)
        attempts = int(step.metadata["recovery_attempts"])

        if attempts >= MAX_RECOVERY_ATTEMPTS:
            logger.info("Recovery limit reached for step %s", step.id)
            return False
        if not classification.get("recoverable"):
            logger.info(
                "Step %s failure not classified as recoverable: %s",
                step.id,
                classification.get("category"),
            )
            return False

        corrected = self._propose_correction(context, step, attempt=attempts + 1)
        if corrected is None:
            return False

        current = step.metadata.get("parameters") or {}
        if corrected == current:
            logger.info("Recovery produced identical arguments for step %s; stopping", step.id)
            return False

        step.metadata["recovery_attempts"] = attempts + 1
        step.metadata["recovery_reason"] = classification.get("category")
        step.metadata["parameters"] = corrected
        context.metadata.setdefault("recovery_history", []).append(
            {
                "step_id": step.id,
                "attempt": attempts + 1,
                "from": current,
                "to": corrected,
                "reason": classification.get("category"),
            }
        )
        logger.info("Recovery rewrote arguments for step %s (attempt %d)", step.id, attempts + 1)
        return True

    def _propose_correction(
        self,
        context: ExecutionContext,
        step: ExecutionStep,
        *,
        attempt: int,
    ) -> Optional[dict]:
        tool = str(step.metadata.get("tool") or "")
        if tool not in PLANNABLE_CAPABILITIES or self._ask is None or not ENABLE_LLM_INTERPRETATION:
            return None

        from reasoning.prompts import RECOVERY_SYSTEM, recovery_user_prompt

        structured = context.structured_intent
        intent_json = json.dumps(structured.to_dict() if structured else {}, ensure_ascii=False)
        
        # Include observations from working state for better recovery context
        observations_summary = self._get_observations_summary(context, step)
        expected_outcome = step.metadata.get("expected_outcome")
        
        data = safe_reasoning_call(
            system_prompt=RECOVERY_SYSTEM,
            user_prompt=recovery_user_prompt(
                intent_json=intent_json,
                failed_capability=tool,
                failed_arguments=json.dumps(step.metadata.get("parameters") or {}, ensure_ascii=False),
                error=str(step.metadata.get("error") or (context.errors[-1] if context.errors else "")),
                attempt=attempt,
                observations=observations_summary,
                expected_outcome=json.dumps(expected_outcome, ensure_ascii=False) if expected_outcome else None,
            ),
            ask=self._ask,
        )
        if not data:
            return None
        if not data.get("recoverable"):
            return None

        corrected = data.get("corrected_arguments")
        if not isinstance(corrected, dict) or not corrected:
            return None

        # Merge over existing arguments so a partial correction keeps context.
        merged = dict(step.metadata.get("parameters") or {})
        merged.update(corrected)
        return merged

    def _get_observations_summary(self, context: ExecutionContext, step: ExecutionStep) -> str:
        """Extract relevant observations for the failed step."""
        if not context.working_state:
            return "No working state available"
        
        relevant_obs = []
        step_id = step.id
        action_id = step.metadata.get("action_id") or step.id
        
        for obs in context.working_state.latest_observations:
            if obs.step_id == step_id or obs.action_id == action_id or obs.tool_name == step.metadata.get("tool"):
                relevant_obs.append({
                    "type": obs.type.value if hasattr(obs.type, 'value') else str(obs.type),
                    "status": obs.status,
                    "summary": obs.summary,
                    "details": obs.details,
                    "tool": obs.tool_name,
                })
        
        if not relevant_obs:
            return "No relevant observations for this step"
        
        return json.dumps(relevant_obs, ensure_ascii=False, indent=2)
