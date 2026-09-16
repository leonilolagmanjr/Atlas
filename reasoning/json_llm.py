"""Structured-output reasoning helpers shared by Atlas LLM stages.

These helpers never execute model text. They only turn a local model response
into a *validated Python object*, repairing obviously malformed JSON and
falling back gracefully when the model refuses to cooperate.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a single JSON object from model output.

    Handles fenced code blocks, leading prose, and common trailing-comma
    mistakes. Returns ``None`` when no object can be recovered.
    """

    if not text or not text.strip():
        return None

    candidates: list[str] = []
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))

    stripped = text.strip()
    candidates.append(stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        parsed = _parse_object(candidate)
        if parsed is not None:
            return parsed
    return None


def _parse_object(candidate: str) -> dict[str, Any] | None:
    for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            value = json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def safe_reasoning_call(
    *,
    system_prompt: str,
    user_prompt: str,
    ask: Any,
) -> dict[str, Any] | None:
    """Call the LLM for JSON and return a validated dict or ``None``.

    ``ask`` is injected so this module never imports the concrete provider and
    remains trivially testable with a fake model.
    """

    try:
        raw = ask(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:  # noqa: BLE001 - LLM availability must never crash a task
        logger.exception("Reasoning LLM call failed")
        return None

    parsed = extract_json_object(raw)
    if parsed is None:
        logger.warning("Reasoning LLM returned non-JSON output")
    return parsed
