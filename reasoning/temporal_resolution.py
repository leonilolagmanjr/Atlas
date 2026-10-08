"""Temporal resolution: map relative temporal expressions to concrete periods.

This module gives Atlas an authoritative notion of "now" and a deterministic
way to resolve relative temporal expressions (``most recent``, ``latest``,
``today``, ``last night``...) into concrete time periods that downstream
research and evidence layers can use.

Design rules:

* The system clock is the single source of truth for "now".  It is obtained
  through :func:`memory.models.system_now`, which returns a timezone-aware
  ``datetime`` in the host's local time.
* Resolution is structural, not entity-specific.  "Most recent NBA Finals"
  and "most recent Super Bowl" use the same resolver; the domain is supplied
  by the caller, not hardcoded here.
* The resolver never guesses at facts.  Its job is to translate temporal
  relations into time windows.  The research layer then retrieves the actual
  facts inside that window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

# ---------------------------------------------------------------------------
# System clock
# ---------------------------------------------------------------------------

def system_now() -> datetime:
    """Return the current local datetime with timezone information.

    The implementation defers to :func:`memory.models.utcnow` converted to
    the host local timezone so every component uses the *same* clock.
    """
    try:
        from memory.models import utcnow
        utc = utcnow()
    except Exception:  # noqa: BLE001 - best-effort fallback
        utc = datetime.now(timezone.utc)
    try:
        return utc.astimezone()
    except Exception:  # noqa: BLE001 - tzinfo-less fallback
        return utc.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalResolution:
    """The resolved temporal meaning of a request.

    Attributes:
        expression: The original temporal expression from the request, if any.
        relation: The resolved temporal relation (e.g. ``most_recent_completed``).
        reference_time: The authoritative "now" used for resolution.
        resolved_period: Human-readable description of the target period.
        resolved_start: Start of the target period (inclusive), or None.
        resolved_end: End of the target period (inclusive), or None.
        completion_state: Whether the target event must be completed.
        confidence: How confident the resolver is in this resolution.
    """

    expression: str = ""
    relation: str = "unknown"
    reference_time: datetime = field(default_factory=system_now)
    resolved_period: str = ""
    resolved_start: date | None = None
    resolved_end: date | None = None
    completion_state: str = "any"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "relation": self.relation,
            "reference_time": self.reference_time.isoformat(),
            "resolved_period": self.resolved_period,
            "resolved_start": self.resolved_start.isoformat() if self.resolved_start else None,
            "resolved_end": self.resolved_end.isoformat() if self.resolved_end else None,
            "completion_state": self.completion_state,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Expression patterns
# ---------------------------------------------------------------------------

#: Ordered by specificity.  Longer phrases first so "last night" wins over
#: "last" in ``_find_expression``.
_TEMPORAL_EXPRESSIONS: tuple[tuple[str, str, str], ...] = (
    # (expression, relation, completion_state)
    # Explicit superlatives
    ("most recent", "most_recent", "completed"),
    ("latest", "latest", "any"),
    ("newest", "newest", "any"),
    ("current", "current", "any"),
    ("currently", "currently", "any"),
    # Relative calendar
    ("today", "today", "completed"),
    ("tonight", "tonight", "completed"),
    ("yesterday", "yesterday", "completed"),
    ("last night", "last_night", "completed"),
    ("this week", "this_week", "completed"),
    ("this month", "this_month", "completed"),
    ("this year", "this_year", "completed"),
    ("last week", "last_week", "completed"),
    ("last month", "last_month", "completed"),
    ("last year", "last_year", "completed"),
    ("next week", "next_week", "upcoming"),
    ("next month", "next_month", "upcoming"),
    ("next year", "next_year", "upcoming"),
    ("recently", "recently", "completed"),
    ("recent", "recent", "completed"),
    ("previous", "previous", "completed"),
    ("upcoming", "upcoming", "upcoming"),
    ("next", "next", "upcoming"),
    ("now", "now", "any"),
    ("right now", "now", "any"),
    ("as of", "as_of", "any"),
    ("so far", "so_far", "completed"),
    ("just released", "just_released", "completed"),
    ("recently released", "recently_released", "completed"),
)

_LOOKUP: dict[str, tuple[str, str]] = {expr: (rel, state) for expr, rel, state in _TEMPORAL_EXPRESSIONS}

_EXPR_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(expr) for expr, _, _ in _TEMPORAL_EXPRESSIONS) + r")\b",
    re.IGNORECASE,
)

#: Verbs that imply the requester wants a *completed* occurrence.
_WINNER_VERBS: frozenset[str] = frozenset(
    {"won", "winner", "winning", "victory", "champion", "championship", "score", "result"}
)

#: Verbs that imply ongoing/current state.
_ONGOING_VERBS: frozenset[str] = frozenset(
    {"winning", "leading", "trailing", "playing", "ongoing", "current", "now"}
)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class TemporalResolver:
    """Resolve relative temporal expressions against the authoritative system clock.

    Usage::

        resolver = TemporalResolver()
        resolution = resolver.resolve("Who won the most recent NBA Finals?")
        print(resolution.resolved_period)   # "2026"
        print(resolution.completion_state)  # "completed"
    """

    def __init__(
        self,
        *,
        reference_time: datetime | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._now_provider = now_provider or system_now
        self._reference_time = reference_time

    @property
    def reference_time(self) -> datetime:
        if self._reference_time is not None:
            return self._reference_time
        return self._now_provider()

    def resolve(self, text: str, *, completion_hint: str = "any") -> TemporalResolution:
        """Resolve temporal expressions in ``text``.

        Args:
            text: The user's request text.
            completion_hint: Caller-supplied hint about whether the requested
                fact requires a completed occurrence (``completed``), an
                ongoing one (``ongoing``), or either (``any``).

        Returns:
            A :class:`TemporalResolution` describing the resolved time.
        """
        lowered = (text or "").casefold()

        # Skip definitional questions where a word that looks temporal is
        # actually the subject of a definition ("what does X mean?", "define X").
        # Only skip when the matched word is the immediate subject, not when
        # it appears later in a normal information question ("What is the
        # latest iPhone?" is not a definition of "latest").
        if re.search(r"\b(?:what\s+does|define|explain\s+what)\b", lowered):
            definiens = re.search(
                r"\b(?:what\s+does|define|explain\s+what)\s+(.+?)\s+(?:mean|refer\s+to|signify)\b",
                lowered,
            )
            if definiens:
                subject = definiens.group(1).strip()
                # Only skip if the subject is a single word that also appears
                # in the temporal expression list.
                if re.match(r"^[a-z]+$", subject):
                    return TemporalResolution(confidence=0.0)

        match = _EXPR_RE.search(lowered)
        if not match:
            return TemporalResolution(confidence=0.0)

        expression = match.group(0)
        relation, default_state = _LOOKUP.get(expression, ("unknown", "any"))
        completion_state = self._completion_state(lowered, default_state, completion_hint)

        return self._apply_relation(
            relation=relation,
            expression=expression,
            completion_state=completion_state,
            reference_time=self.reference_time,
        )

    def resolve_from_structure(
        self,
        text: str,
        *,
        final_event_result: bool = False,
        freshness: str = "any",
        superlative: bool = False,
    ) -> TemporalResolution:
        """Resolve using semantic structure hints from the request.

        When the semantic layer has already classified the request shape,
        those signals refine the resolution.  For example, a
        ``final_event_result`` request with ``freshness=current`` and a
        superlative temporal expression should resolve to the most recent
        *completed* occurrence.
        """
        base = self.resolve(text)
        if base.confidence == 0.0:
            return base

        completion_hint = "any"
        if final_event_result:
            # A winner/result requires a completed occurrence.
            completion_hint = "completed"
        elif freshness == "current" and superlative:
            completion_hint = "completed"
        elif freshness == "stable":
            completion_hint = "any"

        if completion_hint != "any":
            base = self._apply_relation(
                relation=base.relation,
                expression=base.expression,
                completion_state=completion_hint,
                reference_time=base.reference_time,
            )
        return base

    # -- internal ----------------------------------------------------------------

    def _completion_state(self, lowered: str, default: str, hint: str) -> str:
        if hint != "any":
            return hint
        if any(verb in lowered for verb in _WINNER_VERBS):
            return "completed"
        if any(verb in lowered for verb in _ONGOING_VERBS):
            return "ongoing"
        return default

    def _apply_relation(
        self,
        *,
        relation: str,
        expression: str,
        completion_state: str,
        reference_time: datetime,
    ) -> TemporalResolution:
        # Accept suffixed relations like "most_recent_completed" so callers can
        # pass through an already-resolved relation without re-resolution.
        base_relation = relation
        for suffix in ("_completed", "_ongoing", ""):
            if relation.endswith(suffix):
                base_relation = relation[: -len(suffix)] if suffix else relation
                break

        local_now = reference_time.astimezone()
        today = local_now.date()

        if base_relation == "most_recent":
            return self._most_recent(today, completion_state)
        if base_relation == "latest":
            return self._latest(today, completion_state)
        if base_relation == "newest":
            return self._newest(today, completion_state)
        if base_relation == "current":
            return self._current(today, completion_state)
        if base_relation == "currently":
            return self._current(today, completion_state)
        if relation == "today":
            return TemporalResolution(
                expression=expression,
                relation="today",
                reference_time=reference_time,
                resolved_period=today.isoformat(),
                resolved_start=today,
                resolved_end=today,
                completion_state=completion_state,
                confidence=1.0,
            )
        if relation == "tonight":
            return TemporalResolution(
                expression=expression,
                relation="tonight",
                reference_time=reference_time,
                resolved_period=f"evening of {today.isoformat()}",
                resolved_start=today,
                resolved_end=today,
                completion_state=completion_state,
                confidence=0.9,
            )
        if relation == "yesterday":
            d = today - timedelta(days=1)
            return TemporalResolution(
                expression=expression,
                relation="yesterday",
                reference_time=reference_time,
                resolved_period=d.isoformat(),
                resolved_start=d,
                resolved_end=d,
                completion_state=completion_state,
                confidence=1.0,
            )
        if relation == "last_night":
            d = today - timedelta(days=1)
            return TemporalResolution(
                expression=expression,
                relation="last_night",
                reference_time=reference_time,
                resolved_period=f"evening of {d.isoformat()}",
                resolved_start=d,
                resolved_end=d,
                completion_state=completion_state,
                confidence=0.9,
            )
        if relation == "this_week":
            start = today - timedelta(days=today.weekday())
            end = start + timedelta(days=6)
            return TemporalResolution(
                expression=expression,
                relation="this_week",
                reference_time=reference_time,
                resolved_period=f"week of {start.isoformat()}",
                resolved_start=start,
                resolved_end=end,
                completion_state=completion_state,
                confidence=1.0,
            )
        if relation == "this_month":
            start = today.replace(day=1)
            if today.month == 12:
                next_month = today.replace(year=today.year + 1, month=1, day=1)
            else:
                next_month = today.replace(month=today.month + 1, day=1)
            end = next_month - timedelta(days=1)
            return TemporalResolution(
                expression=expression,
                relation="this_month",
                reference_time=reference_time,
                resolved_period=f"{today.year}-{today.month:02d}",
                resolved_start=start,
                resolved_end=end,
                completion_state=completion_state,
                confidence=1.0,
            )
        if relation == "this_year":
            start = today.replace(month=1, day=1)
            end = today.replace(month=12, day=31)
            return TemporalResolution(
                expression=expression,
                relation="this_year",
                reference_time=reference_time,
                resolved_period=str(today.year),
                resolved_start=start,
                resolved_end=end,
                completion_state=completion_state,
                confidence=1.0,
            )
        if relation == "last_week":
            start = today - timedelta(days=today.weekday() + 7)
            end = start + timedelta(days=6)
            return TemporalResolution(
                expression=expression,
                relation="last_week",
                reference_time=reference_time,
                resolved_period=f"week of {start.isoformat()}",
                resolved_start=start,
                resolved_end=end,
                completion_state=completion_state,
                confidence=1.0,
            )
        if relation == "last_month":
            if today.month == 1:
                start = today.replace(year=today.year - 1, month=12, day=1)
            else:
                start = today.replace(month=today.month - 1, day=1)
            end = today.replace(day=1) - timedelta(days=1)
            return TemporalResolution(
                expression=expression,
                relation="last_month",
                reference_time=reference_time,
                resolved_period=f"{start.year}-{start.month:02d}",
                resolved_start=start,
                resolved_end=end,
                completion_state=completion_state,
                confidence=1.0,
            )
        if relation == "last_year":
            start = today.replace(year=today.year - 1, month=1, day=1)
            end = today.replace(year=today.year - 1, month=12, day=31)
            return TemporalResolution(
                expression=expression,
                relation="last_year",
                reference_time=reference_time,
                resolved_period=str(today.year - 1),
                resolved_start=start,
                resolved_end=end,
                completion_state=completion_state,
                confidence=1.0,
            )
        if relation == "next_week":
            start = today - timedelta(days=today.weekday() - 7)
            end = start + timedelta(days=6)
            return TemporalResolution(
                expression=expression,
                relation="next_week",
                reference_time=reference_time,
                resolved_period=f"week of {start.isoformat()}",
                resolved_start=start,
                resolved_end=end,
                completion_state=completion_state,
                confidence=0.8,
            )
        if relation == "next_month":
            if today.month == 12:
                start = today.replace(year=today.year + 1, month=1, day=1)
            else:
                start = today.replace(month=today.month + 1, day=1)
            return TemporalResolution(
                expression=expression,
                relation="next_month",
                reference_time=reference_time,
                resolved_period=f"{start.year}-{start.month:02d}",
                resolved_start=start,
                resolved_end=None,
                completion_state=completion_state,
                confidence=0.8,
            )
        if relation == "next_year":
            start = today.replace(year=today.year + 1, month=1, day=1)
            end = start.replace(month=12, day=31)
            return TemporalResolution(
                expression=expression,
                relation="next_year",
                reference_time=reference_time,
                resolved_period=str(today.year + 1),
                resolved_start=start,
                resolved_end=end,
                completion_state=completion_state,
                confidence=0.8,
            )
        if relation == "recently":
            start = today - timedelta(days=30)
            return TemporalResolution(
                expression=expression,
                relation="recently",
                reference_time=reference_time,
                resolved_period=f"last 30 days from {today.isoformat()}",
                resolved_start=start,
                resolved_end=today,
                completion_state=completion_state,
                confidence=0.7,
            )
        if relation == "recent":
            start = today - timedelta(days=30)
            return TemporalResolution(
                expression=expression,
                relation="recent",
                reference_time=reference_time,
                resolved_period=f"last 30 days from {today.isoformat()}",
                resolved_start=start,
                resolved_end=today,
                completion_state=completion_state,
                confidence=0.7,
            )
        if relation == "previous":
            return self._most_recent(today, completion_state)
        if relation == "now":
            return TemporalResolution(
                expression=expression,
                relation="now",
                reference_time=reference_time,
                resolved_period=today.isoformat(),
                resolved_start=today,
                resolved_end=today,
                completion_state=completion_state,
                confidence=1.0,
            )
        if relation == "as_of":
            return TemporalResolution(
                expression=expression,
                relation="as_of",
                reference_time=reference_time,
                resolved_period=today.isoformat(),
                resolved_start=today,
                resolved_end=today,
                completion_state=completion_state,
                confidence=0.9,
            )
        if relation == "so_far":
            start = today.replace(month=1, day=1)
            return TemporalResolution(
                expression=expression,
                relation="so_far",
                reference_time=reference_time,
                resolved_period=f"year to date {today.year}",
                resolved_start=start,
                resolved_end=today,
                completion_state=completion_state,
                confidence=0.8,
            )
        if relation == "just_released":
            start = today - timedelta(days=7)
            return TemporalResolution(
                expression=expression,
                relation="just_released",
                reference_time=reference_time,
                resolved_period=f"last 7 days from {today.isoformat()}",
                resolved_start=start,
                resolved_end=today,
                completion_state=completion_state,
                confidence=0.9,
            )
        if relation == "recently_released":
            start = today - timedelta(days=30)
            return TemporalResolution(
                expression=expression,
                relation="recently_released",
                reference_time=reference_time,
                resolved_period=f"last 30 days from {today.isoformat()}",
                resolved_start=start,
                resolved_end=today,
                completion_state=completion_state,
                confidence=0.8,
            )

        return TemporalResolution(
            expression=expression,
            relation=relation,
            reference_time=reference_time,
            confidence=0.5,
        )

    def _most_recent(self, today: date, completion_state: str) -> TemporalResolution:
        if completion_state == "completed":
            return TemporalResolution(
                expression="most recent",
                relation="most_recent_completed",
                reference_time=self.reference_time,
                resolved_period=str(today.year),
                resolved_start=today.replace(month=1, day=1),
                resolved_end=today,
                completion_state="completed",
                confidence=0.9,
            )
        return TemporalResolution(
            expression="most recent",
            relation="most_recent",
            reference_time=self.reference_time,
            resolved_period=str(today.year),
            resolved_start=today.replace(month=1, day=1),
            resolved_end=today,
            completion_state="any",
            confidence=0.8,
        )

    def _latest(self, today: date, completion_state: str) -> TemporalResolution:
        if completion_state == "completed":
            return TemporalResolution(
                expression="latest",
                relation="latest_completed",
                reference_time=self.reference_time,
                resolved_period=str(today.year),
                resolved_start=today.replace(month=1, day=1),
                resolved_end=today,
                completion_state="completed",
                confidence=0.9,
            )
        return TemporalResolution(
            expression="latest",
            relation="latest",
            reference_time=self.reference_time,
            resolved_period=str(today.year),
            resolved_start=today.replace(month=1, day=1),
            resolved_end=today,
            completion_state="any",
            confidence=0.8,
        )

    def _newest(self, today: date, completion_state: str) -> TemporalResolution:
        return self._latest(today, completion_state)

    def _current(self, today: date, completion_state: str) -> TemporalResolution:
        return TemporalResolution(
            expression="current",
            relation="current",
            reference_time=self.reference_time,
            resolved_period=str(today.year),
            resolved_start=today.replace(month=1, day=1),
            resolved_end=today,
            completion_state=completion_state,
            confidence=0.8,
        )
