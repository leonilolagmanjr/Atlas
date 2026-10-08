"""Tests for temporal resolution."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone, timedelta

from reasoning.temporal_resolution import (
    TemporalResolution,
    TemporalResolver,
    system_now,
)


class SystemClockTests(unittest.TestCase):
    """The authoritative system clock must be timezone-aware."""

    def test_system_now_returns_timezone_aware(self) -> None:
        now = system_now()
        self.assertIsNotNone(now.tzinfo)

    def test_system_now_is_recent(self) -> None:
        now = system_now()
        self.assertGreaterEqual(now, datetime.now(timezone.utc) - timedelta(seconds=5))


class TemporalResolutionTests(unittest.TestCase):
    """Temporal expressions resolve to concrete periods against a fixed clock."""

    def setUp(self) -> None:
        self.fixed_time = datetime(2026, 10, 8, 14, 30, tzinfo=timezone.utc)
        self.resolver = TemporalResolver(reference_time=self.fixed_time)

    def test_no_expression_returns_zero_confidence(self) -> None:
        result = self.resolver.resolve("What is recursion?")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.relation, "unknown")

    def test_today_resolves_to_fixed_date(self) -> None:
        result = self.resolver.resolve("What happened today?")
        self.assertEqual(result.resolved_start, date(2026, 10, 8))
        self.assertEqual(result.resolved_end, date(2026, 10, 8))
        self.assertEqual(result.relation, "today")
        self.assertEqual(result.confidence, 1.0)

    def test_yesterday_resolves_to_previous_day(self) -> None:
        result = self.resolver.resolve("What happened yesterday?")
        self.assertEqual(result.resolved_start, date(2026, 10, 7))
        self.assertEqual(result.resolved_end, date(2026, 10, 7))
        self.assertEqual(result.relation, "yesterday")
        self.assertEqual(result.confidence, 1.0)

    def test_this_year_resolves_to_calendar_year(self) -> None:
        result = self.resolver.resolve("Who won this year's NBA Finals?")
        self.assertEqual(result.resolved_period, "2026")
        self.assertEqual(result.resolved_start, date(2026, 1, 1))
        self.assertEqual(result.resolved_end, date(2026, 12, 31))
        self.assertEqual(result.relation, "this_year")

    def test_last_year_resolves_to_previous_year(self) -> None:
        result = self.resolver.resolve("What was last year's winner?")
        self.assertEqual(result.resolved_period, "2025")
        self.assertEqual(result.resolved_start, date(2025, 1, 1))
        self.assertEqual(result.resolved_end, date(2025, 12, 31))

    def test_this_month_resolves_to_calendar_month(self) -> None:
        result = self.resolver.resolve("What happened this month?")
        self.assertEqual(result.resolved_period, "2026-10")
        self.assertEqual(result.resolved_start, date(2026, 10, 1))
        self.assertEqual(result.resolved_end, date(2026, 10, 31))

    def test_last_month_resolves_to_previous_month(self) -> None:
        result = self.resolver.resolve("What happened last month?")
        self.assertEqual(result.resolved_period, "2026-09")
        self.assertEqual(result.resolved_start, date(2026, 9, 1))
        self.assertEqual(result.resolved_end, date(2026, 9, 30))

    def test_this_week_resolves_to_week_starting_monday(self) -> None:
        # 2026-10-08 is a Thursday. Monday of that week is 2026-10-05.
        result = self.resolver.resolve("What happened this week?")
        self.assertEqual(result.resolved_start, date(2026, 10, 5))
        self.assertEqual(result.resolved_end, date(2026, 10, 11))
        self.assertEqual(result.resolved_period, "week of 2026-10-05")

    def test_last_night_resolves_to_previous_day(self) -> None:
        result = self.resolver.resolve("Who won last night's game?")
        self.assertEqual(result.resolved_start, date(2026, 10, 7))
        self.assertEqual(result.resolved_end, date(2026, 10, 7))
        self.assertEqual(result.relation, "last_night")
        self.assertEqual(result.completion_state, "completed")

    def test_most_recent_defaults_to_completed(self) -> None:
        result = self.resolver.resolve("Who won the most recent NBA Finals?")
        self.assertEqual(result.relation, "most_recent_completed")
        self.assertEqual(result.completion_state, "completed")
        self.assertEqual(result.resolved_period, "2026")

    def test_most_recent_with_completed_event(self) -> None:
        result = self.resolver.resolve_from_structure(
            "Who won the most recent NBA Finals?",
            final_event_result=True,
            freshness="current",
            superlative=True,
        )
        self.assertEqual(result.relation, "most_recent_completed")
        self.assertEqual(result.completion_state, "completed")
        self.assertEqual(result.confidence, 0.9)

    def test_latest_resolves_with_year_period(self) -> None:
        result = self.resolver.resolve("What is the latest iPhone?")
        self.assertEqual(result.resolved_period, "2026")
        self.assertEqual(result.relation, "latest")
        self.assertEqual(result.completion_state, "any")


class TemporalResolutionBoundaryTests(unittest.TestCase):
    """Year-boundary and month-boundary transitions."""

    def test_january_first_yesterday_is_previous_year(self) -> None:
        fixed = datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc)
        resolver = TemporalResolver(reference_time=fixed)
        result = resolver.resolve("What happened yesterday?")
        self.assertEqual(result.resolved_start, date(2026, 1, 1))
        self.assertEqual(result.resolved_end, date(2026, 1, 1))

    def test_january_first_this_year_is_current_year(self) -> None:
        fixed = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        resolver = TemporalResolver(reference_time=fixed)
        result = resolver.resolve("What happened this year?")
        self.assertEqual(result.resolved_period, "2026")
        self.assertEqual(result.resolved_start, date(2026, 1, 1))
        self.assertEqual(result.resolved_end, date(2026, 12, 31))

    def test_january_first_last_year_is_previous_year(self) -> None:
        fixed = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        resolver = TemporalResolver(reference_time=fixed)
        result = resolver.resolve("What happened last year?")
        self.assertEqual(result.resolved_period, "2025")
        self.assertEqual(result.resolved_start, date(2025, 1, 1))
        self.assertEqual(result.resolved_end, date(2025, 12, 31))


if __name__ == "__main__":
    unittest.main()
