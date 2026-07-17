"""Rule-based intent classification for Atlas V3.2.

NO LLM ALLOWED.

This module classifies user requests into coarse intents so the Planner can
choose a different deterministic execution plan.

The classifier is intentionally lightweight and conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


INTENT_LABELS = {
    "FACT",
    "PERSON",
    "LIST",
    "COUNT",
    "COMPARE",
    "SUMMARIZE",
    "EXPLAIN",
    "DEFINITION",
    "DATE",
    "LOCATION",
    "PROCEDURE",
    "UNKNOWN",
}


@dataclass(frozen=True)
class IntentClassification:
    intent: str
    confidence: float
    rationale: str
    signals: dict[str, str]


class IntentClassifier:
    """Deterministic intent classifier based on keyword and pattern matching."""

    def classify(self, text: str) -> IntentClassification:
        normalized = _normalize(text)
        if not normalized:
            return IntentClassification(
                intent="UNKNOWN",
                confidence=0.0,
                rationale="empty input",
                signals={},
            )

        signals: dict[str, str] = {}

        # ORDER MATTERS: more specific patterns first.
        if _matches_any(normalized, [
            r"\bcompare\b",
            r"\bvs\b",
            r"\bversus\b",
            r"\bversus\b",
        ]):
            signals["pattern"] = "compare"
            return IntentClassification(
                intent="COMPARE",
                confidence=0.95,
                rationale="detected compare intent keywords",
                signals=signals,
            )

        if _matches_any(normalized, [
            r"\bsummarize\b",
            r"\bsum(up)?mary\b",
            r"\bsum\b",
            r"\bsummary\b",
            r"\bsummarise\b",
        ]):
            signals["pattern"] = "summarize"
            return IntentClassification(
                intent="SUMMARIZE",
                confidence=0.9,
                rationale="detected summarize intent keywords",
                signals=signals,
            )

        if _matches_any(normalized, [
            r"\bhow\b.*\bworks\b",
            r"\bexplain\b",
            r"\bwhy\b",
            r"\bwhat does\b",
            r"\bhow to\b",
            r"\bhow do\b",
        ]):
            # Distinguish PROCEDURE vs EXPLAIN.
            if _matches_any(normalized, [r"\bhow to\b", r"\bhow do\b", r"\bsteps?\b", r"\binstructions?\b"]):
                signals["pattern"] = "procedure-how"
                return IntentClassification(
                    intent="PROCEDURE",
                    confidence=0.78,
                    rationale="detected how-to/procedure patterns",
                    signals=signals,
                )
            signals["pattern"] = "explain"
            return IntentClassification(
                intent="EXPLAIN",
                confidence=0.75,
                rationale="detected explain patterns",
                signals=signals,
            )

        if _matches_any(normalized, [
            r"\bdefine\b",
            r"\bmeaning of\b",
            r"\bwhat is\b",
        ]):
            # Avoid misclassifying "what is" as FACT.
            signals["pattern"] = "definition"
            return IntentClassification(
                intent="DEFINITION",
                confidence=0.72,
                rationale="detected definition patterns",
                signals=signals,
            )

        if _matches_any(normalized, [
            r"\bwhere\b",
            r"\blocation\b",
            r"\baddress\b",
            r"\blives\b",
        ]):
            signals["pattern"] = "location"
            return IntentClassification(
                intent="LOCATION",
                confidence=0.7,
                rationale="detected location keywords",
                signals=signals,
            )

        if _matches_any(normalized, [
            r"\bwhen\b",
            r"\bdate\b",
            r"\byear\b",
            r"\bmonth\b",
            r"\bday\b",
        ]):
            signals["pattern"] = "date"
            return IntentClassification(
                intent="DATE",
                confidence=0.7,
                rationale="detected date/time keywords",
                signals=signals,
            )

        if _matches_any(normalized, [
            r"\bwho\b",
            r"\bperson\b",
            r"\bname\b",
        ]):
            signals["pattern"] = "who"
            return IntentClassification(
                intent="PERSON",
                confidence=0.7,
                rationale="detected who/person patterns",
                signals=signals,
            )

        # COUNT (numeric-ish questions)
        if _matches_any(normalized, [
            r"\bhow many\b",
            r"\bcount\b",
            r"\bnumber of\b",
        ]):
            signals["pattern"] = "how-many/count"
            return IntentClassification(
                intent="COUNT",
                confidence=0.88,
                rationale="detected count keywords",
                signals=signals,
            )

        # LIST
        if _matches_any(normalized, [
            r"\bgive\b",
            r"\blist\b",
            r"\bshow\b",
            r"\bwhat\s+are\b",
            r"\bwhich\b",
            r"\binclude\b",
        ]):
            # If it also looks like count, prefer COUNT already.
            if _matches_any(normalized, [r"\bhow many\b", r"\bcount\b", r"\bnumber of\b"]):
                pass
            else:
                signals["pattern"] = "list"
                return IntentClassification(
                    intent="LIST",
                    confidence=0.62,
                    rationale="detected list/show keywords",
                    signals=signals,
                )

        # PROCEDURE (generic)
        if _matches_any(normalized, [
            r"\bsteps?\b",
            r"\binstructions?\b",
            r"\bprocess\b",
            r"\bprocedure\b",
            r"\bhow\b\s+to\b",
        ]):
            signals["pattern"] = "procedure-generic"
            return IntentClassification(
                intent="PROCEDURE",
                confidence=0.65,
                rationale="detected procedure/process keywords",
                signals=signals,
            )

        # FACT default
        if _matches_any(normalized, [
            r"\bwho\b",
            r"\bwhat\b",
            r"\bis\b",
            r"\bare\b",
            r"\bdoes\b",
            r"\bdo\b",
        ]):
            signals["pattern"] = "question-fact"
            return IntentClassification(
                intent="FACT",
                confidence=0.55,
                rationale="defaulting to FACT for question-style prompts",
                signals=signals,
            )

        return IntentClassification(
            intent="UNKNOWN",
            confidence=0.2,
            rationale="no intent matched",
            signals=signals,
        )


def _normalize(text: str) -> str:
    return text.strip().lower()


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)

