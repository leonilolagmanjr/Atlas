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
    "APPLICATION",
    "WRITE_APPLICATION",
    "COMPUTER",
    "WEB_SEARCH",
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
        if _matches_any(normalized, [r"\bopen\b", r"\blaunch\b", r"\bstart\b"]) and ".exe" in normalized:
            signals["pattern"] = "application-launch"
            return IntentClassification(
                intent="APPLICATION",
                confidence=0.92,
                rationale="detected an executable application launch request",
                signals=signals,
            )

        if _looks_like_application_write_request(normalized):
            signals["pattern"] = "application-text-entry"
            return IntentClassification(
                intent="WRITE_APPLICATION",
                confidence=0.88,
                rationale="detected a request to write text into an application",
                signals=signals,
            )

        if _looks_like_web_request(normalized):
            signals["pattern"] = "web-search"
            return IntentClassification(
                intent="WEB_SEARCH",
                confidence=0.86,
                rationale="detected a request for current public web information",
                signals=signals,
            )

        if _matches_any(normalized, [r"\bopen\b", r"\blaunch\b", r"\bstart\b"]) and _looks_like_named_application_request(normalized):
            signals["pattern"] = "named-application-launch"
            return IntentClassification(
                intent="APPLICATION",
                confidence=0.9,
                rationale="detected a named application launch request",
                signals=signals,
            )

        if _matches_any(normalized, [
            r"\bram\b",
            r"\bmemory usage\b",
            r"\bprocess(?:es)?\b",
            r"\bwindows service",
            r"\bnetwork connection",
            r"\btcp connection",
            r"\bwindows version\b",
            r"\bcomputer information\b",
        ]):
            signals["pattern"] = "computer-inspection"
            return IntentClassification(
                intent="COMPUTER",
                confidence=0.9,
                rationale="detected a local computer inspection request",
                signals=signals,
            )

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


def _looks_like_named_application_request(text: str) -> bool:
    match = re.search(r"\b(?:open|launch|start)\s+(.+)$", text)
    if not match:
        return False
    target = match.group(1).strip()
    excluded = {"a file", "the file", "a folder", "the folder", "a directory", "the directory", "a website", "the website"}
    return bool(target) and target not in excluded and not target.startswith(("http://", "https://"))


def _looks_like_application_write_request(text: str) -> bool:
    has_write = _matches_any(text, [r"\bwrite\b", r"\btype\b", r"\benter\b", r"\bcreate\b", r"\bcompose\b"])
    has_target = _matches_any(text, [r"\bin notepad\b", r"\bin \w+\b", r"\binto \w+\b"])
    return has_write and has_target


def _looks_like_web_request(text: str) -> bool:
    return _matches_any(text, [
        r"\bsearch the web\b",
        r"\bgoogle\b",
        r"\blatest\b",
        r"\brecent\b",
        r"\bnews\b",
        r"\bonline\b",
        r"\bwebsite\b",
        r"\bweb\b",
        r"\byoutube\b",
        r"\bcurrent\b",
        r"\btoday\b",
    ])

