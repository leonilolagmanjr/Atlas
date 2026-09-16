"""Semantic interpretation of natural-language requests.

This is the boundary that turns free text into a :class:`StructuredIntent`.

Design rules (from the overhaul spec):

* The LLM proposes *meaning*, never commands.
* Deterministic heuristics provide a fast path and a fallback.
* Confidence decides whether Atlas executes, infers, or asks.
* No synonym dictionary is the primary mechanism; the LLM normalizes wording.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from config import (
    CLARIFICATION_CONFIDENCE_THRESHOLD,
    ENABLE_LLM_INTERPRETATION,
    INTERPRETER_CONFIDENCE_THRESHOLD,
)
from models import StructuredIntent
from reasoning.json_llm import safe_reasoning_call

logger = logging.getLogger(__name__)

# Execution categories are intentionally coarse and extensible.
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "WEB_SEARCH": ("youtube", "search", "find", "google", "web", "online", "videos"),
    "APPLICATION_CONTROL": ("notepad", "calculator", "chrome", "vscode", "open", "launch", "write", "type"),
    "CREATIVE_GENERATION": ("poem", "story", "essay", "song", "joke", "write", "compose", "generate"),
    "FILE_OPERATION": ("file", "folder", "directory", "save", "read", "delete", "rename"),
    "KNOWLEDGE_QUERY": ("what", "who", "when", "where", "why", "how", "explain", "define"),
    "SYSTEM_OPERATION": ("ram", "cpu", "process", "service", "memory usage", "network"),
}

# Canonical action verbs, used only for the deterministic fast path.
_ACTION_VERBS = {
    "create": ("create", "write", "make", "generate", "compose", "produce", "draft"),
    "search": ("search", "find", "look", "show", "get", "pull"),
    "open": ("open", "launch", "start", "run"),
}

_CONTENT_TYPES = ("poem", "story", "essay", "song", "joke", "letter", "email", "summary", "note")
_LENGTH_WORDS = {"short": "short", "brief": "short", "long": "long", "detailed": "long", "tiny": "short"}
_TONE_WORDS = (
    "funny", "serious", "sad", "happy", "formal", "casual", "dramatic",
    "romantic", "angry", "playful", "sarcastic", "friendly",
)
_SORT_WORDS = {
    "latest": "latest",
    "newest": "latest",
    "recent": "latest",
    "popular": "popular",
    "trending": "popular",
    "top": "popular",
    "oldest": "oldest",
}

_ABOUT_RE = re.compile(r"\b(?:about|regarding|concerning|on the topic of|related to)\s+(.+?)(?:\s+(?:in|into|on|using|inside|and put it in|and write it in)\s|$)", re.IGNORECASE)
_IN_APP_RE = re.compile(r"\b(?:in|into|inside|using|with)\s+([A-Za-z][A-Za-z0-9 ._-]*?)(?:\s+and\b|$|[,.])", re.IGNORECASE)
_OPEN_APP_RE = re.compile(r"\b(?:open|launch|start|run)\s+(?:up\s+)?([A-Za-z][A-Za-z0-9 ._-]*?)(?:\s+and\b|$|[,.?])", re.IGNORECASE)
_ON_SITE_RE = re.compile(r"\b(?:on|in|at|from)\s+(youtube|google|the web|the internet)\b", re.IGNORECASE)
_FOR_QUERY_RE = re.compile(r"\b(?:for|find|about)\s+(.+?)(?:\s+(?:videos?|on|in)\b|$)", re.IGNORECASE)
_YOUTUBE_RE = re.compile(r"\byoutube\b", re.IGNORECASE)
_VIDEO_WORD_RE = re.compile(r"\bvideo?s?\b", re.IGNORECASE)

_WEB_TARGETS = {"youtube", "google", "the web", "the internet"}
# Sort/quantity words that must not leak into a search query.
_QUERY_STOPWORDS = {
    "search", "find", "look", "show", "me", "get", "for", "on", "in", "the",
    "youtube", "google", "web", "videos", "video", "latest", "newest", "recent",
    "popular", "trending", "top", "oldest", "about", "some", "a", "an", "please",
}
# Words that signal a web/video search even without an explicit site name.
_SEARCH_TARGET_HINTS = ("youtube", "video", "videos", "channel", "watch")
# Unresolved references that must never be treated as a concrete target.
_PRONOUNS = {"it", "that", "this", "them", "those", "there", "the same", "one"}


class SemanticInterpreter:
    """Turn free text into a structured, composable intent."""

    def __init__(
        self,
        *,
        ask: Callable[..., str] | None = None,
        capability_catalog: str = "",
        enabled: Optional[bool] = None,
    ) -> None:
        self._ask = ask
        self._capability_catalog = capability_catalog
        self._enabled = ENABLE_LLM_INTERPRETATION if enabled is None else enabled

    def interpret(
        self,
        text: str,
        *,
        context: Optional[StructuredIntent] = None,
        history: str = "",
    ) -> StructuredIntent:
        """Interpret ``text`` into a StructuredIntent.

        A high-confidence deterministic read is returned directly. Otherwise the
        LLM is consulted (when enabled and available) and its output is validated
        through :meth:`StructuredIntent.from_mapping`.
        """

        unresolved = _has_unresolved_reference(text)
        heuristic = self._heuristic_interpret(text)
        if heuristic.confidence >= INTERPRETER_CONFIDENCE_THRESHOLD or not self._enabled or self._ask is None:
            return self._finalize(heuristic, context, unresolved_reference=unresolved)

        llm_intent = self._llm_interpret(text, heuristic=heuristic, history=history)
        if llm_intent is None:
            return self._finalize(heuristic, context, unresolved_reference=unresolved)
        # LLM wins on semantic content but inherits a deterministic safety net.
        merged = self._reconcile(llm_intent, heuristic)
        return self._finalize(merged, context, unresolved_reference=unresolved)

    # -- deterministic fast path -------------------------------------------------

    def _heuristic_interpret(self, text: str) -> StructuredIntent:
        normalized = text.strip()
        lowered = normalized.casefold()
        confidence = 0.4
        intent = "unknown"
        action: Optional[str] = None
        content_type: Optional[str] = None
        topic: Optional[str] = None
        destination: Optional[str] = None
        target: Optional[str] = None
        query: Optional[str] = None
        sort: Optional[str] = None
        tone: Optional[str] = None
        length: Optional[str] = None

        for canonical, verbs in _ACTION_VERBS.items():
            if any(re.search(rf"\b{verb}\w*\b", lowered) for verb in verbs):
                action = canonical
                break

        for candidate in _CONTENT_TYPES:
            if re.search(rf"\b{candidate}s?\b", lowered):
                content_type = candidate
                confidence = max(confidence, 0.6)
                break

        topic_match = _ABOUT_RE.search(normalized)
        if topic_match:
            topic = topic_match.group(1).strip()
            confidence = max(confidence, 0.6)

        destination = _extract_destination(normalized, content_type=content_type)
        if destination:
            confidence = max(confidence, 0.7)

        site_match = _ON_SITE_RE.search(normalized)
        if site_match:
            target = site_match.group(1).casefold()
        elif _YOUTUBE_RE.search(normalized):
            target = "youtube"
        elif action == "search" and _VIDEO_WORD_RE.search(normalized):
            target = "youtube"

        wants_web = target in _WEB_TARGETS or (
            action == "search" and _matches_any_token(lowered, _SEARCH_TARGET_HINTS)
        )

        if wants_web:
            intent = "search"
            action = action or "search"
            query = _extract_search_query(normalized)
            confidence = max(confidence, 0.8)
        elif destination and (content_type or action in {"create", "write"}):
            intent = "write_content"
            confidence = max(confidence, 0.85)
        elif content_type or action in {"create", "write"}:
            intent = "write_content" if content_type else "creative_generation"
            confidence = max(confidence, 0.55)
        elif action == "open" and destination:
            intent = "open_application"
            confidence = max(confidence, 0.85)
        elif action == "search":
            intent = "search"
            confidence = max(confidence, 0.55)
        else:
            intent = "knowledge_query"
            confidence = max(confidence, 0.35)

        # Strip a search sort word out of an extracted query.
        if query:
            query = _clean_query(query)
            if not query:
                query = None
        for word in _TONE_WORDS:
            if re.search(rf"\b{word}\b", lowered):
                tone = word
                break
        for word, canonical in _LENGTH_WORDS.items():
            if re.search(rf"\b{word}\b", lowered):
                length = canonical
                break
        for word, canonical in _SORT_WORDS.items():
            if re.search(rf"\b{word}\b", lowered):
                sort = canonical
                break
        if _matches_any_token(lowered, _SEARCH_TARGET_HINTS) and not target and intent == "search":
            target = "youtube"

        return StructuredIntent(
            intent=intent,
            action=action,
            target=target,
            content_type=content_type,
            topic=topic,
            query=query,
            destination=destination,
            tone=tone,
            length=length,
            sort=sort,
            confidence=confidence,
            source="deterministic",
        )

    # -- LLM path ----------------------------------------------------------------

    def _llm_interpret(
        self,
        text: str,
        *,
        heuristic: StructuredIntent,
        history: str,
    ) -> Optional[StructuredIntent]:
        from reasoning.prompts import INTERPRETER_SYSTEM, interpreter_user_prompt

        user_prompt = interpreter_user_prompt(
            request=text,
            capabilities=self._capability_catalog,
            history=history,
            draft=heuristic.to_dict(),
        )
        data = safe_reasoning_call(
            system_prompt=INTERPRETER_SYSTEM,
            user_prompt=user_prompt,
            ask=self._ask,
        )
        if data is None:
            return None
        return StructuredIntent.from_mapping(data, source="llm")

    def _reconcile(self, llm: StructuredIntent, heuristic: StructuredIntent) -> StructuredIntent:
        """Keep LLM meaning but backfill fields deterministic parsing is sure of."""

        for name in ("destination", "content_type", "topic", "target", "query"):
            if not getattr(llm, name) and getattr(heuristic, name):
                setattr(llm, name, getattr(heuristic, name))
        if llm.intent in {"unknown", ""}:
            llm.intent = heuristic.intent
        if not llm.action and heuristic.action:
            llm.action = heuristic.action
        return llm

    def _finalize(
        self,
        intent: StructuredIntent,
        context: Optional[StructuredIntent],
        *,
        unresolved_reference: bool = False,
    ) -> StructuredIntent:
        """Apply conversational context and clarification policy."""

        if context is not None and _looks_like_followup(intent):
            intent = context.merged_with(intent)
            intent.confidence = max(intent.confidence, 0.7)

        # Ambiguity: an action that needs a destination but has none.
        # A clear "in Notepad" request is not ambiguous; an unresolved
        # "open it and write..." is, and must not be hallucinated.
        destination_intents = {"open_application", "write_content"}
        needs_target = intent.intent in destination_intents and not (
            intent.destination or intent.target
        )
        # An explicit but unresolved reference ("open it", "write it in that")
        # is ambiguous even when the content type alone would be fine.
        if unresolved_reference and not (intent.destination or intent.target):
            needs_target = True
        if needs_target:
            intent.needs_clarification = True
            intent.clarification_question = (
                intent.clarification_question
                or "Which application would you like me to use?"
            )
        elif intent.confidence < CLARIFICATION_CONFIDENCE_THRESHOLD and intent.intent == "unknown":
            intent.needs_clarification = True
            intent.clarification_question = (
                intent.clarification_question
                or "Could you rephrase what you'd like Atlas to do?"
            )
        return intent


def _looks_like_followup(intent: StructuredIntent) -> bool:
    """Detect a request that only makes sense against prior context."""

    if intent.intent in {"write_content", "creative_generation"} and not intent.destination:
        return True
    if intent.intent in {"unknown", "knowledge_query"} and (
        intent.topic or intent.tone or intent.length or intent.style
    ):
        return True
    return False


def classify_category(intent: StructuredIntent, text: str) -> str:
    """Map a structured intent onto a coarse execution category."""

    lowered = text.casefold()
    if intent.intent == "search" and (intent.target in _WEB_TARGETS or _YOUTUBE_RE.search(lowered)):
        return "WEB_SEARCH"
    if intent.intent in {"open_application", "write_content"}:
        return "APPLICATION_CONTROL"
    if intent.intent == "creative_generation":
        return "CREATIVE_GENERATION"
    if intent.intent == "knowledge_query":
        scores = {
            category: sum(1 for keyword in keywords if keyword in lowered)
            for category, keywords in CATEGORY_KEYWORDS.items()
        }
        best = max(scores, key=lambda name: scores[name])
        return best if scores[best] > 0 else "KNOWLEDGE_QUERY"
    return "KNOWLEDGE_QUERY"
def _matches_any_token(text: str, tokens: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(token)}\b", text) for token in tokens)


def _extract_destination(text: str, *, content_type: Optional[str]) -> Optional[str]:
    """Extract the target application from either "in X" or "open X" phrasing."""

    app = _OPEN_APP_RE.search(text)
    if app:
        candidate = app.group(1).strip()
        # "open notepad and write..." -> notepad; reject verbs/content words.
        if _is_plausible_application(candidate, content_type=content_type):
            return candidate
    in_app = _IN_APP_RE.search(text)
    if in_app:
        candidate = in_app.group(1).strip()
        if _is_plausible_application(candidate, content_type=content_type):
            return candidate
    return None


def _is_plausible_application(candidate: str, *, content_type: Optional[str]) -> bool:
    lowered = candidate.casefold().strip()
    if not lowered:
        return False
    if lowered in _PRONOUNS:
        # "open it and write..." has no referent -> not a usable destination.
        return False
    if lowered in _CONTENT_TYPES:
        return False
    if lowered in _WEB_TARGETS:
        return False
    # Drop trailing filler and modifier words a regex may have captured.
    blocked_prefixes = ("about ", "funny ", "short ", "long ", "a ", "an ", "the ", "and ")
    if lowered.startswith(blocked_prefixes):
        return False
    if _matches_any_token(lowered, _SEARCH_TARGET_HINTS):
        return False
    return len(lowered) <= 40


def _extract_search_query(text: str) -> Optional[str]:
    """Extract the search phrase, excluding the site and sort/quantity words."""

    cleaned_text = _clean_query(text)
    if not cleaned_text:
        # A query made only of boilerplate ("popular youtube videos") is not a
        # real search term; return None so the plan falls back to the request.
        return None
    match = _FOR_QUERY_RE.search(text)
    if match:
        cleaned = _clean_query(match.group(1))
        if cleaned:
            return cleaned
    return cleaned_text


def _clean_query(value: str) -> str:
    """Remove site, action, and sort words from a candidate search query."""

    lowered = re.sub(r"'s\b", "", value.casefold())
    tokens = [
        token for token in re.findall(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*", lowered)
        if token not in _QUERY_STOPWORDS
    ]
    return " ".join(tokens).strip()
def _has_unresolved_reference(text: str) -> bool:
    """Detect a target phrase that names a pronoun instead of a real target.

    Handles "open it", "write it in that", "put it there" — requests whose
    destination cannot be resolved from the text alone.
    """

    return bool(
        re.search(
            r"\b(?:open|launch|start|write|put|type|save)\s+(?:it|that|this|them|those|there|the same|one)\b",
            text,
            re.IGNORECASE,
        )
    )
