"""Conversation summarization.

A conversation summary is what keeps a long conversation coherent without
stuffing the entire transcript into every prompt. It is *not* a transcript and
not a transcript compression: it records the things that must not be lost —

* decisions the user made
* requirements and constraints they stated
* technical details that matter later
* unresolved questions
* important actions Atlas performed
* the current task state

...and drops small talk.

The summarizer prefers the injected model boundary (``ask``) because a summary
is exactly the kind of language task a model is good at, but it always has a
deterministic fallback: when no model is available, the summary is built from an
extractive selection of the same categories, so a summary is never fabricated
and never silently empty.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable

from config import AUTO_SUMMARIZE_THRESHOLD, CONVERSATION_SUMMARY_MIN_MESSAGES
from memory.models import MemoryMessage
from providers.exceptions import ProviderError

logger = logging.getLogger(__name__)

#: Phrases that mark a sentence as decision/requirement material worth keeping.
_KEEP_MARKERS: tuple[str, ...] = (
    "decide", "decided", "decision", "instead", "prefer", "requirement",
    "requires", "must", "should", "need to", "don't", "do not", "always",
    "never", "remember", "from now on", "the plan is", "we'll", "we will",
    "use ", "switch to", "stick with", "architecture", "database", "api",
    "config", "version", "deadline", "important", "note that",
)

#: Markers of something Atlas could not finish — these must survive summarization.
_UNRESOLVED_MARKERS: tuple[str, ...] = (
    "?", "could not", "couldn't", "failed", "unresolved", "pending", "waiting",
    "not sure", "unclear", "todo", "to do", "next step", "remains",
)

_SMALL_TALK = {
    "hi", "hey", "hello", "thanks", "thank you", "ok", "okay", "cool", "nice",
    "great", "got it", "understood", "bye", "yes", "no", "sure",
}

_WORD_RE = re.compile(r"[a-z0-9']+")

#: Prompt used when a model is available. It asks for structured notes in the
#: same categories the deterministic fallback uses, so the two paths agree.
_SUMMARY_SYSTEM = (
    "You compress a conversation into durable notes for a local assistant. "
    "Record only: decisions made, requirements and constraints stated, "
    "important technical details, unresolved questions, actions performed, and "
    "the current task state. Omit greetings and small talk. Use terse bullet "
    "lines. Never invent anything that was not said."
)


def _sentences(text: str) -> list[str]:
    """Split text into sentences without pulling in a heavy dependency."""

    parts = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def _is_small_talk(sentence: str) -> bool:
    stripped = sentence.strip(" .!?,").casefold()
    return stripped in _SMALL_TALK or len(_WORD_RE.findall(stripped)) <= 2


def extract_notes(messages: Iterable[MemoryMessage], *, limit: int = 14) -> dict[str, list[str]]:
    """Deterministically extract the categories a summary must preserve."""

    decisions: list[str] = []
    requirements: list[str] = []
    unresolved: list[str] = []
    actions: list[str] = []

    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                tool = str(call.get("tool") or "")
                status = str(call.get("status") or "")
                if tool:
                    actions.append(f"{tool} ({status or 'unknown'})")
        if message.role not in {"user", "assistant"}:
            continue
        for sentence in _sentences(message.content):
            if _is_small_talk(sentence):
                continue
            lowered = sentence.casefold()
            if any(marker in lowered for marker in _UNRESOLVED_MARKERS):
                unresolved.append(sentence)
                continue
            if any(marker in lowered for marker in _KEEP_MARKERS):
                if message.role == "user":
                    decisions.append(sentence)
                else:
                    requirements.append(sentence)

    def bounded(items: list[str]) -> list[str]:
        seen: list[str] = []
        for item in items:
            text = item.strip()
            if text and text not in seen:
                seen.append(text)
        return seen[:limit]

    return {
        "decisions": bounded(decisions),
        "requirements": bounded(requirements),
        "unresolved": bounded(unresolved),
        "actions": bounded(actions),
    }


def render_notes(notes: dict[str, list[str]]) -> str:
    """Render extracted notes as the summary text."""

    sections = (
        ("Decisions", notes.get("decisions") or []),
        ("Requirements", notes.get("requirements") or []),
        ("Unresolved", notes.get("unresolved") or []),
        ("Actions", notes.get("actions") or []),
    )
    lines: list[str] = []
    for title, items in sections:
        if not items:
            continue
        lines.append(f"{title}:")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


class MemorySummarizer:
    """Produce and maintain a conversation summary.

    ``ask`` is the same injectable model boundary the rest of Atlas uses. When it
    is ``None`` (or fails, or returns junk), the deterministic extraction is used
    instead, so summarization is never a fake and never a silent no-op.
    """

    def __init__(
        self,
        *,
        ask: Callable[..., str] | None = None,
        model_available: bool = True,
        min_messages: int = CONVERSATION_SUMMARY_MIN_MESSAGES,
    ) -> None:
        self._ask = ask
        self._model_available = bool(model_available)
        self._min_messages = max(2, int(min_messages))

    # -- lifecycle ---------------------------------------------------------------

    def should_summarize(self, *, message_count: int, existing_summary: str | None) -> bool:
        """Return True when the conversation has enough material to summarize."""

        if message_count < self._min_messages:
            return False
        # A conversation with an existing summary is refreshed less often, so a
        # long chat does not pay for a model call on every turn.
        step = max(self._min_messages, AUTO_SUMMARIZE_THRESHOLD // 2)
        if existing_summary:
            return message_count % step == 0
        return True

    def maybe_summarize(
        self,
        *,
        session_id: str,
        message_count: int,
        existing_summary: str | None,
        messages: list[MemoryMessage] | None = None,
    ) -> str | None:
        """Return an updated summary, or the existing one when nothing changed.

        The signature stays compatible with the previous placeholder call site,
        but a real summary is now produced whenever there is material for one.
        """

        if messages is None:
            return existing_summary
        if not self.should_summarize(
            message_count=message_count, existing_summary=existing_summary
        ):
            return existing_summary
        summary = self.summarize(messages, previous=existing_summary)
        return summary or existing_summary

    # -- summarization -----------------------------------------------------------

    def summarize(
        self, messages: list[MemoryMessage], *, previous: str | None = None
    ) -> str:
        """Summarize a conversation, keeping durable content and dropping chatter."""

        if not messages:
            return previous or ""
        notes = extract_notes(messages)
        deterministic = render_notes(notes)

        if not self._model_available or self._ask is None:
            return deterministic

        transcript = _bounded_transcript(messages)
        prompt = (
            "Existing notes (keep what is still true, correct what changed):\n"
            f"{previous or '(none)'}\n\n"
            "Conversation:\n"
            f"{transcript}\n\n"
            "Durable notes:"
        )
        try:
            produced = self._ask(system_prompt=_SUMMARY_SYSTEM, user_prompt=prompt)
        except ProviderError:
            logger.warning("Conversation summarization failed; using deterministic notes")
            return deterministic
        except Exception:  # noqa: BLE001 - a model failure must not lose the summary
            logger.exception("Conversation summarization failed; using deterministic notes")
            return deterministic
        text = (produced or "").strip()
        if not text or len(text) < 20:
            return deterministic
        return text


def _bounded_transcript(messages: list[MemoryMessage], *, max_chars: int = 6000) -> str:
    """Render the most recent messages within a hard character budget."""

    lines: list[str] = []
    total = 0
    for message in reversed(messages):
        content = message.content.strip()
        if not content:
            continue
        line = f"{message.role}: {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(reversed(lines))

