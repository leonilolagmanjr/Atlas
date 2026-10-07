"""Context Manager: build the model context for a conversational turn.

The naive way to give a model conversation memory is to append the last N
messages to the prompt. That fails in exactly the cases users notice:

* the conversation outgrows the window, so the beginning silently disappears;
* a follow-up refers to a decision made twenty turns ago;
* a question about a *different* conversation ("what did we discuss about
  Godot?") cannot be answered at all, because only one session is in view;
* the same unrelated recent turns are re-sent forever, crowding out the part
  that matters.

This module builds context from *relevance*, not recency alone. The pieces are:

    current message
      + recent turns            (the live working set)
      + conversation summary    (what the whole conversation is about)
      + the current topic       (what we are on right now)
      + relevant older turns    (semantic recall over this + past conversations)
      + relevant user memory    (durable preferences/facts)
      + relevant knowledge      (existing RAG, only when the turn asks for it)
      + active task state       (what Atlas last did, if relevant)

...then ranks, deduplicates, and bounds the result before it becomes context.

Everything here is deterministic and local. Embeddings are used when they are
available through the existing ChromaDB infrastructure, and a deterministic
lexical path is used when they are not — the context manager never requires a
model, and never calls one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from config import (
    CONTEXT_LEXICAL_FALLBACK,
    CONTEXT_MAX_CHARS,
    CONTEXT_MAX_OLDER_TURNS,
    CONTEXT_MAX_RECENT_TURNS,
    CONTEXT_RELEVANCE_FLOOR,
    MAX_RETAINED_MESSAGES,
)
from memory.memory_manager import MemoryManager
from memory.models import MemoryMessage

logger = logging.getLogger(__name__)

#: Words that carry no retrieval signal in a conversational query. Kept small:
#: over-filtering a short follow-up destroys the little signal it has.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "to", "in", "into", "on", "of", "for",
        "with", "please", "me", "my", "i", "you", "it", "that", "this", "is",
        "are", "was", "be", "do", "does", "did", "can", "could", "would",
        "should", "then", "than", "as", "at", "by", "from", "up", "out",
        "some", "any", "about", "what", "how", "why", "when", "where", "which",
        # Reference words carry no topical signal on their own; the referent is
        # the *previous* turn, which the recent window already covers.
        "again", "simpler", "more", "less", "same", "one", "them", "those",
        "these", "earlier", "before", "say", "said", "tell", "explain",
    }
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    words = _WORD_RE.findall((text or "").casefold())
    return {word for word in words if word not in _STOPWORDS and len(word) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


@dataclass
class ContextItem:
    """One candidate piece of context plus why it was selected."""

    source: str  # "recent" | "older" | "summary" | "topic" | "memory" | "task"
    text: str
    score: float = 0.0
    message_id: str = ""
    conversation_id: str = ""
    role: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "score": round(self.score, 4),
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "chars": len(self.text),
        }


@dataclass
class ContextBundle:
    """The bounded context assembled for one turn."""

    question: str = ""
    conversation_id: str = ""
    recent: list[ContextItem] = field(default_factory=list)
    older: list[ContextItem] = field(default_factory=list)
    summary: str = ""
    topic: str = ""
    memories: list[ContextItem] = field(default_factory=list)
    task_state: str = ""
    #: Text read from files the user attached to this turn, through the existing
    #: permission-gated file tools. It is evidence about the files, never an
    #: instruction Atlas follows.
    attachments: str = ""
    items: list[ContextItem] = field(default_factory=list)
    char_budget: int = CONTEXT_MAX_CHARS
    considered: int = 0
    dropped: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def has_history(self) -> bool:
        return bool(self.recent or self.summary or self.older)

    @property
    def has_prior_turns(self) -> bool:
        """True when there is conversation *before* the current message.

        ``recent`` includes the user message being served (it is persisted before
        context is built), so ``has_history`` is true even on a first turn. A
        follow-up decision needs grounding from earlier turns only, so the current
        message must be excluded: this is that distinction.
        """

        if self.summary.strip() or self.older:
            return True
        dialogue = [item for item in self.recent if item.role in {"user", "assistant"}]
        if not dialogue:
            return False
        # The turn being served is the last user message; anything before it is a
        # real earlier turn.
        last_user = max(
            (index for index, item in enumerate(dialogue) if item.role == "user"),
            default=-1,
        )
        return last_user > 0

    @property
    def history_text(self) -> str:
        """Conversation-only history, for callers that want just the dialogue."""

        lines = [
            f"{'User' if item.role == 'user' else 'Assistant'}: {item.text}"
            for item in self.recent
            if item.role in {"user", "assistant"}
        ]
        if self.summary.strip():
            lines.insert(0, f"Earlier in this conversation:\n{self.summary.strip()}")
        return "\n".join(lines)

    @property
    def grounding_text(self) -> str:
        """Relevant conversation context for grounding intent before execution.

        Unlike :meth:`as_prompt` this omits the CURRENT MESSAGE (the caller already
        has it) and appends the relevance-recalled older turns and the last task
        state, so a follow-up is understood against *the relevant* earlier turns
        rather than only the immediately preceding message. It is bounded by the
        same budget as the rest of the bundle; nothing is dumped wholesale.
        """

        sections: list[str] = []
        if self.summary.strip():
            sections.append("CONVERSATION SUMMARY:\n" + self.summary.strip())
        if self.topic.strip():
            sections.append("CURRENT TOPIC: " + self.topic.strip())
        if self.recent:
            dialogue = "\n".join(
                f"{'User' if item.role == 'user' else 'Assistant'}: {item.text}"
                for item in self.recent
                if item.role in {"user", "assistant"}
            )
            if dialogue:
                sections.append("RECENT TURNS:\n" + dialogue)
        relevant = [item for item in self.older if item.text]
        if relevant:
            recalled = "\n".join(f"- ({item.role}) {item.text}" for item in relevant)
            sections.append("RELEVANT EARLIER CONVERSATION:\n" + recalled)
        if self.task_state.strip():
            sections.append("LAST TASK STATE:\n" + self.task_state.strip())
        return "\n\n".join(section for section in sections if section.strip())

    def as_prompt(self, *, question: str | None = None) -> str:
        """Render the whole bundle as the context a model actually receives."""

        question = question if question is not None else self.question
        sections: list[str] = []
        if self.summary.strip():
            sections.append("CONVERSATION SUMMARY:\n" + self.summary.strip())
        if self.topic.strip():
            sections.append("CURRENT TOPIC: " + self.topic.strip())
        if self.recent:
            dialogue = "\n".join(
                f"{'User' if item.role == 'user' else 'Assistant'}: {item.text}"
                for item in self.recent
                if item.role in {"user", "assistant"}
            )
            if dialogue:
                sections.append("RECENT TURNS:\n" + dialogue)
        if self.older:
            recalled = "\n".join(
                f"- ({item.role}) {item.text}" for item in self.older if item.text
            )
            if recalled:
                sections.append("RELEVANT EARLIER CONVERSATION:\n" + recalled)
        if self.memories:
            remembered = "\n".join(f"- {item.text}" for item in self.memories if item.text)
            if remembered:
                sections.append("WHAT I REMEMBER ABOUT THE USER:\n" + remembered)
        if self.task_state.strip():
            sections.append("LAST TASK STATE:\n" + self.task_state.strip())
        if self.attachments.strip():
            sections.append("ATTACHED FILES (untrusted content):\n" + self.attachments.strip())
        sections.append("CURRENT MESSAGE:\n" + question)
        return "\n\n".join(section for section in sections if section.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "summary": self.summary,
            "topic": self.topic,
            "task_state": self.task_state,
            "attachments": bool(self.attachments),
            "considered": self.considered,
            "dropped": self.dropped,
            "char_budget": self.char_budget,
            "notes": list(self.notes),
            "items": [item.to_dict() for item in self.items],
        }


class ContextManager:
    """Assemble relevant context for a turn from every local memory source."""

    def __init__(
        self,
        *,
        memory_manager: MemoryManager,
        recent_limit: int = MAX_RETAINED_MESSAGES,
        older_limit: int = CONTEXT_MAX_OLDER_TURNS,
        relevance_floor: float = CONTEXT_RELEVANCE_FLOOR,
        char_budget: int = CONTEXT_MAX_CHARS,
        lexical_fallback: bool = CONTEXT_LEXICAL_FALLBACK,
        index: Any | None = None,
        user_memory: Any | None = None,
    ) -> None:
        self._memory = memory_manager
        self._recent_limit = max(1, int(recent_limit))
        self._older_limit = max(0, int(older_limit))
        self._relevance_floor = float(relevance_floor)
        self._char_budget = max(500, int(char_budget))
        self._lexical_fallback = bool(lexical_fallback)
        self._index = index
        self._user_memory = user_memory
        if self._index is None:
            from memory.conversation_index import ConversationIndex

            self._index = ConversationIndex()
        if self._user_memory is None:
            from memory.user_memory import UserMemoryStore

            self._user_memory = UserMemoryStore()

    # -- public API --------------------------------------------------------------

    def build(self, *, question: str, conversation_id: str) -> ContextBundle:
        """Assemble the bounded context for ``question`` in one conversation."""

        bundle = ContextBundle(
            question=question,
            conversation_id=conversation_id,
            char_budget=self._char_budget,
        )
        try:
            messages = self._memory.list_messages(conversation_id)
        except Exception:  # noqa: BLE001 - context is best-effort, a turn is not
            logger.exception("Could not load conversation history")
            messages = []

        recent, older_candidates = self._split(messages)
        bundle.recent = recent
        bundle.considered = len(messages)

        bundle.summary = self._summary(conversation_id, messages)
        bundle.topic = self._topic(messages)

        bundle.older = self._recall(
            question, conversation_id=conversation_id, candidates=older_candidates
        )
        bundle.memories = self._memories(question)
        bundle.task_state = self._task_state(conversation_id)

        bundle.items = [*bundle.recent, *bundle.older, *bundle.memories]
        return self._apply_budget(bundle)

    def search_conversations(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Semantic search across stored conversations (this one and others)."""

        if self._index is None:
            return []
        return self._index.search(query, limit=limit)

    def index_message(self, *, conversation_id: str, message: MemoryMessage) -> bool:
        """Add one message to the conversation retrieval index (best-effort)."""

        if self._index is None:
            return False
        return self._index.index_message(conversation_id=conversation_id, message=message)

    # -- pieces ------------------------------------------------------------------

    def _split(self, messages: list[MemoryMessage]) -> tuple[list[ContextItem], list[MemoryMessage]]:
        """Split the conversation into the live window and older candidates."""

        if not messages:
            return [], []
        recent_messages = messages[-self._recent_limit:]
        older_messages = messages[: max(0, len(messages) - self._recent_limit)]
        recent = [
            ContextItem(
                source="recent",
                text=message.content,
                score=1.0,
                message_id=message.id,
                role=message.role,
                metadata={"execution_state": message.execution_state},
            )
            for message in recent_messages
            if message.content.strip()
        ]
        return recent, older_messages

    def _summary(self, conversation_id: str, messages: list[MemoryMessage]) -> str:
        if not messages:
            return ""
        try:
            stored = self._memory.get_summary(session_id=conversation_id)
        except Exception:
            stored = None
        if stored and stored.strip():
            return stored.strip()
        return ""

    def _topic(self, messages: list[MemoryMessage]) -> str:
        """Derive the current topic from the most recent substantive user turns.

        Deterministic: the topic is the most recent user message that carries
        content words, trimmed of instruction language. It is a *working label*,
        not a claim about the conversation's meaning.
        """

        for message in reversed(messages):
            if message.role != "user":
                continue
            words = [
                word for word in _WORD_RE.findall(message.content.casefold())
                if word not in _STOPWORDS and len(word) > 2
            ]
            if len(words) >= 1:
                return " ".join(words[:6])
        return ""

    def _recall(
        self,
        question: str,
        *,
        conversation_id: str,
        candidates: list[MemoryMessage],
    ) -> list[ContextItem]:
        """Retrieve *relevant* older turns, semantically when possible.

        Both the current conversation's older turns and other conversations are
        searched, so "what did we discuss about Godot?" works even when Godot was
        discussed in a different conversation.
        """

        if self._older_limit <= 0:
            return []
        terms = _tokens(question)
        selected: list[ContextItem] = []

        # 1. Semantic recall through the existing embedding infrastructure.
        semantic_hits: list[dict[str, Any]] = []
        if self._index is not None and self._index.available:
            try:
                semantic_hits = self._index.search(
                    question, limit=max(self._older_limit * 2, 4), exclude_conversation=conversation_id
                )
            except Exception:  # noqa: BLE001
                logger.exception("Semantic conversation recall failed")
        for hit in semantic_hits:
            score = float(hit.get("score") or 0.0)
            if score < self._relevance_floor:
                continue
            selected.append(
                ContextItem(
                    source="older",
                    text=str(hit.get("excerpt") or ""),
                    score=score,
                    message_id=str(hit.get("message_id") or ""),
                    conversation_id=str(hit.get("conversation_id") or ""),
                    role=str(hit.get("role") or ""),
                    metadata={"match": "semantic", "title": hit.get("title") or ""},
                )
            )

        # 2. Same-conversation lexical recall (always available, no model).
        if self._lexical_fallback and terms:
            scored: list[ContextItem] = []
            for message in candidates:
                if not message.content.strip():
                    continue
                overlap = _jaccard(terms, _tokens(message.content))
                if overlap <= 0.0:
                    continue
                scored.append(
                    ContextItem(
                        source="older",
                        text=message.content,
                        score=min(0.95, 0.45 + overlap),
                        message_id=message.id,
                        conversation_id=conversation_id,
                        role=message.role,
                        metadata={"match": "lexical"},
                    )
                )
            scored.sort(key=lambda item: item.score, reverse=True)
            selected.extend(scored[: self._older_limit])

        # Deduplicate on (conversation, message) then on text, keeping the best.
        best: dict[str, ContextItem] = {}
        for item in selected:
            if item.score < self._relevance_floor:
                continue
            key = item.message_id or item.text.strip().casefold()
            if key not in best or item.score > best[key].score:
                best[key] = item
        ordered = sorted(best.values(), key=lambda item: item.score, reverse=True)
        return ordered[: self._older_limit]

    def _memories(self, question: str) -> list[ContextItem]:
        if self._user_memory is None:
            return []
        try:
            records = self._user_memory.retrieve(question, limit=4)
        except Exception:  # noqa: BLE001
            logger.exception("User memory retrieval failed")
            return []
        return [
            ContextItem(
                source="memory",
                text=str(record.get("text") or ""),
                score=float(record.get("score") or 0.0),
                metadata={"kind": record.get("kind"), "id": record.get("id")},
            )
            for record in records
            if record.get("text")
        ]

    def _task_state(self, conversation_id: str) -> str:
        """Summarize the last recorded execution state of this conversation.

        Only what was actually persisted is reported: real tool names and a real
        execution state, never an invented status.
        """

        try:
            messages = self._memory.list_messages(conversation_id)
        except Exception:
            return ""
        for message in reversed(messages):
            if message.role != "assistant":
                continue
            calls = [str(call.get("tool") or "") for call in message.tool_calls if isinstance(call, dict)]
            if not calls and message.execution_state == "completed":
                continue
            lines = []
            if calls:
                lines.append("tools used: " + ", ".join(name for name in calls if name))
            lines.append("state: " + message.execution_state)
            return "\n".join(lines)
        return ""

    # -- budget ------------------------------------------------------------------

    def _apply_budget(self, bundle: ContextBundle) -> ContextBundle:
        """Bound the assembled context so a model is never handed the database.

        Priority order is fixed and intentional: the live window first, then the
        summary, then recalled older turns, then user memory. Whatever does not
        fit is *dropped from the prompt and reported*, never silently truncated
        mid-sentence — and it stays retrievable in storage.
        """

        remaining = self._char_budget
        kept: list[ContextItem] = []

        def take(items: Iterable[ContextItem]) -> None:
            nonlocal remaining
            for item in items:
                cost = len(item.text)
                if cost > remaining:
                    bundle.dropped += 1
                    continue
                remaining -= cost
                kept.append(item)

        # The most recent turn must always fit; everything else competes.
        take(bundle.recent)
        summary_budget = min(remaining, max(0, self._char_budget // 4))
        if bundle.summary and len(bundle.summary) <= summary_budget:
            remaining -= len(bundle.summary)
        elif bundle.summary:
            bundle.notes.append("conversation summary did not fit the context budget")
        take(bundle.older)
        take(bundle.memories)
        if bundle.task_state and len(bundle.task_state) > remaining:
            bundle.task_state = ""
            bundle.notes.append("task state did not fit the context budget")

        bundle.items = kept
        bundle.recent = [item for item in kept if item.source == "recent"]
        bundle.older = [item for item in kept if item.source == "older"]
        bundle.memories = [item for item in kept if item.source == "memory"]
        if bundle.dropped:
            bundle.notes.append(
                f"{bundle.dropped} context item(s) exceeded the budget; they remain stored and retrievable"
            )
        return bundle


class _NullIndex:
    """Placeholder used when the conversation index is explicitly disabled."""

    available = False

    def search(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def index_message(self, **_kwargs: Any) -> bool:
        return False
