"""Conversation Runtime: the layer ABOVE the existing Atlas agent.

Atlas already had a complete agent: Brain -> interpreter -> Intent Engine ->
validator -> planner -> executor -> tools, plus a reasoning engine, RAG, web
research, computer control, verification and recovery. What it did not have was a
*conversation*: every user message went straight into interpretation and (for
anything the interpreter could not serve as read-only research) into planning.

This module adds the missing layer without replacing any of that:

    user message
      -> load the conversation (MemoryManager / ConversationStore)
      -> resolve context (ContextManager: recent turns, summary, topic,
         relevant older turns, user memory)
      -> decide what the message actually is:
           conversation | retrieval | research | task | computer | hybrid
      -> answer conversationally, or delegate to the *existing* Brain
      -> persist the turn (content, tool calls, citations, execution state)
      -> offer durable facts to long-term memory (never automatically)

Crucially:

* The Conversation Runtime never plans, executes, or selects tools itself. For
  anything that is not a plain answer it calls :meth:`Brain.process`, which is
  the single existing execution path.
* Classification is **deterministic and conservative**. A conversational
  message may be answered directly; anything that could act on the computer is
  delegated to Brain, and Brain's own validator decides whether a task is even
  buildable. When in doubt this layer delegates rather than answering, because a
  missed answer is recoverable and a missed action is not.
* A plain answer is produced by the existing
  :class:`reasoning.answer_generator.AnswerGenerator` (the model boundary Atlas
  already injects), so no second model client exists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from memory.memory_manager import MemoryManager
from memory.models import RESPONSE_KINDS

logger = logging.getLogger(__name__)


#: The kinds of turn the runtime can serve. Kept deliberately small; each one
#: maps onto an existing Atlas capability rather than a new subsystem.
KINDS: frozenset[str] = RESPONSE_KINDS

# ---------------------------------------------------------------------------
# Deterministic classification vocabulary
# ---------------------------------------------------------------------------

#: Verbs/constructions that require *acting on the computer*. If any of these is
#: present the runtime must delegate: a local answer cannot open, write, move or
#: install anything, and answering "conversationally" would be a lie.
_ACTION_LEADS: tuple[str, ...] = (
    "open ", "launch ", "start ", "close ", "quit ", "kill ", "install ",
    "uninstall ", "download ", "run ", "execute ",
)

_ACTION_PHRASES: tuple[str, ...] = (
    "write ", "write it", "type it", "type this", "put it", "put that",
    "put the", "save it", "save that", "save the", "create a file",
    "create the file", "make a file", "move the", "move it", "copy the",
    "copy it", "rename the", "rename it", "delete the", "delete it",
    "remove the file", "send it", "click ", "press ", "scroll ",
    "drag ", "focus ", "paste ", "copy this", "in notepad", "into notepad",
    "to notepad", "in word", "into word", "in vscode", "in the editor",
    "to a file", "into a file", "as a file",
)

#: Verbs that ask for retrieval/research rather than a local answer.
_RESEARCH_PHRASES: tuple[str, ...] = (
    "research ", "search the web", "search online", "look it up",
    "look up the latest", "search the internet", "find the latest",
    "google ", "browse ", "find out about", "what's the latest",
    "whats the latest", "what is the latest",
)

#: Reinforcement and simple acknowledgements are conversation, not requests.
_SMALL_TALK: frozenset[str] = frozenset(
    {
        "hi", "hey", "hello", "yo", "thanks", "thank you", "ta", "ok", "okay",
        "cool", "nice", "great", "got it", "understood", "bye", "goodbye",
        "good morning", "good evening", "good night", "how are you",
        "who are you", "what are you", "yes", "no", "sure", "please",
    }
)

#: Back-reference vocabulary used by :func:`_is_context_dependent`.
_REFERENCE_TERMS: tuple[str, ...] = (
    "that more simply", "that simply", "explain that", "simpler", "shorter",
    "longer", "more detail", "you said", "we discussed", "what about",
    "how about", "which one", "the first", "the second", "the same",
    "do the same", "what did we", "what did you",
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def _is_context_dependent(lowered: str) -> bool:
    """Return True when the message only makes sense with earlier turns.

    This is deliberately narrow: a *whole-word* back-reference or a turn that is
    essentially nothing but a reference ("what about mods?", "explain that more
    simply") is context-dependent. A sentence that merely contains the word
    "that" as a determiner ("Which node type renders sprites?") is not.
    """

    tokens = _WORD_RE.findall(lowered)
    if not tokens:
        return False
    words = set(tokens)
    if words & {"it", "them", "those", "these", "earlier", "previously", "again"}:
        return True
    for phrase in _REFERENCE_TERMS:
        if phrase in lowered:
            return True
    # A very short turn that is mostly a reference ("why?", "and that?").
    if len(tokens) <= 3 and tokens[-1] in {"that", "this", "one"}:
        return True
    return False


@dataclass(frozen=True)
class TurnClassification:
    """What the runtime decided the message is, and why."""

    kind: str = "conversation"
    delegate: bool = False
    reason: str = ""
    signals: tuple[str, ...] = ()
    #: True when the answer depends on earlier turns to be meaningful.
    context_dependent: bool = False
    #: True when the message is an instruction whose *object* is only in earlier
    #: turns ("install it"). This is the case where guessing would be wrong and
    #: asking is right, so it is named separately from topical back-references.
    anaphoric: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "delegate": self.delegate,
            "reason": self.reason,
            "signals": list(self.signals),
            "context_dependent": self.context_dependent,
            "anaphoric": self.anaphoric,
        }


def classify_turn(text: str) -> TurnClassification:
    """Deterministically decide whether a message needs the agent or an answer.

    The bias is explicit: **when in doubt, delegate**. A conversational answer
    is only produced for text this function can positively show is not an
    instruction to act. Everything else goes to the existing Brain, whose
    interpreter, validator and permission engine remain the authority on what is
    actually allowed to happen.
    """

    lowered = " ".join((text or "").casefold().split())
    if not lowered:
        return TurnClassification(kind="conversation", reason="empty message")

    signals: list[str] = []
    tokens = _WORD_RE.findall(lowered)
    stripped = lowered.strip(" .!?,")

    if stripped in _SMALL_TALK:
        return TurnClassification(
            kind="conversation",
            reason="greeting or acknowledgement; a local answer is sufficient",
            signals=("small_talk",),
        )

    # Build-variable so the "open chrmoe" style request (no trailing space) is
    # matched by the same rules as "open chrome".
    padded = f" {lowered} "
    action = any(lowered.startswith(lead) for lead in _ACTION_LEADS) or any(
        phrase in padded for phrase in _ACTION_PHRASES
    )
    if action:
        signals.append("action_language")

    research = any(phrase in lowered for phrase in _RESEARCH_PHRASES)
    if research:
        signals.append("research_language")

    time_sensitive = bool(
        re.search(
            r"\b(latest|newest|current|today|this week|this month|right now|up to date|up-to-date|recently)\b",
            lowered,
        )
        or re.search(r"\b20\d{2}\b", lowered)
    )
    if time_sensitive:
        signals.append("time_sensitive")

    context_dependent = _is_context_dependent(lowered)
    if context_dependent:
        signals.append("context_dependent")

    imperative = bool(
        re.match(
            r"^(?:please\s+|can you\s+|could you\s+|would you\s+)*"
            r"(?:create|write|make|save|move|copy|delete|remove|rename|type|put|send|"
            r"install|open|launch|start|run|download|research|search|find|go|show|list)\b",
            lowered,
        )
    )
    if imperative:
        signals.append("imperative")

    if action:
        # Mutation or application control: always the agent's job.
        return TurnClassification(
            kind="hybrid" if (research or time_sensitive) else "computer",
            delegate=True,
            reason="the request asks Atlas to act on the computer; only the existing permission-gated execution path may do that",
            signals=tuple(signals),
            context_dependent=context_dependent,
            anaphoric=context_dependent and len(tokens) <= 5,
        )

    if research or time_sensitive:
        # Research is delegated to the *existing* reasoning engine (which owns
        # source selection, retrieval, evidence and citations) rather than being
        # answered from model memory. Answering a research request locally would
        # bypass RAG/web research entirely, which is exactly what must not happen.
        return TurnClassification(
            kind="research",
            delegate=True,
            reason=(
                "the request needs retrieval of current information; the existing "
                "reasoning engine owns source selection and citations"
            ),
            signals=tuple(signals),
            context_dependent=context_dependent,
        )

    if imperative:
        # An unmapped imperative is not something to answer conversationally.
        return TurnClassification(
            kind="task",
            delegate=True,
            reason="the request is phrased as an instruction; the interpreter decides whether it is buildable",
            signals=tuple(signals),
            context_dependent=context_dependent,
        )

    return TurnClassification(
        kind="conversation",
        delegate=False,
        reason="no action, retrieval, or instruction was detected; a conversational answer applies",
        signals=tuple(signals),
        context_dependent=context_dependent,
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

EVENT_CONVERSATION_STARTED = "conversation_started"
EVENT_ASSISTANT_STARTED = "assistant_started"
EVENT_TOKEN = "token"
EVENT_TOOL_STARTED = "tool_started"
EVENT_TOOL_PROGRESS = "tool_progress"
EVENT_OBSERVATION = "observation"
EVENT_VERIFICATION = "verification"
EVENT_TOOL_COMPLETED = "tool_completed"
EVENT_ASSISTANT_COMPLETED = "assistant_completed"
EVENT_ERROR = "error"
EVENT_CANCELLED = "cancelled"
EVENT_ATTACHMENT = "attachment"

#: Real events the existing agent emits while it works, mapped onto the stream
#: vocabulary. Only work the agent actually performed is forwarded.
_LIVE_EVENT_TYPES: dict[str, str] = {
    "tool_started": EVENT_TOOL_STARTED,
    "tool_completed": EVENT_TOOL_COMPLETED,
    "observation": EVENT_OBSERVATION,
    "verification": EVENT_VERIFICATION,
}


@dataclass
class TurnEvent:
    """One real execution event emitted while a turn is being served.

    Events carry what actually happened. A token event carries a real token from
    the model stream; a tool event carries a real tool invocation. Nothing here
    fabricates progress for a UI.
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data}


EventSink = Callable[[TurnEvent], None]


@dataclass
class TurnResult:
    """Everything one served turn produced."""

    conversation_id: str
    message_id: str
    user_message_id: str
    text: str
    kind: str = "conversation"
    execution_state: str = "completed"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    activity: list[str] = field(default_factory=list)
    classification: TurnClassification = field(default_factory=TurnClassification)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "user_message_id": self.user_message_id,
            "text": self.text,
            "kind": self.kind,
            "execution_state": self.execution_state,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "citations": self.citations,
            "activity": self.activity,
            "classification": self.classification.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class CancellationToken:
    """Cooperative cancellation shared by the runtime and the executor path.

    This is a real token, not a UI flag: :meth:`is_cancelled` is consulted
    between steps and before tool execution, and the recorded execution state is
    ``cancelled``. It is deliberately *cooperative* — an in-flight blocking model
    or network call is not preempted mid-call, which is stated honestly in the
    README rather than claimed away.
    """

    _cancelled: bool = False
    reason: str = ""

    def cancel(self, reason: str = "cancelled by user") -> None:
        self._cancelled = True
        self.reason = reason

    def is_cancelled(self) -> bool:
        return self._cancelled

    def __call__(self) -> bool:
        return self._cancelled


def _activity_from_context(context: Any) -> tuple[list[dict], list[dict], list[str], list[str]]:
    """Derive real tool activity, citations and human-readable steps from a context.

    Only what the execution path *recorded* is reported. A plan step that never
    ran is not shown as an activity.
    """

    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    activity: list[str] = []
    citations: list[str] = []

    if context is None:
        return tool_calls, tool_results, activity, citations

    for call in list(getattr(context, "tool_calls", []) or []):
        if not isinstance(call, dict):
            continue
        tool_calls.append({
            "tool": call.get("tool"),
            "status": call.get("status"),
            "success": call.get("success"),
            "parameters": call.get("parameters") or {},
        })
        tool_results.append({
            "tool": call.get("tool"),
            "output": call.get("output"),
            "error": call.get("error"),
        })
        name = str(call.get("tool") or "tool")
        status = str(call.get("status") or "unknown")
        activity.append(_humanize_tool(name, status))

        # Real verification evidence, when the executor recorded any, so the UI
        # can show "Verification: ..." from data rather than from a guess.
        verification = (call.get("verification") or (call.get("metadata") or {}).get("verification"))
        if isinstance(verification, dict) and verification.get("status"):
            activity.append(f"Verification: {verification['status']}")

    plan = getattr(context, "execution_plan", None)
    if plan is not None:
        for step in getattr(plan, "steps", []) or []:
            status = getattr(getattr(step, "status", None), "value", "")
            if status in {"COMPLETED", "FAILED"}:
                activity.append(f"{step.name} ({status.lower()})")
            verification = (getattr(step, "metadata", {}) or {}).get("verification")
            if isinstance(verification, dict) and verification.get("status"):
                activity.append(f"Verification: {verification['status']}")

    citations.extend(str(item) for item in (getattr(context, "web_sources", []) or []) if item)
    return tool_calls, tool_results, activity, list(dict.fromkeys(citations))


def _humanize_tool(name: str, status: str) -> str:
    """Render a tool invocation as one readable activity line."""

    labels = {
        "web.search": "Web search",
        "web.research": "Web research",
        "web.fetch": "Read web page",
        "applications.launch": "Opened application",
        "applications.launch_named": "Opened application",
        "applications.write_text": "Wrote content into application",
        "filesystem.write": "Wrote file",
        "filesystem.read": "Read file",
        "filesystem.list": "Listed files",
        "filesystem.search": "Searched files",
        "filesystem.search_content": "Searched file contents",
        "content.generate": "Generated content",
        "content.format": "Formatted content",
        "computer.observe": "Observed screen",
        "computer.vision_observe": "Captured screen",
    }
    label = labels.get(name, name)
    return f"{label} — {status.replace('_', ' ')}"


class ConversationRuntime:
    """Serve one user message inside one persistent conversation.

    The runtime owns conversation state and turn orchestration. It does not own
    planning, execution, retrieval, tools, permissions, verification or memory
    storage: those are the existing Atlas subsystems, injected here so every one
    of them stays the single source of truth.
    """

    def __init__(
        self,
        *,
        memory_manager: MemoryManager,
        brain: Any | None = None,
        context_manager: Any | None = None,
        user_memory: Any | None = None,
        model_name: str = "",
        ask: Callable[..., str] | None = None,
        index: Any | None = None,
        tool_runner: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self._memory = memory_manager
        self._brain = brain
        self._model_name = model_name
        # Attachment reading delegates to the existing permission-gated file
        # tools; when none is wired, attachments degrade honestly.
        from memory.attachments import AttachmentReader

        self._attachments = AttachmentReader(tool_runner=tool_runner)
        # Non-streaming model boundary used for ordinary answers. When it is not
        # injected, the configured provider is used; when there is no provider
        # (a supported state), the answer generator returns an honest limitation
        # rather than inventing an answer.
        if ask is None:
            try:
                from config import ENABLE_LLM_INTERPRETATION

                if ENABLE_LLM_INTERPRETATION:
                    from llm import ask as default_ask

                    ask = default_ask
            except Exception:  # noqa: BLE001 - no model is a supported state
                ask = None
        self._ask = ask
        if user_memory is None:
            from memory.user_memory import UserMemoryStore

            user_memory = UserMemoryStore()
        self._user_memory = user_memory
        if context_manager is None:
            from memory.context_manager import ContextManager

            context_manager = ContextManager(
                memory_manager=memory_manager,
                index=index,
                user_memory=self._user_memory,
            )
        self._context = context_manager

    # -- accessors ---------------------------------------------------------------

    @property
    def context_manager(self) -> Any:
        return self._context

    @property
    def user_memory(self) -> Any:
        return self._user_memory

    def set_brain(self, brain: Any) -> None:
        """Attach the existing Brain after construction (API wires it lazily)."""

        self._brain = brain

    # -- conversations -----------------------------------------------------------

    def start_conversation(self, *, title: str | None = None) -> str:
        """Create a new conversation and make it active."""

        metadata = self._memory.create_session(title=title or "New conversation")
        return metadata.id

    def new_conversation(self, *, title: str | None = None) -> dict[str, Any]:
        """Create a new conversation, recording the model that will serve it."""

        metadata = self._memory.create_session(title=title or "New conversation")
        if self._model_name:
            metadata.model = self._model_name
            self._memory._store.save_metadata(metadata.id, metadata)  # noqa: SLF001
        return metadata.to_dict()

    def list_conversations(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._memory.list_sessions()]

    def open_conversation(self, conversation_id: str) -> dict[str, Any]:
        return self._memory.open_session(conversation_id).to_dict()

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        self._memory.rename_session(conversation_id, new_title=title)

    def delete_conversation(self, conversation_id: str) -> None:
        self._memory.delete_session(conversation_id)

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self._memory.list_messages(conversation_id)]

    def resolve_confirmation(
        self,
        conversation_id: str,
        task_id: str,
        *,
        approve: bool,
    ) -> dict[str, Any]:
        """Resolve a confirmation-paused conversational turn.

        The paused plan belongs to the Brain (keyed by the execution context
        ``task_id`` the turn recorded), not to the API task queue, so this calls
        the existing :meth:`Brain.approve_pending` / :meth:`Brain.deny_pending`
        contract rather than synthesising a task record. The persisted assistant
        message for that turn is then updated to the real state that occurred, so
        the transcript the user sees is the transcript the backend recorded.
        """

        if self._brain is None:
            raise RuntimeError("The Atlas runtime is not available yet")

        self._memory.open_session(conversation_id)
        if approve:
            response = self._brain.approve_pending(task_id)
        else:
            response = self._brain.deny_pending(task_id)

        context = getattr(self._brain, "last_context", None)
        state = _execution_state(context, CancellationToken())
        if context is None:
            # The Brain had no matching pending plan: the action was already
            # resolved. Report the honest outcome rather than a fabricated one.
            state = "completed" if approve else "cancelled"
        # A resumed plan that pauses again (a further consequential step) stays
        # truthfully ``waiting_for_confirmation`` so the control remains visible.

        conversation = self._ensure_conversation()
        target = self._find_confirmation_message(conversation.id, task_id)
        if target is None:
            return {
                "conversation_id": conversation.id,
                "task_id": task_id,
                "execution_state": state,
                "message": None,
                "messages": self.get_messages(conversation.id),
            }

        metadata = dict(target.metadata or {})
        metadata.pop("task_id", None)
        updated = self._memory.update_message(
            target.id,
            content=response or target.content,
            execution_state=state,
            metadata=metadata,
        )
        return {
            "conversation_id": conversation.id,
            "task_id": task_id,
            "execution_state": state,
            "message": updated.to_dict() if updated is not None else None,
            "messages": self.get_messages(conversation.id),
        }

    def _find_confirmation_message(self, conversation_id: str, task_id: str) -> Any:
        """Find the assistant message awaiting the given execution-context plan."""

        for message in self._memory.list_messages(conversation_id):
            if str(message.metadata.get("task_id") or "") == task_id:
                return message
        return None

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Search past conversations: semantic first, deterministic always."""

        semantic: list[dict[str, Any]] = []
        try:
            semantic = self._context.search_conversations(query, limit=limit)
        except Exception:  # noqa: BLE001 - search must never break the conversation
            logger.exception("Semantic conversation search failed")
        lexical = self._memory.search_conversations(query, limit=limit)
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for hit in semantic + lexical:
            key = (str(hit.get("conversation_id") or ""), str(hit.get("message_id") or ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
            if len(merged) >= limit:
                break
        return merged

    # -- serving a turn ----------------------------------------------------------

    def handle_message(
        self,
        text: str,
        *,
        conversation_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        events: EventSink | None = None,
        token: CancellationToken | None = None,
        stream: Callable[..., Any] | None = None,
    ) -> TurnResult:
        """Serve one user message and persist everything that happened.

        ``stream`` is an optional injected streaming model call
        (``stream(system_prompt=..., user_prompt=..., on_token=...)``). When it is
        supplied, the conversational answer is streamed as real tokens; when it is
        not, the same answer path is used without token events. Nothing is faked
        either way.
        """

        emit = events or (lambda event: None)
        cancellation = token or CancellationToken()

        if conversation_id:
            self._memory.open_session(conversation_id)
        conversation = self._ensure_conversation()

        if cancellation.is_cancelled():
            return self._cancelled_result(conversation.id, "")

        emit(TurnEvent(EVENT_CONVERSATION_STARTED, {"conversation_id": conversation.id, "message": text}))

        # Attachments are read through the existing file tools before the turn is
        # classified, so content the user handed Atlas is genuinely available to
        # the answer rather than merely mentioned in the prompt.
        from memory.attachments import normalize_attachments

        attachment_records = normalize_attachments(attachments)
        read_attachments: list[Any] = []
        if attachment_records:
            try:
                read_attachments = self._attachments.read(attachment_records)
            except Exception:  # noqa: BLE001 - an attachment never breaks a turn
                logger.exception("Attachment reading failed")
            emit(TurnEvent(EVENT_ATTACHMENT, {
                "count": len(read_attachments),
                "names": [item.name for item in read_attachments],
            }))

        user_message = self._memory.append_message(
            role="user",
            content=text,
            attachments=[item.to_dict() for item in read_attachments] or attachment_records,
            metadata={"response_kind": ""},
        )
        self._maybe_title_conversation(conversation.id, text)

        # Long-term memory: only the *durable* part of the turn is offered, and
        # only when the user actually stated something lasting.
        remembered: list[dict[str, Any]] = []
        try:
            remembered = self._user_memory.observe_user_message(
                text, conversation_id=conversation.id, message_id=user_message.id
            )
        except Exception:  # noqa: BLE001 - memory is never allowed to break a turn
            logger.exception("User memory observation failed")

        context_bundle = self._context.build(question=text, conversation_id=conversation.id)
        if read_attachments:
            from memory.attachments import AttachmentReader

            attachment_context = AttachmentReader.context_text(read_attachments)
            if attachment_context:
                context_bundle.attachments = attachment_context
        classification = _refine_with_context(classify_turn(text), context_bundle)

        emit(TurnEvent(EVENT_ASSISTANT_STARTED, {"kind": classification.kind}))

        if cancellation.is_cancelled():
            return self._cancelled_result(conversation.id, user_message.id)

        if classification.delegate:
            result = self._delegate(
                text,
                conversation_id=conversation.id,
                user_message_id=user_message.id,
                classification=classification,
                context_bundle=context_bundle,
                emit=emit,
                token=cancellation,
            )
        else:
            result = self._answer_conversationally(
                text,
                conversation_id=conversation.id,
                user_message_id=user_message.id,
                classification=classification,
                context_bundle=context_bundle,
                emit=emit,
                token=cancellation,
                stream=stream,
            )

        if result.execution_state == "cancelled":
            emit(TurnEvent(EVENT_CANCELLED, {"conversation_id": conversation.id}))
        elif result.execution_state == "failed":
            emit(TurnEvent(EVENT_ERROR, {"conversation_id": conversation.id, "text": result.text}))
        else:
            emit(TurnEvent(EVENT_ASSISTANT_COMPLETED, {
                "conversation_id": conversation.id,
                "message_id": result.message_id,
                "execution_state": result.execution_state,
            }))

        if remembered:
            result.metadata["memory_written"] = remembered
            result.activity.append(
                "Remembered: " + "; ".join(str(item.get("summary") or item.get("key")) for item in remembered)
            )
        return result

    # -- internal paths ----------------------------------------------------------

    def _ensure_conversation(self):
        active = self._memory.get_active_session_id()
        if active:
            return self._memory.get_active_metadata() or self._memory.create_session()
        return self._memory.create_session(title="New conversation")

    def _maybe_title_conversation(self, conversation_id: str, text: str) -> None:
        """Give a fresh conversation a real title derived from its first message.

        A title is derived deterministically from the user's own words; nothing
        is invented and an existing title is never overwritten.
        """

        try:
            metadata = self._memory._store.load_metadata(conversation_id)  # noqa: SLF001
        except Exception:
            return
        if metadata.message_count > 1 or metadata.title not in {"New conversation", "Untitled"}:
            return
        title = " ".join(text.strip().split())
        if not title:
            return
        if len(title) > 60:
            title = title[:57].rstrip() + "..."
        metadata.title = title
        metadata.touch()
        try:
            self._memory._store.save_metadata(conversation_id, metadata)  # noqa: SLF001
        except Exception:
            logger.exception("Could not title conversation")

    def _answer_conversationally(
        self,
        text: str,
        *,
        conversation_id: str,
        user_message_id: str,
        classification: TurnClassification,
        context_bundle: Any,
        emit: EventSink,
        token: CancellationToken,
        stream: Callable[..., Any] | None,
    ) -> TurnResult:
        """Answer a conversational turn through the existing answer generator."""

        from reasoning.answer_generator import AnswerGenerator

        history = context_bundle.history_text
        prompt = context_bundle.as_prompt(question=text)
        # The non-streaming askable (the model boundary) and the streaming
        # boundary are separate; the streamer is never passed as the askable,
        # because their call signatures differ.
        generator = AnswerGenerator(ask=self._ask)

        if stream is not None:
            answer_text = self._stream_answer(
                generator=generator,
                streamer=stream,
                prompt=prompt,
                system_prompt=generator._answer_prompt,  # noqa: SLF001 - same boundary object
                emit=emit,
                token=token,
            )
            state = _stream_state(answer_text, token)
            used_model = bool(answer_text)
            if state == "failed":
                answer_text = generator.limitation_text(
                    text, notes=["the local model produced no output"]
                )
            elif state == "cancelled":
                answer_text = answer_text or "Stopped."
            message = self._memory.append_message(
                role="assistant",
                content=answer_text,
                execution_state=state,
                metadata={"response_kind": classification.kind, "used_model": used_model},
            )
            return TurnResult(
                conversation_id=conversation_id,
                message_id=message.id,
                user_message_id=user_message_id,
                text=answer_text,
                kind=classification.kind,
                execution_state=state,
                classification=classification,
            )

        answer = generator.direct(text, history=history)
        if not answer.text.strip():
            answer = generator.direct(text, history="")
        state = "completed" if answer.text.strip() else "failed"
        message = self._memory.append_message(
            role="assistant",
            content=answer.text,
            citations=list(answer.citations),
            execution_state=state,
            metadata={
                "response_kind": classification.kind,
                "used_model": bool(answer.used_model),
                "provenance": [source.value for source in answer.provenance],
            },
        )
        return TurnResult(
            conversation_id=conversation_id,
            message_id=message.id,
            user_message_id=user_message_id,
            text=answer.text,
            kind=classification.kind,
            execution_state=state,
            citations=list(answer.citations),
            classification=classification,
        )

    def _stream_answer(
        self,
        *,
        generator: Any,
        streamer: Callable[..., Any],
        prompt: str,
        system_prompt: str,
        emit: EventSink,
        token: CancellationToken,
    ) -> str:
        """Stream real model tokens, stopping as soon as cancellation is raised.

        A stream that fails *before producing anything* is retried once through
        the ordinary blocking answer path. A transport hiccup should not be
        reported to the user as "the model produced no output", which is what a
        silent empty return turns into; if the retry also produces nothing, the
        caller reports a limitation, and the real error is logged with its
        traceback rather than discarded.
        """

        pieces: list[str] = []

        def on_token(piece: str) -> None:
            if token.is_cancelled():
                return
            pieces.append(piece)
            emit(TurnEvent(EVENT_TOKEN, {"text": piece}))

        try:
            produced = generator.stream(
                streamer=streamer,
                system_prompt=system_prompt,
                user_prompt=prompt,
                on_token=on_token,
                should_stop=token.is_cancelled,
            )
        except Exception:  # noqa: BLE001 - a model failure is reported, not hidden
            logger.exception("Streaming answer failed")
            return self._blocking_fallback(
                generator=generator,
                prompt=prompt,
                system_prompt=system_prompt,
                token=token,
                emit=emit,
            )
        if isinstance(produced, str) and produced and not pieces:
            return produced
        if pieces:
            return "".join(pieces)
        if token.is_cancelled():
            return ""
        # The stream completed but yielded nothing. Retry once without streaming
        # rather than presenting the silence as the model's answer.
        return self._blocking_fallback(
            generator=generator,
            prompt=prompt,
            system_prompt=system_prompt,
            token=token,
            emit=emit,
        )

    @staticmethod
    def _blocking_fallback(
        *,
        generator: Any,
        prompt: str,
        system_prompt: str,
        token: CancellationToken,
        emit: EventSink,
    ) -> str:
        """One non-streamed attempt, used when a stream yielded no usable output."""

        if token.is_cancelled():
            return ""
        ask = getattr(generator, "_ask", None)
        if ask is None:
            return ""
        try:
            text = ask(system_prompt=system_prompt, user_prompt=prompt)
        except Exception:  # noqa: BLE001
            logger.exception("Blocking fallback answer failed")
            return ""
        text = (text or "").strip()
        if text:
            logger.info("Streaming produced no tokens; answered with one blocking call")
            emit(TurnEvent(EVENT_TOKEN, {"text": text}))
        return text

    def _delegate(
        self,
        text: str,
        *,
        conversation_id: str,
        user_message_id: str,
        classification: TurnClassification,
        context_bundle: Any,
        emit: EventSink,
        token: CancellationToken,
    ) -> TurnResult:
        """Hand the message to the existing Brain and persist what it did."""

        if self._brain is None:
            message = self._memory.append_message(
                role="assistant",
                content="The Atlas runtime is not available yet, so I cannot carry that out.",
                execution_state="failed",
                metadata={"response_kind": "error"},
            )
            return TurnResult(
                conversation_id=conversation_id,
                message_id=message.id,
                user_message_id=user_message_id,
                text=message.content,
                kind="error",
                execution_state="failed",
                classification=classification,
            )

        emit(TurnEvent(EVENT_TOOL_STARTED, {"kind": classification.kind, "detail": classification.reason}))
        live_events: list[str] = []

        def on_event(event_type: str, payload: dict[str, Any]) -> None:
            """Forward a real execution event from the agent to the client."""

            live_events.append(event_type)
            if event_type not in _LIVE_EVENT_TYPES:
                return
            emit(TurnEvent(_LIVE_EVENT_TYPES[event_type], dict(payload)))

        try:
            response = self._process_with_brain(text, token, on_event)
        except Exception:  # noqa: BLE001 - reported honestly, never swallowed
            logger.exception("Delegated turn failed")
            message = self._memory.append_message(
                role="assistant",
                content="Atlas could not complete that request.",
                execution_state="failed",
                metadata={"response_kind": "error"},
            )
            return TurnResult(
                conversation_id=conversation_id,
                message_id=message.id,
                user_message_id=user_message_id,
                text=message.content,
                kind="error",
                execution_state="failed",
                classification=classification,
            )

        context = getattr(self._brain, "last_context", None)
        tool_calls, tool_results, activity, citations = _activity_from_context(context)
        state = _execution_state(context, token)
        if not live_events:
            # No live events arrived (an injected Brain, or a path that runs no
            # tools): report the recorded tool calls once so the client still
            # sees what actually ran.
            for call in tool_calls:
                emit(TurnEvent(EVENT_TOOL_COMPLETED, {"tool": call.get("tool"), "status": call.get("status")}))

        kind = _kind_for_context(context, classification)
        message = self._memory.append_message(
            role="assistant",
            content=response,
            tool_calls=tool_calls,
            tool_results=tool_results,
            citations=citations,
            execution_state=state,
            metadata={
                "response_kind": kind,
                "task_id": getattr(context, "task_id", None),
                "activity": activity,
            },
        )
        return TurnResult(
            conversation_id=conversation_id,
            message_id=message.id,
            user_message_id=user_message_id,
            text=response,
            kind=kind,
            execution_state=state,
            tool_calls=tool_calls,
            tool_results=tool_results,
            citations=citations,
            activity=activity,
            classification=classification,
        )

    def _process_with_brain(
        self,
        text: str,
        token: CancellationToken,
        on_event: Callable[[str, dict[str, Any]], None],
    ) -> Any:
        """Call the existing Brain, passing the real cancellation and event hooks.

        Both are handed to Brain only when Brain actually accepts them, so an
        injected double that only implements ``process(user_input)`` keeps
        working. Cancellation is not decoration here: Brain consults it after
        interpretation, after planning, and inside the executor before every
        step, so a stopped turn stops doing work.
        """

        import inspect

        process = self._brain.process
        try:
            parameters = inspect.signature(process).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
            return process(text)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        extra: dict[str, Any] = {}
        if "cancel" in parameters or accepts_kwargs:
            extra["cancel"] = token.is_cancelled
        if "on_event" in parameters or accepts_kwargs:
            extra["on_event"] = on_event
        return process(text, **extra)

    def _cancelled_result(self, conversation_id: str, user_message_id: str) -> TurnResult:
        message = self._memory.append_message(
            role="assistant",
            content="Stopped before starting.",
            execution_state="cancelled",
            metadata={"response_kind": "error"},
        )
        return TurnResult(
            conversation_id=conversation_id,
            message_id=message.id,
            user_message_id=user_message_id,
            text=message.content,
            kind="error",
            execution_state="cancelled",
        )


def _execution_state(context: Any, token: CancellationToken) -> str:
    """Report the state that actually occurred.

    The recorded context is the authority. A turn that ran to completion before
    the cancel request arrived is ``completed``, not ``cancelled``: reporting it
    as cancelled would claim the work was stopped when it was not. The token is
    only consulted when the context has not settled yet.
    """

    if context is None:
        return "cancelled" if token.is_cancelled() else "failed"
    status = getattr(getattr(context, "status", None), "value", None)
    if status == "COMPLETED":
        return "completed"
    if status == "WAITING_FOR_CONFIRMATION":
        return "waiting_for_confirmation"
    if status == "CANCELLED":
        return "cancelled"
    if status == "FAILED":
        return "failed"
    if status == "UNCERTAIN":
        # A clarification turn finished: Atlas asked rather than guessed. The
        # recorded kind says "needs clarification", so the state stays truthful
        # instead of claiming a completed answer.
        return "completed"
    if status in {"RUNNING", "PENDING", None}:
        return "cancelled" if token.is_cancelled() else "running"
    return "unknown"


def _kind_for_context(context: Any, classification: TurnClassification) -> str:
    """Report the kind that actually occurred, not the one that was predicted."""

    if context is None:
        return classification.kind
    # A clarification is decided by the agent (validator / reasoning), and it is
    # reported as one even when the message was predicted to be an action.
    status = getattr(getattr(context, "status", None), "value", None)
    mode = str(getattr(context, "metadata", {}).get("reasoning_answer", {}).get("mode", "") or "")
    task_view = getattr(context, "metadata", {}).get("task", {}) or {}
    if (
        mode == "clarification"
        or task_view.get("needs_clarification")
        or task_view.get("response_mode") == "clarification"
        or status == "UNCERTAIN"
    ):
        return "clarification"
    plan = getattr(context, "execution_plan", None)
    tool_names = {
        str(step.metadata.get("tool") or "")
        for step in (getattr(plan, "steps", []) or [])
        if getattr(step, "action", "") == "invoke_tool"
    }
    if not tool_names:
        mode = str(getattr(context, "metadata", {}).get("reasoning_answer", {}).get("mode", "") or "")
        if mode == "clarification":
            return "clarification"
        return classification.kind if classification.kind != "conversation" else "conversation"
    if any(name.startswith(("applications.", "computer.", "filesystem.write", "filesystem.create", "filesystem.move", "filesystem.copy")) for name in tool_names):
        return "hybrid" if any(name.startswith("web.") for name in tool_names) else "computer"
    if any(name.startswith("web.") for name in tool_names):
        return "research"
    if any(name.startswith("filesystem.") for name in tool_names):
        return "retrieval"
    return classification.kind


def _stream_state(answer_text: str, token: CancellationToken) -> str:
    """Return the truthful execution state after a streamed answer."""

    if token.is_cancelled():
        return "cancelled"
    return "completed" if answer_text.strip() else "failed"


def _refine_with_context(
    classification: TurnClassification, context_bundle: Any
) -> TurnClassification:
    """Fold context signals into the deterministic classification.

    A message that is a *bare* instruction referring to earlier context
    ("install it", "put that in Notepad") is escalated to the existing agent,
    because that is where reference resolution and the permission gate already
    live. A message that is merely context-dependent ("explain that more
    simply") stays conversational — the context manager supplies the referent.
    """

    if not getattr(context_bundle, "has_history", False):
        return classification
    if classification.delegate:
        return classification

    question = " ".join(str(getattr(context_bundle, "question", "") or "").casefold().split())
    if not question:
        return classification
    tokens = _WORD_RE.findall(question)
    if len(tokens) > 6:
        return classification

    bare_action = any(question.startswith(lead) for lead in _ACTION_LEADS) or bool(
        re.match(
            r"^(?:please\s+|can you\s+|could you\s+)?"
            r"(?:run|install|uninstall|open|launch|start|write|type|put|save|create|delete|move|copy|do)\s+"
            r"(?:it|that|them|this|those)\b",
            question,
        )
    )
    if not bare_action:
        return classification
    return TurnClassification(
        kind="computer",
        delegate=True,
        reason=(
            "a bare instruction referring to earlier context; the existing agent "
            "resolves the referent and applies the permission gate"
        ),
        signals=classification.signals + ("anaphoric_action",),
        context_dependent=True,
        anaphoric=True,
    )
