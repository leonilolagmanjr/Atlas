"""Query routing: decide what kind of request Atlas is looking at.

Routing is *semantic first*: the structured Task IR produced by the interpreter
(goal, task_type, actions, entities) is the primary signal. Lexical feature
extraction is used only for the things the IR cannot express - freshness,
self-reference, memory-reference, file/system domains, and unresolved
references. No single keyword ever decides a route on its own; the router
combines features, the IR, and the live capability registry, and records which
features fired so the decision stays inspectable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from models_task import Task
from reasoning.self_introspection import SelfIntrospection

#: Words that indicate information that changes over time.
#: Explicit requests to use the internet as the information source.
_WEB_REQUEST_TERMS: tuple[str, ...] = (
    "search the web", "search the internet", "web search", "internet search",
    "search online", "look online", "look it up online", "browse the web",
    "go online", "google it", "use the internet", "use the web",
    "on the internet", "on the web",
)

_TIME_SENSITIVE_TERMS: tuple[str, ...] = (
    "latest", "newest", "most recent", "current", "currently", "up to date",
    "up-to-date", "today", "tonight", "this year", "this month", "this week",
    "right now", "as of", "so far", "breaking", "interest rate", "price",
    "release date", "recently released", "just released", "news", "trending",
)

#: Words that refer back to an earlier turn.
_MEMORY_TERMS: tuple[str, ...] = (
    "earlier", "before", "previously", "last time", "we discussed", "you said",
    "i said", "i asked", "remember", "my previous question", "our conversation",
    "last question", "you told me", "recall",
)

#: Domain vocabulary for local system state (resources and diagnostics).
_SYSTEM_TERMS: tuple[str, ...] = (
    "cpu", "processor", "memory usage", "ram", "disk", "disk space", "storage",
    "free space", "processes", "services", "startup", "boot", "slow", "sluggish",
    "lag", "lagging", "performance", "overheating", "battery", "fan", "network",
    "wifi", "internet connection", "hardware", "gpu", "graphics card",
    "windows version", "os version", "installed applications", "installed programs",
)

#: First-person framing that turns a technology term into a local-state question.
#: "What is the latest GPU?" is general knowledge; "why is my GPU hot?" is local.
_SYSTEM_LOCAL_MARKERS: tuple[str, ...] = (
    "my ", "this ", "am i ", "do i ", "does my ", "is my ", "on my ", "i have ",
    "the machine", "the computer",
)

#: Diagnostics that are unambiguously about the local machine even without
#: an explicit possessive.
_SYSTEM_DIAGNOSTIC_TERMS: tuple[str, ...] = (
    "slow", "sluggish", "lagging", "overheating", "memory usage", "cpu usage",
    "disk space", "free space", "startup", "boot time", "battery drain",
)


#: Domain vocabulary for local files and documents.
_FILE_TERMS: tuple[str, ...] = (
    "file", "files", "folder", "folders", "directory", "pdf", "pdfs", "document",
    "documents", "docx", "spreadsheet", "resume", "report", "notes", "notes.txt",
    "my downloads", "downloads folder", "desktop", "documents folder", "attachment",
)

#: Vocabulary of content that can be generated rather than retrieved.
_CREATION_TERMS: tuple[str, ...] = (
    "poem", "story", "essay", "song", "haiku", "joke", "letter", "caption", "plan",
    "outline", "script", "lyrics", "blog", "paragraph", "summary", "note", "email",
)

_QUESTION_LEADS: tuple[str, ...] = (
    "what", "why", "how", "when", "where", "who", "which", "whose", "is", "are",
    "does", "do", "can", "could", "should", "will", "explain", "define", "describe",
    "summarize", "summarise", "compare", "tell me", "help me understand",
)

_IMPERATIVE_LEADS: tuple[str, ...] = (
    "open", "launch", "start", "create", "write", "make", "compose", "generate",
    "move", "copy", "delete", "remove", "rename", "search", "find", "look", "show",
    "list", "run", "type", "save", "put", "summarize", "summarise", "explain",
)

#: Object pronouns whose referent must come from earlier context.
_PRONOUN_OBJECTS: frozenset[str] = frozenset(
    {"it", "that", "this", "them", "those", "there", "one", "these"}
)

_QUOTED_RE = re.compile(r"[\"'](.+?)[\"']", re.DOTALL)
_EXTENSION_RE = re.compile(
    r"\b(pdf|txt|text|docx?|xlsx?|csv|pptx?|png|jpe?g|gif|mp4|mp3|zip|json|py|md)\b",
    re.IGNORECASE,
)
_TARGET_NOUN_RE = re.compile(r"\b(?:my|the|this|that|a|an)\s+([A-Za-z][A-Za-z0-9._-]{2,})", re.IGNORECASE)

_READ_VERBS: tuple[str, ...] = (
    "say", "says", "read", "explain", "summarize", "summarise", "describe",
    "contain", "contains", "mention", "mentions", "tell me about", "walk me through",
)
_LIST_VERBS: tuple[str, ...] = ("list", "show", "what files", "what's in", "whats in", "show me")
_CONTENT_SEARCH_MARKERS: tuple[str, ...] = (
    "containing", "that mentions", "mentions", "for the word", "for the phrase",
    "search ... for", "where is", "find the section", "which file",
)

#: Folders users name conversationally, resolved relative to the home directory.
_KNOWN_FOLDERS: dict[str, str] = {
    "downloads": "Downloads",
    "documents": "Documents",
    "desktop": "Desktop",
    "pictures": "Pictures",
    "music": "Music",
    "videos": "Videos",
}

#: Extension per file-type noun, used to build a targeted glob.
_EXTENSION_BY_TYPE: dict[str, str] = {
    "pdf": "pdf", "txt": "txt", "text": "txt", "doc": "docx", "docx": "docx",
    "xls": "xlsx", "xlsx": "xlsx", "csv": "csv", "ppt": "pptx", "pptx": "pptx",
    "png": "png", "jpg": "jpg", "jpeg": "jpg", "gif": "gif", "mp4": "mp4",
    "mp3": "mp3", "zip": "zip", "json": "json", "py": "py", "md": "md",
}


@dataclass(frozen=True)
class FileIntent:
    """What Atlas needs to do with local files, if anything."""

    folder: str | None = None
    extension: str | None = None
    subject: str | None = None
    content_query: str | None = None
    wants_list: bool = False
    wants_read: bool = False
    wants_content_search: bool = False

    @property
    def pattern(self) -> str:
        """Return a filename glob built from the extension, if known."""

        return f"*.{self.extension}" if self.extension else "*"

    def to_dict(self) -> dict[str, Any]:
        return {
            "folder": self.folder,
            "extension": self.extension,
            "subject": self.subject,
            "content_query": self.content_query,
            "wants_list": self.wants_list,
            "wants_read": self.wants_read,
            "wants_content_search": self.wants_content_search,
        }


@dataclass(frozen=True)
class RoutingSignals:
    """Everything the source selector and reasoning engine need to decide.

    ``features`` records which signals fired so the route stays explainable.
    """

    text: str = ""
    is_question: bool = False
    is_self_query: bool = False
    self_kind: str | None = None
    is_memory_query: bool = False
    time_sensitive: bool = False
    explicit_web_request: bool = False
    system_reference: bool = False
    file_intent: FileIntent | None = None
    ambiguous_reference: bool = False
    wants_creation: bool = False
    wants_mutation: bool = False
    has_actions: bool = False
    sources: tuple[str, ...] = ()
    request_type: str = ""
    needs_clarification: bool = False
    features: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_question": self.is_question,
            "is_self_query": self.is_self_query,
            "self_kind": self.self_kind,
            "is_memory_query": self.is_memory_query,
            "time_sensitive": self.time_sensitive,
            "explicit_web_request": self.explicit_web_request,
            "system_reference": self.system_reference,
            "file_intent": self.file_intent.to_dict() if self.file_intent else None,
            "ambiguous_reference": self.ambiguous_reference,
            "wants_creation": self.wants_creation,
            "wants_mutation": self.wants_mutation,
            "has_actions": self.has_actions,
            "features": list(self.features),
        }


class QueryRouter:
    """Compute :class:`RoutingSignals` for a request.

    The router is deterministic. When the deterministic read is inconclusive and
    the reasoning engine is allowed to use the model, the engine may ask the
    model to confirm the *goal* (never to invent capabilities).
    """

    def route(
        self,
        text: str,
        *,
        task: Task,
        prior_task: Task | None = None,
        history: str = "",
        self_introspection: SelfIntrospection | None = None,
    ) -> RoutingSignals:
        lowered = text.casefold().strip()
        tokens = _tokens(lowered)
        features: list[str] = []

        is_question = _looks_interrogative(lowered, tokens)
        if is_question:
            features.append("question_form")

        has_actions = bool(task.actions)
        if has_actions:
            features.append("task_ir_actions")

        time_sensitive = _matches_terms(lowered, _TIME_SENSITIVE_TERMS) or str(
            task.entities.get("sort") or ""
        ) in {"latest", "newest"}
        if time_sensitive:
            features.append("time_sensitive")

        site = str(task.entities.get("site") or "").casefold()
        explicit_web_request = _matches_terms(lowered, _WEB_REQUEST_TERMS) or bool(site) or any(
            action.capability.startswith("web.") for action in task.actions
        )
        if explicit_web_request:
            features.append("explicit_web_request")

        is_self_query = False
        self_kind: str | None = None
        if self_introspection is not None:
            is_self_query = self_introspection.is_self_query(text)
            if is_self_query:
                self_kind = self_introspection.requested_kind(text)
                features.append("self_query")

        is_memory_query = _matches_terms(lowered, _MEMORY_TERMS) and not has_actions
        if is_memory_query:
            features.append("memory_query")

        file_intent = self._file_intent(lowered, task, features)
        system_reference = (
            _is_local_system_question(lowered)
            and not explicit_web_request
            and not (file_intent and file_intent.wants_list)
        )
        if system_reference:
            features.append("system_reference")

        wants_creation = any(term in lowered for term in _CREATION_TERMS)
        if wants_creation:
            features.append("creation_content")

        wants_mutation = _wants_mutation(task) or _has_mutation_verb(lowered)
        if wants_mutation:
            features.append("mutation")

        ambiguous_reference = _has_unresolved_reference(
            lowered,
            tokens,
            prior_task=prior_task,
            task=task,
            history=history,
            file_intent=file_intent,
        )
        if ambiguous_reference:
            features.append("unresolved_reference")

        return RoutingSignals(
            text=text,
            is_question=is_question,
            is_self_query=is_self_query,
            self_kind=self_kind,
            is_memory_query=is_memory_query,
            time_sensitive=time_sensitive or task.current_information_required,
            explicit_web_request=explicit_web_request,
            system_reference=system_reference,
            file_intent=file_intent,
            ambiguous_reference=ambiguous_reference,
            wants_creation=wants_creation,
            wants_mutation=wants_mutation,
            has_actions=has_actions,
            sources=tuple(task.sources) if task.source == "llm" or task.sources != ["model"] else (),
            request_type=task.request_type,
            needs_clarification=task.needs_clarification,
            features=tuple(features),
        )

    # -- file intent --------------------------------------------------------------

    def _file_intent(self, lowered: str, task: Task, features: list[str]) -> FileIntent | None:
        entities = task.entities
        file_terminology = _matches_terms(lowered, _FILE_TERMS)
        folder = entities.get("folder")
        filename = entities.get("filename")
        file_type = entities.get("file_type")

        if not (file_terminology or folder or filename or file_type):
            return None

        # Content written *into an application* is not a file lookup: the
        # destination entity already models it.
        if entities.get("application") and not (folder or file_type):
            return None

        extension: str | None = None
        if file_type:
            key = str(file_type).casefold()
            extension = _EXTENSION_BY_TYPE.get(key, key)
        else:
            match = _EXTENSION_RE.search(lowered)
            if match:
                key = match.group(1).casefold()
                extension = _EXTENSION_BY_TYPE.get(key, key)

        folder_name: str | None = None
        if folder:
            folder_name = _KNOWN_FOLDERS.get(str(folder).casefold(), str(folder))
        else:
            for word, canonical in _KNOWN_FOLDERS.items():
                if re.search(rf"\b{word}\b", lowered):
                    folder_name = canonical
                    break

        wants_list = _matches_terms(lowered, _LIST_VERBS) or bool(
            re.search(r"\bwhat(?:'s| is)?\s+(?:in|files)\b", lowered)
        )
        wants_read = _matches_terms(lowered, _READ_VERBS)
        content_query = self._content_query(lowered, file_context=file_terminology)
        wants_content_search = bool(content_query) or (
            _matches_terms(lowered, ("search", "find"))
            and _matches_terms(lowered, ("document", "documents", "file", "files", "folder"))
        )

        features.append("file_domain")
        if wants_list:
            features.append("file_list")
        if wants_read:
            features.append("file_read")
        if wants_content_search:
            features.append("file_content_search")

        subject: str | None = None
        if filename:
            subject = str(filename)
        else:
            match = _TARGET_NOUN_RE.search(lowered)
            if match:
                candidate = match.group(1).strip(".,?!")
                if candidate not in {"file", "files", "folder", "pdf", "document", "documents"}:
                    subject = candidate

        return FileIntent(
            folder=folder_name,
            extension=extension,
            subject=subject,
            content_query=content_query,
            wants_list=wants_list,
            wants_read=wants_read,
            wants_content_search=wants_content_search,
        )

    @staticmethod
    def _content_query(lowered: str, file_context: bool = False) -> str | None:
        quoted = _QUOTED_RE.search(lowered)
        if quoted:
            return quoted.group(1).strip()
        match = re.search(
            r"\b(?:containing|mentions?|about the word|for the phrase)\s+([a-z0-9 '\-]{3,40})[.?!]?$",
            lowered,
        )
        if match:
            candidate = match.group(1).strip()
            if candidate and not candidate.startswith(("the ", "my ", "a ")):
                return candidate
        if file_context:
            # "search my documents for X" / "what does my PDF say about X"
            match = re.search(
                r"\b(?:for|about|regarding|on)\s+([a-z0-9 '\-]{3,40})[.?!]?$",
                lowered,
            )
            if match:
                candidate = match.group(1).strip().strip("?.")
                if candidate and not candidate.startswith(("the ", "my ", "a ", "me ")):
                    return candidate
        return None


# -- module helpers ---------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text)


def _matches_terms(text: str, terms: Iterable[str]) -> bool:
    """Return True when any vocabulary term appears as a whole-word phrase."""

    for term in terms:
        if " " in term:
            if term in text:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", text):
            return True
    return False


def _looks_interrogative(lowered: str, tokens: list[str]) -> bool:
    if "?" in lowered:
        return True
    if not tokens:
        return False
    joined = " ".join(tokens[:3])
    return any(joined.startswith(lead) or tokens[0] == lead for lead in _QUESTION_LEADS)


def _wants_mutation(task: Task) -> bool:
    return bool(task.requires_confirmation) or any(
        action.risk_level != "read_only" for action in task.actions
    )


def _is_local_system_question(lowered: str) -> bool:
    """Return True for questions about *this machine's* state, not the topic.

    A technology term alone is general knowledge; local framing or a diagnostic
    phrase is what makes the request about the machine Atlas is running on.
    """

    if _matches_terms(lowered, _SYSTEM_DIAGNOSTIC_TERMS):
        return True
    if not _matches_terms(lowered, _SYSTEM_TERMS):
        return False
    return any(marker in lowered for marker in _SYSTEM_LOCAL_MARKERS)


def _has_mutation_verb(lowered: str) -> bool:
    return bool(
        re.search(
            r"\b(?:create|write|make|save|move|copy|delete|remove|rename|type|put|open|launch|start|install)\b",
            lowered,
        )
    )


def _has_unresolved_reference(
    lowered: str,
    tokens: list[str],
    *,
    prior_task: Task | None,
    task: Task,
    history: str,
    file_intent: "FileIntent | None" = None,
) -> bool:
    """Detect a target that only exists in earlier context.

    A pronoun object ("open it", "summarize that") is resolvable when a prior
    turn or conversation history exists and the Task IR carries a target. With
    no referent at all, the reference is genuinely ambiguous.
    """

    if not any(token in _PRONOUN_OBJECTS for token in tokens):
        return False
    entities = task.entities
    if entities.get("application") or entities.get("filename") or entities.get("folder"):
        return False
    # An explicitly named local file ("read my notes.txt and summarize it") gives
    # the pronoun a concrete referent even though it is not a filename entity.
    if file_intent is not None and (file_intent.subject or file_intent.extension or file_intent.folder):
        return False
    if any(action.capability.startswith("filesystem.") for action in task.actions):
        return False
    # A pronoun whose referent is the content a planned step produces ("summarize
    # them" of a web search/research result, "write it" from generated text) is
    # resolvable, so it is not an ambiguous reference.
    producing = {"web.search", "web.research", "web.fetch", "content.generate"}
    if any(action.capability in producing for action in task.actions):
        return False
    referent_available = bool(
        prior_task is not None or (history or "").strip() or task.context.get("prior_task")
    )
    if referent_available:
        return False
    return _matches_terms(lowered, _IMPERATIVE_LEADS) or _looks_interrogative(lowered, tokens)
