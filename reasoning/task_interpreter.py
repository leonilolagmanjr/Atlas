"""Semantic task interpretation: natural language -> structured :class:`Task`.

This is the boundary that turns free text into the Task IR defined in
``models_task``. It is LLM-first with a deterministic fallback:

* The local Qwen model proposes *meaning* (a structured task), never commands.
* Deterministic parsing provides a fast path and a safety net when the model is
  unavailable, slow, or returns something unusable.
* The result is always a valid :class:`Task`; malformed model output can never
  reach the planner or executor.

The deterministic fallback is deliberately *semantic*, not a phrase table: it
splits the request into a verb/content/destination/topic structure and never
keys on specific words like "poem" or "Notepad". That is what lets an unseen
combination such as "write a short story about space in Notepad" work without a
new rule.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Iterable, Optional

from config import ENABLE_LLM_INTERPRETATION, INTERPRETER_CONFIDENCE_THRESHOLD
from models_task import (
    Task, TaskAction, TaskState, EvidenceState, CompletionCriteria
)
from reasoning.json_llm import safe_reasoning_call
from tools.capabilities import CapabilityRegistry

logger = logging.getLogger(__name__)

#: Verbs mapped to a canonical action, grouped by what the request targets.
_OPEN_VERBS = ("open", "launch", "start", "run")
_SEARCH_VERBS = ("search", "find", "look", "show", "get", "pull", "browse", "google", "research")
_CREATE_VERBS = ("create", "write", "make", "generate", "compose", "produce", "draft", "put")
_MOVE_VERBS = ("move", "rename", "relocate")
_COPY_VERBS = ("copy", "duplicate")
_DELETE_VERBS = ("delete", "remove", "erase")

_WEB_HOSTS = ("youtube", "google", "the web", "the internet", "online")
_SEARCH_PLATFORMS = frozenset({"youtube", "google", "the web", "the internet"})

# Video-related terms that imply YouTube search when no explicit site is mentioned
_VIDEO_SEARCH_TERMS = ("video", "videos", "latest video", "newest video", "video url", "video link")

# Known application names to avoid misclassifying as topics.
_KNOWN_APPLICATIONS = frozenset({
    "notepad", "wordpad", "calculator", "calc", "paint", "mspaint",
    "explorer", "file explorer", "task manager", "command prompt", "cmd",
    "terminal", "powershell", "windows terminal", "vscode", "vs code",
    "visual studio code", "chrome", "google chrome", "edge", "microsoft edge",
    "firefox", "word", "excel", "powerpoint", "outlook", "spotify", "discord",
    "steam", "vlc", "settings"
})

_TONE_WORDS = (
    "funny", "serious", "sad", "happy", "formal", "casual", "dramatic",
    "romantic", "angry", "playful", "sarcastic", "friendly", "professional",
    "poetic", "witty", "dark", "optimistic",
)
_LENGTH_WORDS = {"short": "short", "brief": "short", "concise": "short", "long": "long", "detailed": "long"}
_SORT_WORDS = {
    "latest": "latest", "newest": "latest", "recent": "latest",
    "popular": "popular", "trending": "popular", "top": "popular",
    "oldest": "oldest", "largest": "largest", "biggest": "largest",
    "smallest": "smallest",
}

# Folders users name conversationally -> resolved relative to the home directory.
_KNOWN_FOLDERS = {
    "downloads": "Downloads",
    "documents": "Documents",
    "desktop": "Desktop",
    "pictures": "Pictures",
    "music": "Music",
    "videos": "Videos",
}

_CONTENT_NOUNS = (
    "poem", "story", "essay", "song", "joke", "letter", "email", "summary",
    "note", "report", "article", "script", "message", "list", "plan", "caption",
    "bio", "resume", "paragraph", "haiku", "limerick", "verse",
    "code", "snippet",
)
#: Lexical normalizer, not a command table: adjective/noun variants map onto a
#: canonical content type so "poetic", "poetry", or "verse" all mean a poem.
_CONTENT_TYPE_SYNONYMS = {
    "poetic": "poem", "poetry": "poem", "rhyme": "poem", "verse": "poem",
    "tale": "story", "narrative": "story", "article": "essay", "tune": "song",
    "lyrics": "song", "memo": "note", "newsletter": "email",
}
#: Document/report nouns a user refers to as one of *their own* files. Combined
#: with a possessive ("my resume") or a location ("in Downloads") these describe
#: a local file lookup, not a web search, even when the verb is "find"/"search".
_PERSONAL_DOCUMENT_NOUNS = (
    "resume", "cv", "rapport", "report", "notes", "note", "document", "doc",
    "spreadsheet", "presentation", "attachment", "invoice", "contract", "draft",
    "thesis", "dissertation", "essay", "paper", "project", "config", "readme",
    "spreadsheet", "log", "transcript", "receipt",
)
#: Best-effort extension for a personal document noun, used only to build a
#: targeted search glob. Missing/unknown nouns fall back to a noun-based glob.
_DOCUMENT_NAME_TO_EXTENSION = {
    "resume": "pdf", "cv": "pdf", "report": "pdf", "spreadsheet": "xlsx",
    "presentation": "pptx", "notes": "txt", "note": "txt", "transcript": "txt",
    "receipt": "pdf", "invoice": "pdf", "log": "log",
}
#: Possessive / location framing that marks a noun as *the user's own* file.
#: A bare article ("the ", "this ", "that ") is deliberately NOT a marker: it
#: matches almost any sentence and turned "write an essay about the election"
#: into a local-file lookup. Only a real possessive or an explicit local-folder
#: reference marks a document as the user's own.
_POSSESSIVE_MARKERS = ("my ", "our ", "in downloads", "from downloads",
                       "on my desktop", "in documents", "in my downloads",
                       "in my documents", "on the desktop")
_QUESTION_WORDS = ("what", "who", "when", "where", "why", "how", "explain",
                    "define", "compare", "summarize", "summarise", "describe")
#: Transformation verbs that indicate content should be transformed (summarized, condensed, etc.)
#: before being placed in a destination. These are distinct from placement verbs.
_TRANSFORM_VERBS = ("summarize", "summarise", "condense", "extract", "abstract", "digest")
#: Verbs that save/place a result somewhere. Kept separate from _CREATE_VERBS because
#: "save" is a *placement* verb ("save the results to a file") and must not imply
#: that new content be generated.
_PLACEMENT_VERBS = _CREATE_VERBS + ("copy", "paste", "type", "put", "place", "add", "insert", "save", "store", "export", "dump", "drop")
#: Transformation nouns that indicate the output should be a transformed version
_TRANSFORM_NOUNS = ("summary", "summarisation", "explanation", "overview", "synopsis", "digest", "abstract")
_PRONOUNS = {"it", "that", "this", "them", "those", "there", "the same", "one", "something"}


_ABOUT_RE = re.compile(
    r"\b(?:about|regarding|concerning|on the topic of|related to|re:)\s+"
    r"(.+?)"
    r"(?:\s+(?:in|into|on|using|inside|for|with|and\s+(?:put|write|save))\b"
    r"|\s+(?:save|store|export)\b"
    r"|\s*[.,?!]\s*$"
    r"|\s*$)",
    re.IGNORECASE,
)
#: A topic introduced by "of"/"for" after a content noun: "a list of 5 workout
#: exercises", "an essay on the election". Anchored on a content noun so a bare
#: "of France" ("the capital of France") is never captured as a creation topic.
_NOUN_OF_RE = re.compile(
    r"\b[a-z]+s?\s+(?:of|on|for)\s+(?:the\s+|a\s+|an\s+)?"
    r"((?:\d+\s+)?[A-Za-z][A-Za-z0-9' -]{1,60}?)"
    r"(?:\s+(?:in|into|inside|using|with|and\b)\b|\s*[.,?!]\s*$|\s*$)",
    re.IGNORECASE,
)
_IN_APP_RE = re.compile(
    r"\b(?:in|into|inside|using|with|within)\s+"
    r"([A-Za-z][A-Za-z0-9._+-]*(?: [A-Za-z][A-Za-z0-9._+-]*){0,3})"
    r"(?:\s+(?:and\b|about\b|that\b|which\b)|$|[,.?!])",
    re.IGNORECASE,
)
#: A placement verb followed by a destination preposition ("copy to Notepad",
#: "write it into Notepad", "paste this in Notepad"). "to" is only a
#: destination marker *after* an explicit placement verb, so the match is
#: anchored on the verb instead of adding "to" to the generic preposition list.
_PLACE_TO_APP_RE = re.compile(
    r"\b(?:copy|paste|write|save|put|place|add|insert|type|dump|drop)\s+"
    r"(?:it\s+|this\s+|that\s+|everything\s+|"
    r"the\s+(?:text|content|result|results|answer|script|code|list)\s+)?"
    r"(?:to|into|onto)\s+"
    r"([A-Za-z][A-Za-z0-9._+-]*(?: [A-Za-z][A-Za-z0-9._+-]*){0,3})"
    r"(?:\s+(?:and\b|about\b|that\b|which\b)|$|[,.?!])",
    re.IGNORECASE,
)
#: A destination like "in my Downloads folder" names a location, not an app.
_LOCATION_WORDS = frozenset(
    {"folder", "directory", "file", "my", "the", "a", "an", "terms", "detail",
     "simple", "general", "plain", "english", "downloads", "documents", "desktop",
     "pictures", "music", "videos", "online", "web", "internet"}
)
_OPEN_APP_RE = re.compile(
    r"\b(?:open|launch|start|run)\s+(?:up\s+)?([A-Za-z][A-Za-z0-9 ._+-]*?)"
    r"(?:\s+and\b|$|[,.?!])",
    re.IGNORECASE,
)
_SITE_RE = re.compile(
    r"\b(?:on|in|at|from)\s+(youtube|google|the web|the internet)\b"
    r"|\b(?:search|browse|look up|google)\s+(?:the\s+)?(web|internet|online)\b",
    re.IGNORECASE,
)
_SEARCH_WEB_RE = re.compile(
    r"\b(?:search|look up|google|browse|find)\b[^.]*?\b(?:web|internet|online)\b",
    re.IGNORECASE,
)
# YouTube as a bare word is NOT a search trigger; it requires an explicit search verb.
# The bare word "youtube" appearing in a prompt about content creation is a TOPIC.
_YOUTUBE_RE = re.compile(r"\byoutube\b", re.IGNORECASE)
_VIDEO_RE = re.compile(r"\bvideos?\b", re.IGNORECASE)

# Create/generate verbs that indicate content creation.
_CREATE_VERBS = ("create", "write", "make", "generate", "compose", "produce", "draft", "put")
# Open/launch verbs for application control.

# "save it as cars.txt" / "save as cars.txt" / "named cars.txt"
_SAVE_AS_RE = re.compile(
    r"\b(?:save|store|export)\b[^.]*?\b(?:as|to|into|named|called)\s+"
    r"((?:[A-Za-z0-9._-]+\.[A-Za-z0-9]{1,6})|(?:[A-Za-z0-9 _-]+?))\s*(?:$|[,!?])",
    re.IGNORECASE,
)
_AS_NAME_RE = re.compile(
    r"\b(?:as|named|called)\s+([A-Za-z0-9][A-Za-z0-9._-]*\.[A-Za-z0-9]{1,6})\b",
    re.IGNORECASE,
)
_FOLDER_NAMED_RE = re.compile(
    r"\b(?:folder|directory)\s+(?:called|named)?\s*([A-Za-z0-9._ -]+?)"
    r"(?:\s+(?:on|in|at|under)\b|$|[,.!?])",
    re.IGNORECASE,
)
_ON_DESKTOP_RE = re.compile(r"\b(?:on|to|in)\s+(?:my\s+|the\s+)?(desktop|downloads|documents|pictures|music|videos)\b", re.IGNORECASE)
_IN_FOLDER_RE = re.compile(r"\b(?:in|inside|from|under)\s+(?:my\s+|the\s+)?(downloads|documents|desktop|pictures|music|videos)\b", re.IGNORECASE)
_LARGEST_RE = re.compile(r"\b(largest|biggest|smallest|newest|oldest|most recent)\b", re.IGNORECASE)
_FILE_TYPE_RE = re.compile(r"\b(pdf|txt|text|docx?|xlsx?|csv|pptx?|png|jpe?g|gif|mp4|mp3|zip|json|py|md)\b", re.IGNORECASE)


class SemanticTaskInterpreter:
    """Turn free text into a validated-by-construction :class:`Task`."""

    def __init__(
        self,
        *,
        ask: Callable[..., str] | None = None,
        capabilities: CapabilityRegistry | None = None,
        capability_catalog: str = "",
        enabled: Optional[bool] = None,
    ) -> None:
        self._ask = ask
        self._capabilities = capabilities or CapabilityRegistry(None)
        self._capability_catalog = capability_catalog or self._capabilities.render_catalog()
        self._enabled = ENABLE_LLM_INTERPRETATION if enabled is None else enabled

    def interpret(self, text: str, *, context: Task | None = None, history: str = "") -> Task:
        """Interpret ``text`` into a :class:`Task`.

        A high-confidence deterministic read is returned directly. Otherwise the
        LLM is consulted (when enabled and available); its structured output is
        recovered defensively via :meth:`Task.from_mapping`.
        """

        prompt = text.strip()
        heuristic = self._deterministic_task(prompt)
        if not self._enabled or self._ask is None or heuristic.confidence >= INTERPRETER_CONFIDENCE_THRESHOLD:
            return self._reasoning_fields(self._with_context(heuristic, context))

        llm_task = self._llm_task(prompt, heuristic=heuristic, history=history)
        if llm_task is None:
            return self._reasoning_fields(self._with_context(heuristic, context))
        return self._reasoning_fields(self._with_context(self._reconcile(llm_task, heuristic), context))

    @staticmethod
    def _instructional_question(text: str) -> bool:
        return bool(re.match(
            r"^(?:please\s+)?(?:what\s+(?:is|are)\b|how\s+(?:do\s+I|does|can\s+I|to)\b|"
            r"why\b|define\b|explain\s+(?:how|what|why)\b|tell\s+me\s+how\b)",
            text.strip(), re.IGNORECASE,
        ))

    def _reasoning_fields(self, task: Task) -> Task:
        lowered = task.original_prompt.casefold()
        capabilities = {action.capability for action in task.actions}
        if capabilities:
            sources = []
            # Only add "model" if content.generate is NOT a transformation of
            # retrieved content (i.e., it generates from model knowledge, not
            # from input_content referencing another action's output).
            generate_is_transformation = any(
                action.capability == "content.generate"
                and action.parameters.get("input_content", "").startswith("$")
                for action in task.actions
            )
            if "content.generate" in capabilities and not generate_is_transformation:
                sources.append("model")
            if any(cap.startswith("web.") for cap in capabilities):
                sources.append("web")
            if any(cap.startswith("filesystem.") for cap in capabilities):
                sources.append("files")
            if any(cap.startswith("system.") for cap in capabilities):
                sources.append("system")
            # Read-only capabilities do not make a task an action/hybrid: they
            # observe or produce text. Only a capability that mutates local state
            # or drives an application adds the "computer" source. This list must
            # stay aligned with the reasoning engine's own research-only set.
            read_only = {
                "content.generate", "content.format", "web.search", "web.fetch",
                "web.research", "filesystem.search", "filesystem.read",
                "filesystem.list", "filesystem.metadata", "filesystem.search_content",
                "system.info", "processes.list",
            }
            if capabilities - read_only:
                sources.append("computer")
            task.sources = list(dict.fromkeys(task.sources + sources))
            task.request_type = "hybrid" if "computer" in sources and len(sources) > 1 else "action" if "computer" in sources or "content.generate" in capabilities else "question"
        elif not task.sources:
            if re.search(r"\b(?:you|your|atlas)\b", lowered) and re.search(r"\b(?:who|capabilities|tools|model|can you do|able to)\b", lowered):
                task.sources = ["self"]
            elif re.search(r"\b(?:earlier|previously|remember|we discussed|i (?:said|told|asked)|my name)\b", lowered):
                task.sources = ["memory", "conversation"]
            elif task.entities.get("filename") or task.entities.get("folder") or re.search(r"\b(?:my|this|that|our)\s+(?:file|document|folder|pdf|report)\b", lowered):
                task.sources = ["files"]
            elif re.search(r"\b(?:my|this|our)\s+(?:computer|pc|system|cpu|ram|disk)\b", lowered):
                task.sources = ["system"]
            elif task.entities.get("site") or re.search(r"\b(?:search|browse|look up)\b.*\b(?:web|internet|online)\b", lowered):
                task.sources = ["web"]
            elif re.search(r"\b(?:knowledge base|indexed documents|local knowledge)\b", lowered):
                task.sources = ["knowledge"]
            else:
                task.sources = ["model"]
        if task.needs_clarification:
            task.request_type = "clarification"
        elif task.request_type == "unknown":
            task.request_type = "self_query" if "self" in task.sources else "memory_query" if "memory" in task.sources or "conversation" in task.sources else "question"
        if not (capabilities & {"content.generate"}) and re.search(r"\b(?:latest|current|today|right now|recent|news|weather)\b", lowered):
            task.current_information_required = True
        for name, source in (
            ("requires_web", "web"), ("requires_files", "files"),
            ("requires_memory", "memory"), ("requires_knowledge", "knowledge"),
            ("requires_computer", "computer"), ("requires_system", "system"),
            ("requires_self_introspection", "self"), ("requires_model_knowledge", "model"),
        ):
            setattr(task, name, source in task.sources)
        task.requires_memory = task.requires_memory or "conversation" in task.sources
        return task

    def _with_context(self, task: Task, context: Task | None) -> Task:
        # Inherit a destination and shape for a follow-up that does not name one
        # ("make it about dogs"). The inherited entities are merged first, then
        # the content actions are rebuilt with the canonical generator so the
        # follow-up runs the same generate -> format -> write pipeline as a full
        # request. Patching a single write step onto the old actions would drop
        # both the inherited content type and the formatting stage.
        if context is None or not self._is_follow_up(task):
            return task
        inherited = False
        for key in ("application", "destination", "content_type", "folder"):
            value = context.entities.get(key)
            if value and not task.entities.get(key):
                task.entities[key] = value
                inherited = True
        if inherited:
            rebuilt = self._content_actions(task.entities)
            if rebuilt:
                task.actions = rebuilt
        return task
    @staticmethod
    def _is_follow_up(task: Task) -> bool:
        if task.entities.get("application") or task.entities.get("filename"):
            return False
        if not task.actions:
            return False
        return all(a.capability == "content.generate" for a in task.actions)

            # -- LLM path ----------------------------------------------------------------

    def _llm_task(self, text: str, *, heuristic: Task, history: str) -> Optional[Task]:
        from reasoning.prompts import TASK_INTERPRETER_SYSTEM, task_interpreter_user_prompt

        data = safe_reasoning_call(
            system_prompt=TASK_INTERPRETER_SYSTEM,
            user_prompt=task_interpreter_user_prompt(
                request=text,
                capabilities=self._capability_catalog,
                history=history,
                draft=heuristic.to_dict(),
            ),
            ask=self._ask,
        )
        if data is None:
            return None
        return Task.from_mapping(data, prompt=text, source="llm")

    def _reconcile(self, llm: Task, heuristic: Task) -> Task:
        """Keep LLM meaning but backfill fields deterministic parsing is sure of.

        The deterministic pass is trusted for concrete lexical facts (a named
        application, an explicit filename, a topic phrase) because those come
        from the literal user text; the model is trusted for structure and
        normalization. Neither side may drop a modifier the other found.
        """

        if self._instructional_question(heuristic.original_prompt):
            llm.actions = []
            llm.task_type = "informational"
            llm.execution_required = False
            llm.requires_confirmation = False
            llm.request_type = "question"
            llm.sources = [source for source in llm.sources if source != "computer"]
        elif not llm.actions and heuristic.actions:
            llm.actions = heuristic.actions
        if llm.task_type in {"unknown", ""}:
            llm.task_type = heuristic.task_type
        if llm.goal in {"unknown", ""}:
            llm.goal = heuristic.goal
        # Merge entities without overwriting concrete values the model found.
        for key, value in heuristic.entities.items():
            llm.entities.setdefault(key, value)
        for constraint in heuristic.constraints:
            if constraint not in llm.constraints:
                llm.constraints.append(constraint)
        # Backfill action parameters from matching heuristic actions. When the
        # LLM proposes a web.research action without the artifact parameters
        # (must_be_artifact, goal, target, content_type), the deterministic
        # pass—which parses the literal request text—is authoritative and
        # backfills them so task-aware retrieval stays correct.
        # Retrieval intent ("is the artifact itself wanted?") is read from the
        # literal request text, so a deterministic value always wins there: a
        # model guess of "information about the film" must not turn a script
        # request into a page about the film. Every other parameter is only
        # backfilled, leaving the model's own value in place.
        authoritative = ("must_be_artifact", "content_type", "goal", "target")
        for llm_action in llm.actions:
            for heuristic_action in heuristic.actions:
                if llm_action.capability == heuristic_action.capability:
                    for key, value in heuristic_action.parameters.items():
                        if key in authoritative and value is not None:
                            llm_action.parameters[key] = value
                        else:
                            llm_action.parameters.setdefault(key, value)
                    llm_action.depends_on = list(dict.fromkeys(
                        llm_action.depends_on + heuristic_action.depends_on
                    ))
                    if not llm_action.requires_confirmation and heuristic_action.requires_confirmation:
                        llm_action.requires_confirmation = heuristic_action.requires_confirmation
                    break
        llm.confidence = max(llm.confidence, heuristic.confidence)
        llm.source = "hybrid" if heuristic.actions else "llm"
        # A destination the deterministic pass is certain about wins over a
        # model guess, because it is copied from the literal request.
        if not llm.entities.get("application") and heuristic.entities.get("application"):
            llm.entities["application"] = heuristic.entities["application"]
        return llm

    # -- deterministic path ------------------------------------------------------

    def _deterministic_task(self, text: str) -> Task:
        entities = self._extract_entities(text)
        actions = self._build_actions(text, entities)
        task_type, goal, execution_required = self._classify(text, entities, actions)
        constraints = self._extract_constraints(text, entities)

        needs_clarification = self._needs_clarification(text, entities, actions)

        # Build semantic decomposition for complex tasks
        subtasks = self._decompose_into_subtasks(text, entities, actions, goal)

        # Initialize evidence state for retrieval tasks
        evidence_state = self._create_evidence_state(text, entities, goal) if self._needs_retrieval(actions) else None

        # Initialize completion criteria
        completion_criteria = self._create_completion_criteria(text, entities, actions, goal, subtasks)

        task = Task(
            task_type=task_type,
            goal=goal,
            original_prompt=text,
            actions=actions,
            entities=entities,
            constraints=constraints,
            confidence=self._confidence(text, entities, actions, task_type),
            requires_confirmation=any(a.requires_confirmation for a in actions),
            execution_required=execution_required,
            needs_clarification=needs_clarification,
            clarification_question=(
                self._clarification_question(entities, actions) if needs_clarification else None
            ),
            source="deterministic",
            task_state=TaskState.RECEIVED.value,
            subtasks=subtasks,
            evidence_state=evidence_state,
            completion_criteria=completion_criteria,
        )

        # Add execution trace entry
        task.execution_trace.append({
            "stage": "interpretation",
            "action": "deterministic_interpretation",
            "result": "task_created",
            "details": {"task_type": task_type, "goal": goal, "actions_count": len(actions), "subtasks_count": len(subtasks)}
        })

        return task

    def _needs_retrieval(self, actions: list[TaskAction]) -> bool:
        """Check if any action requires web/file retrieval."""
        retrieval_capabilities = {"web.search", "web.fetch", "web.research",
                                  "filesystem.search", "filesystem.read",
                                  "filesystem.search_content", "filesystem.list"}
        return any(a.capability in retrieval_capabilities for a in actions)

    def _create_evidence_state(self, text: str, entities: dict[str, Any], goal: str) -> EvidenceState:
        """Create initial evidence state for retrieval tasks."""
        target = entities.get("topic") or entities.get("content_type") or ""
        content_type = entities.get("content_type", "generic")
        must_be_artifact = self._must_be_artifact(text, content_type)

        # Determine retrieval goal based on task
        retrieval_goal = "find_information"
        if "script" in goal or "transcript" in goal or content_type in {"movie_script", "transcript", "lyrics", "code"}:
            retrieval_goal = "retrieve_document"
        elif "review" in goal:
            retrieval_goal = "find_review"
        elif "reference" in goal or "wiki" in goal:
            retrieval_goal = "find_reference"
        elif "media" in goal or content_type in {"video", "image"}:
            retrieval_goal = "find_media"

        return EvidenceState(
            target=target,
            goal=retrieval_goal,
            required_content_type=content_type,
            must_be_artifact=must_be_artifact,
        )

    def _must_be_artifact(self, text: str, content_type: str) -> bool:
        """Determine if the user wants the artifact itself vs information about it."""
        lowered = text.casefold()
        # Explicit artifact phrases
        artifact_phrases = ("full script", "complete script", "entire script", "the script",
                           "full transcript", "complete transcript", "entire transcript", "the transcript",
                           "full lyrics", "the lyrics", "full text", "complete text", "whole text",
                           "the source code", "full code", "entire code", "the raw text")
        if any(phrase in lowered for phrase in artifact_phrases):
            return True
        # Info phrases mean they DON'T want the artifact
        info_phrases = ("information about", "info about", "information on", "learn about",
                       "tell me about", "explain", "what is", "who is", "overview of",
                       "background on", "facts about", "details about")
        if any(phrase in lowered for phrase in info_phrases):
            return False
        # Question openers ask about the subject
        if re.match(r'^(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|will\s+you\s+)*(?:what|who|when|where|why|how|which|whose|is|are|was|were|does|do|did)\b', lowered):
            return False
        # Document types without info phrases = artifact request
        if content_type in {"movie_script", "transcript", "lyrics", "code", "documentation", "list"}:
            return True
        return False

    def _create_completion_criteria(self, text: str, entities: dict[str, Any], actions: list[TaskAction], goal: str, subtasks: list[dict]) -> CompletionCriteria:
        """Create explicit completion criteria for the task."""
        criteria = CompletionCriteria()

        # Base criteria for all tasks
        criteria.add_criterion("task_understood", "User intent correctly interpreted", met=True)

        # Retrieval criteria
        if self._needs_retrieval(actions):
            criteria.add_criterion("correct_target_identified", f"Correct target identified: {entities.get('topic', 'unknown')}")
            criteria.add_criterion("relevant_sources_retrieved", "Relevant source material retrieved")
            criteria.add_criterion("evidence_sufficient", "Sufficient evidence collected for output")

        # Content generation criteria
        if any(a.capability == "content.generate" for a in actions):
            criteria.add_criterion("content_generated", "Content successfully generated")
            criteria.add_criterion("content_formatted", "Content formatted for destination")

        # Delivery criteria
        if any(a.capability == "applications.write_text" for a in actions):
            app = entities.get("application", "the application")
            criteria.add_criterion("destination_opened", f"{app} opened successfully")
            criteria.add_criterion("content_delivered", f"Content written to {app}")
            criteria.add_criterion("delivery_verified", "Delivery verified")

        if any(a.capability == "filesystem.write" for a in actions):
            criteria.add_criterion("file_created", "Output file created")
            criteria.add_criterion("file_verified", "File content verified")

        # Subtask criteria
        for i, subtask in enumerate(subtasks):
            criteria.add_criterion(f"subtask_{i}_completed", f"Subtask completed: {subtask.get('description', 'unknown')}")

        return criteria

    def _decompose_into_subtasks(self, text: str, entities: dict[str, Any], actions: list[TaskAction], goal: str) -> list[dict[str, Any]]:
        """Decompose complex request into explicit subtasks."""
        subtasks = []

        # Analyze the request to identify distinct phases
        has_retrieval = self._needs_retrieval(actions)
        has_generation = any(a.capability == "content.generate" for a in actions)
        has_delivery = any(a.capability in {"applications.write_text", "filesystem.write"} for a in actions)

        if has_retrieval:
            target = entities.get("topic", "the requested content")
            subtasks.append({
                "id": "subtask_1",
                "description": f"Identify and locate {target}",
                "type": "retrieval",
                "capability": "web.research" if any(a.capability.startswith("web.") for a in actions) else "filesystem.search",
                "status": "pending",
                "depends_on": [],
            })
            subtasks.append({
                "id": "subtask_2",
                "description": "Evaluate retrieved source relevance and quality",
                "type": "evaluation",
                "capability": "internal",
                "status": "pending",
                "depends_on": ["subtask_1"],
            })
            subtasks.append({
                "id": "subtask_3",
                "description": "Refine search if evidence insufficient",
                "type": "retrieval_refinement",
                "capability": "web.research" if any(a.capability.startswith("web.") for a in actions) else "filesystem.search",
                "status": "pending",
                "depends_on": ["subtask_2"],
            })

        if has_generation:
            subtask_id = f"subtask_{len(subtasks) + 1}"
            subtasks.append({
                "id": subtask_id,
                "description": f"Generate {entities.get('content_type', 'content')} about {entities.get('topic', 'the topic')}",
                "type": "generation",
                "capability": "content.generate",
                "status": "pending",
                "depends_on": [subtasks[-1]["id"]] if subtasks else [],
            })

        if has_delivery:
            dest = entities.get("application") or entities.get("filename") or "destination"
            subtask_id = f"subtask_{len(subtasks) + 1}"
            subtasks.append({
                "id": subtask_id,
                "description": f"Open {dest}" if entities.get("application") else f"Create file {dest}",
                "type": "delivery_prep",
                "capability": "applications.launch_named" if entities.get("application") else "filesystem.write",
                "status": "pending",
                "depends_on": [subtasks[-1]["id"]] if subtasks else [],
            })
            subtask_id = f"subtask_{len(subtasks) + 1}"
            subtasks.append({
                "id": subtask_id,
                "description": f"Write content to {dest}",
                "type": "delivery",
                "capability": "applications.write_text" if entities.get("application") else "filesystem.write",
                "status": "pending",
                "depends_on": [subtasks[-1]["id"]] if subtasks else [],
            })
            subtask_id = f"subtask_{len(subtasks) + 1}"
            subtasks.append({
                "id": subtask_id,
                "description": f"Verify content in {dest}",
                "type": "verification",
                "capability": "internal",
                "status": "pending",
                "depends_on": [subtasks[-1]["id"]] if subtasks else [],
            })

        # Simple tasks without retrieval/generation/delivery
        if not subtasks and actions:
            for i, action in enumerate(actions):
                subtasks.append({
                    "id": f"subtask_{i+1}",
                    "description": action.description or action.capability,
                    "type": "action",
                    "capability": action.capability,
                    "status": "pending",
                    "depends_on": action.depends_on,
                })

        return subtasks

    # -- entity extraction -------------------------------------------------------

    def _extract_entities(self, text: str) -> dict[str, Any]:
        lowered = text.casefold()
        entities: dict[str, Any] = {}

        # Detect explicit search intent first.
        has_search_verb = not self._instructional_question(text) and any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _SEARCH_VERBS)
        has_create_verb = any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _CREATE_VERBS)
        has_open_verb = any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _OPEN_VERBS)

        # Explicit application: "open X", else "in X" / "using X".
        application = None
        open_match = _OPEN_APP_RE.search(text)
        if open_match:
            candidate = open_match.group(1).strip()
            if self._plausible_application(candidate):
                application = candidate
        if application is None:
            in_match = _IN_APP_RE.search(text)
            if in_match:
                candidate = in_match.group(1).strip()
                # A bare "in X" is ambiguous between an application ("in Notepad")
                # and a topic/location qualifier ("the weather in Tokyo"). Only a
                # *known* application is trusted here; an unknown destination app
                # arrives from the "open X"/"copy to X" forms below or the LLM.
                if self._plausible_application(candidate) and self._known_application(candidate):
                    application = candidate
        if application is None:
            # A placement verb with "to"/"into" ("copy to Notepad") names the
            # same kind of destination as "in Notepad"; _IN_APP_RE only covers
            # the generic prepositions, so this case is matched verb-anchored.
            to_match = _PLACE_TO_APP_RE.search(lowered)
            if to_match:
                candidate = to_match.group(1).strip()
                if self._plausible_application(candidate):
                    application = candidate
        if application:
            # A captured application must not carry trailing sentence
            # punctuation or a clause that reverses meaning ("Notepad about
            # cars" when the topic arrives first).
            application = application.strip(" .,;:!?")
            application = re.split(r"\s+(?:about|and|that|which)\b", application, maxsplit=1)[0].strip()
        if application and self._plausible_application(application):
            entities["application"] = application
            entities["destination"] = application

        # Content type: the first known content noun (any noun works, known ones
        # are canonicalized; unknown ones are captured as a "content" entity).
        content_type = None
        # Transformation nouns ("summary", "overview") describe how the result
        # should be *reshaped*, not what kind of source to look for. "summarize
        # the car videos" asks for a summary *of videos*, so the source type is a
        # video, and the transformation belongs on the transform step. Capturing
        # "summary" as the content type made task-aware retrieval hunt for a
        # "summary artifact", which does not exist.
        for noun in _CONTENT_NOUNS:
            if noun in _TRANSFORM_NOUNS:
                continue
            if re.search(rf"\b{noun}s?\b", lowered):
                content_type = noun
                break
        if content_type is None:
            for synonym, canonical in _CONTENT_TYPE_SYNONYMS.items():
                if re.search(rf"\b{synonym}\b", lowered):
                    content_type = canonical
                    break
        if content_type:
            entities["content_type"] = content_type
        # Record the requested transformation shape separately (summary,
        # overview, explanation, digest) so planning can consume the source
        # content type and the output shape independently.
        for noun in _TRANSFORM_NOUNS:
            if re.search(rf"\b{noun}s?\b", lowered):
                entities["transform"] = noun
                break

        # Topic: "about X" is the strongest signal for content creation.
        # Only apply this for create requests, not search requests.
        if has_create_verb and not has_search_verb:
            topic_match = _ABOUT_RE.search(text)
            if topic_match:
                topic = topic_match.group(1).strip(" ,.")
                if topic:
                    entities["topic"] = topic
            elif content_type:
                # "make a list of 5 workout exercises" -- the topic follows "of"
                # after the content noun, not "about". Only consulted when a
                # content noun was actually found, so an unrelated "of France"
                # is not promoted to a creation topic.
                noun_of = _NOUN_OF_RE.search(text)
                if noun_of:
                    topic = noun_of.group(1).strip(" ,.")
                    if topic:
                        entities["topic"] = topic
        # For search requests without "about", extract the search target as the topic.
        elif has_search_verb and not entities.get("topic"):
            search_target = self._extract_search_target(text)
            if search_target:
                entities["topic"] = search_target

        # Search platform/site: ONLY when there's an explicit search verb.
        # A bare mention of "youtube" or "google" without a search verb is a TOPIC,
        # not a search target.
        site = None
        if has_search_verb:
            # A personal document reference ("find my resume") is a local file
            # lookup, not a web search, even though the verb is "find"/"search".
            personal_document = self._personal_document_request(text)
            site_match = _SITE_RE.search(text)
            if site_match:
                site = (site_match.group(1) or site_match.group(2) or "the web").casefold()
                if site in {"web", "internet", "online"}:
                    site = "the web"
            elif has_search_verb and _YOUTUBE_RE.search(lowered):
                # Explicit "search youtube" or "find on youtube" -> YouTube is the platform.
                site = "youtube"
            elif _SEARCH_WEB_RE.search(text):
                site = "the web"
            elif personal_document:
                # Leave site unset: the request is about the user's own files.
                site = None
            elif self._has_search_target(text, entities):
                # Search verb present, no explicit site, but a concrete thing to
                # look for -> default to the web.
                # But if the query mentions video-related terms, default to YouTube
                # Check original text for video terms since topic/query may have them stripped
                lowered_text = text.casefold()
                if any(term in lowered_text for term in _VIDEO_SEARCH_TERMS):
                    site = "youtube"
                else:
                    site = "the web"
            else:
                # A bare "find/search" with only a pronoun ("find that file") has
                # no target; leave it unresolved so the reasoning layer asks.
                site = None
        # If a site was detected and we have a topic, remove the site name from the topic
        # (e.g., "search youtube for mrbeast videos" -> topic="mrbeast videos", site="youtube")
        if site and entities.get("topic"):
            site_words = set(site.replace(".", " ").split())
            topic_words = entities["topic"].split()
            # Remove leading site words from topic
            while topic_words and topic_words[0].casefold() in site_words:
                topic_words.pop(0)
            # Also remove "for" if it's the first word after site removal
            if topic_words and topic_words[0].casefold() == "for":
                topic_words.pop(0)
            entities["topic"] = " ".join(topic_words) if topic_words else ""
            if not entities["topic"]:
                entities.pop("topic", None)
        # If no search verb but we have a create verb + "about youtube", youtube is the TOPIC.
        # The topic is already captured by _ABOUT_RE above.
        if site:
            entities["site"] = site
            # Only default content_type to "video" for actual search requests.
            if site == "youtube" and content_type is None and not has_create_verb:
                entities["content_type"] = "video"
        if self._personal_document_request(text):
            # Mark the request as a local-document lookup so downstream source
            # selection reads it as files, not web/knowledge.
            entities["file_intent"] = "lookup"
            # The referenced document noun is the file subject ("resume" ->
            # "*resume*"). A personal document noun is a *file*, not generic
            # content to create, so it never defaults to a content type.
            if content_type in _PERSONAL_DOCUMENT_NOUNS:
                entities["file_subject"] = content_type
                entities.pop("content_type", None)
                extension = _DOCUMENT_NAME_TO_EXTENSION.get(content_type)
                if extension:
                    entities.setdefault("file_type", extension)

        # Sort/filter qualifiers (apply to both search and file operations).
        sort_match = _LARGEST_RE.search(lowered)
        if sort_match:
            entities["sort"] = _SORT_WORDS.get(sort_match.group(1).casefold(), sort_match.group(1).casefold())
        for word, canonical in _SORT_WORDS.items():
            if re.search(rf"\b{word}\b", lowered):
                entities.setdefault("sort", canonical)
                break

        # Tone / length.
        for word in _TONE_WORDS:
            if re.search(rf"\b{word}\b", lowered):
                entities["tone"] = word
                break
        for word, canonical in _LENGTH_WORDS.items():
            if re.search(rf"\b{word}\b", lowered):
                entities["length"] = canonical
                break

        # File type and explicit filename. "text" is normalized to the "txt"
        # extension so a requested file type maps to a real suffix.
        file_type = _FILE_TYPE_RE.search(lowered)
        if file_type:
            raw = file_type.group(1).casefold()
            entities["file_type"] = {"text": "txt", "jpeg": "jpg"}.get(raw, raw)
        save_as = _SAVE_AS_RE.search(text)
        if save_as:
            filename = save_as.group(1).strip().strip("\"'")
            if filename and "not" not in filename.casefold().split():
                if self._plausible_filename(filename):
                    entities["filename"] = filename
        if "filename" not in entities:
            # Only treat "as/named X.ext" as a filename, and only when X is a
            # real filename (has an extension); "a folder called Projects" must
            # not become a filename.
            as_match = _AS_NAME_RE.search(text)
            if as_match and "folder" not in text[: as_match.start()].casefold()[-12:]:
                entities["filename"] = as_match.group(1).strip()

        # Named folder creation: "a folder called Projects".
        folder_match = _FOLDER_NAMED_RE.search(text)
        if folder_match:
            folder_name = folder_match.group(1).strip()
            entities["folder_name"] = folder_name
            # A folder name is not a file to save.
            if str(entities.get("filename", "")).casefold() == folder_name.casefold():
                entities.pop("filename", None)

        # Location: "on my desktop", "in Downloads". A folder introduced by
        # "to"/"into" in a move/copy request is the *destination*, not the place
        # to search: "move the largest pdf to Documents" searches the root and
        # moves into Documents, it does not search inside Documents.
        location_match = _ON_DESKTOP_RE.search(text) or _IN_FOLDER_RE.search(text)
        if location_match:
            folder_value = _KNOWN_FOLDERS.get(location_match.group(1).casefold(), location_match.group(1))
            dest_clause = re.search(
                r"\b(?:to|into|onto)\s+(?:my\s+|the\s+)?(downloads|documents|desktop|pictures|music|videos)\b",
                text, re.IGNORECASE,
            )
            if dest_clause and folder_value.casefold() == dest_clause.group(1).casefold():
                entities["dest_folder"] = folder_value
            else:
                entities["folder"] = folder_value

        # Quantity: "with 5 examples". A standalone number of 1-3 digits only; a
        # 4-digit year (2024) or a longer number (RTX 5090) is not a quantity.
        quantity = re.search(r"(?<![\d.])(\d{1,3})(?![\d.])", text)
        if quantity:
            entities["quantity"] = int(quantity.group(1))

        # Content query for a local-document content search: a quoted phrase or
        # an explicit "for/about/containing X" clause. Only attached when the
        # request is about files, so a general question is not mistaken for one.
        # A content query describes searching *local documents*. It must not be
        # attached to a web request that merely mentions a file as an output
        # ("search the web for car reviews and summarize them in a file").
        if not entities.get("site"):
            if (entities.get("file_intent") == "lookup" or entities.get("file_type")
                    or any(word in text.casefold() for word in ("file", "files", "folder", "document", "documents"))):
                content_query = self._content_query(text)
                if content_query:
                    entities["content_query"] = content_query
        return entities
    @staticmethod
    def _content_query(text: str) -> str | None:
        # A quoted phrase is the strongest content-query signal.
        quoted = re.search(r"[\"'](.+?)[\"']", text, re.DOTALL)
        if quoted:
            return quoted.group(1).strip()
        # "for/about/containing/mentions X" at the end of the request.
        match = re.search(
            r"\b(?:for|about|regarding|containing|mentions?|on)\s+([A-Za-z0-9 '\-]{3,40})[.?!]?$",
            text.casefold(),
        )
        if match:
            candidate = match.group(1).strip().strip("?.")
            if candidate and not candidate.startswith(("the ", "my ", "a ", "me ")):
                return candidate
        return None

    def _is_file_request(self, text: str, entities: dict[str, Any]) -> bool:
        # A *local* file lookup. A requested file type alone is not enough: the
        # extension may describe where web content should be *saved* ("search for
        # X and save it as Y.txt"), which is a web task with a write, not a search
        # of the user's machine. Local framing (a possessive document, a folder,
        # or the explicit word "file") is what makes the object local.
        if self._personal_document_request(text):
            return True
        if entities.get("file_intent") or entities.get("file_subject"):
            return True
        if entities.get("folder"):
            return True
        lowered = text.casefold()
        names_file = any(re.search(rf"\b{word}\b", lowered) for word in ("file", "files", "folder", "directory"))
        has_local_cue = bool(entities.get("sort") or entities.get("folder") or entities.get("file_type"))
        return names_file and has_local_cue

    def _transforms_current_info(self, lowered: str, entities: dict[str, Any]) -> bool:
        # A transformation verb plus a current-information noun: "summarize the
        # news about tesla", "condense the latest updates on X". The request is
        # about information that changes, so it must be retrieved, not invented.
        if not any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _TRANSFORM_VERBS):
            return False
        return bool(
            re.search(
                r"\b(?:news|headlines?|updates?|developments?|latest|recent|current|"
                r"today|this week|this month|trending)\b",
                lowered,
            )
        )

    def _has_search_target(self, text: str, entities: dict[str, Any]) -> bool:
        # A search defaults to the web only when there is something concrete to
        # look for: a captured topic, a content noun, or a non-pronoun noun after
        # the search verb. "find that file" / "search it" have no target.
        if entities.get("topic") or entities.get("content_type"):
            return True
        lowered = text.casefold()
        match = re.search(
            r"\b(?:search|find|look up|look for|google|browse|show|get|research)\b\s+(?:for\s+|me\s+|the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9 '\-]{1,40})",
            lowered,
        )
        if not match:
            return False
        candidate = match.group(1).strip()
        words = [word for word in candidate.split() if word]
        if not words or words[0] in _PRONOUNS:
            return False
        # "that file" / "the document" name a local object, not a web query.
        if any(word in {"file", "files", "folder", "document", "documents"} for word in words):
            return False
        return True

    def _extract_search_target(self, text: str) -> str | None:
        """Extract the search target from a search request.

        Returns the noun phrase after the search verb, or None if not found.
        Stops at clause boundaries like "and write", "in notepad", "and summarize", etc.
        """
        lowered = text.casefold()
        # Cut the text at clause boundaries first (placement verbs, destination prepositions, transformation verbs)
        body = re.split(
            r"\b(?:and\s+)?(?:write|save|put|store|type|paste|copy|add|summarize|summarise|"
            r"condense|extract|digest|export|create|make|insert|place|open|launch|start|"
            r"in|into|to|on|using|with|from|display|show)\b",
            text, maxsplit=1, flags=re.IGNORECASE,
        )[0]
        match = re.search(
            r"\b(?:search|find|look up|look for|google|browse|show|get|research)\b\s+(?:for\s+|me\s+|the\s+|a\s+|an\s+)?([a-z0-9][a-z0-9 '\-]{1,40})",
            body.casefold(),
        )
        if not match:
            return None
        candidate = match.group(1).strip()
        words = [word for word in candidate.split() if word]
        if not words or words[0] in _PRONOUNS:
            return None
        # "that file" / "the document" name a local object, not a web query.
        if any(word in {"file", "files", "folder", "document", "documents"} for word in words):
            return None
        return candidate
    def _personal_document_request(self, text: str) -> bool:
        """Return True when the request refers to one of the user's own files.

        Detection is structural, not a phrase table: a possessive (or an explicit
        local-folder reference) attached to a document noun means the user is
        talking about a local file. "find my resume" is a file lookup; "find
        reviews" is a web search. An explicitly named web platform always wins,
        so "find my resume on youtube" is not treated as a local file request.
        """

        lowered = text.casefold()
        if _matches_any(lowered, _WEB_HOSTS):
            return False
        if any(host in lowered for host in ("the web", "the internet", "online")):
            return False
        if not any(re.search(rf"\b{noun}s?\b", lowered) for noun in _PERSONAL_DOCUMENT_NOUNS):
            return False
        return any(marker in lowered for marker in _POSSESSIVE_MARKERS)

    def _plausible_application(self, candidate: str) -> bool:
        lowered = candidate.casefold().strip()
        if not lowered or lowered in _PRONOUNS:
            return False
        if lowered in _CONTENT_NOUNS or lowered in _SEARCH_PLATFORMS:
            return False
        if lowered.startswith(("about ", "funny ", "short ", "long ", "a ", "an ", "the ", "and ")):
            return False
        # A destination that names a location or a modifier phrase is not an
        # application: "in my Downloads folder", "in simple terms".
        tokens = set(lowered.split())
        if tokens & _LOCATION_WORDS:
            return False
        if any(noun in lowered for noun in ("folder", "file", "document", "directory", "terms")):
            return False
        if any(word in tokens for word in _TONE_WORDS) or tokens & set(_LENGTH_WORDS):
            return False
        # A filename ("results.txt") is a file destination, not an application.
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.[A-Za-z0-9]{1,6}", candidate.strip()):
            return False
        # A destination should be a short proper name, not a sentence fragment.
        return len(lowered) <= 40 and len(lowered.split()) <= 4
    @staticmethod
    def _known_application(candidate: str) -> bool:
        # Whole-word match against the application list, so "in Notepad" and
        # "in VS Code" resolve while "in Tokyo" does not.
        lowered = candidate.casefold().strip()
        if lowered in _KNOWN_APPLICATIONS:
            return True
        return any(
            re.search(rf"\b{re.escape(app)}\b", lowered) for app in _KNOWN_APPLICATIONS
        )
    @staticmethod
    def _plausible_filename(candidate: str) -> bool:
        # A captured filename must look like a name, not a generic noun phrase.
        # "save the results to a file" must not capture "a file" as the target;
        # "save it as cars.txt" must. An extension is the strong signal; a bare
        # word is accepted only when it is not a stop-noun/pronoun.
        name = candidate.strip().strip("\"'")
        if not name or len(name) > 80:
            return False
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.[A-Za-z0-9]{1,6}", name):
            return True
        generic = {
            "a file", "the file", "file", "a text file", "text file", "results",
            "the results", "it", "that", "this", "them", "a document", "document",
            "a summary", "summary", "the output", "output", "something", "there",
        }
        # "save it to notepad" captures an application name, not a filename.
        return name.casefold() not in generic and name.casefold() not in _KNOWN_APPLICATIONS

    # -- action construction -----------------------------------------------------

    def _build_actions(self, text: str, entities: dict[str, Any]) -> list[TaskAction]:
        lowered = text.casefold()
        actions: list[TaskAction] = []
        if self._instructional_question(text):
            return actions

        has_create = any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _CREATE_VERBS)
        has_open = any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _OPEN_VERBS)
        has_search = any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _SEARCH_VERBS)
        has_move = any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _MOVE_VERBS)
        has_copy = any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _COPY_VERBS)
        has_folder_word = "folder" in lowered or "directory" in lowered
        has_delete = any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _DELETE_VERBS)

        site = entities.get("site")
        content_type = entities.get("content_type")
        topic = entities.get("topic")

        # Check for file operations FIRST (before web search) because search verbs
        # like "find" can apply to both files and web.
        # 1. Folder creation.
        if has_folder_word and has_create and not has_delete:
            path = self._folder_path(entities)
            if path:
                return [self._create_folder_action(path)]

        # 2. File search / move / copy. A bare move/copy verb only means a file
        # operation when a file context exists; otherwise ("search the web for X
        # and copy it in Notepad") the verb means "place the found text", and the
        # web branch below handles it as a hybrid.
        has_file_context = bool(
            entities.get("file_type") or entities.get("folder") or entities.get("filename")
            or entities.get("file_subject") or entities.get("file_intent")
        )
        file_move_or_copy = (has_move or has_copy) and has_file_context
        # An explicitly resolved web target ("search the web for X and save it as
        # Y.txt") is a web task with a write, not a local file lookup. A file type
        # in that request describes the *output* file, so it must not send Atlas
        # to search the user's own disk. Genuine personal-file requests still win.
        web_target = site in _WEB_HOSTS and not self._personal_document_request(text)
        if not web_target and (file_move_or_copy or (has_search and self._is_file_request(text, entities))):
            return self._file_actions(text, entities, move=has_move, copy=has_copy, search=has_search or not (has_move or has_copy))

        # 2b. Summarize/transform *a local file*: "create a summary of my
        # report.pdf and put it in notepad". This reads the file and feeds its
        # content to the generator, instead of generating unrelated content from
        # the model's own knowledge. Triggered only by a transformation plus a
        # local-file reference with no web target.
        if (
            not web_target
            and self._transformation_requested(text)
            and has_file_context
            and not entities.get("site")
        ):
            local_plan = self._local_file_transform(text, entities)
            if local_plan:
                return local_plan
        # 3. Web search dominates when a site/video is explicitly named as a search platform.
        # A site is only a search target if it was explicitly set (which only
        # happens when there's a search verb). If there's no site, we don't
        # want web search.
        wants_web = site in _WEB_HOSTS
        # Also treat as web search if there's a search verb + topic + destination
        # (e.g., "research Python decorators and put in Notepad")
        has_destination = entities.get("application") or entities.get("filename")
        if has_search and topic and has_destination:
            wants_web = True
            if not site:
                site = "the web"
        # "summarize the news about X" / "summarise the latest updates on Y" is a
        # transformation of *current* information, so it needs web research even
        # though it names no search verb or destination. Only a transformation
        # plus a current-information noun triggers this, so an ordinary
        # "summarize this document" is not sent to the web.
        if not wants_web and self._transforms_current_info(lowered, entities):
            wants_web = True
            site = site or "the web"
            if not entities.get("topic"):
                # "summarize the news about tesla" -> topic "tesla"; the "about"
                # phrase is the subject even though no search verb was used.
                about = _ABOUT_RE.search(text)
                target = about.group(1).strip(" ,.") if about else self._extract_search_target(text)
                if target:
                    entities["topic"] = target
        if wants_web:
            query = self._search_query(text, entities)
            # "search the web for X and write it into Notepad" is a hybrid task:
            # the content the user wants placed must be *read and isolated*, not
            # just the links. web.research searches and reads several top pages
            # itself, so the hybrid uses it as the primary step instead of a
            # separate web.search.
            follow_up = self._search_follow_up(text, entities, query=query, site=site)
            if follow_up:
                actions.extend(follow_up)
                if has_open and entities.get("application"):
                    actions.insert(
                        0,
                        self._launch_action(entities["application"], action_id="a0"),
                    )
                return actions
            actions.append(
                TaskAction(
                    action_id="a1",
                    capability="web.search",
                    parameters={
                        "query": query,
                        "site": site,
                        "sort": entities.get("sort"),
                        "max_results": 8,
                    },
                    description=f"Search {site or 'the web'} for {query}.",
                    produces="search_results",
                    expected_output="result links",
                )
            )
            # "open chrome and search youtube..." -> launch the browser first.
            if has_open and entities.get("application"):
                actions.insert(
                    0,
                    self._launch_action(entities["application"], action_id="a0"),
                )
            return actions
        # 4. Content creation, optionally written into an application.
        # Triggers on a create verb, or on a bare content noun with no competing
        # search/file intent: "tell me a joke", "give me a haiku" are generation
        # requests even though they name no create verb.
        if has_create and (content_type or topic or entities.get("application")):
            return self._content_actions(entities)
        if content_type and not has_search and not has_move and not has_copy and "folder" not in lowered:
            return self._content_actions(entities)

        # 4b. Create a plain new file ("create a text file") with no named
        # content to generate. This is a bare file touch: it must NOT preempt a
        # content-generation request that merely names an output type ("write
        # python code to reverse a string and save it as reverse.py"), so it only
        # fires when no content type or topic was extracted.
        if (
            has_create and not has_delete
            and not content_type and not topic
            and (entities.get("file_type") or "file" in lowered)
        ):
            created = self._create_file_action(entities)
            if created is not None:
                return [created]

        # 5. Plain application launch.
        if has_open and entities.get("application"):
            return [self._launch_action(entities["application"], action_id="a1")]

        return []

    def _launch_action(self, application: str, *, action_id: str) -> TaskAction:
        return TaskAction(
            action_id=action_id,
            capability="applications.launch_named",
            parameters={"application": application},
            description=f"Open {application}.",
            expected_output=f"{application} running",
            risk_level="low_risk",
            requires_confirmation=True,
        )

    def _content_actions(self, entities: dict[str, Any]) -> list[TaskAction]:
        content_type = entities.get("content_type") or "text"
        topic = entities.get("topic")
        actions = [
            TaskAction(
                action_id="a1",
                capability="content.generate",
                parameters={
                    "content_type": content_type,
                    "topic": topic,
                    "tone": entities.get("tone"),
                    "style": entities.get("style"),
                    "length": entities.get("length"),
                    "instructions": self._generation_instructions(entities),
                },
                description=f"Generate a {content_type}"
                + (f" about {topic}" if topic else "")
                + ".",
                produces="generated_text",
                expected_output="generated content",
            )
        ]
        application = entities.get("application")
        filename = entities.get("filename")
        if application:
            format_step = self._format_step(
                action_id="a2",
                source="$generated_text",
                destination=application,
                title=topic or "",
                depends_on=["a1"],
            )
            if format_step is not None:
                actions.append(format_step)
                write_input, write_depends = "$formatted_text", ["a2"]
                write_id = "a3"
            else:
                # content.format is not registered: write the generated text
                # directly rather than planning a step against a missing tool.
                write_input, write_depends, write_id = "$generated_text", ["a1"], "a2"
            actions.append(
                TaskAction(
                    action_id=write_id,
                    capability="applications.write_text",
                    parameters={"application": application, "text": write_input},
                    description=f"Write the formatted content into {application}.",
                    depends_on=write_depends,
                    expected_output="content present in the application",
                    risk_level="medium_risk",
                    requires_confirmation=True,
                )
            )
        elif filename:
            format_step = self._format_step(
                action_id="a2",
                source="$generated_text",
                destination=filename,
                title=topic or "",
                depends_on=["a1"],
            )
            if format_step is not None:
                actions.append(format_step)
                write_input, write_depends, write_id = "$formatted_text", ["a2"], "a3"
            else:
                write_input, write_depends, write_id = "$generated_text", ["a1"], "a2"
            actions.append(
                TaskAction(
                    action_id=write_id,
                    capability="filesystem.write",
                    parameters={"path": filename, "text": write_input},
                    description=f"Save the formatted content as {filename}.",
                    depends_on=write_depends,
                    expected_output=f"{filename} created",
                    risk_level="medium_risk",
                    requires_confirmation=True,
                )
            )
        return actions
    @staticmethod
    def _generation_instructions(entities: dict[str, Any]) -> Optional[str]:
        # Structural constraints the user stated that the model must honour: an
        # explicit item count ("a list of 5 exercises"). Without this the model
        # has no way to know how many items were requested.
        parts: list[str] = []
        quantity = entities.get("quantity")
        if isinstance(quantity, int) and quantity > 0:
            parts.append(f"Include exactly {quantity} item(s).")
        if entities.get("length") == "short":
            parts.append("Keep it short.")
        elif entities.get("length") == "long":
            parts.append("Make it detailed.")
        return " ".join(parts) if parts else None
    @staticmethod
    def _transformation_requested(text: str) -> bool:
        lowered = text.casefold()
        if any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _TRANSFORM_VERBS):
            return True
        return any(re.search(rf"\b{noun}s?\b", lowered) for noun in _TRANSFORM_NOUNS)
    def _local_file_transform(self, text: str, entities: dict[str, Any]) -> list[TaskAction]:
        # Read the referenced local file, summarize/transform its content, then
        # optionally deliver it. The file reference decides the search pattern;
        # the transformation decides the output content type.
        filename = entities.get("filename")
        subject = entities.get("file_subject")
        extension = entities.get("file_type")
        folder = entities.get("folder")
        if subject:
            pattern = "*" + str(subject).strip("*?") + "*"
            if extension:
                pattern += f".{extension}"
        elif extension:
            pattern = f"*.{extension}"
        else:
            pattern = "*"
        actions: list[TaskAction] = []
        read_path = filename
        if not read_path:
            actions.append(
                TaskAction(
                    action_id="a1",
                    capability="filesystem.search",
                    parameters={"pattern": pattern, "path": folder, "max_results": 50},
                    description=f"Find {pattern} files.",
                    produces="file_matches",
                    expected_output="matching file paths",
                )
            )
            read_path = "$file_matches"
        actions.append(
            TaskAction(
                action_id="a2" if not filename else "a1",
                capability="filesystem.read",
                parameters={"path": read_path, "max_bytes": 200000},
                description="Read the file content.",
                depends_on=["a1"] if not filename else [],
                produces="file_content",
                expected_output="the file text",
            )
        )
        transform_type = str(entities.get("transform") or "summary")
        if transform_type not in {"summary", "explanation", "overview", "digest"}:
            transform_type = "summary"
        read_id = "a2" if not filename else "a1"
        generate = TaskAction(
            action_id="a3" if not filename else "a2",
            capability="content.generate",
            parameters={
                "content_type": transform_type,
                "instructions": f"Summarize the following content, focusing on its key points.",
                "input_content": "$file_content",
                "topic": entities.get("topic") or "",
            },
            description=f"Produce a {transform_type} of the file content.",
            depends_on=[read_id],
            produces="generated_text",
            expected_output=f"{transform_type} of the file content",
        )
        actions.append(generate)
        destination = entities.get("application") or entities.get("filename")
        if destination and entities.get("application"):
            format_step = self._format_step(
                action_id="a4", source="$generated_text", destination=destination,
                title=entities.get("topic") or "", depends_on=[generate.action_id],
            )
            if format_step is not None:
                actions.append(format_step)
                write_text, write_deps = "$formatted_text", [format_step.action_id]
            else:
                write_text, write_deps = "$generated_text", [generate.action_id]
            actions.append(
                TaskAction(
                    action_id="a5",
                    capability="applications.write_text",
                    parameters={"application": destination, "text": write_text},
                    description=f"Write the {transform_type} into {destination}.",
                    depends_on=write_deps,
                    expected_output="content present in the application",
                    risk_level="medium_risk",
                    requires_confirmation=True,
                )
            )
        return actions
    def _format_step(
        self, *, action_id: str, source: str, destination: str, title: str, depends_on: list[str]
    ) -> Optional[TaskAction]:
        # Build the content.format step, or None when the capability is not
        # registered so the caller writes the source text directly. Keeping the
        # composition registry-authoritative means a runtime without the
        # formatter still produces a correct (if unformatted) result.
        if not self._capabilities.exists("content.format"):
            return None
        return TaskAction(
            action_id=action_id,
            capability="content.format",
            parameters={
                "content": source,
                "destination": destination,
                "title": title,
                "repair": True,
            },
            description=f"Format the generated content for {destination}.",
            depends_on=list(depends_on),
            produces="formatted_text",
            expected_output="formatted content ready for destination",
        )
    def _file_destination_requested(self, text: str, entities: dict[str, Any]) -> bool:
        # True when the request names a *file* destination (not an application).
        # Structural, not a phrase table: a save/export verb plus an explicit file
        # word, a file-type extension, or a *.ext filename means a file is the
        # destination. "save the results to a file" and "save it as results.txt"
        # both qualify; "save it in Notepad" (an application) does not."

        lowered = text.casefold()
        if entities.get("filename") or entities.get("file_type"):
            return True
        # A placement verb (save/write/put/...) or a transformation verb
        # (summarize/condense/...) plus an explicit file word means a file is the
        # destination: "save the results to a file", "summarize them in a file".
        if not re.search(
            r"\b(?:save|store|export|write|put|dump|summarize|summarise|condense|digest|abstract)\w*\b",
            lowered,
        ):
            return False
        if re.search(r"\b(?:into\s+)?(?:a\s+|the\s+|an\s+)?(?:file|files|txt|text file|document)\b", lowered):
            return True
        # "save it as cars.txt" style extension already captured by _AS_NAME_RE.
        if re.search(r"\bas\s+[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,6}\b", text):
            return True
        # A bare save/store/export of "the results"/"them"/"it" with no named
        # destination means persistence ("search X and save the results"); the
        # filename is derived from the topic.
        if re.search(
            r"\b(?:save|store|export|dump)\s+(?:the\s+|all\s+|these\s+|those\s+)?"
            r"(?:results?|links?|output|findings?|it|them)\b",
            lowered,
        ):
            return True
        return False

    def _default_results_filename(self, text: str, entities: dict[str, Any]) -> str:
        # Derive a safe default filename for saved search results. The name comes
        # from the topic (or the query body), slugified and bounded; the extension
        # honours a requested file type. No path separators or traversal
        # characters can enter it."

        topic = str(entities.get("topic") or "").strip()
        if not topic:
            topic = self._search_query(text, entities)
        slug = re.sub(r"[^a-z0-9]+", "_", topic.casefold()).strip("_")
        slug = slug[:60].strip("_") or "atlas_search"
        extension = str(entities.get("file_type") or "txt")
        return f"{slug}.{extension}"

    def _search_results_to_file(
        self, text: str, entities: dict[str, Any], *, query: str = "", site: str | None = None
    ) -> list[TaskAction]:
        # Plan a plain web search whose results are saved to a file. The saved
        # content is the rendered result list, published by the search as
        # $search_results and consumed by filesystem.write -- the search links,
        # not a full-page artifact (which a script/lyrics request needs and
        # _search_follow_up handles separately)."

        query = query or self._search_query(text, entities)
        path = entities.get("filename") or self._default_results_filename(text, entities)
        folder = entities.get("folder")
        if folder:
            path = f"{folder}/{path}"
        return [
            TaskAction(
                action_id="a1",
                capability="web.search",
                parameters={
                    "query": query,
                    "site": site,
                    "sort": entities.get("sort"),
                    "max_results": 8,
                },
                description=f"Search {site or 'the web'} for {query}.",
                produces="search_results",
                expected_output="result links",
            ),
            TaskAction(
                action_id="a2",
                capability="filesystem.write",
                parameters={"path": path, "text": "$search_results", "overwrite": False},
                description=f"Save the search results to {path}.",
                depends_on=["a1"],
                expected_output=f"{path} created",
                risk_level="medium_risk",
                requires_confirmation=True,
            ),
        ]

    def _search_follow_up(
        self, text: str, entities: dict[str, Any], *, query: str = "", site: str | None = None
    ) -> list[TaskAction]:
        # Post-search actions for a hybrid search-then-place task. When the user
        # asks for the *content* of something found on the web to be placed in an
        # application or file ("get the skyrim script and copy it in Notepad"),
        # Atlas must read and isolate the content, not dump the links. So the
        # plan researches the web (search + read several pages + rank) and the
        # write consumes the best-matching content.
        #
        # Also handles search-then-transform tasks ("search X and summarize in Notepad")
        # where a transformation verb indicates the retrieved content should be
        # processed before writing.
        #
        # Only an explicit placement verb (write/save/copy/put/...) OR a
        # transformation verb (summarize/condense/...) OR a transformation noun
        # (explanation/summary/...) plus a named destination produces these steps,
        # so a plain search stays plain. If there's a transformation but no
        # destination, we still produce a research+generate pipeline for the
        # reasoning engine to execute and return as an answer.
        lowered = text.casefold()
        has_placement = any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _PLACEMENT_VERBS)
        has_transformation_verb = any(re.search(rf"\b{verb}\w*\b", lowered) for verb in _TRANSFORM_VERBS)
        has_transformation_noun = any(re.search(rf"\b{noun}\b", lowered) for noun in _TRANSFORM_NOUNS)
        has_transformation = has_transformation_verb or has_transformation_noun
        if not (has_placement or has_transformation):
            return []
        application = entities.get("application")
        filename = entities.get("filename")
        # A file destination (explicit filename, a requested file type, or a
        # save-verb with a file word) means the *results* are persisted unless a
        # transformation reshapes them first. This is independent of whether the
        # filename was explicit: "save them in results.txt" and "save the results
        # to a file" are the same task.
        wants_file_results = (
            not application and self._file_destination_requested(text, entities)
        )
        if wants_file_results and not has_transformation:
            return self._search_results_to_file(text, entities, query=query, site=site)
        if wants_file_results and has_transformation and not filename:
            # "search X, summarize, save to a file": the summary is the artifact
            # to persist, so derive the filename and keep the transform pipeline.
            filename = self._default_results_filename(text, entities)
        has_destination = bool(application or filename or wants_file_results)

        # Determine the type of transformation for tailored instructions. The
        # interpreter records the requested shape as an entity; fall back to the
        # lexical scan so an LLM-sourced task still resolves a transform type.
        transform_type = str(entities.get("transform") or "summary")
        if transform_type not in {"summary", "explanation", "overview", "digest"}:
            if re.search(r"\bexplanation\b", lowered):
                transform_type = "explanation"
            elif re.search(r"\b(overview|synopsis)\b", lowered):
                transform_type = "overview"
            elif re.search(r"\bdigest\b", lowered):
                transform_type = "digest"
            else:
                transform_type = "summary"

        # Determine the retrieval goal based on whether this is a direct artifact
        # request or a transformation request.
        topic = entities.get("topic")
        content_type = entities.get("content_type")
        # A transformation request ("summarize these videos") wants information
        # about the subject that is then reshaped -- not the source artifact
        # itself. Retrieval must target the source content type and not demand an
        # artifact, or it would hunt for a "summary document".
        source_content_type = "generic" if has_transformation else content_type
        # Only a document content type can be an "artifact" to retrieve. "save the
        # results" / "save it" is a placement of search output, not a request for
        # a document, so it must not set must_be_artifact (which would send
        # retrieval hunting for a "best laptops document").
        document_requested = str(content_type or "").casefold() in {
            "script", "transcript", "lyrics", "song", "code", "documentation",
            "list", "movie_script",
        }
        wants_artifact = has_placement and not has_transformation and document_requested
        # A pure "save the results to a file" request does not want an artifact
        # read; it wants the search results written out, which is handled above.
        # A placement request wants the content itself (the script/transcript
        # article), not a page that merely discusses it, so the retrieval task
        # is marked as an artifact request.
        research = TaskAction(
            action_id="a1",
            capability="web.research",
            parameters={
                "query": query or self._search_query(text, entities),
                "site": site,
                "max_results": 8,
                "max_pages": 5,
                "target": topic,
                "content_type": source_content_type,
                "must_be_artifact": wants_artifact,
                "goal": "retrieve_document" if wants_artifact else "find_information",
            },
            description="Retrieve the requested content and isolate it."
            + ("" if has_placement else " for transformation."),
            produces="web_content",
            expected_output="the requested content",
        )
        
        actions = [research]
        
        # If transformation is requested, add a content.generate step to transform
        # the retrieved content before formatting (or as final output if no destination).
        if has_transformation:
            transform_instructions = {
                "summary": f"Summarize the following content about {topic or 'the topic'}. Focus on key points and main ideas.",
                "explanation": f"Explain the following content about {topic or 'the topic'} clearly and comprehensively.",
                "overview": f"Provide an overview of the following content about {topic or 'the topic'}.",
                "digest": f"Create a concise digest of the following content about {topic or 'the topic'}.",
            }.get(transform_type, f"Summarize the following content about {topic or 'the topic'}.")
            
            transform = TaskAction(
                action_id="a2",
                capability="content.generate",
                parameters={
                    "content_type": transform_type,
                    "input_content_type": source_content_type,
                    "topic": topic or "",
                    "instructions": transform_instructions,
                    "input_content": "$web_content",
                },
                description=f"Generate {transform_type} of the retrieved content about {topic or 'the topic'}.",
                depends_on=["a1"],
                produces="generated_text",
                expected_output=f"{transform_type} of the retrieved content",
            )
            actions.append(transform)
            format_input = "$generated_text"
            format_depends_on = ["a2"]
        else:
            format_input = "$web_content"
            format_depends_on = ["a1"]
        
        # If there's a destination, format and write. Otherwise, the generated
        # text (or web_content if no transformation) is the final output for the
        # reasoning engine to return as an answer.
        if has_destination:
            # Format the (transformed or raw) content for the destination
            format_action = TaskAction(
                action_id="a3" if has_transformation else "a2",
                capability="content.format",
                parameters={
                    "content": format_input,
                    "destination": application or filename,
                    "title": topic or "",
                    "repair": True,
                },
                description="Format the content for the destination.",
                depends_on=format_depends_on,
                produces="formatted_text",
                expected_output="formatted content ready for destination",
            )
            actions.append(format_action)
            
            write_id = "a4" if has_transformation else "a3"
            write_depends_on = ["a3"] if has_transformation else ["a2"]
            if application:
                write = TaskAction(
                    action_id=write_id,
                    capability="applications.write_text",
                    parameters={"application": application, "text": "$formatted_text"},
                    description=f"Write the formatted content into {application}.",
                    depends_on=write_depends_on,
                    expected_output="content present in the application",
                    risk_level="medium_risk",
                    requires_confirmation=True,
                )
            else:
                write = TaskAction(
                    action_id=write_id,
                    capability="filesystem.write",
                    parameters={"path": filename, "text": "$formatted_text"},
                    description=f"Save the formatted content as {filename}.",
                    depends_on=write_depends_on,
                    expected_output=f"{filename} created",
                    risk_level="medium_risk",
                    requires_confirmation=True,
                )
            actions.append(write)
        return actions

    def _create_file_action(self, entities: dict[str, Any]) -> Optional[TaskAction]:
        # Build a filesystem.write action for a create-a-file request. No content
        # was supplied and no name was given, so a default name is derived from the
        # requested type and a short explicit note is written. The action is
        # confirmation-gated like every other mutation.
        filename = entities.get("filename")
        extension = entities.get("file_type") or "txt"
        if not filename:
            filename = f"atlas_created_{extension}.{extension}"
        elif "." not in filename:
            filename = f"{filename}.{extension}"
        topic = str(entities.get("topic") or "").strip()
        text = f"Created by Atlas about {topic}.\n" if topic else "Created by Atlas.\n"
        folder = entities.get("folder")
        path = f"{folder}/{filename}" if folder else filename
        return TaskAction(
            action_id="a1",
            capability="filesystem.write",
            parameters={"path": path, "text": text, "overwrite": False},
            description=f"Create the file {path}.",
            expected_output=f"{path} created",
            risk_level="medium_risk",
            requires_confirmation=True,
        )

    def _create_folder_action(self, path: str) -> TaskAction:
        return TaskAction(
            action_id="a1",
            capability="filesystem.create_folder",
            parameters={"path": path},
            description=f"Create the folder {path}.",
            expected_output=f"{path} exists",
            risk_level="medium_risk",
            requires_confirmation=True,
        )

    def _file_actions(
        self,
        text: str,
        entities: dict[str, Any],
        *,
        move: bool,
        copy: bool,
        search: bool,
    ) -> list[TaskAction]:
        actions: list[TaskAction] = []
        source_dir = entities.get("folder")
        file_type = entities.get("file_type")
        subject = entities.get("file_subject")
        content_query = entities.get("content_query")
        if subject:
            pattern = "*" + str(subject).strip("*?") + "*"
            if file_type and f".{file_type}" not in pattern.casefold():
                pattern += f".{file_type}"
        else:
            pattern = f"*.{file_type}" if file_type else "*"

        # A content query ("search my documents for 'climate change'") is a
        # full-text search, a distinct contract from a filename lookup.
        if content_query and not (move or copy):
            actions.append(
                TaskAction(
                    action_id="a1",
                    capability="filesystem.search_content",
                    parameters={
                        "query": str(content_query),
                        "path": source_dir,
                        "pattern": None if pattern == "*" else pattern,
                        "max_results": 50,
                    },
                    description=f"Search file contents for '{content_query}'"
                    + (f" in {source_dir}" if source_dir else "")
                    + ".",
                    produces="content_matches",
                    expected_output="matching files with excerpts",
                )
            )
            return actions
        sort = entities.get("sort")
        # When the request ranks results ("the largest PDF"), the search step
        # publishes the selected match under a name later steps can reference.
        produces = "largest_match" if sort in {"largest", "smallest", "newest", "oldest"} else "search_results"
        actions.append(
            TaskAction(
                action_id="a1",
                capability="filesystem.search",
                parameters={
                    "pattern": pattern,
                    "path": source_dir,
                    "max_results": 200,
                    "select": sort,
                },
                description=f"Find {pattern} files"
                + (f" in {source_dir}" if source_dir else "")
                + ".",
                produces=produces,
                expected_output="matching file paths",
            )
        )
        if move or copy:
            # The destination folder is parsed as a distinct entity; fall back to
            # a "to/into <folder>" clause when the entity was not set.
            destination = entities.get("dest_folder")
            if not destination:
                dest_match = re.search(r"\b(?:to|into)\s+(?:my\s+|the\s+)?(downloads|documents|desktop|pictures|music|videos)\b", text, re.IGNORECASE)
                if dest_match:
                    destination = _KNOWN_FOLDERS.get(dest_match.group(1).casefold(), dest_match.group(1))
            capability = "filesystem.copy" if copy else "filesystem.move"
            actions.append(
                TaskAction(
                    action_id="a2",
                    capability=capability,
                    parameters={
                        "source": "$largest_match",
                        "destination": destination,
                        "overwrite": False,
                    },
                    description=f"{'Copy' if copy else 'Move'} the found file"
                    + (f" to {destination}" if destination else "")
                    + ".",
                    depends_on=["a1"],
                    expected_output="file relocated",
                    risk_level="medium_risk",
                    requires_confirmation=True,
                )
            )
        return actions

    def _folder_path(self, entities: dict[str, Any]) -> Optional[str]:
        name = entities.get("folder_name")
        folder = entities.get("folder")
        if not name and not folder:
            return None
        if folder and name:
            return f"{folder}/{name}"
        return folder or name

    def _search_query(self, text: str, entities: dict[str, Any]) -> str:
        query = entities.get("topic")
        if query:
            return query
        # A hybrid request such as "search the web for X and write it into
        # Notepad" is a search *for X*; the trailing write clause is not part
        # of the query. Cut the text at the clause boundary first.
        body = re.split(
            r"\b(?:and\s+)?(?:write|save|put|store|type|paste|copy|add|open|launch|start|"
            r"summarize|summarise|condense|digest|export|create|make|insert|place|display|show)\b",
            text, maxsplit=1, flags=re.IGNORECASE,
        )[0]
        # Fall back to stripping site/action/boilerplate words from the request.
        stop = {
            "search", "find", "look", "show", "me", "get", "for", "on", "in", "the",
            "youtube", "google", "web", "internet", "videos", "video", "please", "and",
            "open", "launch", "start", "browse", "some", "a", "an", "about",
            "latest", "newest", "recent", "popular", "trending", "top", "largest",
            "biggest", "smallest", "oldest", "it", "into", "onto", "them", "that", "this",
        }
        # A named application that is merely the search host ("open Chrome and
        # search ...") must not leak into the query text.
        application = str(entities.get("application") or "").casefold()
        stop.update(token for token in re.findall(r"[a-z0-9]+", application))
        if not body.strip():
            # The clause verb led the sentence ("open chrome and search for
            # weather"): there is no prefix clause, so strip the scaffolding
            # from the whole text rather than echoing the entire prompt back.
            body = text
        tokens = [t for t in re.findall(r"[A-Za-z0-9]+", body.casefold()) if t not in stop]
        return " ".join(tokens).strip() or text.strip()

    # -- classification ----------------------------------------------------------

    def _classify(
        self,
        text: str,
        entities: dict[str, Any],
        actions: list[TaskAction],
    ) -> tuple[str, str, bool]:
        if actions:
            capabilities = {action.capability for action in actions}
            if capabilities <= {"filesystem.search"} and not entities.get("topic"):
                # A pure search still needs a target to be useful; treat a
                # request with no actionable target as a lookup.
                return "computer_action", "search_files", True
            if any(cap.startswith("content.") for cap in capabilities):
                goal = "create_content" if "applications.write_text" in capabilities or "filesystem.write" in capabilities else "generate_content"
                return "content_creation", goal, True
            if "applications.launch_named" in capabilities and len(actions) == 1:
                return "computer_action", "open_application", True
            if "web.search" in capabilities:
                return "search", "search_web", True
            if "filesystem.create_folder" in capabilities:
                return "computer_action", "create_folder", True
            if capabilities & {"filesystem.move", "filesystem.copy"}:
                return "computer_action", "organize_files", True
            return "computer_action", "perform_action", True

        lowered = text.casefold()
        if any(re.search(rf"\b{word}\b", lowered) for word in _QUESTION_WORDS):
            return "informational", "answer_question", False
        if not text.strip():
            return "unknown", "unknown", False
        return "conversation", "respond", False

    def _extract_constraints(self, text: str, entities: dict[str, Any]) -> list[str]:
        constraints: list[str] = []
        lowered = text.casefold()
        if "without deleting" in lowered or "don't delete" in lowered or "do not delete" in lowered:
            constraints.append("do not delete files")
        if "overwrite" in lowered:
            constraints.append("overwrite allowed")
        for key in ("tone", "length", "style", "sort", "quantity"):
            value = entities.get(key)
            if value:
                constraints.append(f"{key}={value}")
        return constraints

    def _needs_clarification(
        self,
        text: str,
        entities: dict[str, Any],
        actions: list[TaskAction],
    ) -> bool:
        lowered = text.casefold()
        if self._instructional_question(text):
            return False
        # An action verb that targets a pronoun ("open it", "write it there")
        # with no resolvable target is ambiguous.
        if re.search(r"\b(?:open|launch|start|write|put|type|save|move|copy)\s+(?:it|that|this|them|those|there)\b", lowered):
            # "move it to Documents" is unambiguous only because a prior search
            # step locates the object. A pronoun whose referent is produced by an
            # earlier planned step (a file search, or the results a web search
            # returns) is resolvable; a pronoun with no such step ("open it and
            # write something") has no referent and is genuinely ambiguous.
            resolving = {"filesystem.search", "filesystem.search_content", "web.search", "web.research"}
            has_resolving_search = any(a.capability in resolving for a in actions)
            if (
                not has_resolving_search
                and not entities.get("application")
                and not entities.get("filename")
            ):
                return True
        # A content request is only unactionable when it names neither a content
        # type nor a topic nor a destination ("write something"). "write a
        # poem" is fully satisfiable: the model generates the poem and Atlas
        # returns it, so it must NOT be blocked waiting for a destination.
        if actions and all(a.capability == "content.generate" for a in actions):
            if not (
                entities.get("content_type")
                or entities.get("topic")
                or entities.get("application")
                or entities.get("filename")
            ):
                return True
        return False

    def _clarification_question(
        self,
        entities: dict[str, Any],
        actions: list[TaskAction],
    ) -> str:
        if actions and all(a.capability == "content.generate" for a in actions):
            return "Where should I put the content, or what should it be about?"
        return "Which application or file location should I use?"

    def _confidence(
        self,
        text: str,
        entities: dict[str, Any],
        actions: list[TaskAction],
        task_type: str,
    ) -> float:
        if not text.strip():
            return 0.0
        if actions:
            base = 0.7
            if entities.get("application") or entities.get("filename"):
                base = 0.85
            if entities.get("topic"):
                base = max(base, 0.85)
            if len(actions) > 1:
                base = max(base, 0.8)
            return base
        if task_type == "informational":
            return 0.7
        return 0.5


def _matches_any(text: str, terms: Iterable[str]) -> bool:
    """Return True when any term appears as a whole word or phrase in ``text``."""

    for term in terms:
        if " " in term:
            if term in text:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", text):
            return True
    return False

def task_to_intent_shim(task: Task) -> dict[str, Any]:
    """Adapt a task into the flat mapping the legacy planner expects.

    Lets the new Task pipeline reuse deterministic legacy plans (retrieval,
    comparison, summarization) without keeping two interpreters.
    """

    entities = task.entities
    return {
        "intent": {
            "content_creation": "write_content",
            "computer_action": "computer_action",
            "search": "search",
            "informational": "knowledge_query",
            "conversation": "knowledge_query",
        }.get(task.task_type, "unknown"),
        "action": None,
        "target": entities.get("site"),
        "content_type": entities.get("content_type"),
        "topic": entities.get("topic"),
        "query": entities.get("topic"),
        "destination": entities.get("application") or entities.get("folder"),
        "tone": entities.get("tone"),
        "length": entities.get("length"),
        "sort": entities.get("sort"),
        "confidence": task.confidence,
        "source": task.source,
    }
