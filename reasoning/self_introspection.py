"""Self-introspection: Atlas's knowledge about itself.

Self-knowledge is *derived from the live capability registry and runtime
configuration*, never from a hand-written paragraph. If a capability is added,
removed, or re-scoped, Atlas's answers about itself change automatically.

This module is deterministic: answering "what can you do?" costs no model call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from config import (
    COLLECTION_NAME,
    DATABASE_FOLDER,
    EMBEDDING_MODEL_NAME,
    EXECUTION_MODE,
    KNOWLEDGE_FOLDER,
    OLLAMA_MODEL,
)
from tools.base import ToolMetadata
from tools.capabilities import Capability, CapabilityRegistry

#: Answer domains Atlas can describe about itself. The vocabulary maps a domain
#: onto the words a user may use to ask about it.
_KIND_TERMS: dict[str, tuple[str, ...]] = {
    "capabilities": (
        "capabilities", "capability", "features", "feature", "abilities", "ability",
        "what can you do", "what do you do", "skills", "functions", "supported",
    ),
    "tools": ("tools", "tool", "commands", "integrations"),
    "applications": (
        "applications", "application", "apps", "app", "programs", "program",
        "launch", "open", "automate",
    ),
    "web": (
        "internet", "web", "online", "browse", "google", "websites", "website",
        "url", "webpage", "search the web",
    ),
    "files": (
        "files", "file", "folders", "folder", "documents", "document", "pdf", "pdfs",
        "filesystem", "downloads",
    ),
    "model": ("model", "llm", "language model", "engine", "ollama"),
    "knowledge": (
        "knowledge", "knowledge base", "index", "indexed", "vector", "database",
        "embeddings",
    ),
    "memory": ("memory", "conversation history", "session", "remember"),
    "system": (
        "cpu", "ram", "processes", "services", "disk space", "hardware",
        "os version", "windows version",
    ),
}

#: Self-reference words. A question is about Atlas when the subject is Atlas.
_SELF_SUBJECTS = ("you", "your", "yourself", "atlas", "u")
_ABILITY_VERBS = (
    "can", "could", "able to", "do you", "does atlas", "would you", "supported",
    "available to", "have you",
)


@dataclass(frozen=True)
class SelfKnowledge:
    """Structured snapshot of what Atlas can currently do."""

    capabilities: tuple[Capability, ...]
    tool_count: int
    model_name: str
    knowledge_folder: str
    database_folder: str
    collection_name: str
    embedding_model: str
    execution_mode: str
    categories: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilities": [capability.to_dict() for capability in self.capabilities],
            "tool_count": self.tool_count,
            "model_name": self.model_name,
            "knowledge_folder": self.knowledge_folder,
            "database_folder": self.database_folder,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "execution_mode": self.execution_mode,
            "categories": list(self.categories),
        }


class SelfIntrospection:
    """Answer questions about Atlas from Atlas's own registry and state."""

    def __init__(
        self,
        capabilities: CapabilityRegistry,
        *,
        registered: Sequence[ToolMetadata] = (),
        model_name: str = OLLAMA_MODEL,
        knowledge_folder: Path | str = KNOWLEDGE_FOLDER,
        database_folder: Path | str = DATABASE_FOLDER,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        execution_mode: str = EXECUTION_MODE,
    ) -> None:
        self._capabilities = capabilities
        self._registered = tuple(registered)
        self._model_name = model_name
        self._knowledge_folder = str(knowledge_folder)
        self._database_folder = str(database_folder)
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._execution_mode = execution_mode

    # -- registry-derived self knowledge ----------------------------------------

    @property
    def capabilities(self) -> tuple[Capability, ...]:
        """Every capability the runtime exposes, derived from the live registry."""

        return tuple(sorted(self._capabilities.plannable(), key=lambda capability: capability.name))

    def names_in_category(self, prefix: str) -> tuple[Capability, ...]:
        return tuple(
            capability
            for capability in self.capabilities
            if capability.category.startswith(prefix) or capability.name.startswith(prefix)
        )

    # -- routing support ---------------------------------------------------------

    @staticmethod
    def kinds() -> tuple[str, ...]:
        """Return the self-answer domains Atlas understands."""

        return tuple(_KIND_TERMS)

    def is_self_query(self, question: str) -> bool:
        """Return True when the question's subject is Atlas itself."""

        lowered = question.casefold()
        tokens = _tokens(lowered)
        if not tokens:
            return False
        if not any(token in tokens for token in _SELF_SUBJECTS):
            return False
        if re.search(r"\b(?:my|this|that|the)\s+(?:file|pdf|report|resume|document)\b", lowered):
            return False
        if re.match(r"(?:please\s+)?(?:open|write|create|search|find|read|summarize)\b", lowered):
            return False
        # A self-subject plus either an ability phrasing or an Atlas-domain term
        # makes the question about Atlas rather than about its subject matter.
        if any(phrase in lowered for phrase in _ABILITY_VERBS):
            return True
        return self.requested_kind(question) is not None

    def requested_kind(self, question: str) -> str | None:
        """Return which aspect of itself the user asked about, if any.

        Selection is by strongest term overlap against the kind vocabulary, so a
        question mentioning several domains resolves deterministically.
        """

        lowered = question.casefold()
        tokens = set(_tokens(lowered))
        best: tuple[int, str] | None = None
        for kind, terms in _KIND_TERMS.items():
            score = 0
            for term in terms:
                if " " in term:
                    if term in lowered:
                        score += 2
                elif term in tokens:
                    score += 1
            if score and (best is None or score > best[0]):
                best = (score, kind)
        return best[1] if best else None

    # -- answers -----------------------------------------------------------------

    def describe(self) -> str:
        """Return a capability summary generated from the live registry."""

        capabilities = self.capabilities
        if not capabilities:
            return "I have no capabilities registered in this runtime."

        grouped: dict[str, list[Capability]] = {}
        for capability in capabilities:
            grouped.setdefault(capability.category, []).append(capability)

        lines = [
            f"I am Atlas, a local-first assistant running the {self._model_name} model.",
            f"I currently expose {len(capabilities)} capabilities:",
        ]
        for category in sorted(grouped):
            lines.append(f"- {category}:")
            for capability in grouped[category]:
                confirmation = " (needs your confirmation)" if capability.requires_confirmation else ""
                lines.append(f"  - {capability.name}: {capability.description}{confirmation}")
        lines.append(
            "Ask me about a specific area (tools, applications, internet, files, "
            "knowledge, memory, or the model) and I will describe exactly what is "
            "registered right now."
        )
        return "\n".join(lines)

    def answer(self, kind: str | None) -> str:
        """Answer a self-query for ``kind`` (defaults to a full overview)."""

        handlers = {
            "capabilities": self.describe,
            "tools": self._describe_tools,
            "applications": self._describe_applications,
            "web": self._describe_web,
            "files": self._describe_files,
            "model": self._describe_model,
            "knowledge": self._describe_knowledge,
            "memory": self._describe_memory,
            "system": self._describe_system,
        }
        handler = handlers.get(kind or "")
        return handler() if handler is not None else self.describe()

    def ability_answer(self, question: str) -> str:
        """Answer a yes/no ability question from the registry (no model call)."""

        return self.answer(self.requested_kind(question))

    def has(self, name: str) -> bool:
        return self._capabilities.exists(name)

    def self_knowledge(self) -> SelfKnowledge:
        capabilities = self.capabilities
        return SelfKnowledge(
            capabilities=capabilities,
            tool_count=len(capabilities),
            model_name=self._model_name,
            knowledge_folder=self._knowledge_folder,
            database_folder=self._database_folder,
            collection_name=self._collection_name,
            embedding_model=self._embedding_model,
            execution_mode=self._execution_mode,
            categories=tuple(sorted({capability.category for capability in capabilities})),
        )

    # -- per-domain descriptions -------------------------------------------------

    def _describe_tools(self) -> str:
        capabilities = self.capabilities
        lines = [f"I have {len(capabilities)} registered capabilities:"]
        for capability in capabilities:
            lines.append(
                f"- {capability.name} [{capability.category}] risk={capability.risk_level}"
                + (" confirmation required" if capability.requires_confirmation else "")
                + f": {capability.description}"
            )
        return "\n".join(lines)

    def _describe_applications(self) -> str:
        launch = self.names_in_category("applications")
        if not launch:
            return "I do not have application-control capabilities registered right now."
        lines = ["I can control applications through these registered capabilities:"]
        for capability in launch:
            lines.append(f"- {capability.name}: {capability.description}")
        if any(metadata.name == "applications.list_installed" for metadata in self._registered):
            lines.append(
                "I can also enumerate the applications installed on this machine "
                "(applications.list_installed) and resolve one by name."
            )
        lines.append("Launching an application and writing text into it always asks for your confirmation.")
        return "\n".join(lines)

    def _describe_web(self) -> str:
        web = self.names_in_category("web")
        if not web:
            return "I do not have internet capabilities registered right now."
        return (
            "Yes. Internet access is read-only and registered as:\n"
            + "\n".join(f"- {capability.name}: {capability.description}" for capability in web)
            + "\nPage content is treated as untrusted evidence: I never follow "
            "instructions found inside a webpage."
        )

    def _describe_files(self) -> str:
        files = self.names_in_category("filesystem")
        if not files:
            return "I do not have filesystem capabilities registered right now."
        lines = ["Yes. I can work with your local files (always inside the allowed root):"]
        for capability in files:
            lines.append(f"- {capability.name}: {capability.description}")
        if any(capability.name == "filesystem.read" for capability in files):
            lines.append(
                f"I also index documents from {self._knowledge_folder} so I can answer "
                "questions about their contents."
            )
        return "\n".join(lines)

    def _describe_model(self) -> str:
        return (
            f"I am currently using the local model '{self._model_name}' for reasoning, "
            "generation, and answer synthesis. The model is interchangeable: Atlas talks "
            "to it through a provider abstraction, so another local or remote model can be "
            "substituted without changing the reasoning, planning, or execution layers."
        )

    def _describe_knowledge(self) -> str:
        return (
            "My local knowledge lives in a vector database:\n"
            f"- source documents: {self._knowledge_folder}\n"
            f"- vector store: {self._database_folder} (collection '{self._collection_name}')\n"
            f"- embedding model: {self._embedding_model}\n"
            "Conversation memory is kept separate from this knowledge base: memory holds "
            "what we discussed, the knowledge base holds your documents."
        )

    def _describe_memory(self) -> str:
        return (
            "I keep conversation memory per session on disk, under the memory folder. "
            "Memory stores what you and I said; it stays separate from the document "
            "knowledge base and from the web."
        )

    def _describe_system(self) -> str:
        dynamic = [
            capability
            for capability in self.capabilities
            if capability.category.startswith(("computer.system", "computer.process"))
            or capability.name.startswith(("system.", "processes."))
        ]
        lines = ["I can inspect (never modify) your local system state:"]
        for capability in dynamic:
            lines.append(f"- {capability.name}: {capability.description}")
        if not dynamic:
            lines.append("- (no system inspection capability is registered in this runtime)")
        lines.append(f"Execution mode for consequential actions: {self._execution_mode}.")
        return "\n".join(lines)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text)

