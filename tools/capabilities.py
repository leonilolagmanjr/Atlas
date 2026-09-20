"""Render registered tools as a machine-readable capability catalog.

The planner and interpreter receive this catalog so they can select from real
capabilities instead of inventing tools.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Iterable
from tools.base import ToolMetadata
from tools.registry import ToolRegistry
#: Capabilities the interpreter/planner may reference when reasoning over a
#: natural-language request. Capabilities absent from this set are still
#: registered and routable (e.g. admin/inspection tools) but are invisible to
#: the LLM, so free text can never select them.
#:
#: The set is intentionally high-level. Mutating PowerShell, process mutation,
#: and other low-level backends stay out of the free-text-driven surface, though
#: ``powershell.execute`` and ``processes.*`` remain valid *legacy* plan
#: capabilities constructed by deterministic rules (see ``brain._validate_plan``).
PLANNABLE_CAPABILITIES: frozenset[str] = frozenset(
    {
        # Content and application control
        "content.generate",
        "content.format",
        "applications.write_text",
        "applications.launch_named",
        "applications.launch",
        # Computer perception (read-only observation of the desktop)
        "computer.windows",
        "computer.observe",
        "computer.vision_observe",
        "computer.find",
        # Visual interaction (permission-gated, validated against observations)
        "computer.click",
        "computer.double_click",
        "computer.right_click",
        "computer.move",
        "computer.drag",
        "computer.type",
        "computer.keypress",
        "computer.scroll",
        "computer.focus",
        # Filesystem (high-level, permission-gated)
        "filesystem.list",
        "filesystem.read",
        "filesystem.metadata",
        "filesystem.search",
        "filesystem.search_content",
        "filesystem.write",
        "filesystem.create_folder",
        "filesystem.move",
        "filesystem.copy",
        # Internet
        "web.search",
        "web.fetch",
        "web.research",
        # Local system inspection (read-only observation for reasoning)
        "system.info",
    }
)

@dataclass(frozen=True)
class Capability:
    """A validated, describable capability derived from a registered tool."""

    name: str
    description: str
    category: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required_parameters: tuple[str, ...] = ()
    risk_level: str = "read_only"
    requires_confirmation: bool = False
    verifiable: bool = False
    produces: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": dict(self.parameters),
            "required_parameters": list(self.required_parameters),
            "risk_level": self.risk_level,
            "requires_confirmation": self.requires_confirmation,
            "verifiable": self.verifiable,
            "produces": list(self.produces),
        }


def capability_from_metadata(metadata: ToolMetadata) -> Capability:
    """Build a capability descriptor from registered tool metadata."""

    requires_confirmation = metadata.permission_level.value != "read_only"
    overlay = _FALLBACK_CAPABILITIES.get(metadata.name)
    return Capability(
        name=metadata.name,
        description=metadata.description or (overlay.description if overlay else ""),
        category=metadata.category,
        parameters=_render_parameters(metadata.input_schema)
        or (overlay.parameters if overlay else {}),
        required_parameters=(
            overlay.required_parameters
            if overlay
            else tuple(metadata.required_parameters)
        ),
        risk_level=metadata.risk_level.value,
        requires_confirmation=requires_confirmation,
        verifiable=overlay.verifiable if overlay else metadata.verifiable,
        produces=overlay.produces if overlay else tuple(metadata.produces),
    )

class CapabilityRegistry:
    """Look up and describe the capabilities Atlas may plan against.

    Wraps an optional :class:`ToolRegistry`. When no registry is supplied the
    descriptors are synthesized from the built-in plannable set so the
    interpreter/planner/validator always have a coherent catalog even in tests.
    """

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry
        self._capabilities: dict[str, Capability] = {}
        self._reload()

    def _reload(self) -> None:
        capabilities: dict[str, Capability] = {}
        if self._registry is not None:
            for metadata in self._registry.list_metadata():
                if not isinstance(metadata, ToolMetadata):
                    continue
                capabilities[metadata.name] = capability_from_metadata(metadata)
        else:
            for name in PLANNABLE_CAPABILITIES:
                capabilities[name] = _fallback_capability(name)
        self._capabilities = capabilities
    def get(self, name: str) -> Capability | None:
        self._reload()
        return self._capabilities.get(str(name).strip())

    def exists(self, name: str) -> bool:
        self._reload()
        return str(name).strip() in self._capabilities
    def plannable(self) -> list[Capability]:
        self._reload()
        return [
            capability
            for name, capability in sorted(self._capabilities.items())
            if name in PLANNABLE_CAPABILITIES
        ]

    def render_catalog(self) -> str:
        catalog = [capability.to_dict() for capability in self.plannable()]
        if not catalog:
            return "(no capabilities registered)"
        return json.dumps(catalog, indent=2, ensure_ascii=False)


def capabilities_for_planning(registry: ToolRegistry) -> list[ToolMetadata]:
    """Return the subset of tools the planner may select."""

    return [
        metadata
        for metadata in registry.list_metadata()
        if isinstance(metadata, ToolMetadata) and metadata.name in PLANNABLE_CAPABILITIES
    ]


def render_capability_catalog(
    registry: ToolRegistry | None,
    *,
    fallback: list[ToolMetadata] | None = None,
) -> str:
    """Render available capabilities as compact JSON for an LLM prompt."""

    if registry is not None:
        return CapabilityRegistry(registry).render_catalog()
    if fallback:
        capabilities = {
            item.name: capability_from_metadata(item)
            for item in fallback
            if item.name in PLANNABLE_CAPABILITIES
        }
        for name in PLANNABLE_CAPABILITIES:
            capabilities.setdefault(name, _fallback_capability(name))
        catalog = [
            capability.to_dict() for _, capability in sorted(capabilities.items())
        ]
        return json.dumps(catalog, indent=2, ensure_ascii=False)
    return CapabilityRegistry(None).render_catalog()


def _render_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for name, spec in (schema or {}).items():
        if isinstance(spec, dict):
            rendered[name] = {
                "type": spec.get("type", "string"),
                "description": spec.get("description", ""),
            }
        else:
            rendered[name] = {"type": str(spec)}
    return rendered
#: Minimal descriptors for capabilities that may not be present in a given
#: registry (e.g. a unit-test registry). They keep the catalog coherent without
#: duplicating real tool metadata when tools *are* registered.
_FALLBACK_CAPABILITIES: dict[str, Capability] = {
    "content.generate": Capability(
        name="content.generate",
        description="Generate creative or informational text from a request.",
        category="content.generation",
        parameters={
            "content_type": {"type": "string", "description": "poem, story, essay, ..."},
            "topic": {"type": "string", "description": "subject the content is about"},
            "tone": {"type": "string", "description": "e.g. funny, serious"},
            "style": {"type": "string", "description": "e.g. futuristic, formal"},
            "length": {"type": "string", "description": "short or long"},
            "instructions": {"type": "string", "description": "extra free-form guidance"},
        },
        required_parameters=("content_type",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
        produces=("generated_text",),
    ),
    "content.format": Capability(
        name="content.format",
        description="Normalize, structure, and render raw content for a specific destination.",
        category="content.formatting",
        parameters={
            "content": {"type": "string", "description": "Raw content to format"},
            "destination": {"type": "string", "description": "Target application (notepad, word, markdown, code_editor, terminal)"},
            "title": {"type": "string", "description": "Optional title for the content"},
            "source": {"type": "string", "description": "Source identifier (URL, filename, etc.)"},
            "url": {"type": "string", "description": "Source URL if from web"},
            "repair": {"type": "boolean", "description": "Enable automatic formatting repair"},
        },
        required_parameters=("content",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
        produces=("formatted_text",),
    ),
    "applications.launch_named": Capability(
        name="applications.launch_named",
        description="Resolve and launch a trusted installed application by name.",
        category="computer.applications",
        parameters={"application": {"type": "string", "description": "application name"}},
        required_parameters=("application",),
        risk_level="low_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "applications.launch": Capability(
        name="applications.launch",
        description="Launch one explicitly selected Windows executable.",
        category="computer.applications",
        parameters={"executable": {"type": "string"}, "arguments": {"type": "array"}},
        required_parameters=("executable",),
        risk_level="low_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "applications.write_text": Capability(
        name="applications.write_text",
        description="Write supplied text into a resolved application and read it back.",
        category="computer.applications",
        parameters={
            "application": {"type": "string", "description": "target application name"},
            "text": {"type": "string", "description": "text to enter"},
            "delivery": {
                "type": "string",
                "description": "auto (default), direct (window message) or paste",
            },
            "wait_seconds": {
                "type": "integer",
                "description": "bounded seconds to wait for the application window (1-15)",
            },
        },
        required_parameters=("application", "text"),
        risk_level="medium_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "computer.windows": Capability(
        name="computer.windows",
        description=(
            "List open desktop windows (title, class, process, focus state) "
            "without changing anything."
        ),
        category="computer.perception",
        parameters={
            "application": {"type": "string", "description": "optional name filter"},
            "include_hidden": {"type": "boolean"},
            "max_results": {"type": "integer"},
        },
        required_parameters=(),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "computer.observe": Capability(
        name="computer.observe",
        description=(
            "Observe the current or named application's state: window, controls, "
            "focused control, and readable text; optionally a full visual-state "
            "observation (UI Automation + local OCR + image processing + VLM)."
        ),
        category="computer.perception",
        parameters={
            "application": {"type": "string", "description": "optional application name"},
            "wait_seconds": {"type": "integer"},
            "text_limit": {"type": "integer"},
            "full_screen": {"type": "boolean"},
            "region": {"type": "object"},
            "include_ocr": {"type": "boolean"},
            "include_uia": {"type": "boolean"},
            "allow_vlm": {"type": "boolean"},
            "vlm_question": {"type": "string"},
        },
        required_parameters=(),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "computer.vision_observe": Capability(
        name="computer.vision_observe",
        description=(
            "Capture and analyze the current screen into a structured visual state "
            "(windows, UI elements, text) via UI Automation, local OCR, image "
            "processing, and optionally a local vision model."
        ),
        category="computer.perception",
        parameters={
            "application": {"type": "string"},
            "full_screen": {"type": "boolean"},
            "region": {"type": "object"},
            "include_uia": {"type": "boolean"},
            "include_ocr": {"type": "boolean"},
            "allow_vlm": {"type": "boolean"},
            "vlm_question": {"type": "string"},
        },
        required_parameters=(),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "computer.find": Capability(
        name="computer.find",
        description=(
            "Find a visible UI element by label, semantic description, or type, "
            "returning structured candidates with bounds and confidence."
        ),
        category="computer.perception",
        parameters={
            "query": {"type": "string", "description": "label or description to find"},
            "element_type": {"type": "string"},
            "application": {"type": "string"},
            "region": {"type": "object"},
            "max_results": {"type": "integer"},
            "allow_vlm": {"type": "boolean"},
        },
        required_parameters=("query",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "computer.click": Capability(
        name="computer.click",
        description="Click a UI element located in the current observation (validated coordinates).",
        category="computer.interaction",
        parameters={
            "element_id": {"type": "string"},
            "description": {"type": "string"},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "observation_id": {"type": "string"},
            "application": {"type": "string"},
            "button": {"type": "string"},
        },
        required_parameters=(),
        risk_level="medium_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "computer.double_click": Capability(
        name="computer.double_click",
        description="Double-click a UI element located in the current observation.",
        category="computer.interaction",
        parameters={"element_id": {"type": "string"}, "description": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}},
        required_parameters=(),
        risk_level="medium_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "computer.right_click": Capability(
        name="computer.right_click",
        description="Right-click a UI element located in the current observation.",
        category="computer.interaction",
        parameters={"element_id": {"type": "string"}, "description": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}},
        required_parameters=(),
        risk_level="medium_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "computer.move": Capability(
        name="computer.move",
        description="Move the mouse pointer to a UI element located in the current observation.",
        category="computer.interaction",
        parameters={"element_id": {"type": "string"}, "description": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}},
        required_parameters=(),
        risk_level="low_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "computer.drag": Capability(
        name="computer.drag",
        description="Drag from a source element to a destination element in the current observation.",
        category="computer.interaction",
        parameters={
            "from_element_id": {"type": "string"},
            "from_description": {"type": "string"},
            "to_element_id": {"type": "string"},
            "to_description": {"type": "string"},
            "application": {"type": "string"},
        },
        required_parameters=(),
        risk_level="medium_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "computer.type": Capability(
        name="computer.type",
        description="Type text into a UI element located in the current observation.",
        category="computer.interaction",
        parameters={
            "text": {"type": "string"},
            "element_id": {"type": "string"},
            "description": {"type": "string"},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "application": {"type": "string"},
        },
        required_parameters=("text",),
        risk_level="high_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "computer.keypress": Capability(
        name="computer.keypress",
        description="Send a named key press (enter, tab, escape, up, down, ...) to the active window.",
        category="computer.interaction",
        parameters={"key": {"type": "string"}, "application": {"type": "string"}},
        required_parameters=("key",),
        risk_level="medium_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "computer.scroll": Capability(
        name="computer.scroll",
        description="Scroll the active view up or down, optionally after moving to an element.",
        category="computer.interaction",
        parameters={
            "direction": {"type": "string"},
            "amount": {"type": "integer"},
            "element_id": {"type": "string"},
            "description": {"type": "string"},
            "application": {"type": "string"},
        },
        required_parameters=(),
        risk_level="low_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "computer.focus": Capability(
        name="computer.focus",
        description="Focus (bring to the foreground) an observed application window.",
        category="computer.interaction",
        parameters={"application": {"type": "string"}},
        required_parameters=("application",),
        risk_level="low_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "web.search": Capability(
        name="web.search",
        description="Search the public web and return attributed result links.",
        category="internet.search",
        parameters={
            "query": {"type": "string", "description": "what to search for"},
            "max_results": {"type": "integer"},
            "site": {"type": "string", "description": "optional site, e.g. youtube"},
            "sort": {"type": "string", "description": "latest or popular"},
        },
        required_parameters=("query",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "web.fetch": Capability(
        name="web.fetch",
        description="Fetch a public web page as bounded, untrusted text.",
        category="internet.retrieval",
        parameters={"url": {"type": "string"}, "max_bytes": {"type": "integer"}},
        required_parameters=("url",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "web.research": Capability(
        name="web.research",
        description=(
            "Search the web, read several top pages, and return the most relevant "
            "content for the request (not just result links)."
        ),
        category="internet.research",
        parameters={
            "query": {"type": "string", "description": "what the content should be about"},
            "max_results": {"type": "integer"},
            "max_pages": {"type": "integer"},
            "site": {"type": "string"},
        },
        required_parameters=("query",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
        produces=("web_content",),
    ),
    "filesystem.search": Capability(
        name="filesystem.search",
        description="Search filenames below an allowed root without modifying files.",
        category="computer.filesystem",
        parameters={
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        required_parameters=("pattern",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "filesystem.list": Capability(
        name="filesystem.list",
        description="List entries in an allowed directory.",
        category="computer.filesystem",
        parameters={"path": {"type": "string"}},
        required_parameters=("path",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "filesystem.metadata": Capability(
        name="filesystem.metadata",
        description="Inspect metadata (size, kind, modified) for a path.",
        category="computer.filesystem",
        parameters={"path": {"type": "string"}},
        required_parameters=("path",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "filesystem.read": Capability(
        name="filesystem.read",
        description="Read bounded UTF-8 text or PDF page text under the allowed root.",
        category="computer.filesystem",
        parameters={"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
        required_parameters=("path",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "filesystem.write": Capability(
        name="filesystem.write",
        description="Write UTF-8 text to a file under the allowed root.",
        category="computer.filesystem",
        parameters={
            "path": {"type": "string"},
            "text": {"type": "string"},
            "overwrite": {"type": "boolean"},
        },
        required_parameters=("path", "text"),
        risk_level="medium_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "filesystem.create_folder": Capability(
        name="filesystem.create_folder",
        description="Create a directory under the allowed root.",
        category="computer.filesystem",
        parameters={"path": {"type": "string"}},
        required_parameters=("path",),
        risk_level="medium_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "filesystem.move": Capability(
        name="filesystem.move",
        description="Move or rename a file/directory under the allowed root.",
        category="computer.filesystem",
        parameters={
            "source": {"type": "string"},
            "destination": {"type": "string"},
            "overwrite": {"type": "boolean"},
        },
        required_parameters=("source", "destination"),
        risk_level="medium_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "filesystem.copy": Capability(
        name="filesystem.copy",
        description="Copy a file/directory under the allowed root.",
        category="computer.filesystem",
        parameters={
            "source": {"type": "string"},
            "destination": {"type": "string"},
            "overwrite": {"type": "boolean"},
        },
        required_parameters=("source", "destination"),
        risk_level="medium_risk",
        requires_confirmation=True,
        verifiable=True,
    ),
    "filesystem.search_content": Capability(
        name="filesystem.search_content",
        description=(
            "Search the text contents of files below an allowed folder, returning "
            "matching file paths with a short excerpt."
        ),
        category="computer.filesystem",
        parameters={
            "query": {"type": "string", "description": "text to look for"},
            "path": {"type": "string", "description": "optional subfolder to search"},
            "pattern": {"type": "string", "description": "optional filename glob"},
            "max_results": {"type": "integer"},
            "max_files": {"type": "integer"},
            "max_bytes": {"type": "integer"},
        },
        required_parameters=("query",),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
    "system.info": Capability(
        name="system.info",
        description="Inspect read-only operating-system, CPU, memory, and disk state.",
        category="computer.system",
        parameters={"path": {"type": "string"}},
        required_parameters=(),
        risk_level="read_only",
        requires_confirmation=False,
        verifiable=True,
    ),
}


def _fallback_capability(name: str) -> Capability:
    if name in _FALLBACK_CAPABILITIES:
        return _FALLBACK_CAPABILITIES[name]
    # Generic descriptor so an unknown-to-the-fallback-map name still validates
    # structurally; the router remains the authority on whether it can run.
    return Capability(
        name=name,
        description=f"Capability {name}.",
        category=name.split(".", 1)[0],
        parameters={},
        required_parameters=(),
        risk_level="read_only",
        requires_confirmation=False,
    )


def iter_capability_names() -> Iterable[str]:
    """Yield the names of the built-in plannable capabilities."""

    return iter(sorted(PLANNABLE_CAPABILITIES))
