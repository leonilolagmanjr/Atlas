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
        "applications.write_text",
        "applications.launch_named",
        "applications.launch",
        # Filesystem (high-level, permission-gated)
        "filesystem.list",
        "filesystem.read",
        "filesystem.metadata",
        "filesystem.search",
        "filesystem.write",
        "filesystem.create_folder",
        "filesystem.move",
        "filesystem.copy",
        # Internet
        "web.search",
        "web.fetch",
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
        for name in PLANNABLE_CAPABILITIES:
            capabilities.setdefault(name, _fallback_capability(name))
        self._capabilities = capabilities
    def get(self, name: str) -> Capability | None:
        return self._capabilities.get(str(name).strip())

    def exists(self, name: str) -> bool:
        return str(name).strip() in self._capabilities
    def plannable(self) -> list[Capability]:
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
        description="Write supplied text into a resolved application.",
        category="computer.applications",
        parameters={
            "application": {"type": "string", "description": "target application name"},
            "text": {"type": "string", "description": "text to enter"},
        },
        required_parameters=("application", "text"),
        risk_level="medium_risk",
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
        description="Read a UTF-8 text file under the allowed root.",
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
