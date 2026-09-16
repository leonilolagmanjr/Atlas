"""Render registered tools as a machine-readable capability catalog.

The planner and interpreter receive this catalog so they can select from real
capabilities instead of inventing tools.
"""

from __future__ import annotations

import json
from typing import Any

from tools.base import ToolMetadata
from tools.registry import ToolRegistry

#: Capabilities the planner is allowed to reference in a plan. Anything not in
#: this set is invisible to the LLM, which prevents it selecting admin/inspection
#: tools that are not meant to be driven by free text.
PLANNABLE_CAPABILITIES: frozenset[str] = frozenset(
    {
        # Free-text-driven capabilities only. Inspection tools (filesystem,
        # processes, system, powershell) are deliberately excluded so the LLM
        # never selects them from a natural-language request.
        "content.generate",
        "applications.write_text",
        "applications.launch_named",
        "applications.launch",
        "web.search",
        "web.fetch",
    }
)


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

    metadata_list: list[ToolMetadata] = []
    if registry is not None:
        metadata_list = capabilities_for_planning(registry)
    elif fallback:
        metadata_list = [
            item for item in fallback if item.name in PLANNABLE_CAPABILITIES
        ]

    catalog: list[dict[str, Any]] = [
        {
            "name": metadata.name,
            "description": metadata.description,
            "parameters": _render_parameters(metadata.input_schema),
        }
        for metadata in metadata_list
    ]
    if not catalog:
        return "(no capabilities registered)"
    return json.dumps(catalog, indent=2, ensure_ascii=False)


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
