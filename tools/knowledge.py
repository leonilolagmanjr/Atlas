"""Data-driven knowledge records for discoverable Atlas tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ToolParameter:
    name: str
    description: str = ""
    required: bool = False
    parameter_type: str = "string"
    default: Any = None


@dataclass(frozen=True)
class ToolKnowledgeRecord:
    name: str
    tool_type: str
    category: str
    description: str
    purpose: tuple[str, ...] = ()
    parameters: tuple[ToolParameter, ...] = ()
    examples: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    related_tools: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    requires_admin: bool = False
    risk_level: str = "safe"
    destructive: bool = False
    read_only: bool = True
    expected_output: str = ""
    platforms: tuple[str, ...] = ()
    powershell_versions: tuple[str, ...] = ()
    documentation_url: str | None = None
    source: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolKnowledgeRecord":
        parameters = tuple(
            ToolParameter(
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
                required=bool(item.get("required", False)),
                parameter_type=str(item.get("type", item.get("parameter_type", "string"))),
                default=item.get("default"),
            )
            for item in payload.get("parameters", [])
            if isinstance(item, dict) and item.get("name")
        )
        return cls(
            name=str(payload["name"]),
            tool_type=str(payload.get("type", payload.get("tool_type", "custom"))),
            category=str(payload.get("category", "general")),
            description=str(payload.get("description", "")),
            purpose=_strings(payload.get("purpose", [])),
            parameters=parameters,
            examples=_strings(payload.get("examples", [])),
            aliases=_strings(payload.get("aliases", [])),
            related_tools=_strings(payload.get("related_tools", [])),
            prerequisites=_strings(payload.get("prerequisites", [])),
            requires_admin=bool(payload.get("requires_admin", False)),
            risk_level=str(payload.get("risk_level", "safe")).lower(),
            destructive=bool(payload.get("destructive", False)),
            read_only=bool(payload.get("read_only", not payload.get("destructive", False))),
            expected_output=str(payload.get("expected_output", "")),
            platforms=_strings(payload.get("platforms", [])),
            powershell_versions=_strings(payload.get("powershell_versions", [])),
            documentation_url=payload.get("documentation_url"),
            source=str(payload.get("source", "local")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.tool_type,
            "category": self.category,
            "description": self.description,
            "purpose": list(self.purpose),
            "parameters": [
                {
                    "name": parameter.name,
                    "description": parameter.description,
                    "required": parameter.required,
                    "type": parameter.parameter_type,
                    "default": parameter.default,
                }
                for parameter in self.parameters
            ],
            "examples": list(self.examples),
            "aliases": list(self.aliases),
            "related_tools": list(self.related_tools),
            "prerequisites": list(self.prerequisites),
            "requires_admin": self.requires_admin,
            "risk_level": self.risk_level,
            "destructive": self.destructive,
            "read_only": self.read_only,
            "expected_output": self.expected_output,
            "platforms": list(self.platforms),
            "powershell_versions": list(self.powershell_versions),
            "documentation_url": self.documentation_url,
            "source": self.source,
            "metadata": self.metadata,
        }


class ToolKnowledgeStore:
    """Load and search general tool records without a second vector database."""

    def __init__(self, records: Iterable[ToolKnowledgeRecord] = ()) -> None:
        self._records: dict[tuple[str, str], ToolKnowledgeRecord] = {}
        self.add_many(records)

    def add(self, record: ToolKnowledgeRecord) -> None:
        key = (record.tool_type.casefold(), record.name.casefold())
        self._records[key] = record

    def add_many(self, records: Iterable[ToolKnowledgeRecord]) -> None:
        for record in records:
            self.add(record)

    def get(self, name: str, *, tool_type: str | None = None) -> ToolKnowledgeRecord:
        if tool_type:
            return self._records[(tool_type.casefold(), name.casefold())]
        matches = [record for (_, record_name), record in self._records.items() if record_name == name.casefold()]
        if not matches:
            raise KeyError(name)
        return matches[0]

    def list_records(self) -> list[ToolKnowledgeRecord]:
        return list(self._records.values())

    def search(self, query: str, *, tool_type: str | None = None, limit: int = 10) -> list[ToolKnowledgeRecord]:
        terms = {term for term in query.casefold().split() if term}
        scored: list[tuple[int, ToolKnowledgeRecord]] = []
        for record in self._records.values():
            if tool_type and record.tool_type.casefold() != tool_type.casefold():
                continue
            searchable = " ".join(
                [record.name, record.description, record.category, *record.purpose, *record.aliases]
            ).casefold()
            score = sum(1 for term in terms if term in searchable)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].name.casefold()))
        return [record for _, record in scored[: max(limit, 0)]]


def load_json(path: Path) -> list[ToolKnowledgeRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("tools", payload) if isinstance(payload, (dict, list)) else []
    if not isinstance(items, list):
        raise ValueError("Tool knowledge JSON must contain a list or a 'tools' list")
    return [ToolKnowledgeRecord.from_dict(item) for item in items if isinstance(item, dict)]


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if item is not None)
