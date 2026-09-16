"""Tool capability discovery over the shared tool knowledge store."""

from __future__ import annotations

from dataclasses import dataclass

from tools.knowledge import ToolKnowledgeRecord, ToolKnowledgeStore


@dataclass(frozen=True)
class ToolCandidate:
    tool: str
    tool_type: str
    reason: str
    risk_level: str
    read_only: bool
    score: int


class ToolDiscovery:
    def __init__(self, store: ToolKnowledgeStore) -> None:
        self._store = store

    def discover(self, task: str, *, tool_type: str | None = None, limit: int = 5) -> list[ToolCandidate]:
        records = self._store.search(task, tool_type=tool_type, limit=limit)
        return [self._candidate(task, record, index) for index, record in enumerate(records)]

    @staticmethod
    def _candidate(task: str, record: ToolKnowledgeRecord, index: int) -> ToolCandidate:
        reason = record.description
        if record.purpose:
            matching = [purpose for purpose in record.purpose if any(term in purpose.casefold() for term in task.casefold().split())]
            if matching:
                reason = matching[0]
        return ToolCandidate(
            tool=record.name,
            tool_type=record.tool_type,
            reason=reason,
            risk_level=record.risk_level,
            read_only=record.read_only,
            score=max(1, 100 - index),
        )
