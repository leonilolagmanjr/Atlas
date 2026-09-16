"""Small deterministic result interpreters for tool output."""

from __future__ import annotations

import json

from tools.base import ToolResult


def interpret_tool_result(tool_name: str, result: ToolResult) -> str:
    if not result.success:
        return result.error or f"{tool_name} failed."
    if tool_name == "powershell.execute" and isinstance(result.output, dict):
        stdout = str(result.output.get("stdout", "")).strip()
        if stdout:
            structured = _try_json(stdout)
            if structured is not None:
                return _summarize_structured(structured)
            return stdout
        return "PowerShell completed successfully with no output."
    return f"Completed tool action: {tool_name}."


def _try_json(value: str) -> object | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _summarize_structured(value: object) -> str:
    rows = value if isinstance(value, list) else [value]
    if not rows or not all(isinstance(row, dict) for row in rows):
        return json.dumps(value, indent=2)
    if all({"Name", "Id", "WorkingSet64"}.issubset(row) for row in rows):
        lines = ["Top processes by memory usage:"]
        for index, row in enumerate(rows[:10], start=1):
            memory = _format_bytes(row.get("WorkingSet64"))
            lines.append(f"{index}. {row.get('Name', 'unknown')} (PID {row.get('Id', 'unknown')}) - {memory}")
        return "\n".join(lines)
    if all({"Status", "Name"}.issubset(row) for row in rows):
        return f"Found {len(rows)} Windows services."
    return json.dumps(value, indent=2)


def _format_bytes(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    units = ("B", "KB", "MB", "GB", "TB")
    index = 0
    while number >= 1024 and index < len(units) - 1:
        number /= 1024
        index += 1
    return f"{number:.1f} {units[index]}"
