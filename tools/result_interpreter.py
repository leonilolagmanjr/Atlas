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
    if tool_name == "applications.write_text" and isinstance(result.output, dict):
        application = result.output.get("application", "the application")
        verification = result.output.get("verification")
        if verification == "confirmed":
            return (
                f"Wrote {result.output.get('characters', 'the')} characters into {application} "
                "and read the text back to confirm it."
            )
        if verification == "failed":
            return (
                f"The text could not be found in {application} after writing it, "
                "so the write is not verified."
            )
        characters = result.output.get("characters")
        if characters is not None:
            return (
                f"Wrote {characters} characters into {application}. "
                "I could not read the control back, so this is not independently verified."
            )
        return f"Wrote the requested text into {application}."
    if tool_name == "computer.observe" and isinstance(result.output, dict):
        status = result.output.get("status")
        if status == "observed":
            return str(result.output.get("summary") or "Observed the application window.")
        return str(
            result.output.get("summary")
            or "I could not observe an application window for this request."
        )
    if tool_name == "computer.windows" and isinstance(result.output, dict):
        windows = result.output.get("windows") or []
        if not windows:
            return "No open windows matched that request."
        titles = [
            str(window.get("title") or window.get("class_name") or "untitled")
            for window in windows[:10]
            if isinstance(window, dict)
        ]
        lines = [f"{len(windows)} open window(s):"]
        lines.extend(f"- {title}" for title in titles)
        return "\n".join(lines)
    if tool_name == "content.generate" and isinstance(result.output, dict):
        return str(result.output.get("text") or "Content generated.")
    if tool_name == "applications.launch_named" and isinstance(result.output, dict):
        return f"Opened {result.output.get('application', 'the application')}."
    if tool_name == "applications.launch" and isinstance(result.output, dict):
        return f"Launched {result.output.get('executable', 'the application')}."
    if tool_name == "filesystem.write" and isinstance(result.output, dict):
        path = result.output.get("path")
        if path:
            return f"Saved the content to {path}."
        return "Saved the content to a file."
    if tool_name == "filesystem.search" and isinstance(result.output, dict):
        matches = result.output.get("matches") or []
        if not matches:
            return "No matching files were found."
        return f"Found {len(matches)} matching file(s)."
    if tool_name == "web.search" and isinstance(result.output, dict):
        results = result.output.get("results") or []
        query = result.output.get("query", "the request")
        if not results:
            site = result.metadata.get("source") if isinstance(result.metadata, dict) else None
            if site == "YouTube":
                return f"I searched YouTube but couldn't find videos for: {query}"
            return (
                f"I couldn't find public web results for: {query}. "
                "The search backends may be rate-limiting this machine."
            )
        lines = [f"Found {len(results)} public web results:"]
        for index, item in enumerate(results[:8], start=1):
            if not isinstance(item, dict):
                continue
            lines.append(f"{index}. {item.get('title', 'Untitled')}\n   {item.get('url', '')}\n   {item.get('snippet', '')}")
        return "\n".join(lines)
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
