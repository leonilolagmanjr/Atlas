# Atlas Tool Intelligence

Atlas tools are registered runtime capabilities. Tool execution remains behind
`ToolRouter`, permission policy, and `Executor`; the planner does not execute
subprocesses directly.

## Knowledge records

`tools/knowledge.py` defines data-driven `ToolKnowledgeRecord` records. A record
can describe any provider, including PowerShell, Python, Windows APIs, browser,
Git, networking, or custom tools. Records include:

- name, provider type, category, and description
- purposes, aliases, examples, and related tools
- parameters and parameter metadata
- prerequisites, platform and version compatibility
- risk, destructive, read-only, and privilege metadata
- expected output, source, documentation URL, and extensible metadata

JSON records are imported with `load_json()` and searched with
`ToolKnowledgeStore`. This reuses the existing local application architecture
without creating a second vector database. Embedding-backed search can be
added behind the same store later.

## Discovery and planning

`ToolDiscovery` returns structured candidates for a requested capability. The
current deterministic `PowerShellTaskPlanner` uses discovery to select a
provider record and construct commands for supported local inspection tasks.
The LLM is not the authority for available commands.

## PowerShell provider

The initial dataset is [tools/powershell_commands.json](tools/powershell_commands.json).
It is intentionally small and updateable; adding command knowledge does not
require changing Python source.

The current provider supports documented, read-only commands such as:

- `Get-Process`
- `Get-Service`
- `Get-ChildItem`
- `Get-ComputerInfo`
- `Get-NetTCPConnection`

`PowerShellTool` returns structured success, exit code, stdout, stderr, and
duration fields. It uses a non-shell subprocess with `-NoProfile` and a bounded
timeout.

Read-only pipelines are supported when every stage is documented in the local
knowledge base. Planned process and service queries append `ConvertTo-Json` so
the interpreter can produce concise summaries while the task panel retains
expandable raw output.

## Validation and permissions

`PowerShellValidator` runs before subprocess creation. Unknown commands and
known destructive operations are blocked. The initial provider rejects
redirection, script blocks, command substitution, encoded commands, executable
launches, downloads, registry changes, service/process mutation, firewall
changes, account changes, and scheduled-task changes.

The provider is registered as read-only, so safe inspection runs without
confirmation under the existing permission engine. Mutating PowerShell is not
implemented and must be added as a separate risk-aware capability.

## Named applications

`applications.launch_named` resolves arbitrary application names such as
Discord, Notepad, Calculator, Chrome, or other installed desktop programs to
trusted executable locations using PATH, common install folders, and Windows
uninstall metadata. It uses the same non-shell process boundary and low-risk
confirmation requirement as explicit executable launch. If resolution fails,
Atlas reports the failure without launching anything.
