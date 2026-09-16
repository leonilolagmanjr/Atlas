# Atlas Tool Intelligence
Atlas tools are registered runtime capabilities. Tool execution remains behind
`ToolRouter`, permission policy, and `Executor`; the planner does not execute
subprocesses directly.

## Capability registry
`tools/capabilities.py` turns registered tool metadata into validated
`Capability` descriptors and renders the compact catalog injected into LLM
prompts. Each capability exposes:

- name and description
- parameter schema and required parameters
- risk level and whether confirmation is required
- whether it is verifiable and which outputs it produces
`PLANNABLE_CAPABILITIES` defines the subset the LLM may select. It is a
high-level surface (`content.generate`, `applications.*`, `filesystem.*`,
`web.*`); low-level admin backends (`powershell.execute`, `processes.*`) remain
registered and routable but are invisible to free-text interpretation, so the
model can never select them from a natural-language request.

The capability descriptor merges the real tool metadata (description, parameter
schema, risk) with a contract overlay for the facts a tool class does not carry
(required parameters, verifiability, produced output names). The router remains
the authority on whether a capability can actually run.

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

## Filesystem capabilities
Read-only: `filesystem.list`, `filesystem.read`, `filesystem.metadata`,
`filesystem.search` (with an optional `path` subfolder scope).

Permission-gated (medium risk, require confirmation): `filesystem.write`,
`filesystem.create_folder`, `filesystem.move`, `filesystem.copy`. All are bounded
to the configured `COMPUTER_ROOT`; a path outside the root is rejected. `move`
and `copy` treat an existing destination directory as "move into it", preserving
the file name.

These back high-level natural-language tasks such as "create a folder called
Projects on my desktop" or "find the largest PDF in Downloads and move it to my
Documents folder" without exposing raw shell commands.

## Named applications

`applications.launch_named` resolves arbitrary application names such as
Discord, Notepad, Calculator, Chrome, or other installed desktop programs to
trusted executable locations using PATH, common install folders, and Windows
uninstall metadata. It uses the same non-shell process boundary and low-risk
confirmation requirement as explicit executable launch. If resolution fails,
Atlas reports the failure without launching anything.

## Application text entry

`applications.write_text` opens a resolved application, discovers its window,
focuses it through Win32, and pastes supplied text using the Windows clipboard.
This is a medium-risk operation and always requires user confirmation. The
tool limits text size, uses `shell=False` for the application process, and
does not accept arbitrary keyboard or mouse scripts.

For example, `Create a short poem in Notepad` produces a plan containing the
Notepad target and poem text. Quoted text is entered verbatim; short-poem
requests use a deterministic starter poem until a dedicated content-generation
step is added.

## Web research

`web.search` performs read-only public search. Explicit YouTube/video requests
use YouTube's structured search data directly and return actual watch URLs and
native thumbnails; other searches use DuckDuckGo with a Bing RSS fallback.
Results include attributed titles, URLs, snippets, thumbnails, and provider
names.
`web.fetch` retrieves bounded public HTML or plain text from a validated public
HTTP(S) URL. Private hosts, loopback addresses, local names, unsupported content
types, and oversized responses are rejected.

Web results are untrusted evidence. Page text is never treated as an Atlas
instruction and cannot directly create a computer-control command.
