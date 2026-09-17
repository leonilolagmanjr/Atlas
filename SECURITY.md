# Atlas Security Boundary

> [README.md](README.md) is authoritative for current architecture, setup, configuration, testing, and roadmap status. This document retains the detailed security policy and remaining boundary limitations. Atlas is not an OS, process, filesystem, or network sandbox.

## Deployment and trust

Run the local API on loopback (`127.0.0.1`) as documented in [README: Running](README.md#running). The current deployment does not provide an authenticated multi-user security boundary. Do not expose it to an untrusted network or treat CORS as authentication.

Tools execute with the user's OS privileges. Keep the configured filesystem root narrow, use least-privilege OS accounts, and do not enable autonomous operation based solely on model confidence. Read-only does not mean non-sensitive: documents, process information, requests, responses, history, and logs can contain private data.

## Interpretation and execution policy

The model proposes a structured Task, including sources and registered actions. It does not directly execute shell text. Task validation, planning, concrete plan checks, tool routing, and permission evaluation remain separate boundaries. The reasoning engine's read-only source tools use a narrow allowlist and the router; any `Task.actions`, including read-only actions, are delegated to the existing execution pipeline.

A capability description or imported tool-knowledge record does not authorize execution. Runtime availability comes from the actual registry, and policy is checked at the tool boundary. Evidence from files or the internet is not an instruction source and must not bypass these boundaries.

## Filesystem policy

Filesystem tools resolve paths beneath `COMPUTER_ROOT` and reject paths outside it. Traversal rechecks resolved candidates, avoids recursive symlink/junction traversal, and excludes common runtime/dependency/private-storage and credential locations. These exclusions are defense in depth, not a comprehensive secret detector or protection against all filesystem races.

File walks and reads have result, entry, time, byte, and content limits. Text/PDF content search is distinct from filename search. PDF file tools use pypdf with input-size, page-count, extracted-text, and between-page deadline checks; they do not provide OCR. A single PDF parser/page extraction may still block. Startup knowledge ingestion is a separate pipeline and should not be assumed to inherit all file-tool limits.

Mutating filesystem capabilities use the same root boundary and the permission-aware router. They are not arbitrary whole-machine file management. A request mentioning a home folder does not expand the allowed root.

## PowerShell execution

PowerShell is executed through the registered `powershell.execute` provider. It launches a subprocess with an argument list, `shell=False`, and `-NoProfile`, captures results, and applies a subprocess timeout. `shell=False` avoids an additional shell layer; it does not make a PowerShell command intrinsically safe.

Before execution, the lexical validator requires bounded, non-empty commands and documented read-only/safe command records. It rejects unsupported commands and syntax such as redirection, script blocks, command substitution, null bytes, and escaping. The current boundary blocks destructive process/service operations, registry/system/firewall/account changes, scheduled tasks, downloads, executable launches, encoded commands, and arbitrary expression execution.

This is conservative lexical validation, **not a complete PowerShell AST sandbox**. Adding command knowledge must not silently enable mutation. A future mutating provider needs a separate risk-aware policy, stronger parser-backed validation, explicit permissions, and effect verification. A successful read-only smoke test is not authorization for arbitrary shell execution.

## Applications and text entry

Explicit/named application launch resolves an executable and invokes it without a shell. Application text entry is a separate medium-risk capability, with bounded text and target-window/control discovery. It attempts control-based delivery and uses clipboard/focus fallback where necessary. This is narrower than unrestricted GUI automation, but still changes local application state and can interact with the wrong target if resolution is inadequate.

Use confirmation for consequential actions in the default policy. Do not describe text delivery as independently verified merely because a tool reports a character count. Arbitrary keyboard/mouse scripts are not an implemented capability.

## Web and untrusted content

Search/fetch are read-only. Page scripts are not executed, and controlled file downloads/browser interaction are not implemented. Public provider responses, snippets, extracted pages, and URLs remain untrusted even when relevant to the question.

The default web transport validates HTTP(S) destinations at initial access, redirects, and final page handling. DNS checks inspect IPv4 and IPv6 answers and reject non-public targets, including loopback/private/link-local addresses. Response-size and network timeout limits also apply.

These checks reduce SSRF exposure but **do not eliminate DNS rebinding or TOCTOU**: urllib may resolve again after validation, and proxy behavior can change actual resolution/connection routing. This is not a network sandbox. Search-provider failures are isolated, but bot challenges, provider changes, and misleading source content remain possible.

External text must never directly grant approval, define a new executable capability, or override the user's request. Prompt labeling and grounded-answer prompts are not proof that model prompt injection is impossible. Review source provenance and keep effectful actions behind their independent validation/permission path.

## Permission modes and approval gaps

The permission engine distinguishes read-only, low, medium, high, and critical operations. Defaults:

- `safe`: read-only tools only.
- `confirm`: read-only tools can run; consequential operations require confirmation.
- `autonomous`: low/medium-risk operations may run without confirmation.
- Critical operations are denied by default; explicit policy overrides can change decisions.

Task-specific approve/deny endpoints exist, and resuming a paused plan retains completed steps. However, approval resolution is not yet fully serialized with the submission worker. Existing approval grants are not completely bound to immutable step arguments/targets, including recovery changes. UI duplicate-click controls are not a concurrency proof.

Priority fixes are atomic queue-aware approval/resume and argument-bound authorization, with reapproval whenever a proposed recovery changes an effectful target or parameter. Until then, do not claim global action serialization, one-click exactly-once execution, or a security sandbox.

## Budgets, cancellation and verification

Reasoning budgets count each actual routed source-tool call and bound ordered source iterations. They do not globally meter every internal provider request, retrieval subquery, or legacy executor operation.

The reasoning engine accepts a cooperative cancellation callback/token and checks deadlines between calls. Full running-task cancellation is not wired through Brain/API/UI. Denial applies to a paused approval, not arbitrary in-flight work. Ollama transport timeouts bound transport behavior, not an absolute whole-generation wall-clock deadline. Blocking file/PDF/model calls need stronger isolation for hard preemption.

Verification records `verified`, `unverified`, or `failed` according to current capability contracts. Many checks validate tool-reported output shape, not an independent observation of the requested effect. Retrieval sufficiency and citations likewise do not prove factual correctness.

## Storage and diagnostics

Conversation sessions and API task snapshots are persisted locally. Snapshots are history, not resumable live checkpoints; restart marks in-flight tasks interrupted rather than replaying effects without context.

New API/UI reasoning telemetry is bounded and sanitized: allowed stages, source names, response modes, evidence counts, and reduced citations are exposed instead of raw evidence/prompt/history objects. Restored reasoning snapshots are sanitized again. This does not comprehensively redact all requests, responses, legacy tool outputs, logs, or conversation files. Diagnostic key redaction catches configured keys, not every embedded secret.

Keep private knowledge, sessions, task data, logs, exports, and credentials out of commits. Review actual ignore rules rather than assuming all runtime paths are ignored. Delete or retain local data according to the user's needs; automatic long-term retention/privacy policy is not implemented.

## Remaining hardening

- Atomic serialized approval/resume and argument-bound grants with recovery reapproval.
- Independent effect verification and stronger target/resource identity checks.
- End-to-end cancellation, blocking-work isolation, and durable safe execution checkpoints.
- DNS/connection/proxy boundary hardening and broader untrusted-content tests.
- Parser-backed PowerShell validation before considering any mutation.
- Stronger process isolation, comprehensive output/stream limits, and least-privilege policies.
- Controlled download/browser protections before adding those capabilities.
- Broader privacy/secret-handling review of all persisted and exposed task data.

See [README: Testing](README.md#testing) for current validation commands and [README: Roadmap](README.md#roadmap) for prioritized work. Tests demonstrate specific cases, not complete security assurance.
