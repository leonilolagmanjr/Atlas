# Atlas Security Boundary

## PowerShell execution

PowerShell is executed only by the registered `powershell.execute` tool. The
executor uses an argument list with `shell=False`, captures stdout and stderr,
and enforces a timeout and output process boundary.

Before execution, `PowerShellValidator`:

- requires a non-empty bounded command
- requires the first command to exist in the local PowerShell knowledge base
- permits only records marked read-only and safe/low risk
- rejects redirection, script blocks, substitution, null bytes, and escaping
- blocks deletion, process/service control, system configuration, registry and
  firewall changes, scheduled tasks, account changes, downloads, executable
  launches, encoded commands, and arbitrary expression execution

This is a conservative lexical boundary, not a complete PowerShell AST
sandbox. A future mutating provider must use a separate policy and stronger
parser-backed validation.

## Tool knowledge and internet content

Tool knowledge is data-driven and can be imported from local JSON. It is not an
execution instruction by itself. Future webpage or internet content must remain
untrusted evidence; webpage text must never be copied into an executable tool
request without local planning, validation, and permission evaluation.

## Permissions

The existing permission engine distinguishes read-only, low, medium, high, and
critical operations. Read-only PowerShell inspection is registered as
read-only. Destructive PowerShell is currently unavailable rather than silently
approved.

## Remaining hardening

- PowerShell AST-aware validation
- durable live-task checkpoints and cancellation
- output-size limits and streaming policy
- stronger process isolation or sandboxing
- explicit trusted path and target-resource policies
- browser automation and internet SSRF/download protections

The current Windows smoke check successfully ran one documented process
inspection pipeline. This does not constitute authorization for mutation or
arbitrary shell execution.
