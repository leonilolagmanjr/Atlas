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

## Application text entry

Text entry is separate from application launch and is classified as medium
risk. Atlas resolves the application first, launches it without a shell, waits
for startup, discovers and focuses the target window through Win32, then uses
a bounded clipboard paste only after the user approves the plan. Arbitrary
keyboard/mouse scripts and unapproved application targets are not accepted.

## Tool knowledge and internet content

Tool knowledge is data-driven and can be imported from local JSON. It is not an
execution instruction by itself. Future webpage or internet content must remain
untrusted evidence; webpage text must never be copied into an executable tool
request without local planning, validation, and permission evaluation.

The web runtime enforces HTTP(S)-only URLs, rejects loopback/private/link-local
targets to reduce SSRF risk, applies response-size and timeout limits, and
labels returned page content as untrusted. Search and fetch are read-only and
do not download files or execute page scripts.

## Permissions

The existing permission engine distinguishes read-only, low, medium, high, and
critical operations. Read-only PowerShell inspection is registered as
read-only. Destructive PowerShell is currently unavailable rather than silently
approved.

## Remaining hardening
- PowerShell AST-aware validation (lexical validation is in place today)
- durable live-task checkpoints and cancellation
- output-size limits and streaming policy
- stronger process isolation or sandboxing
- explicit trusted path and target-resource policies
- web download safety and browser-automation protections (search/fetch SSRF checks are already implemented)

The current Windows smoke check successfully ran one documented process
inspection pipeline. This does not constitute authorization for mutation or
arbitrary shell execution.
