# Atlas Testing

## Automated checks

Run from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in pathlib.Path('.').rglob('*.py') if '.venv' not in p.parts and '__pycache__' not in p.parts]; print('syntax ok')"
.\.venv\Scripts\python.exe -c "import atlas, api, brain, cli, executor, intent_classifier, planner, models, tools, computer; print('imports ok')"
Push-Location frontend; npm run build; Pop-Location
```

The Python suite uses the standard library `unittest` runner and covers:

- Execution context serialization and lifecycle states.
- Tool metadata, validation, registration, routing, and permission decisions.
- Bounded filesystem inspection and traversal rejection.
- Process/system/application inspection contracts.
- Non-shell application launch and confirmation gating.
- Application intent, deterministic planning, executor pause/resume boundaries, and CLI approval callbacks.
- FastAPI health and system smoke endpoints.
- Durable task snapshot restoration and interrupted-task handling.
- Tool knowledge JSON import and capability discovery.
- PowerShell command safety validation, structured execution, and timeout boundaries.
- Natural-language computer intent and knowledge-backed PowerShell planning.
- Real Windows PowerShell smoke execution through `PowerShellTool` with JSON result interpretation.
- Frontend API contracts for tool knowledge and non-executing tool discovery.
- Natural-language arbitrary named application resolution and confirmation-gated launch planning.

## Manual runtime checks completed

On 2026-09-16:

- Frontend production build passed with Vite and strict TypeScript.
- API health returned `online`, local mode, model `qwen2.5:7b`.
- API system returned real Windows OS, CPU, Python, and disk data.
- API tools returned the registered computer capabilities and permission metadata.
- API applications returned installed Windows registry entries.
- A real explicit executable task reached `WAITING_FOR_CONFIRMATION` and was cancelled through the API without launching the application.
- The browser console rendered the command workspace, approval state, cancellation state, and live Applications view.
- Ollama 0.31.2 was present and `qwen2.5:7b` was installed.

## Not verified in this audit

- A successful Ollama generation through `Brain`, because it depends on the running Ollama service and a retrieval context accepted by the local Chroma index.
- Retrieval quality against a representative PDF corpus.
- Internet search, URL retrieval, downloads, or webpage prompt-injection handling: no internet runtime exists in the current source.
- Broad live PowerShell coverage: one safe process pipeline was verified; the unit suite continues to mock subprocesses for deterministic failure and timeout cases.
- Resuming a live execution context after an API restart: task snapshots are
	durable, but execution checkpoints are not implemented yet.
- Browser automated regression coverage: the live browser smoke check was manual through the VS Code browser tool.

## Security checks

The implemented computer boundary was checked for:

- Filesystem traversal outside the configured repository root: rejected.
- Application launch shell injection: launch uses an executable path plus argument list with `shell=False`.
- Unapproved application launch: router returns `confirmation_required` and Executor pauses the plan.
- Critical permission defaults: denied.

Internet, download, webpage-instruction, secret-handling, and terminal-command security checks remain pending because those capabilities are not implemented.
PowerShell mutation, AST-aware validation, and live command cancellation remain pending.
