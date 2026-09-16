$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$frontend = Join-Path $root "frontend"

if (-not (Test-Path $python)) {
    Write-Host "Atlas virtual environment not found at $python" -ForegroundColor Red
    Write-Host "Create it with: python -m venv .venv" -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

if (-not (Test-Path (Join-Path $frontend "package.json"))) {
    Write-Host "Atlas frontend was not found at $frontend" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "npm was not found. Install Node.js before launching the Atlas frontend." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Frontend dependencies are not installed." -ForegroundColor Red
    Write-Host "Run: Set-Location frontend; npm install" -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

$backendPort = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($backendPort) {
    $owner = Get-Process -Id $backendPort[0].OwningProcess -ErrorAction SilentlyContinue
    $ownerName = if ($owner) { $owner.ProcessName } else { "unknown process" }
    Write-Host "Port 8000 is already in use by $ownerName (PID $($backendPort[0].OwningProcess))." -ForegroundColor Yellow
    Write-Host "Stop the existing Atlas backend before launching this copy." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

$backendArguments = @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command",
    "Set-Location -LiteralPath '$root'; & '$python' -m uvicorn api:app --host 127.0.0.1 --port 8000"
)

$frontendArguments = @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command",
    "Set-Location -LiteralPath '$frontend'; npm run dev -- --host 127.0.0.1"
)

Start-Process powershell.exe -ArgumentList $backendArguments -WorkingDirectory $root
Start-Process powershell.exe -ArgumentList $frontendArguments -WorkingDirectory $frontend
Start-Process "http://127.0.0.1:5173"

Write-Host "Atlas backend and frontend are starting." -ForegroundColor Cyan
Write-Host "Backend:  http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"
