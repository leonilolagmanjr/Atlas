"""Durable ground-truth dump for the section-25 reasoning repair.

Writes repair_report.txt with: pytest failure lines, def/class index of the
engine and router, signal-construction and module-constant regions, engine
gather/delegate/handle regions, real RetrievalResult/Task shapes, and failing
test bodies. Everything survives compaction on disk.
"""
import dataclasses
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = []


def log(*parts):
    out.append(" ".join(str(p) for p in parts))


# 1. pytest failure summary -------------------------------------------------
try:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_reasoning_behavior.py",
         "-q", "--tb=line", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(("FAILED", "ERROR")) or "AssertionError" in line or "Error" in line and line.strip().startswith(("E", ">")):
            log("PYTEST|", line)
    tail = [l for l in proc.stdout.splitlines() if l.strip()][-6:]
    log("PYTEST-TAIL|", *tail)
except Exception as exc:  # noqa: BLE001
    log("pytest failed:", repr(exc))

# 2. def/class/constant index ----------------------------------------------
engine = (ROOT / "reasoning" / "reasoning_engine.py").read_text(encoding="utf-8")
router = (ROOT / "reasoning" / "query_router.py").read_text(encoding="utf-8")
tests = (ROOT / "tests" / "test_reasoning_behavior.py").read_text(encoding="utf-8")

for name, src in (("ENGINE", engine), ("ROUTER", router)):
    log(f"== {name} def index ==")
    for i, line in enumerate(src.splitlines(), 1):
        if re.match(r"\s*(class |def |_[A-Z_]+\s*=)", line):
            log(f"{i}: {line.strip()[:100]}")

# 3. router: explicit_web_request + signals construction -------------------
rlines = router.splitlines()
log("== ROUTER explicit_web_request / RoutingSignals sites ==")
for i, line in enumerate(rlines, 1):
    if "explicit_web_request" in line or "RoutingSignals(" in line or "class RoutingSignals" in line:
        lo, hi = max(0, i - 6), min(len(rlines), i + 6)
        log(f"--- router {i} ---")
        for j in range(lo, hi):
            log(f"{j + 1}: {rlines[j]}")

# 4. engine: gather / delegate / handle_request regions ---------------------
elines = engine.splitlines()
for pat in (r"def handle_request", r"def _gather", r"def _execute", r"def _delegate"):
    for i, line in enumerate(elines, 1):
        if re.search(pat, line):
            lo, hi = max(0, i - 1), min(len(elines), i + 34)
            log(f"--- engine {pat} at {i} ---")
            for j in range(lo, hi):
                log(f"{j + 1}: {elines[j]}")
            break

# 5. real shapes ------------------------------------------------------------
log("== RetrievalResult shape ==")
try:
    sys.path.insert(0, str(ROOT))
    from knowledge_search import RetrievalResult
    for f in dataclasses.fields(RetrievalResult):
        log("field:", f.name, ":", str(f.type))
except Exception as exc:  # noqa: BLE001
    log("RetrievalResult introspection failed:", repr(exc))

log("== Task shape ==")
try:
    from models_task import Task
    for f in dataclasses.fields(Task):
        log("field:", f.name, ":", str(f.type))
except Exception as exc:  # noqa: BLE001
    log("Task introspection failed:", repr(exc))

# 6. failing test bodies ----------------------------------------------------
log("== failing test bodies ==")
for m in re.finditer(r"def (test_[a-z0-9_]+)", tests):
    name = m.group(1)
    tlines = tests.splitlines()
    start = tests[: m.start()].count("\n")
    # find next top-level def/class to bound the body
    end = len(tlines)
    for i in range(start + 1, len(tlines)):
        if re.match(r"(def |class )", tlines[i]):
            end = i
            break
    log(f"--- {name} ({start + 1}-{end}) ---")
    for j in range(start, end):
        log(f"{j + 1}: {tlines[j]}")

report = ROOT / "repair_report.txt"
report.write_text("\n".join(out), encoding="utf-8")
print("REPORT-WRITTEN:", report, "lines:", len(out))
