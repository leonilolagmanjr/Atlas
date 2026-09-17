import re, subprocess, sys, pathlib
root = pathlib.Path(r"c:\Users\leonilolagmanjr\Documents\Atlas")
out = []
def section(path, pats, ctx=8):
    try:
        text = (root / path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        out.append(f"### {path} MISSING: {exc}")
        return
    out.append(f"### {path}")
    for i, l in enumerate(text):
        for p in pats:
            if re.search(p, l):
                lo = max(0, i - ctx)
                hi = min(len(text), i + ctx + 1)
                out.append(f"-- match /{p}/ @L{i+1}")
                out.extend(f"{j+1}: {text[j]}" for j in range(lo, hi))
                out.append("---")
                break
section("reasoning/reasoning_engine.py", [r"class ReasoningEngine"], ctx=34)
section("reasoning/reasoning_engine.py", [r"def handle_request"], ctx=44)
section("reasoning/reasoning_engine.py", [r"def _gather"], ctx=24)
section("reasoning/reasoning_engine.py", [r"def _execute|def _run_action|def _act|def _delegate"], ctx=18)
section("reasoning/reasoning_engine.py", [r"retrieved_chunks"], ctx=6)
section("reasoning/query_router.py", [r"explicit_web_request"], ctx=8)
section("reasoning/query_router.py", [r"time_sensitive\s*[:=]|web_hint|search the (web|internet)|_WEB_SEARCH"], ctx=6)
section("reasoning/query_router.py", [r"def route"], ctx=30)
section("knowledge_search.py", [r"class RetrievalResult"], ctx=26)
section("knowledge_search.py", [r"def to_dict"], ctx=24)
section("tests/test_reasoning_behavior.py", [r"retrieved_chunks|class .*Retriev|def retrieve|def list_files|def search_web|class .*Web|class .*File"], ctx=16)
section("tests/test_reasoning_behavior.py", [r"def test_.*(knowledge|notepad|web|files|downloads|delegat)"], ctx=14)
proc = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_reasoning_behavior.py", "-q", "--tb=line", "-ra", "-p", "no:cacheprovider"],
    cwd=str(root), capture_output=True, text=True, timeout=900,
)
out.append("### PYTEST STDOUT tail")
out.extend(proc.stdout.splitlines()[-34:])
out.append("### PYTEST STDERR tail")
out.extend(proc.stderr.splitlines()[-8:])
report = root / "scripts" / "_fix_report.txt"
report.write_text("\n".join(out), encoding="utf-8")
print("REPORT_LINES", len(out))
print("PYTEST_RETURN", proc.returncode)
