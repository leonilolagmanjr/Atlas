import re, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
out = []
def section(title, path, pattern, ctx=8, maxhits=6, pre=2):
    out.append(f"\n### {title} [{path}]")
    try:
        lines = (root / path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        out.append(f"  !! {e}"); return
    hits = 0
    for i, line in enumerate(lines):
        if hits >= maxhits: break
        if re.search(pattern, line):
            hits += 1
            for j in range(max(0, i-pre), min(len(lines), i+ctx)):
                out.append(f"{j+1}: {lines[j]}")
            out.append("---")
section("ENGINE handle_request head", "reasoning/reasoning_engine.py", r"def handle_request", ctx=50)
section("ENGINE _decide", "reasoning/reasoning_engine.py", r"def _decide\b", ctx=34, maxhits=1)
section("ENGINE _retrieve", "reasoning/reasoning_engine.py", r"def _retrieve\b", ctx=20, maxhits=1)
section("ENGINE _gather_files", "reasoning/reasoning_engine.py", r"def _gather_files", ctx=26, maxhits=1)
section("ENGINE knowledge gather", "reasoning/reasoning_engine.py", r"result\.retrieved_chunks", ctx=9, maxhits=2)
section("ROUTER signal assignments", "reasoning/query_router.py", r"explicit_web_request=|wants_mutation=|has_actions=|signals\.", ctx=9, maxhits=8)
section("FAILING TESTS", "tests/test_reasoning_behavior.py", r"def test_grounded_answer_uses_local_knowledge_when_relevant|def test_downloads_listing_uses_filesystem_tools|def test_open_notepad_delegates_to_task_pipeline|def test_poem_in_notepad_is_not_rerouted_to_web_search", ctx=24, maxhits=4)
report = "\n".join(out)
(root / "scripts/_s25_dump.txt").write_text(report, encoding="utf-8")
print(report)
