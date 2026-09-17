"""Temporary diagnostic: what does the goal layer decide for the target prompts?"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning.goal import build_goal, needs_decomposition, split_clauses  # noqa: E402
from reasoning.task_interpreter import SemanticTaskInterpreter  # noqa: E402

PROMPTS = [
    "Search the Bee Movie script and copy it to Notepad.",
    "Find Jordan B. Peterson's 12 Rules for Life book and summarize it and write the summary to Notepad.",
    "Find 12 Rules for Life and summarize it and write the summary to Notepad.",
    "Create a poem in Notepad about cars.",
    "Create a poem in Notepad about YouTube.",
    "Write a poem about YouTube in Notepad.",
    "Find information about quantum computing and summarize it in Notepad.",
    "Find information about the James Webb Space Telescope and write the key points to Notepad.",
    "Find information about Apple and summarize it.",
    "Write a poem about Apple in Notepad.",
    "Find my resume.",
    "search the web for the skyrim script and copy it in notepad",
    "Open Notepad.",
    "What is quantum computing?",
    "Find a review of the Bee Movie in Notepad.",
    "Find how to write a resume and write the key points to Notepad.",
]

only_goal = "--goal-only" in sys.argv
apps = frozenset(["notepad", "chrome", "edge", "youtube", "google"])
for text in PROMPTS:
    goal = build_goal(text, known_applications=apps)
    print("=" * 78)
    print("PROMPT:", text)
    print("  clauses  :", split_clauses(text))
    print(
        "  intent   :", goal.intent, "| transform:", goal.transform or "-",
        "| dest:", goal.destination or "-", "| output:", goal.requested_output,
        "| artifact:", goal.must_be_artifact, "| decompose:", needs_decomposition(goal),
    )
    for outcome in goal.outcomes:
        print(
            "   outcome :", outcome.action, "| obj:", outcome.object, "| topic:", outcome.topic,
            "| type:", outcome.content_type, "| xform:", outcome.transform,
            "| prev:", outcome.uses_previous_result, "| dest:", outcome.destination,
        )
    print("  objective:", goal.objective)
    print("  caps     :", goal.required_capabilities)
    print("  criteria :", goal.completion_criteria)
    print("  entities :", goal.entities)
    if not only_goal:
        task = SemanticTaskInterpreter(enabled=False).interpret(text)
        for action in task.actions:
            print("   legacy  :", action.capability, action.parameters)
        if not task.actions:
            print("   legacy  : (no actions)")
