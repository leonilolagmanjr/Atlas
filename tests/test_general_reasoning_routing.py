"""Regression tests for general-purpose routing (spec sections 6, 9, 10, 11, 19, 25).

These lock in the fixes that stop ordinary/general requests from falling through
to a web or "not in knowledge base" route when the semantic intent is a local
file, a self query, or a model-knowledge question.

Everything is offline: the deterministic interpreter runs with the LLM disabled,
the model is a canned stub, and tool execution is a recording fake.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from computer.runtime import register_read_only_tools  # noqa: E402
from config import COMPUTER_ROOT  # noqa: E402
from reasoning.answer_generator import AnswerGenerator  # noqa: E402
from reasoning.query_router import QueryRouter  # noqa: E402
from reasoning.reasoning_engine import ReasoningEngine  # noqa: E402
from reasoning.reasoning_models import ResponseMode, SourceType  # noqa: E402
from reasoning.self_introspection import SelfIntrospection  # noqa: E402
from reasoning.source_selector import SourceSelector  # noqa: E402
from reasoning.task_interpreter import SemanticTaskInterpreter  # noqa: E402
from tools.base import ToolResult  # noqa: E402
from tools.capabilities import CapabilityRegistry  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402


class FakeAsk:
    def __init__(self, answer: str = "A general-knowledge answer.") -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self, *, system_prompt: str = "", user_prompt: str = "", **_kw) -> str:
        self.calls += 1
        return self.answer


def _capabilities() -> CapabilityRegistry:
    registry = ToolRegistry()
    register_read_only_tools(registry, root=COMPUTER_ROOT, ask=None)
    return CapabilityRegistry(registry)


def _interpreter() -> SemanticTaskInterpreter:
    caps = _capabilities()
    return SemanticTaskInterpreter(ask=None, capabilities=caps, capability_catalog=caps.render_catalog())


def _route(prompt: str):
    interpreter = _interpreter()
    caps = _capabilities()
    task = interpreter.interpret(prompt)
    signals = QueryRouter().route(
        prompt, task=task, self_introspection=SelfIntrospection(caps, model_name="test-model")
    )
    plan = SourceSelector(caps).select(signals, task_type=task.task_type)
    return task, signals, plan


class FileRoutingRegressionTests(unittest.TestCase):
    """Personal-document lookups must be local file requests, never web search."""

    def test_find_my_resume_is_a_file_lookup(self):
        task, signals, plan = _route("Find my resume.")
        self.assertEqual([a.capability for a in task.actions], ["filesystem.search"])
        self.assertNotIn("web", [s.value for s in plan.sources])
        self.assertIn("files", [s.value for s in plan.sources])
        self.assertIsNotNone(signals.file_intent)
        self.assertEqual(signals.file_intent.subject, "resume")

    def test_search_my_documents_for_phrase_is_content_search(self):
        task, signals, plan = _route("Search my documents for 'climate change'.")
        self.assertEqual(
            [a.capability for a in task.actions], ["filesystem.search_content"]
        )
        self.assertEqual(task.actions[0].parameters["query"], "climate change")
        self.assertNotIn("web", [s.value for s in plan.sources])

    def test_explicit_web_platform_overrides_file_hint(self):
        # A named web platform still wins over a weak personal-document hint.
        _, signals, plan = _route("Search youtube for my resume videos.")
        self.assertIn("web", [s.value for s in plan.sources])

    def test_plain_web_search_still_routes_to_web(self):
        task, _, plan = _route("Find the latest NVIDIA GPU.")
        self.assertIn("web", [s.value for s in plan.sources])


class CreateFileRegressionTests(unittest.TestCase):
    def test_create_text_file_produces_a_write_action(self):
        task, _, plan = _route("Create a text file.")
        self.assertEqual([a.capability for a in task.actions], ["filesystem.write"])
        self.assertEqual(task.actions[0].parameters["path"], "atlas_created_txt.txt")
        self.assertTrue(plan.requires_action)

    def test_create_file_with_name_uses_that_name(self):
        task = _interpreter().interpret("Create a text file named notes.txt")
        capabilities = [a.capability for a in task.actions]
        # A named file request still writes the file; generating content first is
        # an acceptable (and helpful) prefix step.
        self.assertIn("filesystem.write", capabilities)
        write = task.actions[-1]
        self.assertEqual(write.capability, "filesystem.write")
        self.assertEqual(write.parameters["path"], "notes.txt")


class HybridRegressionTests(unittest.TestCase):
    def test_search_then_write_researches_and_writes_the_content(self):
        task, _, plan = _route(
            "Search the web for the latest Python version and write it into Notepad."
        )
        capabilities = [a.capability for a in task.actions]
        # The plan researches the web (search + read pages + isolate the relevant
        # content) and writes that content, not the list of links.
        self.assertEqual(capabilities, ["web.research", "content.format", "applications.write_text"])
        research = task.actions[0]
        self.assertEqual(research.produces, "web_content")
        self.assertNotIn("write", research.parameters["query"])
        write = task.actions[-1]
        self.assertEqual(write.parameters["text"], "$formatted_text")
        self.assertTrue(plan.requires_action)

    def test_plain_search_is_not_a_write(self):
        task = _interpreter().interpret("Search YouTube for popular videos.")
        self.assertEqual([a.capability for a in task.actions], ["web.search"])

class SaveResultsRegressionTests(unittest.TestCase):
    # "Save the results to a file" must persist something, not be dropped. Before
    # this fix the save clause was silently discarded: the task planned a bare
    # web.search and reported the links as if the request were fulfilled.

    CASES = [
        "Search for RTX 5090 benchmarks and save the results to a file.",
        "search for Python decorators and save the results to a file",
        "look up RTX 5090 benchmarks and save them in results.txt",
        "search for climate data and put the results in a text file",
    ]

    def test_save_results_produces_a_search_then_write(self):
        for prompt in self.CASES:
            with self.subTest(prompt=prompt):
                task = _interpreter().interpret(prompt)
                capabilities = [a.capability for a in task.actions]
                self.assertEqual(capabilities, ["web.search", "filesystem.write"], prompt)
                search, write = task.actions
                self.assertEqual(search.produces, "search_results")
                self.assertEqual(write.parameters["text"], "$search_results")
                self.assertTrue(write.requires_confirmation)

    def test_derived_filename_is_safe_and_named_from_topic(self):
        task = _interpreter().interpret(
            "Search for RTX 5090 benchmarks and save the results to a file."
        )
        path = task.actions[1].parameters["path"]
        self.assertEqual(path, "rtx_5090_benchmarks.txt")
        self.assertNotIn("/", path)
        self.assertNotIn("..", path)

    def test_a_generic_file_noun_is_not_captured_as_a_filename(self):
        # "save the results to a file" must not treat "a file" as a real name,
        # and must not become an artifact request.
        task = _interpreter().interpret(
            "search for python decorators and save the results to a file"
        )
        self.assertNotIn("filename", task.entities)
        self.assertNotEqual(task.entities.get("application"), "a file")

    def test_results_saved_to_file_is_not_a_local_file_search(self):
        # A requested output extension (results.txt) must not send Atlas to
        # search the user's own disk instead of the web.
        task = _interpreter().interpret(
            "look up RTX 5090 benchmarks and save them in results.txt"
        )
        capabilities = [a.capability for a in task.actions]
        self.assertIn("web.search", capabilities)
        self.assertNotIn("filesystem.search", capabilities)
        self.assertFalse(task.needs_clarification)

    def test_summarize_then_save_to_file_is_not_a_local_file_search(self):
        # "summarize them in a file" writes the summary; it must not become a
        # local content search and must not drop the write step.
        task = _interpreter().interpret(
            "search the web for car reviews and summarize them in a file"
        )
        capabilities = [a.capability for a in task.actions]
        self.assertEqual(
            capabilities,
            ["web.research", "content.generate", "content.format", "filesystem.write"],
        )
        self.assertEqual(task.actions[-1].parameters["path"], "car_reviews.txt")

    def test_transformation_does_not_request_a_summary_artifact(self):
        # "summarize the car videos" is information about videos, not a request
        # for a "summary document"; must_be_artifact must stay False and the
        # retrieval content type must not be the transformation noun.
        task = _interpreter().interpret(
            "Search YouTube for car videos and put the summary in Notepad"
        )
        research = task.actions[0]
        self.assertFalse(research.parameters["must_be_artifact"])
        self.assertNotEqual(research.parameters["content_type"], "summary")
        self.assertEqual(research.parameters["goal"], "find_information")
        self.assertEqual(
            [a.capability for a in task.actions],
            ["web.research", "content.generate", "content.format", "applications.write_text"],
        )

    def test_summarize_results_has_no_ambiguous_reference(self):
        # "summarize them" refers to the search results, which the plan produces.
        task, signals, _ = _route("Search for RTX 5090 benchmarks and summarize them.")
        self.assertFalse(signals.ambiguous_reference)
        self.assertIn("web.research", [a.capability for a in task.actions])

    def test_create_about_an_article_noun_is_not_a_local_file_lookup(self):
        # "write an essay about the election" is content creation. The bare
        # article "the" must not mark "essay" as the user's own document.
        task = _interpreter().interpret(
            "write an essay about the 2024 election in notepad"
        )
        self.assertEqual(task.entities.get("content_type"), "essay")
        self.assertNotIn("file_intent", task.entities)
        self.assertNotIn("file_subject", task.entities)
        self.assertEqual(task.entities.get("application"), "notepad")
        self.assertEqual(
            [a.capability for a in task.actions],
            ["content.generate", "content.format", "applications.write_text"],
        )

    def test_standalone_number_is_the_only_quantity(self):
        # A 4-digit year or a longer model number must not be read as a count,
        # and must not leak a partial-digit "quantity" from inside it.
        self.assertNotIn("quantity", _interpreter().interpret("write a haiku about 2025").entities)
        self.assertNotIn(
            "quantity",
            _interpreter().interpret("search for RTX 5090 benchmarks").entities,
        )
        self.assertEqual(
            _interpreter().interpret("search for the top 5 GPUs").entities.get("quantity"), 5
        )

    def test_make_a_list_of_topic_is_captured(self):
        # "make a list of 5 workout exercises" introduces the topic after "of",
        # not "about". The topic must survive or Atlas writes about nothing.
        task = _interpreter().interpret(
            "make a list of 5 workout exercises and put it in notepad"
        )
        self.assertEqual(task.entities.get("content_type"), "list")
        self.assertEqual(task.entities.get("topic"), "5 workout exercises")
        self.assertEqual(task.entities.get("quantity"), 5)

    def test_creation_topic_of_clause_does_not_hijack_general_questions(self):
        # A bare "of X" on a non-creation request must not become a topic.
        task = _interpreter().interpret("what is the capital of France")
        self.assertIsNone(task.entities.get("topic"))

    def test_move_destination_is_not_the_search_location(self):
        # "move the largest pdf to Documents" searches for the largest PDF and
        # moves it *into* Documents; it must not search inside Documents.
        task = _interpreter().interpret("move the largest pdf to Documents")
        capabilities = [a.capability for a in task.actions]
        self.assertEqual(capabilities, ["filesystem.search", "filesystem.move"])
        search, move = task.actions
        self.assertIsNone(search.parameters.get("path"))
        self.assertEqual(search.parameters.get("select"), "largest")
        self.assertEqual(move.parameters.get("destination"), "Documents")

    def test_personal_document_lookup_still_detected(self):
        # The stricter possessive set must not lose genuine local-file requests.
        task = _interpreter().interpret("find my resume")
        self.assertEqual(task.entities.get("file_intent"), "lookup")
        self.assertEqual(task.entities.get("file_subject"), "resume")

    def test_bare_save_of_results_writes_a_file(self):
        # "save the results" with no explicit file word still persists.
        task = _interpreter().interpret("search for the best laptops and save the results")
        self.assertEqual(
            [a.capability for a in task.actions], ["web.search", "filesystem.write"]
        )
        # The search is a plain web.search of the *results* (no artifact demand).
        self.assertEqual(task.actions[0].capability, "web.search")
        self.assertEqual(task.actions[1].parameters["path"], "best_laptops.txt")

    def test_explicit_filename_saves_results_not_an_artifact(self):
        # "export the results to recipes.csv" persists the result list; it must
        # not become an artifact retrieval with a polluted query.
        task = _interpreter().interpret(
            "search for recipes and export the results to recipes.csv"
        )
        self.assertEqual(
            [a.capability for a in task.actions], ["web.search", "filesystem.write"]
        )
        self.assertEqual(task.actions[1].parameters["path"], "recipes.csv")
        self.assertNotIn("export", task.actions[0].parameters["query"])

    def test_location_qualifier_is_not_an_application(self):
        # "the weather in Tokyo" must not resolve Tokyo as the destination app.
        task = _interpreter().interpret(
            "look up the weather in Tokyo and save it as weather.txt"
        )
        self.assertNotEqual(task.entities.get("application"), "Tokyo")
        self.assertEqual(
            [a.capability for a in task.actions], ["web.search", "filesystem.write"]
        )
        self.assertEqual(task.actions[1].parameters["path"], "weather.txt")

    def test_trailing_open_clause_is_not_part_of_the_query(self):
        task = _interpreter().interpret("search for cat pictures and open notepad")
        search = next(a for a in task.actions if a.capability == "web.search")
        self.assertEqual(search.parameters["query"], "cat pictures")
        self.assertIn("applications.launch_named", [a.capability for a in task.actions])

    def test_summarize_current_news_uses_web_research(self):
        # "summarize the news" needs current information, so it retrieves it
        # rather than relying on the model's possibly-stale knowledge.
        task = _interpreter().interpret("summarize the news about tesla")
        self.assertEqual(
            [a.capability for a in task.actions], ["web.research", "content.generate"]
        )
        self.assertEqual(task.actions[0].parameters["query"], "tesla")

    def test_summarize_local_document_is_not_sent_to_the_web(self):
        # A transformation without a current-information noun stays local.
        task = _interpreter().interpret("summarize this document")
        self.assertEqual(task.actions, [])

    def test_stated_quantity_reaches_generation_instructions(self):
        # "a list of 5 items" must tell the generator how many to produce.
        task = _interpreter().interpret(
            "make a list of 5 workout exercises and put it in notepad"
        )
        self.assertIn("5", task.actions[0].parameters["instructions"])

    def test_stated_length_reaches_generation_instructions(self):
        short = _interpreter().interpret("write a short poem about cars")
        self.assertIn("short", short.actions[0].parameters["instructions"].casefold())
        plain = _interpreter().interpret("write a poem about cars")
        self.assertIsNone(plain.actions[0].parameters["instructions"])

    def test_follow_up_preserves_content_type_and_pipeline(self):
        # "make it about dogs" after "write a poem about cars in notepad" must
        # keep the poem content type and the full generate->format->write chain,
        # not degrade to generic text with an unwrapped write.
        interpreter = _interpreter()
        first = interpreter.interpret("write a poem about cars in notepad")
        second = interpreter.interpret("make it about dogs", context=first)
        self.assertEqual(
            [a.capability for a in second.actions],
            ["content.generate", "content.format", "applications.write_text"],
        )
        self.assertEqual(second.actions[0].parameters["content_type"], "poem")
        self.assertEqual(second.actions[0].parameters["topic"], "dogs")
        self.assertEqual(second.actions[-1].parameters["application"], "notepad")

    def test_named_file_resolves_pronoun_reference(self):
        # "read my notes.txt and summarize it" has a concrete referent; it must
        # not be reported as an ambiguous reference requiring clarification.
        task, signals, plan = _route("Read my notes.txt and summarize it")
        self.assertFalse(signals.ambiguous_reference)
        self.assertEqual([s.value for s in plan.sources], ["files"])

    def test_generation_without_destination_is_not_blocked(self):
        # "write a poem" is satisfiable in chat; Atlas must not demand a
        # destination before generating content.
        for prompt in ("write a poem", "write a haiku", "tell me a joke"):
            with self.subTest(prompt=prompt):
                task = _interpreter().interpret(prompt)
                self.assertFalse(task.needs_clarification)
                self.assertEqual([a.capability for a in task.actions], ["content.generate"])

    def test_bare_content_noun_generates_without_a_create_verb(self):
        # "tell me a joke" names no create verb but is a generation request.
        task = _interpreter().interpret("tell me a joke")
        self.assertEqual(task.actions[0].parameters["content_type"], "joke")

    def test_generation_request_never_uses_the_web(self):
        # A self-contained generation request must not be mistaken for research.
        task = _interpreter().interpret("write a poem about cars")
        self.assertEqual([a.capability for a in task.actions], ["content.generate"])

    def test_artifact_request_is_unchanged_by_the_save_path(self):
        # "get the X script and copy it in Notepad" still retrieves the artifact
        # itself; the save-results path must not swallow it.
        task = _interpreter().interpret("get the bee movie script and copy it in Notepad")
        capabilities = [a.capability for a in task.actions]
        self.assertEqual(
            capabilities, ["web.research", "content.format", "applications.write_text"]
        )
        self.assertTrue(task.actions[0].parameters.get("must_be_artifact"))

    def test_save_with_local_document_still_searches_files(self):
        # Genuine local-file requests are unaffected by the web-target guard.
        task = _interpreter().interpret("find my resume")
        self.assertEqual([a.capability for a in task.actions], ["filesystem.search"])

    def test_copy_it_in_notepad_is_a_web_hybrid_not_a_file_copy(self):
        # "copy" here means "place the found text", not a filesystem copy; the
        # bare verb must not hijack the search into a file move/copy plan.
        task = _interpreter().interpret(
            "search the web for the bee movie script and copy it in notepad"
        )
        capabilities = [a.capability for a in task.actions]
        self.assertEqual(capabilities, ["web.research", "content.format", "applications.write_text"])
        self.assertNotIn("filesystem.copy", capabilities)
        for action in task.actions:
            for value in action.parameters.values():
                self.assertNotEqual(value, "$largest_match")

    def test_copy_to_notepad_names_notepad_as_the_destination(self):
        # "copy to Notepad" is the same hybrid as "copy it in Notepad": the
        # destination preposition "to" follows a placement verb.
        task = _interpreter().interpret(
            "search the web for bee movie script and copy to notepad"
        )
        self.assertEqual(
            [a.capability for a in task.actions],
            ["web.research", "content.format", "applications.write_text"],
        )
        self.assertEqual(task.entities["application"].casefold(), "notepad")
        research = task.actions[0].parameters
        self.assertTrue(research["must_be_artifact"])
        self.assertEqual(research["goal"], "retrieve_document")
        # The destination "to notepad" must not leak into the search query.
        self.assertNotIn("notepad", research["query"].casefold())


    class ReferenceResolutionTests(unittest.TestCase):
        # A "$name" plan reference must resolve to the value an earlier step
        # published; a literal "$name" reaching a tool is a broken handoff.

        def test_generic_reference_resolves_from_produced_map(self):
            from executor import _resolve_value
            produced = {"search_results": "- Python 3.13 (https://python.org)"}
            self.assertEqual(
                _resolve_value("$search_results", produced),
                "- Python 3.13 (https://python.org)",
            )

        def test_generated_text_reference_still_resolves(self):
            from executor import _resolve_value
            self.assertEqual(
                _resolve_value("$generated_text", {"generated_text": "a poem"}), "a poem"
            )

        def test_unknown_reference_is_left_literal(self):
            from executor import _resolve_value
            self.assertEqual(_resolve_value("$missing", {}), "$missing")

        def test_hybrid_plan_passes_resolved_text_to_the_writer(self):
            from brain import Brain
            from executor import Executor
            from planner import Planner
            from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
            from tools.base import Tool, ToolMetadata, ToolResult
            class FakeVectorStore:
                def search(self, *_a, **_k):
                    return []

            class RecordingTool(Tool):
                def __init__(self, name, output):
                    self.metadata = ToolMetadata(name=name, description=name, category="test")
                    self._output = output
                    self.calls = []

                def execute(self, parameters):
                    self.calls.append(dict(parameters))
                    return ToolResult(success=True, status="completed", output=self._output)

            researcher = RecordingTool(
                "web.research",
                {
                    "query": "python version",
                    "content": "Python 3.13 is the newest release.",
                    "sources": ["https://python.org"],
                },
            )
            writer = RecordingTool("applications.write_text", {"pid": 1, "application": "notepad", "characters": 36})

            registry = ToolRegistry()
            registry.register(researcher)
            registry.register(writer)
            router = ToolRouter(registry=registry, permission_engine=PermissionEngine(mode=ExecutionMode.AUTONOMOUS))
            brain = Brain(
                vector_store=FakeVectorStore(),
                system_prompt="sys",
                retrieval_template="{conversation_history}{context}{question}",
                planner=Planner(registry=router, ask=lambda **_: "{}"),
                executor=Executor(
                    vector_store=FakeVectorStore(),
                    system_prompt="sys",
                    retrieval_template="{conversation_history}{context}{question}",
                    tool_router=router,
                ),
                tool_router=router,
                llm_ask=lambda **_: "{}",
            )
            brain.process("search the web for the latest python version and write it into notepad")

            self.assertEqual(len(researcher.calls), 1)
            self.assertEqual(len(writer.calls), 1)
            written = writer.calls[0]["text"]
            # The write replaced the reference with the researched content, so the
            # destination gets the isolated content, not the links.
            self.assertEqual(written, "Python 3.13 is the newest release.")

class NegativeRoutingTests(unittest.TestCase):
    # Semantic intent must dominate keyword matching (spec section 19).

    def test_poem_about_cars_is_not_a_video_search(self):
        task, signals, plan = _route("Create a poem in Notepad about cars.")
        capabilities = [a.capability for a in task.actions]
        self.assertEqual(capabilities, ["content.generate", "content.format", "applications.write_text"])
        self.assertNotIn("web.search", capabilities)
        self.assertIn("computer", [s.value for s in plan.sources])
        self.assertEqual(task.entities.get("topic"), "cars")

    def test_plain_question_does_not_reach_for_web(self):
        _, signals, plan = _route("Explain recursion.")
        self.assertFalse(signals.time_sensitive)
        self.assertFalse(signals.explicit_web_request)
        self.assertEqual([s.value for s in plan.sources], ["knowledge", "model"])

    def test_current_question_reaches_for_web(self):
        _, signals, plan = _route("What is the latest Python version?")
        self.assertTrue(signals.time_sensitive)
        self.assertEqual(plan.sources[0].value, "web")

        def test_self_query_routes_to_self(self):
            _, signals, plan = _route("What can you do?")
            self.assertTrue(signals.is_self_query)
            self.assertEqual([s.value for s in plan.sources], ["self"])

class AmbiguousRequestTests(unittest.TestCase):
    # Requests with no resolvable target must ask, not guess (spec section 25).

    def test_find_that_file_asks_for_clarification(self):
        task, signals, plan = _route("Find that file.")
        self.assertFalse([a for a in task.actions if a.capability == "web.search"])
        self.assertTrue(signals.ambiguous_reference)
        self.assertEqual(plan.sources, ())

    def test_bare_search_verb_with_no_target_is_not_a_web_action(self):
        task = _interpreter().interpret("Search it.")
        self.assertEqual([a.capability for a in task.actions], [])

    def test_concrete_search_still_produces_a_web_action(self):
        task = _interpreter().interpret("Search for the latest AI news.")
        self.assertEqual([a.capability for a in task.actions], ["web.search"])


class ReasoningDelegationTests(unittest.TestCase):
    # Read-only research is served here; mutations delegate (spec sections 10/22).

    def test_pure_web_search_is_served_by_the_engine(self):
        task = _interpreter().interpret("Search the web for the latest Python version.")
        self.assertFalse(ReasoningEngine._must_delegate(task))

    def test_file_lookup_actions_are_served_by_the_engine(self):
        task = _interpreter().interpret("Find my resume.")
        self.assertFalse(ReasoningEngine._must_delegate(task))

    def test_file_mutation_delegates(self):
        task = _interpreter().interpret("Create a text file.")
        self.assertTrue(ReasoningEngine._must_delegate(task))

    def test_app_launch_delegates(self):
        task = _interpreter().interpret("Open Notepad.")
        self.assertTrue(ReasoningEngine._must_delegate(task))

    def test_content_creation_delegates(self):
        task = _interpreter().interpret("Create a poem in Notepad about cars.")
        self.assertTrue(ReasoningEngine._must_delegate(task))


class ExecutorWebResultRenderingTests(unittest.TestCase):
    # A search-then-write hybrid must hand usable text to the write step.

    def test_named_web_output_is_rendered_as_text(self):
        from executor import _render_web_results

        rendered = _render_web_results(
            {
                "query": "python",
                "results": [
                    {"title": "Python 3.13", "url": "https://python.org", "snippet": "Newest."}
                ],
            }
        )
        self.assertIn("Python 3.13", rendered)
        self.assertIn("https://python.org", rendered)

    def test_empty_web_results_render_honestly(self):
        from executor import _render_web_results

        rendered = _render_web_results({"query": "obscure", "results": []})
        self.assertIn("No web results", rendered)


class TransformToDestinationRegressionTests(unittest.TestCase):
    """Transformation requests with a destination must generate + write content.

    These lock in the fix for the primary regression:
    "summarize the avatar movie in notepad" must produce
    generate -> format -> write actions, not be answered inline.
    """

    NOTEPAD_CASES = [
        "summarize the avatar movie in notepad",
        "summarize Avatar in Notepad",
        "write a poem about cars in Notepad",
        "put a short explanation of black holes into Notepad",
        "create a summary of the matrix in notepad",
        "write an overview of python in notepad",
        "summarize avatar and write it to notepad",
        "open notepad and write a list of my tasks",
    ]

    def test_transform_plus_destination_produces_write_action(self):
        expected_capabilities = ["content.generate", "content.format", "applications.write_text"]
        for text in self.NOTEPAD_CASES:
            with self.subTest(text=text):
                task, _, _ = _route(text)
                caps = [a.capability for a in task.actions]
                self.assertEqual(caps, expected_capabilities, f"Failed for: {text}")
                write_action = next(a for a in task.actions if a.capability == "applications.write_text")
                self.assertEqual(write_action.parameters["application"].casefold(), "notepad")

    def test_summarize_avatar_movie_has_summary_content_type(self):
        task, _, _ = _route("summarize the avatar movie in notepad")
        gen = next(a for a in task.actions if a.capability == "content.generate")
        self.assertEqual(gen.parameters["content_type"], "summary")
        self.assertIn("avatar", str(gen.parameters.get("topic", "")))

    def test_pure_summarize_without_destination_is_informational(self):
        """'summarize Avatar' (no destination) should NOT trigger computer actions."""
        task, _, _ = _route("summarize Avatar")
        self.assertEqual([a.capability for a in task.actions], [])

    def test_explain_to_file_destination(self):
        """'explain X and put it in a file' should generate + write to a file."""
        task, _, _ = _route("explain quantum computing and put it in a file")
        caps = [a.capability for a in task.actions]
        self.assertIn("content.generate", caps)
        self.assertIn("filesystem.write", caps)

    def test_summarize_news_is_web_research(self):
        """'summarize the latest news about X' should use web.research, not local generation."""
        task, _, _ = _route("summarize the latest news about spacex")
        caps = [a.capability for a in task.actions]
        self.assertIn("web.research", caps)
        self.assertIn("content.generate", caps)
        # The query should not contain boilerplate
        research = next(a for a in task.actions if a.capability == "web.research")
        self.assertNotIn("latest news", research.parameters.get("query", ""))


if __name__ == "__main__":
    unittest.main()
