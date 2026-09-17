from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from brain import Brain
from executor import UNKNOWN_RESPONSE
from models import TaskStatus
from models_task import Task, _boolean, _enum, REQUEST_TYPES
from reasoning.answer_generator import Answer
from reasoning.reasoning_models import ConfidenceLevel, ResponseMode
from reasoning.task_interpreter import SemanticTaskInterpreter
from tests.test_task_pipeline_e2e import FakeVectorStore, RecordingTool, build_brain


class SemanticReasoningFieldsTests(unittest.TestCase):
    def test_strict_boolean_and_enum_values(self):
        for value in ("false", "true", 0, 1, [], {}):
            self.assertFalse(_boolean(value))
        self.assertFalse(_boolean(False, True))
        self.assertTrue(_boolean(None, True))
        self.assertEqual(_enum("QUESTION", REQUEST_TYPES, "unknown"), "question")
        self.assertEqual(_enum("powershell", REQUEST_TYPES, "unknown"), "unknown")
        task = Task.from_mapping({
            "request_type": "command", "sources": ["model", "web.search", "computer", {}, "model"],
            "current_information_required": "false", "requires_web": "true",
            "execution_required": "false", "requires_confirmation": "false",
        }, prompt="x")
        self.assertEqual(task.request_type, "unknown")
        self.assertEqual(task.sources, ["model", "computer"])
        self.assertFalse(task.current_information_required)
        self.assertFalse(task.requires_web)
        self.assertTrue(task.execution_required)
        self.assertFalse(task.requires_confirmation)

    def test_decision_validation_and_roundtrip(self):
        task = Task.from_mapping({
            "request_type": "hybrid", "sources": ["model", "computer"],
            "current_information_required": True, "requires_computer": True,
        }, prompt="x")
        task.apply_decision(SimpleNamespace(request_type="invalid", requires_web="false"))
        self.assertEqual(task.request_type, "hybrid")
        self.assertFalse(task.requires_web)
        restored = Task.from_mapping(task.to_dict(), prompt="x")
        self.assertEqual(restored.sources, ["model", "computer"])
        self.assertTrue(restored.current_information_required)

    def test_semantic_model_sources_survive_interpretation(self):
        cases = [
            ("Explain fusion", "question", ["model"], False),
            ("Review my agreement", "question", ["files"], False),
            ("Recall our discussion", "memory_query", ["conversation", "memory"], False),
            ("Describe available integrations", "self_query", ["self"], False),
            ("Assess market developments", "question", ["web"], True),
            ("Describe machine health", "question", ["system"], True),
        ]
        for prompt, request_type, sources, current in cases:
            with self.subTest(prompt=prompt):
                ask = Mock(return_value=json.dumps({
                    "task_type": "informational", "actions": [], "execution_required": False,
                    "request_type": request_type, "sources": sources,
                    "current_information_required": current, "confidence": 0.95,
                }))
                task = SemanticTaskInterpreter(ask=ask, enabled=True).interpret(prompt)
                self.assertEqual(task.sources, sources)
                self.assertEqual(task.request_type, request_type)
                self.assertEqual(task.current_information_required, current)
                self.assertEqual(task.actions, [])
                self.assertIn('"current_information_required"', ask.call_args.kwargs["system_prompt"])

    def test_knowledge_questions_never_launch_even_with_bad_model_action(self):
        for prompt in ("What is Notepad?", "How do I open Notepad?"):
            with self.subTest(prompt=prompt):
                ask = Mock(return_value=json.dumps({
                    "task_type": "computer_action", "request_type": "action", "sources": ["computer"],
                    "actions": [{"capability": "applications.launch_named", "parameters": {"application": "Notepad"}}],
                }))
                task = SemanticTaskInterpreter(ask=ask, enabled=True).interpret(prompt)
                self.assertEqual(task.actions, [])
                self.assertEqual(task.request_type, "question")
                self.assertEqual(task.sources, ["model"])
                self.assertFalse(task.execution_required)

    def test_action_fast_paths_preserve_sources_and_do_not_call_model(self):
        ask = Mock(side_effect=AssertionError("fast path called model"))
        interpreter = SemanticTaskInterpreter(ask=ask, enabled=True)
        task = interpreter.interpret("Create a poem in Notepad about cars.")
        self.assertEqual(task.sources, ["model", "computer"])
        self.assertEqual(task.request_type, "hybrid")
        self.assertFalse(task.requires_web)
        self.assertFalse(task.current_information_required)
        opened = interpreter.interpret("Open Notepad.")
        self.assertEqual(opened.sources, ["computer"])
        self.assertEqual(opened.request_type, "action")
        ask.assert_not_called()


class BrainReasoningIntegrationTests(unittest.TestCase):
    def test_instructional_question_with_non_json_interpreter_fallback(self):
        launcher = RecordingTool("applications.launch_named", {"pid": 1})
        brain = build_brain(lambda **_: "Open Notepad using the Start menu.", [launcher])
        self.assertEqual(brain.process("How do I open Notepad?"), "Open Notepad using the Start menu.")
        self.assertEqual(brain.last_context.metadata["task"]["actions"], [])
        self.assertEqual(launcher.calls, [])
        self.assertEqual(brain.last_context.status, TaskStatus.COMPLETED)

    def test_delegated_actions_receive_reasoning_before_existing_pipeline(self):
        generator = RecordingTool("content.generate", {"text": "cars poem"})
        writer = RecordingTool("applications.write_text", {"pid": 1, "application": "Notepad", "characters": 9})
        brain = build_brain(lambda **_: "{}", [generator, writer])
        brain.process("Create a poem in Notepad about cars.")
        task = brain.last_context.metadata["task"]
        self.assertIn("computer", task["sources"])
        self.assertNotIn("web", task["sources"])
        self.assertTrue(brain.last_context.metadata["reasoning"])
        self.assertEqual(writer.calls[0]["text"], "cars poem")
        self.assertEqual(brain.last_context.status, TaskStatus.COMPLETED)

    def test_early_answer_persists_both_messages_and_followup_history(self):
        memory = Mock()
        memory.build_conversation_history_for_prompt.return_value = "User: prior request"
        answer = Answer(text="A useful answer", mode=ResponseMode.DIRECT_ANSWER,
                        confidence_level=ConfidenceLevel.MEDIUM)
        engine = Mock()
        engine.handle_request.return_value = answer
        brain = Brain(vector_store=FakeVectorStore(), system_prompt="sys",
                      retrieval_template="{context}{question}", memory_manager=memory,
                      reasoning_engine=engine, llm_ask=lambda **_: "{}")
        brain.process("What is quantum computing?")
        memory.append_message.assert_any_call(role="user", content="What is quantum computing?")
        memory.append_message.assert_any_call(role="assistant", content="A useful answer")
        self.assertEqual(memory.append_message.call_count, 2)
        self.assertEqual(engine.handle_request.call_args.kwargs["history"], "User: prior request")
        self.assertEqual(engine.handle_request.call_args.kwargs["task"].sources, ["model"])
        first_task = brain._active_task
        brain.process("What did we discuss?")
        self.assertIs(engine.handle_request.call_args.kwargs["prior_task"], first_task)

    def test_disabled_engine_uses_existing_executor(self):
        brain = build_brain(lambda **_: "{}", [])
        engine = Mock()
        brain._reasoning_engine = engine
        executor = Mock()
        brain._executor = executor
        with patch("brain.ENABLE_REASONING_ENGINE", False):
            brain.process("What is quantum computing?")
        engine.handle_request.assert_not_called()
        executor.execute.assert_called_once()

    def test_disabled_general_fallback_keeps_ordinary_questions_on_retrieval(self):
        brain = build_brain(lambda **_: "{}", [])
        brain._reasoning_engine = Mock()
        brain._executor = Mock()
        with patch("brain.ENABLE_GENERAL_QUESTION_FALLBACK", False):
            brain.process("What is quantum computing?")
        brain._reasoning_engine.handle_request.assert_not_called()
        brain._executor.execute.assert_called_once()

    def test_engine_exception_marks_failure_without_exposing_error_text(self):
        brain = build_brain(lambda **_: "{}", [])
        brain._reasoning_engine = Mock()
        brain._reasoning_engine.handle_request.side_effect = RuntimeError("private exception content")
        self.assertEqual(brain.process("What is quantum computing?"), UNKNOWN_RESPONSE)
        self.assertEqual(brain.last_context.status, TaskStatus.FAILED)
        self.assertEqual(brain.last_context.errors, ["Brain execution failed (RuntimeError)"])

    def test_unknown_model_capability_is_rejected_before_reasoning(self):
        brain = build_brain(lambda **_: json.dumps({
            "task_type": "computer_action", "actions": [{"capability": "arbitrary.command"}],
        }), [])
        brain._reasoning_engine = Mock()
        brain.process("perform an arbitrary operation")
        brain._reasoning_engine.handle_request.assert_not_called()
        self.assertEqual(brain.last_context.status, TaskStatus.FAILED)
