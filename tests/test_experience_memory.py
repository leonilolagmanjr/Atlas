"""Tests for the human feedback -> experience -> retrieval -> improvement loop.

These cover the twelve required scenarios from the specification. They use fake
tools/models and a temporary experience store, so they run with no network, no
real model, and no access to the developer's own experience history.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experience.builder import build_experience, evaluate_completion
from experience.memory import (
    ExperienceEmbedder,
    ExperienceMemory,
    render_experience_context,
)
from experience.models import (
    CompletionChecks,
    Experience,
    FAILURE_CATEGORIES,
    sanitize_text,
)
from experience.service import ExperienceService, FeedbackRequest
from experience.store import ExperienceStore, FeedbackEvent
from executor import classify_failure
from models import (
    Evidence,
    ExecutionContext,
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    StepStatus,
    TaskStatus,
)
from models_task import Task, TaskAction


# -- helpers --------------------------------------------------------------------


def make_tool_call(tool: str, output: dict, *, success: bool = True, status: str = "completed") -> dict:
    return {"tool": tool, "output": output, "success": success, "status": status, "parameters": {}, "error": None}


def research_and_deliver_context(
    *,
    request: str = "Research how to make a resume and put the key points in Notepad.",
    status: TaskStatus = TaskStatus.COMPLETED,
    write_output: dict | None = None,
    verifications: list[dict] | None = None,
) -> ExecutionContext:
    """Build a realistic hybrid web-research -> Notepad execution context."""

    task = Task(
        goal="research_and_deliver",
        task_type="multi_step",
        original_prompt=request,
        desired_outcome="A resume-research summary exists in Notepad",
        entities={"application": "Notepad", "topic": "how to make a resume"},
        request_type="hybrid",
        actions=[
            TaskAction(action_id="a1", capability="web.research", parameters={"query": "resume"}),
            TaskAction(
                action_id="a2",
                capability="applications.write_text",
                parameters={"application": "Notepad", "text": "$research"},
                depends_on=["a1"],
            ),
        ],
    )
    steps = [
        ExecutionStep(id="a1", name="Research", action="invoke_tool", description="",
                      status=StepStatus.COMPLETED, metadata={"tool": "web.research", "parameters": {}}),
        ExecutionStep(id="a2", name="Deliver", action="invoke_tool", description="",
                      status=StepStatus.COMPLETED, metadata={"tool": "applications.write_text", "parameters": {}}),
    ]
    context = ExecutionContext(user_input=request)
    context.status = status
    context.final_response = "I wrote the summary into Notepad."
    context.execution_plan = ExecutionPlan(user_question=request, steps=steps, status=PlanStatus.COMPLETED)
    context.tool_calls = [
        make_tool_call("web.research", {"content": "resume research", "query": "resume"}),
        make_tool_call(
            "applications.write_text",
            write_output if write_output is not None else {"application": "Notepad", "characters": 120, "observed": True},
        ),
    ]
    context.verification_results = verifications if verifications is not None else [
        {"capability": "applications.write_text", "verified": True, "status": "verified", "detail": "read back"}
    ]
    context.metadata["task_object"] = task
    return context


def service_with_store(directory: str) -> tuple[ExperienceService, ExperienceStore]:
    store = ExperienceStore(
        experience_path=Path(directory) / "experiences.jsonl",
        feedback_path=Path(directory) / "feedback.jsonl",
    )
    return ExperienceService(store=store), store


# -- TEST 1: a normal question gets no feedback UI ------------------------------

class FeedbackEligibilityTests(unittest.TestCase):
    def test_plain_conversational_answer_offers_no_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            context = ExecutionContext(user_input="What is quantum computing?")
            context.status = TaskStatus.COMPLETED
            context.final_response = "Quantum computing uses qubits."
            # No plan and no tool calls: a direct answer, not a task outcome.
            self.assertFalse(service.should_offer_feedback(context))

    def test_meaningful_multi_step_task_offers_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            self.assertTrue(service.should_offer_feedback(research_and_deliver_context()))

    def test_task_still_running_offers_no_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            context = research_and_deliver_context(status=TaskStatus.WAITING_FOR_CONFIRMATION)
            self.assertFalse(service.should_offer_feedback(context))


# -- TEST 2 + 3: task completes and a SUCCESS experience persists ---------------

class SuccessPersistenceTests(unittest.TestCase):
    def test_completed_meaningful_task_is_recorded_as_experience(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            context = research_and_deliver_context()
            outcome = service.record_task_outcome(context, record_id="record-1")
            self.assertTrue(outcome.recorded)
            self.assertTrue(outcome.meaningful)
            self.assertEqual(outcome.outcome, "success")
            stored = store.experiences()
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].outcome, "success")

    def test_success_feedback_persists_and_promotes_the_experience(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            result = service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))
            self.assertTrue(result.ok)
            experience = store.find_for_record("record-1")[0]
            self.assertEqual(experience.outcome, "success")
            self.assertEqual(experience.feedback, "success")
            self.assertEqual(experience.experience_type, "successful_task")
            self.assertEqual(experience.state, "evaluated")
            self.assertGreaterEqual(experience.confirmations, 1)

    def test_success_experience_is_retrievable_and_carries_its_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))
            stored = store.experiences()[0]
            self.assertIn("web.research", stored.tools_used)
            self.assertIn("applications.write_text", stored.tools_used)
            self.assertEqual(stored.requested_destination, "Notepad")
            self.assertEqual(stored.completion_checks.delivery_verified, True)


# -- TEST 4 + 5: a FAILURE experience with a user-chosen category ---------------

class FailurePersistenceTests(unittest.TestCase):
    def test_failed_task_is_recorded_as_a_failure_experience(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            context = research_and_deliver_context(
                status=TaskStatus.FAILED,
                write_output={"application": "Notepad", "characters": 120, "observed": False},
                verifications=[
                    {"capability": "applications.write_text", "verified": False, "status": "failed", "detail": "not present"}
                ],
            )
            service.record_task_outcome(context, record_id="record-f")
            result = service.apply_feedback(FeedbackRequest(record_id="record-f", outcome="failure"))
            self.assertTrue(result.ok)
            experience = store.find_for_record("record-f")[0]
            self.assertEqual(experience.outcome, "failure")
            self.assertEqual(experience.experience_type, "failed_task")
            self.assertTrue(experience.is_failure())

    def test_user_supplied_failure_category_is_stored(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-f")
            service.apply_feedback(
                FeedbackRequest(
                    record_id="record-f",
                    outcome="failure",
                    failure_category="did_not_follow_instruction",
                )
            )
            experience = store.find_for_record("record-f")[0]
            self.assertEqual(experience.failure_category, "did_not_follow_instruction")
            self.assertEqual(experience.experience_type, "failed_task")

    def test_derived_failure_category_when_user_gives_no_reason(self):
        # The user is never forced to explain: Atlas derives the category from
        # the verification evidence it already has.
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            context = research_and_deliver_context(
                status=TaskStatus.FAILED,
                write_output={"application": "Notepad", "characters": 120, "observed": False},
                verifications=[
                    {"capability": "applications.write_text", "verified": False, "status": "failed", "detail": "absent"}
                ],
            )
            context.errors.append("applications.write_text did not produce the expected effect: text was not present")
            context.metadata["failure_classification"] = classify_failure(context)
            service.record_task_outcome(context, record_id="record-f")
            service.apply_feedback(FeedbackRequest(record_id="record-f", outcome="failure"))
            experience = store.find_for_record("record-f")[0]
            self.assertEqual(experience.failure_category, "verification_failed")

    def test_only_canonical_categories_are_accepted(self):
        request = FeedbackRequest.from_payload(
            "record-x", {"outcome": "failure", "failure_category": "not_a_real_category"}
        )
        self.assertEqual(request.failure_category, "other")
        self.assertIn("delivery_not_completed", FAILURE_CATEGORIES)


# -- TEST 6: user correction is stored ------------------------------------------

class CorrectionTests(unittest.TestCase):
    def test_correction_is_stored_and_classified_as_a_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-c")
            service.apply_feedback(
                FeedbackRequest(
                    record_id="record-c",
                    outcome="failure",
                    failure_category="did_not_follow_instruction",
                    correction="I specifically asked you to put the summary in Notepad.",
                )
            )
            experience = store.find_for_record("record-c")[0]
            self.assertEqual(
                experience.user_correction,
                "I specifically asked you to put the summary in Notepad.",
            )
            self.assertEqual(experience.experience_type, "correction")
            self.assertEqual(experience.state, "evaluated")

    def test_correction_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiences.jsonl"
            feedback_path = Path(directory) / "feedback.jsonl"
            service = ExperienceService(store=ExperienceStore(experience_path=path, feedback_path=feedback_path))
            service.record_task_outcome(research_and_deliver_context(), record_id="record-c")
            service.apply_feedback(
                FeedbackRequest(record_id="record-c", outcome="failure", correction="Put it in Notepad.")
            )
            # A brand-new service and store instance, as after an application restart.
            reopened = ExperienceService(
                store=ExperienceStore(experience_path=path, feedback_path=feedback_path)
            )
            experiences = reopened.store.experiences()
            self.assertEqual(len(experiences), 1)
            self.assertEqual(experiences[0].user_correction, "Put it in Notepad.")
            self.assertEqual(experiences[0].outcome, "failure")


# -- TEST 7: a later similar task retrieves the relevant experience -------------

class RetrievalTests(unittest.TestCase):
    def test_similar_future_request_retrieves_the_experience(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            service.record_task_outcome(
                research_and_deliver_context(
                    request="Research how to make a resume and put the key points in Notepad."
                ),
                record_id="record-1",
            )
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))

            context = service.retrieve_for_planning(
                "Research how to write a cover letter and put the key points in Notepad."
            )
            self.assertTrue(context.items)
            retrieved = context.items[0].experience
            self.assertIn("resume", retrieved.original_user_request)
            self.assertIn("Notepad", render_experience_context(context.items))

    def test_irrelevant_request_does_not_retrieve_a_distant_experience(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))
            context = service.retrieve_for_planning("What is the capital of Peru?")
            # Nothing comparable was stored, so no context is offered rather than
            # a weakly-related experience being forced onto planning.
            self.assertFalse(context.items)

    def test_retrieval_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            for index in range(12):
                service.record_task_outcome(
                    research_and_deliver_context(
                        request=f"Research topic {index} and put the key points in Notepad."
                    ),
                    record_id=f"record-{index}",
                )
                service.apply_feedback(FeedbackRequest(record_id=f"record-{index}", outcome="success"))
            context = service.retrieve_for_planning(
                "Research topic A and put the key points in Notepad."
            )
            self.assertLessEqual(len(context.items), 4)

    def test_never_retrieves_a_whole_store(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperienceStore(
                experience_path=Path(directory) / "experiences.jsonl",
                feedback_path=Path(directory) / "feedback.jsonl",
            )
            for index in range(30):
                store.add(
                    Experience(
                        original_user_request=f"unrelated archived task {index}",
                        outcome="success",
                        tools_used=["filesystem.write"],
                    )
                )
            memory = ExperienceMemory(store=store)
            context = memory.retrieve("Research resumes and put them in Notepad.")
            self.assertLess(len(context.items), 5)


# -- TEST 8 + 9: a successful/failed experience reaches planning ----------------

class PlanningContextTests(unittest.TestCase):
    def test_successful_experience_is_rendered_as_supporting_context(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))
            context = service.retrieve_for_planning(
                "Research how to write a resume and put the key points in Notepad."
            )
            text = context.text
            self.assertIn("EXPERIENCE MEMORY", text)
            self.assertIn("SUCCESS", text)
            self.assertIn("web.research", text)
            # The current request must be stated as authoritative.
            self.assertIn("current request always wins", text)

    def test_failed_experience_is_rendered_so_planning_avoids_the_mistake(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            service.apply_feedback(
                FeedbackRequest(
                    record_id="record-1",
                    outcome="failure",
                    failure_category="delivery_not_completed",
                    correction="Put the summary in Notepad, not in chat.",
                )
            )
            context = service.retrieve_for_planning(
                "Research how to write a resume and put the key points in Notepad."
            )
            text = context.text
            self.assertIn("FAILURE", text)
            self.assertIn("delivery_not_completed", text)
            self.assertIn("Put the summary in Notepad, not in chat.", text)
            self.assertIn("avoid repeating", text)

    def test_experience_context_reaches_the_planner_prompt(self):
        from reasoning.prompts import planner_user_prompt

        prompt = planner_user_prompt(
            intent_json="{}",
            capabilities="web.research",
            history="",
            experience_context="EXPERIENCE MEMORY: previous SUCCESS workflow",
        )
        self.assertIn("EXPERIENCE MEMORY", prompt)
        self.assertIn("current request always wins", prompt)

    def test_task_planner_forwards_experience_context(self):
        from reasoning.task_planner import TaskPlanner

        captured: dict[str, str] = {}

        def legacy(user_question, *, intent=None, experience_context=""):
            captured["context"] = experience_context
            from models import PlannerDecision

            return PlannerDecision(plan=ExecutionPlan(user_question=user_question, steps=[]))

        planner = TaskPlanner()
        decision = planner.create_plan(
            Task(original_prompt="Summarize X"),
            user_question="Summarize X",
            legacy_planner=legacy,
            experience_context="EXPERIENCE MEMORY: previous FAILURE",
        )
        self.assertEqual(captured["context"], "EXPERIENCE MEMORY: previous FAILURE")
        self.assertTrue(decision.metadata["experience_context_used"])


# -- TEST 10: the current instruction overrides experience ----------------------

class InstructionAuthorityTests(unittest.TestCase):
    def test_experience_never_becomes_a_hard_rule(self):
        # A previous task delivered into Notepad; the current task explicitly asks
        # to answer in chat. The rendered context must state the precedence so the
        # model is instructed to follow the current request.
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))
            context = service.retrieve_for_planning(
                "Research resumes but just answer me here, do not open any application."
            )
            self.assertIn("current request always wins", context.text)
            self.assertIn("must be preserved", context.text)

    def test_retrieval_does_not_mutate_the_current_task(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))
            task = Task(original_prompt="What is Python?", goal="answer", entities={})
            before = task.to_dict()
            service.retrieve_for_planning("What is Python?", task=task)
            self.assertEqual(task.to_dict(), before)


# -- TEST 11: execution succeeded but verification failed -----------------------

class VerificationAuthorityTests(unittest.TestCase):
    def test_unconfirmed_delivery_is_not_a_verified_success(self):
        context = research_and_deliver_context(
            write_output={"application": "Notepad", "characters": 120},  # sent, not read back
            verifications=[
                {"capability": "applications.write_text", "verified": False, "status": "unverified",
                 "detail": "the control could not be read back."}
            ],
        )
        checks = evaluate_completion(context, context.metadata["task_object"])
        # Delivery reached the tool, but nothing confirmed the effect.
        self.assertIsNone(checks.delivery_verified)

    def test_explicit_verification_failure_is_recorded_as_unmet(self):
        context = research_and_deliver_context(
            status=TaskStatus.FAILED,
            write_output={"application": "Notepad", "characters": 0, "observed": False},
            verifications=[
                {"capability": "applications.write_text", "verified": False, "status": "failed",
                 "detail": "the text was not present in the control"}
            ],
        )
        checks = evaluate_completion(context, context.metadata["task_object"])
        self.assertFalse(checks.delivery_verified)
        self.assertFalse(checks.requested_destination_reached)
        self.assertIn("delivery_verified", checks.unmet())

    def test_a_model_claiming_done_is_not_evidence(self):
        # "Done." in the final response must not upgrade anything by itself.
        context = research_and_deliver_context(
            status=TaskStatus.FAILED,
            write_output={"application": "Notepad", "characters": 0},
            verifications=[],
        )
        context.final_response = "Done."
        checks = evaluate_completion(context, context.metadata["task_object"])
        self.assertFalse(checks.execution_completed)
        self.assertIsNone(checks.delivery_verified)

    def test_user_failure_feedback_is_flagged_against_verified_success(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            result = service.apply_feedback(
                FeedbackRequest(record_id="record-1", outcome="failure", failure_category="other")
            )
            # Atlas verified the delivery; the user says it failed. The
            # disagreement is surfaced rather than silently averaged away.
            self.assertTrue(result.disagrees_with_verification)


# -- TEST 12: experiences survive an application restart ------------------------

class RestartPersistenceTests(unittest.TestCase):
    def test_experiences_and_feedback_survive_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiences.jsonl"
            feedback_path = Path(directory) / "feedback.jsonl"
            first = ExperienceService(
                store=ExperienceStore(experience_path=path, feedback_path=feedback_path)
            )
            first.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            first.apply_feedback(
                FeedbackRequest(record_id="record-1", outcome="failure", failure_category="wrong_action")
            )
            self.assertTrue(path.exists())

            second = ExperienceService(
                store=ExperienceStore(experience_path=path, feedback_path=feedback_path)
            )
            experiences = second.store.experiences()
            self.assertEqual(len(experiences), 1)
            self.assertEqual(experiences[0].outcome, "failure")
            self.assertEqual(experiences[0].failure_category, "wrong_action")
            self.assertIsNotNone(second.store.latest_feedback("record-1"))

    def test_api_task_record_keeps_feedback_across_a_store_reload(self):
        # The API task snapshot (which the UI polls) persists the feedback answer,
        # so the selected state is restored rather than reverting to un-answered.
        from api import TaskRecord

        record = TaskRecord(
            id="record-1",
            request="x",
            status="COMPLETED",
            created_at=1,
            updated_at=1,
            feedback_available=True,
            feedback_outcome="failure",
            feedback_category="did_not_follow_instruction",
        )
        reloaded = TaskRecord(**record.model_dump())
        self.assertEqual(reloaded.feedback_outcome, "failure")
        self.assertEqual(reloaded.feedback_category, "did_not_follow_instruction")


# -- feedback lifecycle ---------------------------------------------------------

class FeedbackLifecycleTests(unittest.TestCase):
    def test_feedback_can_be_changed_without_creating_a_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="failure", failure_category="other"))
            experiences = store.find_for_record("record-1")
            self.assertEqual(len(experiences), 1)
            self.assertEqual(experiences[0].outcome, "failure")

    def test_duplicate_success_does_not_inflate_confirmations(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))
            first = store.find_for_record("record-1")[0].confirmations
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))
            second = store.find_for_record("record-1")[0].confirmations
            self.assertEqual(first, second)

    def test_repeated_success_promotes_to_reliable(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            for index in range(3):
                service.record_task_outcome(research_and_deliver_context(), record_id=f"record-{index}")
                service.apply_feedback(FeedbackRequest(record_id=f"record-{index}", outcome="success"))
            promoted = service.promote_repeated_patterns()
            self.assertGreaterEqual(promoted, 1)
            self.assertTrue(any(item.state == "reliable" for item in store.experiences()))

    def test_a_single_interaction_does_not_become_global_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")
            service.apply_feedback(FeedbackRequest(record_id="record-1", outcome="success"))
            service.promote_repeated_patterns()
            self.assertEqual(store.experiences()[0].state, "evaluated")

    def test_feedback_for_an_unrecorded_task_is_still_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            result = service.apply_feedback(FeedbackRequest(record_id="never-recorded", outcome="failure"))
            self.assertTrue(result.ok)
            self.assertIn("not stored as an experience", result.detail)
            self.assertEqual(len(store.feedback_events()), 1)

    def test_invalid_feedback_is_rejected_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            self.assertFalse(service.apply_feedback(FeedbackRequest(record_id="r", outcome="maybe")).ok)
            self.assertFalse(service.apply_feedback(FeedbackRequest(record_id="", outcome="success")).ok)


# -- privacy / data minimization -------------------------------------------------

class PrivacyTests(unittest.TestCase):
    def test_credentials_are_scrubbed_before_persisting(self):
        text = "use password=hunter2 and api_key=sk-abcdefghijklmnop"
        cleaned = sanitize_text(text)
        self.assertNotIn("hunter2", cleaned)
        self.assertNotIn("sk-abcdefghijklmnop", cleaned)
        self.assertIn("[redacted]", cleaned)

    def test_stored_experience_does_not_contain_a_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            context = research_and_deliver_context(
                request="Research it, token=abcdef1234567890"
            )
            service.record_task_outcome(context, record_id="record-1")
            serialized = (Path(directory) / "experiences.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("abcdef1234567890", serialized)

    def test_no_whole_conversation_is_stored(self):
        # The record is a compact task representation: it must not carry a
        # conversation transcript field at all.
        record = build_experience(research_and_deliver_context(), outcome="success")
        self.assertNotIn("conversation", record.to_dict())
        self.assertNotIn("messages", record.to_dict())

    def test_a_corrupt_store_line_does_not_break_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiences.jsonl"
            path.write_text("not json\n" + '{"experience_id": "ok-1", "outcome": "success"}\n', encoding="utf-8")
            store = ExperienceStore(
                experience_path=path, feedback_path=Path(directory) / "feedback.jsonl"
            )
            self.assertEqual(len(store.experiences()), 1)


# -- periodic analysis and safe self-improvement --------------------------------

class AnalysisTests(unittest.TestCase):
    def test_analysis_is_gated_and_never_auto_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            for index in range(4):
                service.record_task_outcome(research_and_deliver_context(), record_id=f"record-{index}")
                service.apply_feedback(
                    FeedbackRequest(
                        record_id=f"record-{index}",
                        outcome="failure",
                        failure_category="delivery_not_completed",
                    )
                )
            report = service.analysis_report()
            self.assertTrue(report["available"])
            self.assertFalse(report["applied"])
            self.assertTrue(any(item["kind"] == "recurring_failure" for item in report["proposals"]))

    def test_analysis_is_not_run_for_every_message(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-0")
            service.apply_feedback(FeedbackRequest(record_id="record-0", outcome="success"))
            self.assertTrue(service.analysis_report()["available"])
            # Immediately afterwards there is nothing new to justify the cost.
            self.assertFalse(service.analysis_report()["available"])

    def test_analysis_does_not_modify_source_or_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-0")
            service.apply_feedback(FeedbackRequest(record_id="record-0", outcome="failure"))
            before = [item.to_dict() for item in store.experiences()]
            service.analysis_report()
            self.assertEqual(before, [item.to_dict() for item in store.experiences()])


# -- retry / recovery trajectory -------------------------------------------------

class RetryTrajectoryTests(unittest.TestCase):
    def test_failure_then_corrected_success_forms_a_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            # Attempt 1: the summary went to chat instead of Notepad.
            first = research_and_deliver_context()
            first.tool_calls = [make_tool_call("web.research", {"content": "resume research"})]
            first.metadata["task_object"].actions = [
                TaskAction(action_id="a1", capability="web.research", parameters={})
            ]
            service.record_task_outcome(first, record_id="attempt-1")
            service.apply_feedback(
                FeedbackRequest(
                    record_id="attempt-1",
                    outcome="failure",
                    failure_category="delivery_not_completed",
                    correction="Put the summary in Notepad.",
                )
            )
            # Attempt 2: the same task, with the correction applied, succeeds.
            service.record_task_outcome(research_and_deliver_context(), record_id="attempt-2")
            service.apply_feedback(FeedbackRequest(record_id="attempt-2", outcome="success"))

            experiences = store.experiences()
            self.assertEqual(len(experiences), 2)
            self.assertEqual(experiences[0].outcome, "failure")
            self.assertEqual(experiences[1].outcome, "success")
            # The correction is retained as the recoverable lesson.
            self.assertEqual(experiences[0].user_correction, "Put the summary in Notepad.")

    def test_retry_attempt_number_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            service, store = service_with_store(directory)
            service.record_task_outcome(research_and_deliver_context(), record_id="record-2", )
            # The builder accepts an explicit attempt number for a retry.
            experience = build_experience(
                research_and_deliver_context(), record_id="record-2", attempt=2
            )
            self.assertEqual(experience.attempt, 2)


# -- behavioural: the whole loop, exercised directly -----------------------------

class EndToEndLoopTests(unittest.TestCase):
    def test_full_loop_improves_the_next_planning_context(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)

            # 1. A meaningful task completes and is recorded.
            service.record_task_outcome(research_and_deliver_context(), record_id="record-1")

            # 2. The user says it failed and explains why.
            result = service.apply_feedback(
                FeedbackRequest(
                    record_id="record-1",
                    outcome="failure",
                    failure_category="did_not_follow_instruction",
                    correction="I specifically asked you to put the summary in Notepad.",
                )
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.failure_category_label, "Did not follow instruction")

            # 3. A comparable later request retrieves that lesson.
            context = service.retrieve_for_planning(
                "Research how to write a resume and put the key points in Notepad."
            )
            self.assertTrue(context.items)
            text = context.text
            self.assertIn("Notepad", text)
            self.assertIn("did_not_follow_instruction", text)
            self.assertIn("I specifically asked you to put the summary in Notepad.", text)

    def test_disabled_loop_is_a_clean_no_op(self):
        service = ExperienceService(enabled=False)
        context = research_and_deliver_context()
        self.assertFalse(service.should_offer_feedback(context))
        self.assertFalse(service.record_task_outcome(context, record_id="r").recorded)
        self.assertFalse(service.retrieve_for_planning("anything").items)
        self.assertFalse(service.apply_feedback(FeedbackRequest(record_id="r", outcome="success")).ok)
        self.assertFalse(service.status()["enabled"])

    def test_recording_never_raises_on_a_malformed_context(self):
        with tempfile.TemporaryDirectory() as directory:
            service, _ = service_with_store(directory)
            outcome = service.record_task_outcome(SimpleNamespace(), record_id="r")
            self.assertFalse(outcome.recorded)

    def test_store_bounds_the_history(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperienceStore(
                experience_path=Path(directory) / "experiences.jsonl",
                feedback_path=Path(directory) / "feedback.jsonl",
                max_records=5,
            )
            for index in range(9):
                store.add(Experience(original_user_request=f"task {index}", outcome="success"))
            self.assertEqual(len(store.experiences()), 5)
            # The most recent records are kept.
            self.assertEqual(store.experiences()[-1].original_user_request, "task 8")


# -- RAG separation --------------------------------------------------------------

class MemorySeparationTests(unittest.TestCase):
    def test_experience_metadata_is_a_distinct_namespace(self):
        from experience.memory import experience_metadata

        record = build_experience(research_and_deliver_context(), outcome="success")
        metadata = experience_metadata(record)
        # Experience must never be indistinguishable from factual knowledge.
        self.assertEqual(metadata["memory_type"], "experience")
        self.assertEqual(metadata["outcome"], "success")
        self.assertEqual(metadata["destination"], "Notepad")

    def test_experience_collection_is_separate_from_knowledge(self):
        from config import COLLECTION_NAME, EXPERIENCE_COLLECTION_NAME

        self.assertNotEqual(COLLECTION_NAME, EXPERIENCE_COLLECTION_NAME)

    def test_embedder_degrades_gracefully_without_chroma(self):
        # An unavailable embedder is a normal state, not an error: retrieval
        # still works on the deterministic lexical signal.
        embedder = ExperienceEmbedder(persist_dir="Z:\\nonexistent-atlas-path")
        self.assertEqual(embedder.similarity_scores("x", ["a"]), {})

    def test_lexical_retrieval_works_without_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperienceStore(
                experience_path=Path(directory) / "experiences.jsonl",
                feedback_path=Path(directory) / "feedback.jsonl",
            )
            store.add(
                Experience(
                    original_user_request="Research how to make a resume and put the key points in Notepad.",
                    outcome="success",
                    tools_used=["web.research", "applications.write_text"],
                    requested_destination="Notepad",
                    desired_outcome="a summary exists in Notepad",
                )
            )
            memory = ExperienceMemory(store=store, embedder=None)
            context = memory.retrieve("Research how to write a resume and put the key points in Notepad.")
            self.assertTrue(context.items)


if __name__ == "__main__":
    unittest.main()


# -- Brain-level integration ----------------------------------------------------
# These exercise the real Brain wiring, proving the loop is connected end to end:
# a meaningful task records an experience, feedback promotes it, and a later
# comparable request retrieves it as planning context.

class BrainLoopIntegrationTests(unittest.TestCase):
    @staticmethod
    def _build_brain(directory: str):
        from brain import Brain
        from executor import Executor
        from planner import Planner
        from tools import ExecutionMode, PermissionEngine, ToolRegistry, ToolRouter
        from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult

        class FakeVectorStore:
            def search(self, *_args, **_kwargs):
                return []

        class FakeTool(Tool):
            def __init__(self, name, output, medium=False):
                self.metadata = ToolMetadata(
                    name=name,
                    description=name,
                    category="content.formatting",
                    permission_level=PermissionLevel.MEDIUM_RISK if medium else PermissionLevel.READ_ONLY,
                    risk_level=RiskLevel.MEDIUM if medium else RiskLevel.READ_ONLY,
                )
                self._output = output
                self.calls = []

            def execute(self, parameters):
                self.calls.append(dict(parameters))
                return ToolResult(success=True, status="completed", output=self._output)

        registry = ToolRegistry()
        for tool in (
            FakeTool("web.research", {"content": "resume tips", "query": "resume"}),
            FakeTool("content.generate", {"text": "summary"}),
            FakeTool("content.format", {"text": "summary", "metadata": {"verification_passed": True, "content_type": "general_prose"}}),
            FakeTool("applications.write_text", {"application": "Notepad", "characters": 7, "observed": True}, medium=True),
        ):
            registry.register(tool)
        router = ToolRouter(registry=registry, permission_engine=PermissionEngine(mode=ExecutionMode.AUTONOMOUS))
        service = ExperienceService(
            store=ExperienceStore(
                experience_path=Path(directory) / "experiences.jsonl",
                feedback_path=Path(directory) / "feedback.jsonl",
            )
        )
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
            experience_service=service,
        )
        return brain, service

    def test_meaningful_task_is_recorded_and_offered_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, service = self._build_brain(directory)
            brain.process("Research resume tips and put the key points in Notepad.")

            context = brain.last_context
            self.assertEqual(context.status, TaskStatus.COMPLETED)
            self.assertTrue(context.metadata.get("experience_recorded"))
            self.assertTrue(context.metadata.get("feedback_eligible"))
            self.assertEqual(len(service.store.experiences()), 1)

    def test_conversation_is_not_recorded_as_a_meaningful_experience(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, service = self._build_brain(directory)
            brain.process("What is quantum computing?")
            # A direct answer has no task outcome, so nothing durable is written
            # for it and no feedback controls appear.
            self.assertFalse(brain.last_context.metadata.get("feedback_eligible", False))

    def test_successful_experience_is_retrieved_on_a_later_similar_request(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, service = self._build_brain(directory)
            brain.process("Research resume tips and put the key points in Notepad.")
            recorded = brain.last_context.metadata["experience_id"]
            service.apply_feedback(FeedbackRequest(record_id="", outcome="success"))

            # A comparable later request must supply retrieved experience context.
            brain.process("Research cover letter tips and put the key points in Notepad.")
            supplied = brain.last_context.metadata.get("experience_context")
            self.assertIsNotNone(supplied)
            self.assertTrue(supplied["items"])
            self.assertIn("Notepad", supplied["text"])
            self.assertTrue(recorded)

    def test_retrieved_experience_ids_are_recorded_on_the_new_experience(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, service = self._build_brain(directory)
            brain.process("Research resume tips and put the key points in Notepad.")
            first_id = brain.last_context.metadata["experience_id"]
            service.apply_feedback(
                FeedbackRequest(record_id="", outcome="success")
            )
            brain.process("Research cover letter tips and put the key points in Notepad.")
            second_id = brain.last_context.metadata["experience_id"]
            second = service.store.find_by_id(second_id)
            self.assertIsNotNone(second)
            self.assertIn(first_id, second.retrieved_experience_ids)

    def test_disabled_experience_service_leaves_brain_working(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, _ = self._build_brain(directory)
            brain._experience = ExperienceService(enabled=False)
            brain.process("Research resume tips and put the key points in Notepad.")
            self.assertEqual(brain.last_context.status, TaskStatus.COMPLETED)
            self.assertFalse(brain.last_context.metadata.get("experience_recorded", False))
