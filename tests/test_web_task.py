"""Tests for task-aware retrieval modelling (web_task).

The retrieval pipeline must distinguish "the thing itself" from "information
about the thing", pick a source accordingly, validate the extracted content, and
reformulate the query from the missing content type when validation fails.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web_task import (  # noqa: E402
    detect_source_type,
    infer_retrieval_task,
    llm_validate_content,
    reformulate_query,
    score_source,
    validate_content,
)


SCRIPT_PAGE = {
    "url": "https://imsdb.com/scripts/Bee-Movie.html",
    "title": "Bee Movie Script",
    "text": "BARRY: You like jazz? VANESSA: You're a bee! BARRY: I'm a bee. KEN: Close the window. " * 60,
}
WIKI_PAGE = {
    "url": "https://en.wikipedia.org/wiki/Bee_Movie",
    "title": "Bee Movie - Wikipedia",
    "text": (
        "Bee Movie is a 2007 American animated comedy film. It was directed by "
        "Simon Smith. The film stars Jerry Seinfeld. It was released in 2007 and "
        "received mixed reviews. The plot concerns a bee who sues humans."
    ),
}


class TaskInferenceTests(unittest.TestCase):
    def test_artifact_request_is_detected(self):
        task = infer_retrieval_task(
            "search the web for bee movie script and copy it in notepad",
            entities={"application": "notepad"},
        )
        self.assertEqual(task.goal, "retrieve_document")
        self.assertEqual(task.content_type, "movie_script")
        self.assertTrue(task.must_be_artifact)
        self.assertEqual(task.desired_output, "full_text")
        self.assertEqual(task.destination, "notepad")

    def test_information_request_is_not_an_artifact_request(self):
        task = infer_retrieval_task("find information about the bee movie")
        self.assertEqual(task.goal, "find_information")
        self.assertFalse(task.must_be_artifact)
        self.assertEqual(task.desired_output, "summary")

    def test_wikipedia_page_request_targets_the_reference_page(self):
        task = infer_retrieval_task("find the bee movie wikipedia page")
        self.assertEqual(task.goal, "find_reference")
        self.assertEqual(task.content_type, "reference")

    def test_review_request(self):
        task = infer_retrieval_task("find a review of bee movie")
        self.assertEqual(task.goal, "find_review")
        self.assertEqual(task.content_type, "review")

    def test_trailer_request_is_media(self):
        task = infer_retrieval_task("find the bee movie trailer")
        self.assertEqual(task.goal, "find_media")
        self.assertEqual(task.content_type, "video")

    def test_youtube_videos_vs_youtube_information(self):
        media = infer_retrieval_task("search youtube for popular car videos", entities={"site": "youtube"})
        self.assertEqual(media.goal, "find_media")
        info = infer_retrieval_task("search youtube for information about cars", entities={"site": "youtube"})
        self.assertEqual(info.goal, "find_information")

    def test_artifact_request_seeds_artifact_queries(self):
        task = infer_retrieval_task(
            "bee movie script", entities={"topic": "bee movie"}, must_be_artifact=True
        )
        self.assertTrue(task.must_be_artifact)
        self.assertTrue(any("script" in query for query in task.queries))


class ContentTypeDetectionTests(unittest.TestCase):
    def test_wikipedia_detected_as_reference(self):
        self.assertEqual(detect_source_type(WIKI_PAGE["url"], WIKI_PAGE["title"], WIKI_PAGE["text"]), "reference")

    def test_script_page_detected_as_a_script_family_type(self):
        # A page with dialogue is detected as a script-family type; transcript and
        # movie_script are treated as the same family for matching purposes.
        detected = detect_source_type(SCRIPT_PAGE["url"], SCRIPT_PAGE["title"], SCRIPT_PAGE["text"])
        self.assertIn(detected, {"movie_script", "transcript"})

    def test_dialogue_without_script_title_detected_as_transcript(self):
        dialogue = ("BARRY: You like jazz? VANESSA: You're a bee! KEN: Close the window. " * 20)
        self.assertEqual(detect_source_type("https://ex/page", "Some Page", dialogue), "transcript")

    def test_prose_page_is_generic(self):
        prose = "This page explains how something works in clear plain sentences. " * 20
        self.assertEqual(detect_source_type("https://ex/page", "Some Page", prose), "generic")

    def test_youtube_detected_as_video(self):
        self.assertEqual(
            detect_source_type("https://www.youtube.com/watch?v=x", "Bee Movie Trailer", "watch"), "video"
        )

    def test_review_title_detected_as_review(self):
        self.assertEqual(
            detect_source_type("https://example.com/x", "Bee Movie review", "text"), "review"
        )


class SourceScoringTests(unittest.TestCase):
    def test_wikipedia_scores_low_when_script_requested(self):
        task = infer_retrieval_task("find the bee movie script")
        score = score_source(task, url=WIKI_PAGE["url"], title=WIKI_PAGE["title"], text=WIKI_PAGE["text"])
        self.assertFalse(score.accepted)
        self.assertLess(score.score, 0.35)

    def test_script_page_scores_high_when_script_requested(self):
        task = infer_retrieval_task("find the bee movie script")
        score = score_source(task, url=SCRIPT_PAGE["url"], title=SCRIPT_PAGE["title"], text=SCRIPT_PAGE["text"])
        self.assertTrue(score.accepted)
        self.assertGreater(score.score, 0.5)

    def test_wikipedia_scores_high_for_reference_request(self):
        task = infer_retrieval_task("find the bee movie wikipedia page")
        score = score_source(task, url=WIKI_PAGE["url"], title=WIKI_PAGE["title"], text=WIKI_PAGE["text"])
        self.assertTrue(score.accepted)

    def test_wikipedia_is_acceptable_for_information_request(self):
        task = infer_retrieval_task("find information about the bee movie")
        score = score_source(task, url=WIKI_PAGE["url"], title=WIKI_PAGE["title"], text=WIKI_PAGE["text"])
        self.assertTrue(score.accepted)


SCRIBD_PAGE = {
    "url": "https://www.scribd.com/document/123/Bee-Movie-Script",
    "title": "Bee Movie Script: Full Transcript | PDF | Bees",
    "text": (
        "0 ratings 0% found this document useful (0 votes) 1K views 11 pages "
        "Bee Movie Script: Full Transcript This document is a transcript of dialogue "
        "from the movie Bee Movie. Uploaded by Mischelle Beerbaum "
    ) * 6,
}

class DocumentHostTests(unittest.TestCase):
    # A document-store landing page advertises the document; it is not the
    # document, even when its title names the artifact.

    def test_document_host_detected_by_host(self):
        self.assertEqual(
            detect_source_type(SCRIBD_PAGE["url"], SCRIBD_PAGE["title"], SCRIBD_PAGE["text"]),
            "document_host",
        )

    def test_listing_detected_by_card_chrome_without_the_domain(self):
        self.assertEqual(
            detect_source_type("https://random.example/x", SCRIBD_PAGE["title"], SCRIBD_PAGE["text"]),
            "document_host",
        )

    def test_real_script_is_not_flagged_as_a_document_host(self):
        self.assertNotIn(
            detect_source_type(SCRIPT_PAGE["url"], SCRIPT_PAGE["title"], SCRIPT_PAGE["text"]),
            {"document_host", "listing"},
        )

    def test_document_host_scores_zero_for_an_artifact_request(self):
        task = infer_retrieval_task("find the bee movie script")
        score = score_source(task, url=SCRIBD_PAGE["url"], title=SCRIBD_PAGE["title"], text=SCRIBD_PAGE["text"])
        self.assertFalse(score.accepted)
        self.assertEqual(score.detected_type, "document_host")

    def test_document_host_content_is_rejected_by_validation(self):
        task = infer_retrieval_task("find the bee movie script")
        result = validate_content(
            task, content=SCRIBD_PAGE["text"], detected_type="document_host", title=SCRIBD_PAGE["title"]
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.action, "search_again")

class ValidationTests(unittest.TestCase):
    def test_descriptive_page_rejected_when_artifact_wanted(self):
        task = infer_retrieval_task("find the bee movie script")
        result = validate_content(task, content=WIKI_PAGE["text"], detected_type="reference", title="Bee Movie - Wikipedia")
        self.assertFalse(result.ok)
        self.assertEqual(result.action, "search_again")

    def test_full_script_accepted_when_artifact_wanted(self):
        task = infer_retrieval_task("find the bee movie script")
        result = validate_content(task, content=SCRIPT_PAGE["text"], detected_type="transcript", title="Bee Movie Script")
        self.assertTrue(result.ok)
        self.assertEqual(result.action, "extract")

    def test_information_request_accepts_reference_page(self):
        task = infer_retrieval_task("find information about the bee movie")
        result = validate_content(task, content=WIKI_PAGE["text"], detected_type="reference", title="Bee Movie - Wikipedia")
        self.assertTrue(result.ok)

    def test_empty_content_is_rejected(self):
        task = infer_retrieval_task("find the bee movie script")
        result = validate_content(task, content="", detected_type="transcript")
        self.assertFalse(result.ok)


class ReformulationTests(unittest.TestCase):
    def test_reformulation_targets_missing_content_type(self):
        task = infer_retrieval_task("find the bee movie script")
        improved = reformulate_query(task, reason="the page describes the film", attempt=0)
        self.assertIn("script", improved.casefold())
        self.assertIn("bee movie", improved.casefold())

    def test_reformulation_for_review(self):
        task = infer_retrieval_task("find a review of bee movie")
        improved = reformulate_query(task, reason="not a review", attempt=0)
        self.assertIn("review", improved.casefold())


class LLMValidationTests(unittest.TestCase):
    def test_llm_acceptance_is_honoured(self):
        task = infer_retrieval_task("find the bee movie script")
        fake = lambda **_: '{"match": true, "content_type_match": true, "contains_target": true, "completeness": 0.9, "confidence": 0.9, "action": "extract", "reason": "contains the script"}'
        result = llm_validate_content(
            task, content=SCRIPT_PAGE["text"], detected_type="transcript",
            title="Bee Movie Script", ask=fake,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.ok)
        self.assertEqual(result.source, "llm")

    def test_llm_rejection_is_honoured(self):
        task = infer_retrieval_task("find the bee movie script")
        fake = lambda **_: '{"match": false, "action": "search_again", "reason": "describes the film"}'
        result = llm_validate_content(
            task, content=WIKI_PAGE["text"], detected_type="reference",
            title="Bee Movie - Wikipedia", ask=fake,
        )
        self.assertIsNotNone(result)
        self.assertFalse(result.ok)
        self.assertEqual(result.action, "search_again")

    def test_unusable_model_output_returns_none(self):
        task = infer_retrieval_task("find the bee movie script")
        result = llm_validate_content(
            task, content="x", detected_type="transcript", ask=lambda **_: "not json"
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
