"""Tests for RAG-style web retrieval (web_research).

A hybrid web task must read several candidate pages and isolate the content that
matches the request, rather than returning the first page or the links.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch  # noqa: E402
from web import WebResearchTool, WebResult  # noqa: E402
from web_research import (  # noqa: E402
    ResearchResult,
    chunk_text,
    query_terms,
    rank_chunks,
    research,
    score_chunk,
)


class QueryTermTests(unittest.TestCase):
    def test_task_words_are_not_query_terms(self):
        terms = query_terms("search the web for the skyrim script and copy it in notepad")
        self.assertIn("skyrim", terms)
        self.assertIn("script", terms)
        for noise in ("search", "web", "copy", "notepad", "the"):
            self.assertNotIn(noise, terms)

    def test_longer_terms_rank_first(self):
        terms = query_terms("nvidia gpu architecture")
        self.assertEqual(terms[0], "architecture")


class ChunkTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("short body"), ["short body"])

    def test_long_text_is_split_with_overlap(self):
        chunks = chunk_text("word " * 600)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_score_rewards_distinct_term_coverage(self):
        terms = query_terms("skyrim script")
        on_topic = score_chunk("The full Skyrim script for the opening scene.", terms)
        off_topic = score_chunk("A trailer for an unrelated racing game.", terms)
        self.assertGreater(on_topic, off_topic)


class ResearchTests(unittest.TestCase):
    def _search(self, *results):
        def search(query, limit):
            return {"query": query, "results": list(results)}

        return search

    def test_relevant_chunk_is_selected_over_the_top_result(self):
        # The top search hit is a trailer page (irrelevant); the third page holds
        # the actual script. Research must pick the script content, not page one.
        pages = {
            "https://ex/trailer": {"url": "https://ex/trailer", "title": "Skyrim Trailer", "text": "Watch the awesome trailer. Buy now."},
            "https://ex/news": {"url": "https://ex/news", "title": "Skyrim News", "text": "Release date announced for the game."},
            "https://ex/script": {"url": "https://ex/script", "title": "Skyrim Script", "text": "SKYRIM SCRIPT: the full script of the elder scrolls v skyrim opening."},
        }

        def fetch(url):
            return pages.get(url)

        outcome = research(
            "skyrim script",
            search=self._search(
                {"title": "Skyrim Trailer", "url": "https://ex/trailer"},
                {"title": "Skyrim News", "url": "https://ex/news"},
                {"title": "Skyrim Script", "url": "https://ex/script"},
            ),
            fetch=fetch,
        )
        self.assertIn("the full script", outcome.content.lower())
        self.assertNotIn("trailer", outcome.content.lower())
        self.assertIn("https://ex/script", outcome.sources)
        self.assertEqual(outcome.pages_read, 3)

    def test_no_results_is_reported_honestly(self):
        outcome = research("obscure thing", search=self._search(), fetch=lambda url: None)
        self.assertEqual(outcome.content, "")
        self.assertTrue(outcome.notes)

    def test_failed_fetches_do_not_produce_content(self):
        outcome = research(
            "skyrim script",
            search=self._search({"title": "X", "url": "https://ex/x"}),
            fetch=lambda url: None,
        )
        self.assertEqual(outcome.content, "")
        self.assertTrue(any("read" in note for note in outcome.notes))

    def test_content_is_ordered_by_reading_position(self):
        page = {
            "url": "https://ex/p",
            "title": "Script",
            "text": ("intro words " * 40) + " the skyrim script here " + ("tail " * 40),
        }
        outcome = research("skyrim script", search=self._search({"title": "Script", "url": "https://ex/p"}), fetch=lambda url: page)
        # The selected content comes from the matching region of the page.
        self.assertIn("skyrim script", outcome.content)


class WebResearchToolTests(unittest.TestCase):
    # Exercise the real tool wiring with only the network layer patched. Using a
    # fake search callable previously hid a bug where the tool referenced a
    # method that lived on a different class.

    def test_tool_searches_reads_and_returns_relevant_content(self):
        results = [
            WebResult("Skyrim Trailer", "https://ex/trailer", "watch the trailer"),
            WebResult("Skyrim Script", "https://ex/script", "the skyrim script"),
        ]
        script_line = "DRAGONBORN: I am the dragonborn. GUARD: Let me guess, someone stole your sweetroll. "
        pages = {
            "https://ex/trailer": {
                "url": "https://ex/trailer", "title": "Skyrim Trailer",
                "text": "Watch the awesome trailer for the game. Buy now.",
            },
            "https://ex/script": {
                "url": "https://ex/script", "title": "Skyrim Script",
                "text": "SKYRIM SCRIPT: the full script of the elder scrolls v skyrim. " + script_line * 80,
            },
        }
        with patch("web.search_duckduckgo", return_value=results), patch(
            "web.search_bing_rss", return_value=[]
        ), patch("web.search_wikipedia", return_value=[]), patch(
            "web.fetch_public_page", side_effect=lambda url, **_kw: pages.get(url)
        ):
            result = WebResearchTool().execute(
                {"query": "skyrim script", "max_pages": 4, "must_be_artifact": True,
                 "content_type": "movie_script", "target": "skyrim"}
            )

        self.assertTrue(result.success, result.error)
        self.assertIn("dragonborn", result.output["content"].lower())
        self.assertNotIn("trailer", result.output["content"].lower())
        self.assertIn("https://ex/script", result.output["sources"])
        self.assertTrue(result.output["validated"])

    def test_tool_reports_failure_when_nothing_readable(self):
        with patch("web.search_duckduckgo", return_value=[]), patch(
            "web.search_bing_rss", return_value=[]
        ), patch("web.search_wikipedia", return_value=[]):
            result = WebResearchTool().execute({"query": "nothing at all"})
        self.assertFalse(result.success)

class RankChunkTests(unittest.TestCase):
    def test_rank_orders_by_score(self):
        pages = [
            {"url": "a", "title": "A", "text": "irrelevant filler content about nothing"},
            {"url": "b", "title": "B", "text": "the skyrim script is here in full"},
        ]
        ranked = rank_chunks(pages, query="skyrim script")
        self.assertEqual(ranked[0].url, "b")
        self.assertGreater(ranked[0].score, ranked[-1].score)


# ---------------------------------------------------------------------------
# Task-aware pipeline: the acceptance cases (A-F, H, retry)
# ---------------------------------------------------------------------------

BEE_SCRIPT = {
    "url": "https://imsdb.com/scripts/Bee-Movie.html",
    "title": "Bee Movie Script",
    "text": (
        "BARRY: You like jazz? VANESSA: You're a bee! BARRY: I'm a bee. "
        "KEN: Could you close the window please? BARRY: Check out my new resume. "
        "VANESSA: Folds out. BARRY: More humans. "
    ) * 100,
}
BEE_WIKI = {
    "url": "https://en.wikipedia.org/wiki/Bee_Movie",
    "title": "Bee Movie - Wikipedia",
    "text": (
        "Bee Movie is a 2007 American animated comedy film. It was directed by "
        "Simon Smith and stars Jerry Seinfeld. It was released in 2007 and received "
        "mixed reviews. The plot concerns a bee who sues humans for honey. "
    ),
}
BEE_REVIEW = {
    "url": "https://www.rogerebert.com/reviews/bee-movie",
    "title": "Bee Movie movie review",
    "text": "Bee Movie review: a lively animated comedy. Our review praises the voice cast and humor. Rating: 3 stars. " * 12,
}
BEE_TRAILER = {
    "url": "https://www.youtube.com/watch?v=bee",
    "title": "Bee Movie Official Trailer",
    "text": "Watch the official trailer for Bee Movie. " * 12,
}


def _search_returning(*pages):
    def search(query, limit, youtube_query=False):
        return [WebResult(page["title"], page["url"], "snippet") for page in pages]
    return search


def _fetch_from(*pages):
    index = {page["url"]: page for page in pages}
    return lambda url, **_kw: index.get(url)


class TaskAwarePipelineTests(unittest.TestCase):
    def _run(self, query, pages, *, params=None, ask=None):
        with patch("web.search_duckduckgo", side_effect=_search_returning(*pages)), patch(
            "web.search_bing_rss", return_value=[]
        ), patch("web.search_wikipedia", return_value=[]), patch(
            "web.search_youtube", side_effect=_search_returning(*pages)
        ), patch("web.fetch_public_page", side_effect=_fetch_from(*pages)):
            return WebResearchTool(ask=ask).execute({"query": query, **(params or {})})

    def test_A_artifact_request_selects_the_script_not_the_wikipedia_page(self):
        result = self._run(
            "bee movie script",
            [BEE_WIKI, BEE_SCRIPT],
            params={"must_be_artifact": True, "content_type": "movie_script", "target": "bee movie"},
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output["sources"], [BEE_SCRIPT["url"]])
        self.assertIn("jazz", result.output["content"].lower())
        self.assertNotIn("animated comedy film", result.output["content"].lower())

    def test_B_information_request_accepts_the_wikipedia_page(self):
        result = self._run("information about the bee movie", [BEE_WIKI], params={"goal": "find_information"})
        self.assertTrue(result.success, result.error)
        self.assertIn(BEE_WIKI["url"], result.output["sources"])

    def test_C_wikipedia_page_request_accepts_the_reference_page(self):
        result = self._run("bee movie wikipedia page", [BEE_WIKI], params={"goal": "find_reference", "content_type": "reference"})
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output["detected_type"], "reference")

    def test_D_script_request_prefers_the_script_page(self):
        result = self._run(
            "bee movie script",
            [BEE_WIKI, BEE_SCRIPT],
            params={"must_be_artifact": True, "content_type": "movie_script", "target": "bee movie"},
        )
        self.assertTrue(result.success, result.error)
        self.assertIn("jazz", result.output["content"].lower())

    def test_D_fails_honestly_when_only_a_describing_page_exists(self):
        result = self._run(
            "bee movie script",
            [BEE_WIKI],
            params={"must_be_artifact": True, "content_type": "movie_script", "target": "bee movie"},
        )
        self.assertFalse(result.success)
        self.assertIn("Could not obtain", result.error)

    def test_E_review_request_selects_review_content(self):
        result = self._run("bee movie review", [BEE_REVIEW], params={"goal": "find_review", "content_type": "review"})
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output["detected_type"], "review")

    def test_F_trailer_request_selects_video_source(self):
        result = self._run("bee movie trailer", [BEE_TRAILER], params={"goal": "find_media", "content_type": "video"})
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output["detected_type"], "video")

    def test_H_youtube_video_search_returns_media(self):
        video = {"url": "https://www.youtube.com/watch?v=car", "title": "Popular car video", "text": "a video about cars " * 20}
        result = self._run("popular car videos", [video], params={"site": "youtube", "goal": "find_media"})
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output["detected_type"], "video")

    def test_S_scribd_listing_is_not_selected_over_the_real_script(self):
        scribd = {
            "url": "https://www.scribd.com/document/123/Bee-Movie-Script",
            "title": "Bee Movie Script: Full Transcript | PDF | Bees",
            "text": (
                "0 ratings 0% found this document useful (0 votes) 1K views 11 pages "
                "Bee Movie Script: Full Transcript This document is a transcript of "
                "dialogue from the movie Bee Movie. Uploaded by Mischelle Beerbaum "
            ) * 6,
        }
        # The listing page ranks first in the results every time.
        result = self._run(
            "bee movie script",
            [scribd, BEE_SCRIPT],
            params={"must_be_artifact": True, "content_type": "movie_script", "target": "bee movie"},
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output["sources"], [BEE_SCRIPT["url"]])
        self.assertIn("jazz", result.output["content"].lower())
        self.assertNotIn("found this document useful", result.output["content"].lower())

    def test_S_fails_honestly_when_only_a_listing_page_exists(self):
        scribd = {
            "url": "https://www.scribd.com/document/123/Bee-Movie-Script",
            "title": "Bee Movie Script: Full Transcript",
            "text": (
                "0 ratings 0% found this document useful (0 votes) 1K views 11 pages "
                "This document is a transcript of dialogue from the movie Bee Movie. "
                "Uploaded by Mischelle Beerbaum "
            ) * 6,
        }
        result = self._run(
            "bee movie script",
            [scribd],
            params={"must_be_artifact": True, "content_type": "movie_script", "target": "bee movie"},
        )
        self.assertFalse(result.success)
        self.assertIn("Could not obtain", result.error)

    def test_retry_reformulates_the_query_when_the_first_attempt_fails(self):
        seen = []
        # Only a *reformulated* query (the screenplay variant) finds the script;
        # the seed query returns the describing page so the first attempt fails.
        def search(query, limit, youtube_query=False):
            seen.append(query)
            # The first search returns the describing page; a later attempt (the
            # reformulated query) finds the script, so the retry must happen.
            if len(seen) == 1:
                return [WebResult(BEE_WIKI["title"], BEE_WIKI["url"], "film")]
            return [WebResult(BEE_SCRIPT["title"], BEE_SCRIPT["url"], "script")]

        pages = {BEE_WIKI["url"]: BEE_WIKI, BEE_SCRIPT["url"]: BEE_SCRIPT}
        with patch("web.search_duckduckgo", side_effect=search), patch(
            "web.search_bing_rss", return_value=[]
        ), patch("web.search_wikipedia", return_value=[]), patch(
            "web.fetch_public_page", side_effect=lambda url, **_kw: pages.get(url)
        ):
            result = WebResearchTool().execute(
                {"query": "bee movie script", "must_be_artifact": True,
                 "content_type": "movie_script", "target": "bee movie"}
            )

        self.assertTrue(result.success, result.error)
        self.assertGreaterEqual(result.output["attempts"], 2)
        self.assertGreaterEqual(len(seen), 2)
        self.assertTrue(any("script" in q for q in seen), seen)


if __name__ == "__main__":
    unittest.main()
