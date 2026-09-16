"""Regression tests for web search robustness and application resolution.

These cover the concrete bugs found while tuning the real runtime:

* YouTube result pages exceeded the small read cap and were truncated, which
  silently produced zero results for any YouTube query.
* DuckDuckGo serves an anti-bot challenge page; the tool must treat that as
  \"no results\" and fall back instead of returning junk.
* Search fallbacks can return stale/unrelated pages; those must be filtered.
* `notepad`/`calculator`/etc. live in System32 or per-user Programs folders and
  were not resolvable by name.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
from computer.launch import _APPLICATION_ALIASES, resolve_application_name  # noqa: F401
from web import (
    WebResult,
    WebSearchTool,
    _filter_relevant,
    _relevance_score,
    search_bing_rss,
    search_wikipedia,
)


class FakeResponse:
    def __init__(self, body: bytes | str, url: str = "https://example.com"):
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self.headers = type("Headers", (), {"get_content_type": lambda self: "text/html"})()
        self._url = url

    def read(self, limit=None):
        return self._body if limit is None else self._body[:limit]

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class YouTubeTruncationTests(unittest.TestCase):
    def test_large_youtube_payload_is_fully_parsed(self):
        # A payload far larger than the old 1MB read cap, with the real video
        # data placed at the end so truncation would have hidden it.
        padding = "x" * 2_000_000
        body = (
            "<html>" + padding +
            'var ytInitialData = {"contents":{"videoRenderer":{"videoId":"abc123",'
            '"title":{"simpleText":"Big Payload Video"},"ownerText":{"simpleText":"Chan"}}}};'
        )
        from web import search_youtube
        results = search_youtube("anything", limit=1, opener=lambda *_a, **_k: FakeResponse(body))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://www.youtube.com/watch?v=abc123")


class DuckDuckGoChallengeTests(unittest.TestCase):
    def test_challenge_page_yields_no_results(self):
        challenge = "<html><body>anomaly detected</body></html>"
        from web import search_duckduckgo
        results = search_duckduckgo("anything", opener=lambda *_a, **_k: FakeResponse(challenge))
        self.assertEqual(results, [])


class RelevanceFilterTests(unittest.TestCase):
    def test_unrelated_results_are_dropped(self):
        results = [
            WebResult("Government portal", "https://gov.example", "unrelated"),
            WebResult("PewDiePie biography", "https://wiki.example/pewdiepie", "Felix"),
        ]
        kept = _filter_relevant(results, "best pewdiepie")

        self.assertEqual([r.title for r in kept], ["PewDiePie biography"])

    def test_distinctive_term_scores_higher_than_generic_term(self):
        generic = WebResult("Python downloads", "https://python.org", "python")
        specific = WebResult(
            "Asyncio tutorial", "https://example.com/asyncio", "learn python asyncio"
        )

        self.assertGreater(
            _relevance_score(specific, "python asyncio tutorial"),
            _relevance_score(generic, "python asyncio tutorial"),
        )

    def test_all_irrelevant_results_produce_empty_output(self):
        with patch("web.search_duckduckgo", return_value=[]), patch(
            "web.search_bing_rss",
            return_value=[WebResult("Stuff", "https://example.com", "nothing relevant")],
        ), patch("web.search_wikipedia", return_value=[]):
            result = WebSearchTool().execute({"query": "best pewdiepie", "max_results": 5})

        self.assertTrue(result.success)
        self.assertEqual(result.output["results"], [])


class WikipediaBackendTests(unittest.TestCase):
    def test_wikipedia_api_results_are_parsed(self):
        body = (
            '{"query":{"search":[{"title":"PewDiePie","snippet":"<b>Felix</b> Kjellberg"}]}}'
        )
        results = search_wikipedia("pewdiepie", limit=5, opener=lambda *_a, **_k: FakeResponse(body))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "PewDiePie")
        self.assertEqual(results[0].url, "https://en.wikipedia.org/wiki/PewDiePie")
        self.assertEqual(results[0].source, "Wikipedia")

    def test_malformed_rss_returns_no_results(self):
        results = search_bing_rss("query", opener=lambda *_a, **_k: FakeResponse("<html>error</html>"))
        self.assertEqual(results, [])


class ApplicationResolutionTests(unittest.TestCase):
    def test_windows_system_application_resolves(self):
        self.assertIsNotNone(resolve_application_name("notepad"))
        self.assertIsNotNone(resolve_application_name("calculator"))

    def test_alias_table_covers_common_friendly_names(self):
        for name in ("notepad", "calculator", "vscode", "chrome", "explorer"):
            with self.subTest(name=name):
                self.assertIn(name, _APPLICATION_ALIASES)

    def test_unknown_application_does_not_resolve(self):
        self.assertIsNone(resolve_application_name("definitely-not-installed-xyz"))

    def test_resolved_paths_are_existing_executables(self):
        resolved = resolve_application_name("notepad")
        self.assertIsNotNone(resolved)
        self.assertTrue(Path(resolved).is_file())
        self.assertEqual(Path(resolved).suffix.casefold(), ".exe")


if __name__ == "__main__":
    unittest.main()
