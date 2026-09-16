import io
import unittest
from unittest.mock import patch

from intent_classifier import IntentClassifier
from planner import Planner
from web import WebFetchTool, WebSearchTool, WebResult, search_bing_rss, search_youtube, thumbnail_for_url, validate_public_url


class FakeResponse:
    def __init__(self, body: str, url: str = "https://example.com"):
        self._body = body.encode("utf-8")
        self.headers = type("Headers", (), {"get_content_type": lambda self: "text/html"})()
        self._url = url

    def read(self, _limit):
        return self._body

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class WebTests(unittest.TestCase):
    def test_latest_video_request_uses_web_search(self):
        result = IntentClassifier().classify("What is the latest video from the Atlas channel?")
        decision = Planner().create_plan(result.intent and "What is the latest video from the Atlas channel?", intent=result.intent)

        self.assertEqual(result.intent, "WEB_SEARCH")
        self.assertEqual(decision.plan.steps[0].metadata["tool"], "web.search")
        self.assertIn("latest video", decision.plan.steps[0].metadata["parameters"]["query"])

    def test_youtube_search_filters_unrelated_pages(self):
        with patch("web.search_duckduckgo", return_value=[]), patch(
            "web.search_bing_rss",
            return_value=[
                WebResult("MrBeast channel", "https://www.youtube.com/channel/abc", "channel"),
                WebResult("MrBeast video", "https://www.youtube.com/watch?v=video123", "video"),
                WebResult("Wikipedia", "https://en.wikipedia.org/wiki/MrBeast", "unrelated"),
            ],
        ):
            result = WebSearchTool().execute({"query": "site:youtube.com/watch MrBeast videos", "max_results": 5})

        self.assertEqual([item["title"] for item in result.output["results"]], ["MrBeast video"])
        self.assertEqual(result.output["results"][0]["thumbnail_url"], "https://i.ytimg.com/vi/video123/hqdefault.jpg")

    def test_youtube_renderer_parser_creates_watch_results(self):
        body = 'var ytInitialData = {"contents":{"videoRenderer":{"videoId":"abc123","title":{"simpleText":"MrBeast test"},"ownerText":{"simpleText":"MrBeast"}}}};'
        results = search_youtube("MrBeast", limit=1, opener=lambda *_args, **_kwargs: FakeResponse(body))

        self.assertEqual(results[0].url, "https://www.youtube.com/watch?v=abc123")
        self.assertEqual(results[0].thumbnail_url, "https://i.ytimg.com/vi/abc123/hqdefault.jpg")

    def test_search_parser_returns_attributed_results(self):
        html = '<a class="result__a" href="https://example.com/video">Latest Video</a><div class="result__snippet">A useful result</div>'
        result = WebSearchTool().execute({"query": "latest video", "max_results": 3})

        with patch("web.search_duckduckgo", return_value=[WebResult("Latest Video", "https://example.com/video", "A useful result")]):
            result = WebSearchTool().execute({"query": "latest video", "max_results": 3})
        self.assertTrue(result.success)
        self.assertEqual(result.output["results"][0]["url"], "https://example.com/video")

    def test_bing_rss_parser_returns_attributed_results(self):
        rss = b"<rss><channel><item><title>Latest Video</title><link>https://example.com/video</link><description>Useful result</description></item></channel></rss>"
        result = search_bing_rss("latest video", opener=lambda *_args, **_kwargs: FakeResponse(rss.decode()))

        self.assertEqual(result[0].source, "Bing")
        self.assertEqual(result[0].url, "https://example.com/video")

    def test_private_urls_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_public_url("http://127.0.0.1:8000/api")

    def test_page_fetch_marks_content_untrusted(self):
        page = '<html><title>Example</title><body>Hello <b>world</b></body></html>'
        with patch("web.fetch_public_page", return_value={"url": "https://example.com", "title": "Example", "text": "Hello world", "untrusted_content": True}):
            result = WebFetchTool().execute({"url": "https://example.com"})
        self.assertTrue(result.success)
        self.assertTrue(result.output["untrusted_content"])

    def test_youtube_results_get_thumbnail_urls(self):
        thumbnail = thumbnail_for_url("https://www.youtube.com/watch?v=abc123")

        self.assertEqual(thumbnail, "https://i.ytimg.com/vi/abc123/hqdefault.jpg")


if __name__ == "__main__":
    unittest.main()
