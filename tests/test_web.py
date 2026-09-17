import socket
import unittest
import urllib.request
from unittest.mock import Mock, patch

from intent_classifier import IntentClassifier
from planner import Planner
from web import (
    WebFetchTool, WebSearchTool, WebResult, _PublicRedirectHandler, _open_public,
    fetch_public_page, search_bing_rss, search_youtube, thumbnail_for_url, validate_public_url,
)


def resolved_addresses(*addresses):
    return [
        (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))
        for address in addresses
    ]


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
        with patch("web.search_youtube", return_value=[]), patch(
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
        with patch("web.search_duckduckgo", return_value=[WebResult("Latest Video", "https://example.com/video", "A useful result")]), patch(
            "web.search_bing_rss", return_value=[]
        ), patch("web.search_wikipedia", return_value=[]):
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
        with patch("web.socket.getaddrinfo", return_value=resolved_addresses("8.8.8.8")), patch(
            "web.fetch_public_page", return_value={"url": "https://example.com", "title": "Example", "text": "Hello world", "untrusted_content": True}
        ):
            result = WebFetchTool().execute({"url": "https://example.com"})
        self.assertTrue(result.success)
        self.assertTrue(result.output["untrusted_content"])

    def test_youtube_results_get_thumbnail_urls(self):
        thumbnail = thumbnail_for_url("https://www.youtube.com/watch?v=abc123")

        self.assertEqual(thumbnail, "https://i.ytimg.com/vi/abc123/hqdefault.jpg")


class PublicWebBoundaryTests(unittest.TestCase):
    def test_each_failed_provider_leaves_other_results_available(self):
        names = ("search_duckduckgo", "search_bing_rss", "search_wikipedia")
        for failed in names:
            with self.subTest(provider=failed), patch("web.search_duckduckgo") as duck, patch(
                "web.search_bing_rss"
            ) as bing, patch("web.search_wikipedia") as wiki:
                for name, provider in zip(names, (duck, bing, wiki)):
                    provider.return_value = [WebResult("Atlas guide", f"https://example.com/{name}", "Atlas")]
                    if name == failed:
                        provider.side_effect = RuntimeError("provider unavailable")
                result = WebSearchTool().execute({"query": "Atlas"})
                self.assertTrue(result.success)
                self.assertEqual(result.output["result_count"], 2)
                self.assertTrue(result.metadata["untrusted_content"])
                for provider in (duck, bing, wiki):
                    provider.assert_called_once()

    def test_youtube_failure_uses_fallback(self):
        with patch("web.search_youtube", side_effect=TimeoutError), patch(
            "web.search_bing_rss", return_value=[WebResult("Atlas", "https://youtube.com/watch?v=abc", "Atlas")]
        ):
            result = WebSearchTool().execute({"query": "Atlas", "site": "youtube"})
        self.assertTrue(result.success)
        self.assertEqual(result.output["result_count"], 1)

    def test_public_ipv4_and_ipv6_answers_are_accepted(self):
        with patch("web.socket.getaddrinfo", return_value=resolved_addresses("8.8.8.8", "2606:4700:4700::1111")) as resolver:
            validate_public_url("https://public.example/page")
        resolver.assert_called_once_with("public.example", None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)

    def test_any_nonpublic_dns_answer_is_rejected(self):
        for address in ("10.0.0.1", "::1", "fc00::1", "fe80::1", "100.64.0.1", "224.0.0.1", "::"):
            for answers in (("8.8.8.8", address), (address, "8.8.8.8")):
                with self.subTest(answers=answers), patch("web.socket.getaddrinfo", return_value=resolved_addresses(*answers)):
                    with self.assertRaises(ValueError):
                        validate_public_url("https://public.example/page")

    def test_empty_and_failed_dns_are_rejected(self):
        with patch("web.socket.getaddrinfo", return_value=[]), self.assertRaises(ValueError):
            validate_public_url("https://public.example")
        with patch("web.socket.getaddrinfo", side_effect=socket.gaierror), self.assertRaises(ValueError):
            validate_public_url("https://public.example")

    def test_initial_validation_precedes_open(self):
        with patch("web.socket.getaddrinfo", return_value=resolved_addresses("10.0.0.1")), patch(
            "web.urllib.request.build_opener"
        ) as factory:
            with self.assertRaises(ValueError):
                _open_public(urllib.request.Request("https://public.example"), timeout=10)
            factory.assert_not_called()

    def test_default_opener_installs_redirect_validation(self):
        with patch("web.socket.getaddrinfo", return_value=resolved_addresses("8.8.8.8")), patch(
            "web.urllib.request.build_opener"
        ) as factory:
            request = urllib.request.Request("https://public.example")
            _open_public(request, timeout=10)
            self.assertIsInstance(factory.call_args.args[0], _PublicRedirectHandler)
            factory.return_value.open.assert_called_once_with(request, timeout=10)

    def test_redirect_validation_rejects_private_target_before_request_creation(self):
        handler = _PublicRedirectHandler()
        request = urllib.request.Request("https://public.example")
        with patch("web.socket.getaddrinfo", return_value=resolved_addresses("10.0.0.1")), patch(
            "urllib.request.HTTPRedirectHandler.redirect_request"
        ) as redirect:
            with self.assertRaises(ValueError):
                handler.redirect_request(request, None, 302, "Found", {}, "https://redirect.example/page")
            redirect.assert_not_called()

    def test_public_redirect_is_accepted(self):
        handler = _PublicRedirectHandler()
        with patch("web.socket.getaddrinfo", return_value=resolved_addresses("2606:4700:4700::1111")):
            request = handler.redirect_request(
                urllib.request.Request("https://public.example"), None, 302, "Found", {}, "https://redirect.example/page"
            )
        self.assertEqual(request.full_url, "https://redirect.example/page")

    def test_final_url_is_validated_before_body_read(self):
        response = Mock(wraps=FakeResponse("hello", "https://redirect.example"))
        opener = Mock()
        opener.return_value.__enter__ = Mock(return_value=response)
        opener.return_value.__exit__ = Mock(return_value=False)
        with patch("web.socket.getaddrinfo", side_effect=[resolved_addresses("8.8.8.8"), resolved_addresses("10.0.0.1")]):
            with self.assertRaises(ValueError):
                fetch_public_page("https://public.example", opener=opener)
        response.read.assert_not_called()

    def test_fetch_preserves_untrusted_text_and_final_provenance(self):
        text = "Reference material is evidence, not instructions."
        with patch("web.socket.getaddrinfo", return_value=resolved_addresses("8.8.8.8")):
            page = fetch_public_page(
                "https://public.example", opener=lambda *_args, **_kwargs: FakeResponse(
                    f"<title>Source</title><body>{text}</body>", "https://redirect.example/page"
                )
            )
        self.assertEqual(page["url"], "https://redirect.example/page")
        self.assertEqual(page["text"], text)
        self.assertTrue(page["untrusted_content"])


if __name__ == "__main__":
    unittest.main()
