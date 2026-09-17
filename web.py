"""Read-only public web search and retrieval tools.

Web content is returned as untrusted evidence. It is never treated as an Atlas
instruction or passed directly into a computer-control command.
"""

from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import xml.etree.ElementTree as ET
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from tools.base import PermissionLevel, RiskLevel, Tool, ToolMetadata, ToolResult


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
MAX_QUERY_LENGTH = 500
MAX_RESULTS = 10
MAX_PAGE_BYTES = 1_000_000
# Search result pages (YouTube especially) embed large JSON payloads that can
# exceed the page-fetch limit; truncating them breaks result parsing entirely.
MAX_SEARCH_BYTES = 6_000_000
# Terms too common to signal relevance on their own.
_SEARCH_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "is", "are", "best",
    "top", "vs", "and", "or", "how", "what", "why", "who", "when", "where",
    "with", "about", "from", "new", "latest", "popular", "videos", "video",
}


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    source: str = "DuckDuckGo"
    thumbnail_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "thumbnail_url": self.thumbnail_url or thumbnail_for_url(self.url),
        }
class WebSearchTool(Tool):
    metadata = ToolMetadata(
        name="web.search",
        description="Search the public web and return attributed result links without executing page instructions.",
        category="internet.search",
        input_schema={
            "query": {"type": "string", "description": "what to search for"},
            "max_results": {"type": "integer", "description": "result count"},
            "site": {"type": "string", "description": "optional site, e.g. youtube"},
            "sort": {"type": "string", "description": "latest or popular"},
        },
        output_schema={"query": {"type": "string"}, "results": {"type": "array"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            query = str(parameters["query"]).strip()
            if not query:
                return ToolResult.failure("Search query cannot be empty")
            if len(query) > MAX_QUERY_LENGTH:
                return ToolResult.failure("Search query exceeds the size limit")
            limit = min(max(int(parameters.get("max_results", 5)), 1), MAX_RESULTS)
            site = str(parameters.get("site") or "").strip().casefold()
            sort = str(parameters.get("sort") or "").strip().casefold()
            # A structured site/sort hint from the planner generalizes the old
            # keyword-only YouTube detection without hard-coding person names.
            search_query = _apply_sort(query, sort)
            youtube_query = site == "youtube" or _is_youtube_video_query(query)
            results = self._collect_results(search_query, youtube_query=youtube_query, limit=limit)
            provider = results[0].source if results else ("YouTube" if youtube_query else "Web")
            return ToolResult(
                success=True,
                status="completed",
                output={
                    "query": query,
                    "results": [result.to_dict() for result in results],
                    "result_count": len(results),
                },
                metadata={"source": provider, "untrusted_content": True},
            )
        except (KeyError, TypeError, ValueError, OSError, TimeoutError, ET.ParseError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)

    @staticmethod
    def _collect_results(
        search_query: str,
        *,
        youtube_query: bool,
        limit: int,
    ) -> list[WebResult]:
        # Query backends in fallback order and filter to relevant results.
        # Backends are tried until one returns usable results. Bing/DDG can
        # occasionally serve stale or unrelated pages, so results are filtered
        # for token overlap with the query before being returned.
        if youtube_query:
            results = _search_provider(search_youtube, search_query, limit=limit)
            results = [result for result in results if _is_youtube_video_url(result.url)]
            if results:
                return results[:limit]
            fallback = _search_provider(search_bing_rss, f"site:youtube.com/watch {search_query}", limit=limit * 2)
            return [result for result in fallback if _is_youtube_video_url(result.url)][:limit]

        # Gather from every backend and keep the most relevant results.
        # Backends differ in trustworthiness per query type, so scoring beats a
        # fixed fallback order: a stale Bing hit cannot crowd out a good one.
        collected: list[tuple[int, WebResult]] = []
        for provider in (search_duckduckgo, search_bing_rss, search_wikipedia):
            for result in _search_provider(provider, search_query, limit=limit):
                score = _relevance_score(result, search_query)
                if score > 0:
                    collected.append((score, result))
        # Highest score first; stable within equal scores to preserve provider order.
        collected.sort(key=lambda item: item[0], reverse=True)
        results: list[WebResult] = []
        seen_urls: set[str] = set()
        for _score, result in collected:
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            results.append(result)
            if len(results) >= limit:
                break
        return results

class WebFetchTool(Tool):
    metadata = ToolMetadata(
        name="web.fetch",
        description="Fetch a public web page as bounded, untrusted text with provenance.",
        category="internet.retrieval",
        input_schema={"url": {"type": "string"}, "max_bytes": {"type": "integer"}},
        output_schema={"url": {"type": "string"}, "title": {"type": "string"}, "text": {"type": "string"}},
        permission_level=PermissionLevel.READ_ONLY,
        risk_level=RiskLevel.READ_ONLY,
    )

    def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            self.validate(parameters)
            url = str(parameters["url"]).strip()
            validate_public_url(url)
            max_bytes = min(max(int(parameters.get("max_bytes", MAX_PAGE_BYTES)), 1_000), MAX_PAGE_BYTES)
            page = fetch_public_page(url, max_bytes=max_bytes)
            return ToolResult(
                success=True,
                status="completed",
                output=page,
                metadata={"source": url, "untrusted_content": True},
            )
        except (KeyError, TypeError, ValueError, OSError, TimeoutError) as exc:
            return ToolResult.failure(str(exc), recoverable=True)


def _search_provider(provider: Any, query: str, *, limit: int) -> list[WebResult]:
    try:
        return list(provider(query, limit=limit))
    except Exception:
        return []


class _PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_public(request: urllib.request.Request, *, timeout: float):
    validate_public_url(request.full_url)
    return urllib.request.build_opener(_PublicRedirectHandler()).open(request, timeout=timeout)


def search_duckduckgo(query: str, *, limit: int = 5, opener: Any = None) -> list[WebResult]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    with (opener or _open_public)(request, timeout=10) as response:
        body = response.read(MAX_SEARCH_BYTES).decode("utf-8", errors="replace")
    # DuckDuckGo sometimes serves an anti-bot challenge page instead of results.
    # Treat that as "no results" so the caller can fall back cleanly.
    if "result__a" not in body:
        return []
    parser = _DuckDuckGoParser()
    parser.feed(body)
    return parser.results[:limit]


def search_youtube(query: str, *, limit: int = 5, opener: Any = None) -> list[WebResult]:
    url = "https://www.youtube.com/results?" + urllib.parse.urlencode({"search_query": query})
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    with (opener or _open_public)(request, timeout=15) as response:
        # YouTube embeds a large JSON blob; a small read cap truncates it and
        # breaks parsing, which previously produced zero results.
        body = response.read(MAX_SEARCH_BYTES).decode("utf-8", errors="replace")
    marker = "var ytInitialData = "
    start = body.find(marker)
    if start < 0:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(body[start + len(marker):])
    except json.JSONDecodeError:
        return []
    results: list[WebResult] = []
    for renderer in _find_video_renderers(payload):
        video_id = renderer.get("videoId")
        if not video_id:
            continue
        title = _renderer_text(renderer.get("title")) or "YouTube video"
        snippet = _renderer_text(renderer.get("descriptionSnippet"))
        owner = _renderer_text(renderer.get("ownerText"))
        details = " · ".join(value for value in (owner, _renderer_text(renderer.get("publishedTimeText")), _renderer_text(renderer.get("viewCountText"))) if value)
        if details:
            snippet = f"{details}. {snippet}".strip()
        results.append(WebResult(
            title=title,
            url=f"https://www.youtube.com/watch?v={video_id}",
            snippet=snippet,
            source="YouTube",
            thumbnail_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        ))
        if len(results) >= limit:
            break
    return results


def _find_video_renderers(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        renderer = value.get("videoRenderer")
        if isinstance(renderer, dict):
            found.append(renderer)
        for child in value.values():
            found.extend(_find_video_renderers(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_video_renderers(child))
    return found


def _renderer_text(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    if isinstance(value.get("simpleText"), str):
        return value["simpleText"].strip()
    runs = value.get("runs")
    if isinstance(runs, list):
        return "".join(str(run.get("text", "")) for run in runs if isinstance(run, dict)).strip()
    return ""


def search_bing_rss(query: str, *, limit: int = 5, opener: Any = None) -> list[WebResult]:
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"format": "rss", "q": query})
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    with (opener or _open_public)(request, timeout=10) as response:
        body = response.read(MAX_SEARCH_BYTES)
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        # Bing occasionally returns an HTML consent/error page instead of RSS.
        return []
    results: list[WebResult] = []
    for item in root.findall(".//item"):
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        description = html.unescape(item.findtext("description", default="").strip())
        if title and link:
            results.append(WebResult(title, link, re.sub(r"\s+", " ", description), source="Bing"))
        if len(results) >= limit:
            break
    return results

def search_wikipedia(query: str, *, limit: int = 5, opener: Any = None) -> list[WebResult]:
    # Search Wikipedia's public API as a reliable general-knowledge backend.
    # This does not depend on scraping an ad-driven search page, so it remains
    # useful when the HTML search endpoints are rate-limited or blocked.
    params = urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": limit}
    )
    url = "https://en.wikipedia.org/w/api.php?" + params
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (opener or _open_public)(request, timeout=10) as response:
        body = response.read(MAX_PAGE_BYTES).decode("utf-8", errors="replace")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    hits = payload.get("query", {}).get("search", [])
    results: list[WebResult] = []
    for hit in hits:
        title = str(hit.get("title", "")).strip()
        if not title:
            continue
        snippet = re.sub(r"<[^>]+>", "", str(hit.get("snippet", "")))
        page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        results.append(WebResult(title, page_url, html.unescape(snippet).strip(), source="Wikipedia"))
    return results


def fetch_public_page(url: str, *, max_bytes: int = MAX_PAGE_BYTES, opener: Any = None) -> dict[str, Any]:
    validate_public_url(url)
    max_bytes = min(max(int(max_bytes), 1), MAX_PAGE_BYTES)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (opener or _open_public)(request, timeout=10) as response:
        final_url = response.geturl()
        validate_public_url(final_url)
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "text/plain", "application/xhtml+xml"}:
            raise ValueError(f"Unsupported web content type: {content_type}")
        body = response.read(max_bytes).decode("utf-8", errors="replace")
    parser = _PageTextParser()
    parser.feed(body)
    return {
        "url": final_url,
        "title": parser.title.strip(),
        "text": re.sub(r"\s+", " ", parser.text.strip()),
        "truncated": len(body.encode("utf-8")) >= max_bytes,
        "untrusted_content": True,
    }


def _resolve_public_addresses(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("Could not resolve public web host") from exc
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError as exc:
            raise ValueError("Could not resolve public web host") from exc
    if not addresses:
        raise ValueError("Could not resolve public web host")
    return addresses


def validate_public_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP(S) URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL credentials are not allowed")
    if parsed.port is not None and parsed.port < 1:
        raise ValueError("Invalid web port")
    hostname = parsed.hostname.casefold().rstrip(".")
    if "%" in hostname:
        raise ValueError("Scoped web hosts are not allowed")
    blocked_names = {"localhost", "localhost.localdomain", "metadata.google.internal"}
    if hostname in blocked_names or hostname.endswith(".local"):
        raise ValueError("Local and metadata hosts are not allowed")
    try:
        address = ipaddress.ip_address(hostname)
        addresses = [address]
    except ValueError:
        addresses = _resolve_public_addresses(hostname)
    for address in addresses:
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        if not address.is_global or address.is_multicast or address.is_reserved:
            raise ValueError("Private and local network targets are not allowed")


def thumbnail_for_url(url: str) -> str | None:
    """Return a stable preview image for supported public video URLs."""

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname.casefold() if parsed.hostname else ""
    video_id = urllib.parse.parse_qs(parsed.query).get("v", [None])[0]
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"} and video_id:
        return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
    if host == "youtu.be" and parsed.path.strip("/"):
        return f"https://i.ytimg.com/vi/{parsed.path.strip('/').split('/')[0]}/hqdefault.jpg"
    return None


def _is_youtube_video_query(query: str) -> bool:
    return bool(re.search(r"\byoutube\b|site:youtube\.com", query, re.IGNORECASE))


def _query_terms(query: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", query.casefold())
        if token not in _SEARCH_STOPWORDS and len(token) >= 2
    ]

def _relevance_score(result: WebResult, query: str) -> int:
    # Score a result by weighted query-term matches.
    # Longer terms are more specific, so they weigh more: matching "asyncio"
    # should outrank merely matching the generic word "python". Results that
    # match nothing distinctive score 0 and are dropped entirely.
    terms = _query_terms(query)
    if not terms:
        return 1
    title_and_snippet = f"{result.title} {result.snippet}".casefold()
    body = f"{title_and_snippet} {result.url}".casefold()
    score = 0
    for term in terms:
        weight = 2 if len(term) >= 6 else 1
        if term in title_and_snippet:
            score += weight * 2
        elif term in body:
            score += weight
    return score

def _filter_relevant(results: list[WebResult], query: str) -> list[WebResult]:
    # Keep only results that match the query's most distinctive term.
    # Generic fallback pages (e.g. "Python.org" for "python asyncio tutorial")
    # match a common word but miss the point of the query, so results are
    # scored by matching the longest/most specific query token first.
    terms = _query_terms(query)
    if not terms:
        return list(results)
    # Longer tokens are more distinctive; require at least the best match found.
    distinctive = [term for term in terms if len(term) >= 4]
    required = distinctive or terms
    relevant: list[WebResult] = []
    for result in results:
        haystack = f"{result.title} {result.snippet} {result.url}".casefold()
        if any(term in haystack for term in required):
            relevant.append(result)
    return relevant

def _apply_sort(query: str, sort: str) -> str:
    # Append a natural-language sort hint to a search query.
    # The backends accept free text, so a structured sort from the planner is
    # expressed as a word the engine already understands rather than a bespoke
    # parameter the providers cannot honor.
    if not sort:
        return query
    lowered = query.casefold()
    if sort in lowered:
        return query
    return f"{query} {sort}"


def _is_youtube_video_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname.casefold() if parsed.hostname else ""
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        return parsed.path == "/watch" and bool(urllib.parse.parse_qs(parsed.query).get("v"))
    return host == "youtu.be" and bool(parsed.path.strip("/"))


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebResult] = []
        self._current_url: str | None = None
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._current_url = _decode_result_url(attributes.get("href", ""))
            self._title = []
            self._in_title = True
        elif tag in {"a", "div"} and "result__snippet" in classes:
            self._snippet = []
            self._in_snippet = True

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        if self._in_snippet:
            self._snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            if self._current_url and self._title:
                self.results.append(WebResult("".join(self._title).strip(), self._current_url, ""))
        if tag in {"a", "div"} and self._in_snippet:
            self._in_snippet = False
            if self.results:
                self.results[-1] = WebResult(
                    self.results[-1].title,
                    self.results[-1].url,
                    " ".join("".join(self._snippet).split()),
                )


def _decode_result_url(value: str) -> str:
    parsed = urllib.parse.urlparse(html.unescape(value))
    if parsed.path.startswith("/l/"):
        query = urllib.parse.parse_qs(parsed.query)
        return query.get("uddg", [value])[0]
    return html.unescape(value)


class _PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.text_parts: list[str] = []
        self._in_title = False
        self._ignored_depth = 0

    @property
    def text(self) -> str:
        return " ".join(self.text_parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._ignored_depth:
            self.text_parts.append(data)
