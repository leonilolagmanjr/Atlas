"""Readable-main-content extraction for fetched web pages.

Pages fetched by Atlas are dumped as flat text by a naive HTML strip, which
includes navigation, headers, footers, cookie/consent banners, adverts,
related/comment/share widgets, and media alt text. This module keeps only the
page's readable main content.

The extractor is deterministic and dependency-free:

* a page that marks a main region (``<main>``, ``<article>``, ``role=main``)
  has only that region kept;
* otherwise known chrome (by tag, role, or class/id hint) and all media are
  dropped, and the remaining text is kept.

All string literals use single quotes so the source survives quote-normalizing
editors without corruption.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Tags whose entire subtree is chrome or non-content and is dropped.
_BOILERPLATE_TAGS = frozenset({
    'nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption',
    'script', 'style', 'noscript', 'template', 'svg', 'canvas',
    'iframe', 'video', 'audio', 'picture', 'button', 'select',
    'option', 'dialog', 'menu', 'map', 'object', 'embed',
})

# Media/void tags whose alt text or caption is not part of the page content.
_MEDIA_TAGS = frozenset({'img', 'source', 'track', 'area', 'input'})

#: Void elements that never have an end tag, so they must not raise the
#: persistent nesting depth used for skip/main bookkeeping.
_VOID_TAGS = frozenset({
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
    'meta', 'param', 'source', 'track', 'wbr',
})

# Class/id/role substrings that mark a region as chrome rather than content.
_BOILERPLATE_HINTS = (
    'nav', 'menu', 'sidebar', 'side-bar', 'footer', 'header',
    'masthead', 'breadcrumb', 'cookie', 'consent', 'banner',
    'advert', 'ads-', '-ads', 'promo', 'sponsor', 'social',
    'share', 'comment', 'related', 'recommend', 'toolbar',
    'pagination', 'subscribe', 'newsletter', 'signin', 'login',
    'modal', 'popup', 'overlay', 'skip-link',
    # Document-store / upload card chrome (Scribd-style landing pages).
    'ratings', 'rating', 'views', 'upload', 'uploaded', 'download',
    'document-info', 'doc-info', 'metadata', 'stats', 'engagement',
    'carousel', 'preview', 'recommendations', 'read_more', 'read-more',
)

# Roles that denote structural chrome.
_BOILERPLATE_ROLES = frozenset({
    'navigation', 'banner', 'contentinfo', 'complementary',
    'search', 'form', 'dialog', 'alertdialog', 'menu', 'menubar',
})

_MAIN_TAGS = frozenset({'main', 'article'})
_MAIN_ROLES = frozenset({'main', 'article'})

_WS_RE = re.compile(r'\s+')
_TOKEN_RE = re.compile(r'[^a-z0-9]+')


def collapse_text(value: str) -> str:
    """Collapse runs of whitespace in ``value`` and trim the ends."""

    return _WS_RE.sub(' ', value).strip()


class PageContentParser(HTMLParser):
    """Extract the readable main content of an HTML page.

    Usage: feed the document, then read :attr:`title` and :attr:`text`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title = ''
        self._all_parts: list[str] = []
        self._main_parts: list[str] = []
        self._in_title = False
        self._parts: list[str] = self._all_parts
        # Nesting depth of elements currently open, and the depth at which the
        # outermost main-content element opened (None when not inside main).
        self._depth = 0
        self._skip_since: int | None = None
        self._main_since: int | None = None

    @property
    def text(self) -> str:
        # Prefer the explicit main region when the page provided one and it
        # carries the bulk of the readable text; otherwise use all non-chrome
        # text (some pages have no main-region markup at all).
        main_text = collapse_text(' '.join(self._main_parts))
        all_text = collapse_text(' '.join(self._all_parts))
        if main_text and (not all_text or len(main_text) >= max(200, int(len(all_text) * 0.5))):
            return main_text
        return all_text or main_text

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): (value or '') for key, value in attrs}
        if tag == 'title':
            self._in_title = True
        void = tag in _VOID_TAGS
        if not void:
            self._depth += 1
        if self._skip_since is not None:
            return
        if tag == 'br' and not self._in_title:
            self._parts.append(' ')
            return
        if _is_boilerplate(tag, attributes):
            # A void boilerplate tag (img/input) has no end tag to close it, so
            # only non-void chrome opens a skip region.
            if not void:
                self._skip_since = self._depth
            return
        if self._main_since is None and not void and (tag in _MAIN_TAGS or _is_main(attributes)):
            self._main_since = self._depth
            self._parts = self._main_parts
    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # A self-closing tag (img, source, br, ...). Only a line break changes
        # how the surrounding text reads; media contributes no content.
        if tag == 'br' and self._skip_since is None and not self._in_title:
            self._parts.append(' ')

    def handle_endtag(self, tag: str) -> None:
        if tag == 'title':
            self._in_title = False
        if self._skip_since is not None and self._depth <= self._skip_since:
            self._skip_since = None
        if self._main_since is not None and self._depth <= self._main_since:
            self._main_since = None
            self._parts = self._all_parts
        if self._depth:
            self._depth -= 1
    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_since is not None:
            return
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)


def _is_main(attributes: dict[str, str]) -> bool:
    return attributes.get('role', '').lower() in _MAIN_ROLES


def _is_boilerplate(tag: str, attributes: dict[str, str]) -> bool:
    if tag in _BOILERPLATE_TAGS or tag in _MEDIA_TAGS:
        return True
    if attributes.get('role', '').lower() in _BOILERPLATE_ROLES:
        return True
    haystack = (attributes.get('class', '') + chr(32) + attributes.get('id', '')).lower()
    if not haystack.strip():
        return False
    tokens = set(_TOKEN_RE.split(haystack))
    return any(hint in tokens or hint in haystack for hint in _BOILERPLATE_HINTS)
