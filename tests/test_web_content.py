"""Tests for readable-main-content extraction (web_content.PageContentParser).

A fetched page must yield the readable content, not the surrounding chrome
(navigation, headers, footers, cookie banners, adverts, share widgets) or media
alt text.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web_content import PageContentParser  # noqa: E402


def extract(html: str) -> PageContentParser:
    parser = PageContentParser()
    parser.feed(html)
    return parser


class MainRegionTests(unittest.TestCase):
    def test_main_region_is_preferred_and_chrome_dropped(self):
        parser = extract(
            "<html><head><title>Bee Movie Script</title></head><body>"
            "<nav class='site-nav'>Home Movies</nav>"
            "<header id='masthead'>ScriptsDB</header>"
            "<div class='cookie-consent'>We use cookies <button>Accept</button></div>"
            "<main><h1>Bee Movie (2007) Script</h1>"
            "<figure><img src='p.jpg' alt='Bee Movie poster'><figcaption>Poster</figcaption></figure>"
            "<p>BARRY: You like jazz?</p><p>VANESSA: You're a bee!</p>"
            "</main>"
            "<div class='social-share'>Share on Facebook</div>"
            "<footer class='site-footer'>Privacy Policy</footer>"
            "</body></html>"
        )
        text = parser.text
        self.assertEqual(parser.title, "Bee Movie Script")
        self.assertIn("BARRY: You like jazz?", text)
        self.assertIn("VANESSA: You're a bee!", text)
        for noise in ("Home", "cookies", "Accept", "poster", "Facebook", "Privacy Policy"):
            self.assertNotIn(noise, text)

    def test_page_without_main_keeps_body_but_drops_chrome(self):
        parser = extract(
            "<body><nav>Menu</nav><p>just the body text</p><footer>end</footer></body>"
        )
        self.assertEqual(parser.text, "just the body text")

    def test_media_alt_text_and_captions_are_not_content(self):
        parser = extract(
            "<body><main><h1>Title</h1>"
            "<img src='x.png' alt='a large red banner image'>"
            "<video src='v.mp4'>fallback caption</video>"
            "<p>Real content here.</p></main></body>"
        )
        text = parser.text
        self.assertIn("Real content here.", text)
        self.assertNotIn("banner image", text)
        self.assertNotIn("fallback caption", text)

    def test_script_and_style_are_excluded(self):
        parser = extract(
            "<body><main><script>var x='ignore me';</script>"
            "<style>.a{color:red}</style><p>Visible text</p></main></body>"
        )
        self.assertIn("Visible text", parser.text)
        self.assertNotIn("ignore me", parser.text)
        self.assertNotIn("color:red", parser.text)

    def test_role_navigation_is_chrome(self):
        parser = extract(
            "<body><div role='navigation'>Links</div>"
            "<div role='main'><p>Content</p></div></body>"
        )
        self.assertIn("Content", parser.text)
        self.assertNotIn("Links", parser.text)

    def test_br_becomes_a_space_not_a_join(self):
        parser = extract("<body><main><p>line one<br>line two</p></main></body>")
        self.assertIn("line one line two", parser.text)


if __name__ == "__main__":
    unittest.main()
