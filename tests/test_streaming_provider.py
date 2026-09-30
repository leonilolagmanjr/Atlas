"""Streaming provider contract (Milestone 6).

The bug these tests exist for was found by running the real streaming path: the
installed ``ollama`` client yields ``ChatResponse`` objects, not dicts, so a
``chunk.get("message")`` read dropped every token and Atlas silently answered
"the local model produced no output". A silent empty stream is worse than a
loud failure, so both the extraction and the fallback are asserted here.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from providers.base_provider import BaseProvider
from providers.ollama_provider import _chunk_text


@dataclass
class _Message:
    role: str
    content: str


@dataclass
class _ChatResponse:
    """Mirror of the installed client's object shape (not a dict)."""

    model: str
    message: _Message


class ChunkExtractionTests(unittest.TestCase):
    def test_object_chunks_are_read(self) -> None:
        # This is the shape the installed ollama client actually returns.
        self.assertEqual(_chunk_text(_ChatResponse("m", _Message("assistant", "Hello"))), "Hello")

    def test_dict_chunks_are_read(self) -> None:
        # Older releases returned plain dicts; both must work.
        self.assertEqual(_chunk_text({"message": {"content": "World"}}), "World")

    def test_empty_and_unknown_shapes_are_empty_not_errors(self) -> None:
        for chunk in ({}, {"message": {}}, {"message": None}, object(), None, "text"):
            with self.subTest(chunk=chunk):
                self.assertEqual(_chunk_text(chunk), "")

    def test_missing_content_is_empty(self) -> None:
        self.assertEqual(_chunk_text(_ChatResponse("m", _Message("assistant", ""))), "")


class _FakeClient:
    """Stands in for ``ollama.Client`` with a configurable stream shape."""

    def __init__(self, chunks: list[Any], *, error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def chat(self, **_kwargs: Any):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


class StreamingFallbackTests(unittest.TestCase):
    """A stream that yields nothing must not become a silent empty answer."""

    def _provider(self, chunks: list[Any], *, error: Exception | None = None):
        from providers import ollama_provider

        original = ollama_provider.ollama.Client
        ollama_provider.ollama.Client = lambda **_kwargs: _FakeClient(chunks, error=error)  # type: ignore[assignment]
        self.addCleanup(setattr, ollama_provider.ollama, "Client", original)
        return ollama_provider.OllamaProvider()

    def test_object_chunks_are_streamed(self) -> None:
        provider = self._provider([
            _ChatResponse("m", _Message("assistant", "Hello")),
            _ChatResponse("m", _Message("assistant", " world")),
        ])
        tokens: list[str] = []
        text = provider.stream(
            system_prompt="s", user_prompt="u", on_token=tokens.append, should_stop=lambda: False
        )
        self.assertEqual(text, "Hello world")
        self.assertEqual(tokens, ["Hello", " world"])

    def test_an_empty_stream_falls_back_to_one_blocking_call(self) -> None:
        provider = self._provider([])
        calls: list[str] = []

        def fake_ask(*, system_prompt: str, user_prompt: str) -> str:
            calls.append(user_prompt)
            return "A real answer."

        provider.ask = fake_ask  # type: ignore[method-assign]
        tokens: list[str] = []
        text = provider.stream(
            system_prompt="s", user_prompt="u", on_token=tokens.append, should_stop=lambda: False
        )
        self.assertEqual(text, "A real answer.")
        self.assertEqual(tokens, ["A real answer."])
        self.assertTrue(calls)

    def test_cancellation_stops_pulling_further_chunks(self) -> None:
        provider = self._provider([
            _ChatResponse("m", _Message("assistant", "one")),
            _ChatResponse("m", _Message("assistant", "two")),
            _ChatResponse("m", _Message("assistant", "three")),
        ])
        state = {"stop": False}

        def on_token(piece: str) -> None:
            if piece == "one":
                state["stop"] = True

        text = provider.stream(
            system_prompt="s",
            user_prompt="u",
            on_token=on_token,
            should_stop=lambda: state["stop"],
        )
        # Generation is abandoned after the first token rather than completed
        # and hidden.
        self.assertEqual(text, "one")


class BaseProviderContractTests(unittest.TestCase):
    def test_default_stream_is_honest_about_not_streaming(self) -> None:
        class _Blocking(BaseProvider):
            def ask(self, *, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
                return "whole answer"

        provider = _Blocking()
        self.assertFalse(provider.supports_streaming())
        tokens: list[str] = []
        text = provider.stream(system_prompt="s", user_prompt="u", on_token=tokens.append)
        # One real token, not a fabricated per-word sequence.
        self.assertEqual(text, "whole answer")
        self.assertEqual(tokens, ["whole answer"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
