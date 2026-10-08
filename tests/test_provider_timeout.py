import unittest
from unittest.mock import patch

import httpx
import ollama

from config import (
    OLLAMA_CONNECT_TIMEOUT_SECONDS,
    OLLAMA_MODEL,
    OLLAMA_READ_TIMEOUT_SECONDS,
    REASONING_TIMEOUT_SECONDS,
)
from providers.exceptions import (
    ProviderConnectionFailure,
    ProviderModelError,
    ProviderTimeout,
)
from providers.ollama_provider import OllamaProvider


class ProviderTimeoutTests(unittest.TestCase):
    def test_client_receives_configured_timeout_and_model(self):
        for timeout in (None, 2.5):
            with self.subTest(timeout=timeout), patch("providers.ollama_provider.ollama.Client") as factory:
                client = factory.return_value.__enter__.return_value
                client.chat.return_value = {"message": {"content": "answer"}}
                provider = OllamaProvider(timeout_seconds=timeout)
                self.assertEqual(provider.ask(system_prompt="system", user_prompt="question"), "answer")
                # The timeout passed to ollama.Client is now an httpx.Timeout object
                # distinguishing connect from read, rather than a bare float.
                call_kwargs = factory.call_args.kwargs
                self.assertIn("timeout", call_kwargs)
                injected = call_kwargs["timeout"]
                self.assertIsInstance(injected, httpx.Timeout)
                expected_read = REASONING_TIMEOUT_SECONDS if timeout is None else timeout
                self.assertEqual(injected.read, expected_read)
                client.chat.assert_called_once_with(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "question"}],
                )
                factory.return_value.__exit__.assert_called_once()

    def test_default_connect_and_read_timeouts(self):
        provider = OllamaProvider()
        self.assertEqual(provider.connect_timeout, OLLAMA_CONNECT_TIMEOUT_SECONDS)
        self.assertEqual(provider.read_timeout, REASONING_TIMEOUT_SECONDS)
        self.assertEqual(provider.timeout_seconds, REASONING_TIMEOUT_SECONDS)

    def test_explicit_connect_and_read_are_distinct(self):
        provider = OllamaProvider(connect_timeout=5.0, read_timeout=120.0)
        self.assertEqual(provider.connect_timeout, 5.0)
        self.assertEqual(provider.read_timeout, 120.0)
        timeout = provider._timeout()
        self.assertEqual(timeout.connect, 5.0)
        self.assertEqual(timeout.read, 120.0)

    def test_installed_client_accepts_httpx_timeout(self):
        import ollama

        timeout = httpx.Timeout(timeout=120.0, connect=10.0, read=120.0, write=120.0, pool=120.0)
        with ollama.Client(timeout=timeout) as client:
            self.assertEqual(client._client.timeout.connect, 10.0)
            self.assertEqual(client._client.timeout.read, 120.0)

    def test_read_timeout_translated_to_provider_timeout(self):
        with patch("providers.ollama_provider.ollama.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = httpx.ReadTimeout("timed out")
            with self.assertRaises(ProviderTimeout):
                OllamaProvider(timeout_seconds=1).ask(system_prompt="system", user_prompt="question")
            factory.return_value.__exit__.assert_called_once()

    def test_connect_error_translated_to_connection_failure(self):
        with patch("providers.ollama_provider.ollama.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = httpx.ConnectError("refused")
            with self.assertRaises(ProviderConnectionFailure):
                OllamaProvider().ask(system_prompt="system", user_prompt="question")

    def test_model_not_found_translated_to_model_error(self):
        with patch("providers.ollama_provider.ollama.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = ollama.ResponseError("model not found", status_code=404)
            with self.assertRaises(ProviderModelError):
                OllamaProvider().ask(system_prompt="system", user_prompt="question")

    def test_invalid_timeout_is_rejected(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                OllamaProvider(timeout_seconds=timeout)


if __name__ == "__main__":
    unittest.main()
