import unittest
from unittest.mock import patch

from config import OLLAMA_MODEL, REASONING_TIMEOUT_SECONDS
from providers.ollama_provider import OllamaProvider


class ProviderTimeoutTests(unittest.TestCase):
    def test_client_receives_configured_timeout_and_model(self):
        for timeout in (None, 2.5):
            with self.subTest(timeout=timeout), patch("providers.ollama_provider.ollama.Client") as factory:
                client = factory.return_value.__enter__.return_value
                client.chat.return_value = {"message": {"content": "answer"}}
                provider = OllamaProvider(timeout_seconds=timeout)
                self.assertEqual(provider.ask(system_prompt="system", user_prompt="question"), "answer")
                factory.assert_called_once_with(timeout=REASONING_TIMEOUT_SECONDS if timeout is None else timeout)
                client.chat.assert_called_once_with(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "question"}],
                )
                factory.return_value.__exit__.assert_called_once()

    def test_timeout_propagates_and_client_is_closed(self):
        with patch("providers.ollama_provider.ollama.Client") as factory, patch("providers.ollama_provider.logger"):
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = TimeoutError("request timed out")
            with self.assertRaises(TimeoutError):
                OllamaProvider(timeout_seconds=1).ask(system_prompt="system", user_prompt="question")
            factory.return_value.__exit__.assert_called_once()

    def test_invalid_timeout_is_rejected(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                OllamaProvider(timeout_seconds=timeout)

    def test_installed_client_accepts_transport_timeout(self):
        import ollama

        with ollama.Client(timeout=2.5) as client:
            self.assertEqual(client._client.timeout.connect, 2.5)
            self.assertEqual(client._client.timeout.read, 2.5)


if __name__ == "__main__":
    unittest.main()
