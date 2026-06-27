import unittest
from unittest.mock import Mock, patch, MagicMock
from nudging.models import OllamaClient
import json
import requests

class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient(
            model="qwen3:0.6b",
            base_url="http://localhost:11434",
            timeout=120
        )
    
    def test_client_initialisation(self):
        """Test if client starts with correct defaults"""

        self.assertEqual(self.client.model, "qwen3:0.6b")
        self.assertEqual(self.client.base_url, "http://localhost:11434")
        self.assertEqual(self.client.timeout, 120)

    @patch('requests.post')
    def test_generate_with_temperature(self, mock_post):
        """Test generate method applies temperature parameter correctly."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Response"}
        mock_post.return_value = mock_response

        self.client.generate("Hello", temperature=0.9)

        call_args = mock_post.call_args
        payload = call_args.kwargs['json']
        self.assertEqual(payload['options']['temperature'], 0.9)




    @patch('requests.get')
    def test_is_running_returns_true_when_ollama_responds(self, mock_get):
        """Test Ollama readiness check."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        self.assertTrue(self.client.is_running())
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=2,
        )

    @patch('requests.get')
    def test_is_running_returns_false_when_ollama_unavailable(self, mock_get):
        """Test failed Ollama readiness check."""
        mock_get.side_effect = requests.RequestException("unavailable")

        self.assertFalse(self.client.is_running())

    @patch('time.sleep')
    @patch('subprocess.Popen')
    @patch.object(OllamaClient, 'is_running')
    def test_ensure_running_starts_ollama_when_needed(
            self,
            mock_is_running,
            mock_popen,
            mock_sleep,
    ):
        """Test local Ollama startup path."""
        mock_is_running.side_effect = [False, True]

        self.assertTrue(self.client.ensure_running())

        mock_popen.assert_called_once()
        mock_sleep.assert_called_once_with(3.0)

    @patch('subprocess.Popen')
    @patch.object(OllamaClient, 'is_running')
    def test_ensure_running_can_skip_startup(self, mock_is_running, mock_popen):
        """Test readiness check without attempting startup."""
        mock_is_running.return_value = False

        self.assertFalse(self.client.ensure_running(start_if_needed=False))
        mock_popen.assert_not_called()

    @patch('requests.post')
    def _test_generate_stream(self, mock_post):
        """Test generate method in streaming mode"""
        mock_response = Mock()
        mock_lines = [
            b'{"response": "Hello"}',
            b'{"response": " world"}',
            b'{"response": "!"}',
            b'{"response": "", "done": true}'
        ]
        mock_response.iter_lines.return_value = mock_lines
        mock_post.return_value = mock_response

        result = self.client.generate("Say hello", stream=True)

        # Result should be a generator
        chunks = list(result)
        self.assertEqual(chunks, ["Hello", " world", "!"])


    @patch('requests.post')
    def _test_chat_stream(self, mock_post):
        """Test chat method in streaming mode"""
        mock_response = Mock()
        mock_lines = [
            b'{"message": {"content": "Hi"}}',
            b'{"message": {"content": " there"}}',
            b'{"message": {"content": "!"}}',
        ]
        mock_response.iter_lines.return_value = mock_lines
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = self.client.chat(messages, stream=True)

        chunks = list(result)
        self.assertEqual(chunks, ["Hi", " there", "!"])


class TestOllamaIntegration(unittest.TestCase):
    """Integration tests that require a running Ollama instance"""

    def test_ollama_connection(self):
        """Check connection to Ollama (skip if not available)"""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = response.json()
            self.assertTrue(response.ok)
            self.assertIn('models', models)
            print(f"\nAvailable models: {[m['name'] for m in models['models']]}")
        except Exception as e:
            self.skipTest(f"Ollama not running: {e}")


if __name__ == '__main__':
    unittest.main()
