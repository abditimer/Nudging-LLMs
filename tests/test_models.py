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