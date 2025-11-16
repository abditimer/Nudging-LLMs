import requests
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
import json

@dataclass
class OllamaClient:
    """
    client for local Ollama instance.
    
    default host: http://localhost:11434
    """
    model: str = "qwen3:0.6b"
    base_url: str = "http://localhost:11434"
    timeout: int = 120

    def _post(self, path:str, payload:Dict, stream:bool=False):
        url = f"{self.base_url.rstrip('/')}{path}"
        resp = requests.post(
            url,
            json=payload,
            timeout=self.timeout,
            stream=stream
        )
        resp.raise_for_status()
        return resp

    def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stream: bool = False,
            **extra
    ) -> str | Iterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "think": False,
            "options": {
                "temperature": temperature
            }
        }

        if system:
            payload["system"] = system
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        payload["options"].update(extra)

        resp = self._post("/api/generate", payload, stream=stream)

        if not stream:
            return resp.json().get("response", "")
        
        def _iter_chunks() -> Iterator[str]:
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                chunk = data.get("response")
                if chunk:
                    yield chunk
        return _iter_chunks()
    
    def chat(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens : Optional[int] = None,
            stream: bool = False,
            **extra,
    ) -> str | Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        payload["options"].update(extra)

        resp = self._post("/api/chat", payload, stream=stream)

        if not stream:
            data = resp.json()
            return data.get("message", {}).get("content", "")
        
        def _iter_chunks() -> Iterator[str]:
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                msg = data.get("message", {})
                chunk = msg.get("content")
                if chunk:
                    yield chunk
        return _iter_chunks()