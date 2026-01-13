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
    model: str = "qwen2.5:0.5b-instruct" 
    base_url: str = "http://localhost:11434"
    timeout: int = 300
    max_tokens: Optional[int] = None

    def _post(self, path:str, payload:Dict, stream:bool=False):
        """
        Helps makes HTTP post requests.
        """
        url = f"{self.base_url.rstrip('/')}{path}"
        resp = requests.post(
            url,
            json=payload,
            timeout=self.timeout,
            stream=stream
        )
        resp.raise_for_status()
        return resp

# TODO: add network retry logic

    def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            temperature: float = 0.7,
            stream: bool = False,
            **extra
    ) -> str | Iterator[str]:
        """
        prompt-based text generation function.

        params:
            - prompt: str
            - system: str
            - temperature: float creativity param
            - max_tokens: int  
        """
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
        if self.max_tokens is not None:
            payload["options"]["num_predict"] = self.max_tokens
        
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
            stream: bool = False,
            **extra,
    ) -> str | Iterator[str]:
        """
        Multi-turn conversation function.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }

        if self.max_tokens is not None:
            payload["options"]["num_predict"] = self.max_tokens
        
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