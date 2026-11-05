
"""
ZENTIQ - Ollama Client

Handles communication between ZENTIQ and locally hosted
Ollama models, including text generation and chat completion.

Developed by EverMark Creative Solutions.
"""

import requests
from typing import Optional, Dict, Any, List


class OllamaClient:
    """Client for communicating with the Ollama API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_connection(self) -> bool:
        """Check whether the Ollama service is available."""

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            return True

        except requests.RequestException:
            return False

    def list_models(self) -> list:
        """Return the models currently available in Ollama."""

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            return [
                model["name"]
                for model in data.get("models", [])
            ]

        except requests.RequestException as error:
            raise ConnectionError(
                f"Could not retrieve Ollama models: {error}"
            ) from error

    def generate(
        self,
        prompt: str,
        model: str = "llama3.2:1b",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """Generate a response using Ollama's generate endpoint."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("response", "")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    "Ollama returned an empty response."
                )

            return content.strip()

        except requests.RequestException as error:
            raise ConnectionError(
                f"Ollama generation failed: {error}"
            ) from error

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama3.2:1b",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a response using Ollama's chat endpoint.

        Supports structured conversation history.
        """

        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "Chat messages must be a non-empty list."
            )

        valid_roles = {"system", "user", "assistant", "tool"}

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each chat message must be a dictionary.")

            if message.get("role") not in valid_roles:
                raise ValueError("Invalid chat message role.")

            if not isinstance(message.get("content"), str):
                raise ValueError(
                    "Each chat message must contain text content."
                )

        chat_messages = [dict(message) for message in messages]

        if system_prompt:
            chat_messages.insert(
                0,
                {
                    "role": "system",
                    "content": system_prompt
                }
            )

        payload = {
            "model": model,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("message", {}).get("content", "")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    "Ollama chat returned an empty response."
                )

            return content.strip()

        except requests.RequestException as error:
            raise ConnectionError(
                f"Ollama chat failed: {error}"
            ) from error
