"""
ZENTIQ - Ollama Client
Handles communication between ZENTIQ and locally hosted Ollama models.
"""

import requests
from typing import Optional, Dict, Any


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
        """Generate a response using an Ollama model."""

        if not prompt or not prompt.strip():
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
            return result.get("response", "").strip()

        except requests.RequestException as error:
            raise ConnectionError(
                f"Ollama generation failed: {error}"
            ) from error