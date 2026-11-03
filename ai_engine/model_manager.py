
"""
ZENTIQ - Hybrid Model Manager

Manages Groq as the primary AI provider
and Ollama as the secondary provider.
"""

from ai_engine.groq_client import GroqClient
from ai_engine.ollama_client import OllamaClient


class ModelManager:

    def __init__(
        self,
        groq_client=None,
        ollama_client=None
    ):
        self.groq_client = groq_client or GroqClient()
        self.ollama_client = ollama_client or OllamaClient()

        self.default_model = "llama-3.3-70b-versatile"
        self.ollama_model = "llama3.2:1b"
        self.guard_model = "llama-guard3:1b"

        self.active_provider = None

    def select_provider(self):
        """Select Groq first, then Ollama as fallback."""

        if self.groq_client.check_connection():
            self.active_provider = "groq"
            return "groq"

        if self.ollama_client.check_connection():
            self.active_provider = "ollama"
            return "ollama"

        raise ConnectionError(
            "Neither Groq nor Ollama is currently available."
        )

    def select_model(self, task="general", provider=None):

        provider = provider or self.select_provider()

        if provider == "groq":
            return self.default_model

        if provider == "ollama":
            if task == "safety":
                return self.guard_model
            return self.ollama_model

        raise ValueError(f"Unknown AI provider: {provider}")

    def get_available_models(self, provider=None):

        provider = provider or self.select_provider()

        if provider == "groq":
            return self.groq_client.list_models()

        if provider == "ollama":
            return self.ollama_client.list_models()

        return []

