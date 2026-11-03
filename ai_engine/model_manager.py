
"""
ZENTIQ - Hybrid Model Manager

Groq is the primary AI provider.
Ollama is the secondary AI provider.
"""

from ai_engine.groq_client import GroqClient
from ai_engine.ollama_client import OllamaClient


class ModelManager:

    def __init__(
        self,
        groq_client=None,
        ollama_client=None
    ):
        # Initialize Groq without preventing Ollama from starting
        self.groq_client = groq_client

        if self.groq_client is None:
            try:
                self.groq_client = GroqClient()
            except Exception:
                self.groq_client = None

        self.ollama_client = (
            ollama_client or OllamaClient()
        )

        self.default_model = "openai/gpt-oss-20b"
        self.ollama_model = "llama3.2:1b"
        self.guard_model = "llama-guard3:1b"

        self.active_provider = None

    def select_provider(self):
        """Try Groq first, then Ollama."""

        if self.groq_client is not None:
            if self.groq_client.check_connection():
                self.active_provider = "groq"
                return "groq"

        if self.ollama_client.check_connection():
            self.active_provider = "ollama"
            return "ollama"

        raise ConnectionError(
            "Neither Groq nor Ollama is available."
        )

    def select_model(self, task="general", provider=None):

        provider = provider or self.select_provider()

        if provider == "groq":
            return self.default_model

        if provider == "ollama":
            if task == "safety":
                return self.guard_model

            return self.ollama_model

        raise ValueError(
            f"Unknown AI provider: {provider}"
        )

    def get_available_models(self, provider=None):

        provider = provider or self.select_provider()

        if provider == "groq":
            return self.groq_client.list_models()

        if provider == "ollama":
            return self.ollama_client.list_models()

        return []
