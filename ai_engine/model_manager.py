"""
ZENTIQ - Model Manager

Manages available language models and model selection.
"""

from ai_engine.ollama_client import OllamaClient


class ModelManager:

    def __init__(self, client=None):
        self.client = client or OllamaClient()

        self.default_model = "llama3.2:1b"
        self.guard_model = "llama-guard3:1b"

    def get_available_models(self):
        """Return all models currently installed in Ollama."""
        return self.client.list_models()

    def is_model_available(self, model_name):
        """Check whether a requested model is installed."""
        return model_name in self.get_available_models()

    def select_model(self, task="general"):
        """Select a model according to the requested task."""

        if task == "safety":
            selected_model = self.guard_model
        else:
            selected_model = self.default_model

        if not self.is_model_available(selected_model):
            raise ValueError(
                f"Required model is not installed: {selected_model}"
            )

        return selected_model