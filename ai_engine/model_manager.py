"""
ZENTIQ - Hybrid Model Manager

PRODUCTION:
    Primary: Cloudflare Workers AI (Gemma 4)
    Fallback: Groq Cloud

LOCAL DEVELOPMENT:
    Primary: Ollama (Llama 3.2:1B)
    Fallback: Groq Cloud

Developed by EverMark Creative Solutions.
"""

import streamlit as st

from ai_engine.cloudflare_client import CloudflareClient
from ai_engine.groq_client import GroqClient
from ai_engine.ollama_client import OllamaClient


class ModelManager:
    """Manages ZENTIQ's conversational AI providers."""

    def __init__(
        self,
        groq_client=None,
        ollama_client=None,
        cloudflare_client=None
    ):
        self.cloudflare_enabled = self._has_cloudflare_secrets()

        self.cloudflare_client = cloudflare_client
        self.cloudflare_error = None

        if self.cloudflare_enabled and self.cloudflare_client is None:
            try:
                self.cloudflare_client = CloudflareClient()
            except Exception as error:
                self.cloudflare_error = error

        self.groq_client = groq_client

        if self.groq_client is None:
            try:
                self.groq_client = GroqClient()
            except Exception:
                self.groq_client = None

        self.ollama_client = (
            ollama_client or OllamaClient()
        )

        self.cloudflare_model = (
            "@cf/google/gemma-4-26b-a4b-it"
        )

        self.default_model = "openai/gpt-oss-20b"
        self.ollama_model = "llama3.2:1b"
        self.guard_model = "llama-guard3:1b"

        self.active_provider = None

        self.safety_provider = (
            "cloudflare"
            if self.cloudflare_enabled
            else "ollama"
        )

    @staticmethod
    def _has_cloudflare_secrets():
        """Check whether Cloudflare credentials are configured."""

        try:
            account_id = st.secrets.get(
                "CLOUDFLARE_ACCOUNT_ID"
            )

            api_token = st.secrets.get(
                "CLOUDFLARE_API_TOKEN"
            )

            return bool(
                account_id
                and str(account_id).strip()
                and api_token
                and str(api_token).strip()
            )

        except Exception:
            return False

    def get_provider_order(self):
        """Return providers in priority order."""

        if self.cloudflare_enabled:
            return ["cloudflare", "groq"]

        return ["ollama", "groq"]

    def select_provider(self):
        """Return the preferred conversational provider."""

        providers = self.get_provider_order()

        if not providers:
            raise ConnectionError(
                "No conversational AI provider is configured."
            )

        return providers[0]

    def select_model(self, task="general", provider=None):
        """Select the model associated with a provider."""

        provider = provider or self.select_provider()

        if provider == "cloudflare":
            return self.cloudflare_model

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
        """List models available through a provider."""

        provider = provider or self.select_provider()

        if provider == "cloudflare":
            if self.cloudflare_client is None:
                raise ConnectionError(
                    "Cloudflare client is unavailable."
                )

            return self.cloudflare_client.list_models()

        if provider == "groq":
            if self.groq_client is None:
                raise ConnectionError(
                    "Groq client is unavailable."
                )

            return self.groq_client.list_models()

        if provider == "ollama":
            return self.ollama_client.list_models()

        raise ValueError(
            f"Unknown AI provider: {provider}"
        )
        
