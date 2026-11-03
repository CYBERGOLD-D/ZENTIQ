
"""
ZENTIQ - Hybrid LLM Service

Groq is the primary AI provider.
Ollama is the automatic backup provider.

ZENTIQ automatically identifies and displays the provider
that successfully generates each response.
"""

from ai_engine.model_manager import ModelManager
from ai_engine.prompt_templates import PromptTemplates


class LLMService:

    def __init__(
        self,
        client=None,
        model_manager=None,
        groq_client=None,
        ollama_client=None
    ):
        # Maintain compatibility with the previous client setup.
        if client is not None and ollama_client is None:
            ollama_client = client

        self.model_manager = model_manager or ModelManager(
            groq_client=groq_client,
            ollama_client=ollama_client
        )

        self.groq_client = self.model_manager.groq_client
        self.ollama_client = self.model_manager.ollama_client

        self.prompt_templates = PromptTemplates()

    def get_system_prompt(self, template_name="general"):
        """Retrieve the appropriate ZENTIQ assistant instructions."""

        templates = {
            "general": self.prompt_templates.general_business_assistant,
            "customer_support": self.prompt_templates.customer_support,
            "financial": self.prompt_templates.financial_assistant,
            "sales": self.prompt_templates.sales_assistant
        }

        if template_name not in templates:
            raise ValueError(
                f"Unknown prompt template: {template_name}"
            )

        return templates[template_name]()

    def generate_response(
        self,
        prompt,
        task="general",
        system_prompt=None,
        temperature=0.3,
        template_name="general"
    ):
        """
        Generate an AI response using Groq or Ollama.

        Groq is attempted first.
        Ollama automatically takes over when Groq fails.
        The active provider is updated after successful generation.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if system_prompt is None:
            system_prompt = self.get_system_prompt(template_name)

        # Identify the first available AI provider.
        provider = self.model_manager.select_provider()

        # ------------------------------------------------
        # PRIMARY PROVIDER: GROQ
        # ------------------------------------------------
        if provider == "groq":

            model = self.model_manager.select_model(
                task=task,
                provider="groq"
            )

            try:
                response = self.groq_client.generate(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature
                )

                # Groq successfully generated the response.
                self.model_manager.active_provider = "groq"

                return response

            except Exception as groq_error:

                # Groq failed. Check whether Ollama can take over.
                if self.ollama_client.check_connection():

                    try:
                        fallback_model = self.model_manager.select_model(
                            task=task,
                            provider="ollama"
                        )

                        response = self.ollama_client.generate(
                            prompt=prompt,
                            model=fallback_model,
                            system_prompt=system_prompt,
                            temperature=temperature
                        )

                        # Ollama successfully took over.
                        self.model_manager.active_provider = "ollama"

                        return response

                    except Exception as ollama_error:
                        raise ConnectionError(
                            "Both AI providers failed. "
                            f"Groq error: {groq_error}. "
                            f"Ollama error: {ollama_error}"
                        ) from ollama_error

                raise ConnectionError(
                    "Groq failed and Ollama is unavailable. "
                    f"Groq error: {groq_error}"
                ) from groq_error

        # ------------------------------------------------
        # BACKUP PROVIDER: OLLAMA
        # ------------------------------------------------
        if provider == "ollama":

            model = self.model_manager.select_model(
                task=task,
                provider="ollama"
            )

            response = self.ollama_client.generate(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature
            )

            # Ollama is handling the request.
            self.model_manager.active_provider = "ollama"

            return response

        raise ConnectionError(
            "No supported AI provider is available."
        )
