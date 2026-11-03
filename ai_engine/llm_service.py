
"""
ZENTIQ - Hybrid LLM Service

Coordinates AI requests between Groq and Ollama.
Groq is the primary provider.
Ollama is the automatic fallback provider.
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
        # Preserve compatibility with the previous client argument.
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
        """Retrieve the requested ZENTIQ prompt template."""

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
        """Generate a response with automatic provider fallback."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if system_prompt is None:
            system_prompt = self.get_system_prompt(template_name)

        # Select the primary available provider.
        provider = self.model_manager.select_provider()

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

                return response

            except Exception as groq_error:
                # Automatically try Ollama if Groq fails.
                if self.ollama_client.check_connection():

                    try:
                        fallback_model = (
                            self.model_manager.select_model(
                                task=task,
                                provider="ollama"
                            )
                        )

                        response = self.ollama_client.generate(
                            prompt=prompt,
                            model=fallback_model,
                            system_prompt=system_prompt,
                            temperature=temperature
                        )

                        return response

                    except Exception as ollama_error:
                        raise ConnectionError(
                            "Both Groq and Ollama failed. "
                            f"Groq error: {groq_error}. "
                            f"Ollama error: {ollama_error}"
                        ) from ollama_error

                raise ConnectionError(
                    "Groq failed and Ollama is unavailable. "
                    f"Groq error: {groq_error}"
                ) from groq_error

        # Use Ollama when Groq is unavailable.
        if provider == "ollama":

            model = self.model_manager.select_model(
                task=task,
                provider="ollama"
            )

            return self.ollama_client.generate(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature
            )

        raise ConnectionError(
            "No supported AI provider is available."
        )
