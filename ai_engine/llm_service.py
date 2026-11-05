
"""
ZENTIQ - Hybrid LLM Service

Primary conversational provider: Ollama
Primary conversational model: Llama 3.2:1B

Safety model: Llama Guard 3:1B
Secondary conversational provider: Groq Cloud

All incoming messages and generated responses
are screened through Llama Guard.
"""

from ai_engine.model_manager import ModelManager
from ai_engine.prompt_templates import PromptTemplates
from safety.llama_guard import LlamaGuard


class LLMService:

    def __init__(
        self,
        client=None,
        model_manager=None,
        groq_client=None,
        ollama_client=None,
        safety_guard=None
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

        # Use the same Ollama connection for Llama Guard.
        self.safety_guard = safety_guard or LlamaGuard(
            ollama_client=self.ollama_client,
            guard_model=self.model_manager.guard_model
        )

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

    def _generate_with_provider(
        self,
        provider,
        prompt,
        model,
        system_prompt,
        temperature
    ):
        """Generate a response using the specified provider."""

        if provider == "ollama":
            return self.ollama_client.generate(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature
            )

        if provider == "groq":
            return self.groq_client.generate(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature
            )

        raise ValueError(
            f"Unsupported AI provider: {provider}"
        )

    def _check_safety(self, content, stage):
        """Screen content using Llama Guard."""

        if stage == "input":
            result = self.safety_guard.check_input(content)
        else:
            result = self.safety_guard.check_output(content)

        if not isinstance(result, dict):
            raise RuntimeError(
                "Llama Guard returned an invalid safety result."
            )

        if result.get("safe") is not True:
            return False

        return True

    def generate_response(
        self,
        prompt,
        task="general",
        system_prompt=None,
        temperature=0.3,
        template_name="general"
    ):
        """
        Generate a response using Ollama first and Groq
        as fallback.

        Screen the user's message before generation and
        the AI response before returning it.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if system_prompt is None:
            system_prompt = self.get_system_prompt(template_name)

        # ------------------------------------------------
        # STAGE 1: SCREEN USER INPUT
        # ------------------------------------------------

        try:
            input_safe = self._check_safety(
                prompt,
                stage="input"
            )

        except Exception as error:
            raise RuntimeError(
                "ZENTIQ could not complete input safety screening. "
                "The request was not processed."
            ) from error

        if not input_safe:
            return (
                "I’m sorry, but I cannot process this request "
                "because it did not pass ZENTIQ's safety checks."
            )

        # ------------------------------------------------
        # STAGE 2: SELECT CONVERSATIONAL PROVIDER
        # ------------------------------------------------

        try:
            provider = self.model_manager.select_provider()

        except Exception as error:
            raise ConnectionError(
                "Neither Ollama nor Groq is currently available."
            ) from error

        # ------------------------------------------------
        # STAGE 3: GENERATE RESPONSE
        # ------------------------------------------------

        if provider == "ollama":

            model = self.model_manager.select_model(
                task=task,
                provider="ollama"
            )

            try:
                response = self._generate_with_provider(
                    provider="ollama",
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature
                )

                active_provider = "ollama"

            except Exception as ollama_error:

                # Ollama failed. Attempt Groq fallback.
                try:
                    if (
                        self.groq_client is None
                        or not self.groq_client.check_connection()
                    ):
                        raise ConnectionError(
                            "Groq is unavailable."
                        )

                    fallback_model = self.model_manager.select_model(
                        task=task,
                        provider="groq"
                    )

                    response = self._generate_with_provider(
                        provider="groq",
                        prompt=prompt,
                        model=fallback_model,
                        system_prompt=system_prompt,
                        temperature=temperature
                    )

                    active_provider = "groq"

                except Exception as groq_error:
                    raise ConnectionError(
                        "Both conversational providers failed. "
                        f"Ollama error: {ollama_error}. "
                        f"Groq error: {groq_error}."
                    ) from groq_error

        elif provider == "groq":

            model = self.model_manager.select_model(
                task=task,
                provider="groq"
            )

            try:
                response = self._generate_with_provider(
                    provider="groq",
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature
                )

                active_provider = "groq"

            except Exception as error:
                raise ConnectionError(
                    f"Groq generation failed: {error}"
                ) from error

        else:
            raise ConnectionError(
                "No supported conversational provider is available."
            )

        # ------------------------------------------------
        # STAGE 4: VALIDATE GENERATED RESPONSE
        # ------------------------------------------------

        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(
                "The AI provider returned an empty response."
            )

        # ------------------------------------------------
        # STAGE 5: SCREEN AI RESPONSE
        # ------------------------------------------------

        try:
            output_safe = self._check_safety(
                response,
                stage="output"
            )

        except Exception as error:
            raise RuntimeError(
                "ZENTIQ could not complete response safety screening. "
                "The generated response was not released."
            ) from error

        if not output_safe:
            return (
                "ZENTIQ generated a response that did not pass "
                "the safety checks. The response has been withheld."
            )

        # Record the provider that generated the approved response.
        self.model_manager.active_provider = active_provider

        return response
