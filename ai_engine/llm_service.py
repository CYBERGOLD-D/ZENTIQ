"""
ZENTIQ - Hybrid LLM Service

Production:
    Primary conversational provider: Cloudflare Gemma 4
    Fallback conversational provider: Groq Cloud
    Safety provider: Cloudflare Llama Guard 3 8B

Local development:
    Primary conversational provider: Ollama Llama 3.2:1B
    Fallback conversational provider: Groq Cloud
    Safety provider: Local Ollama Llama Guard 3:1B

All incoming messages and generated responses
are screened before processing or release.

Developed by EverMark Creative Solutions.
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
        cloudflare_client=None,
        safety_guard=None
    ):
        # Preserve compatibility with the previous client argument.
        if client is not None and ollama_client is None:
            ollama_client = client

        self.model_manager = model_manager or ModelManager(
            groq_client=groq_client,
            ollama_client=ollama_client,
            cloudflare_client=cloudflare_client
        )

        self.groq_client = self.model_manager.groq_client
        self.ollama_client = self.model_manager.ollama_client

        self.cloudflare_client = getattr(
            self.model_manager,
            "cloudflare_client",
            None
        )

        self.prompt_templates = PromptTemplates()

        safety_provider = getattr(
            self.model_manager,
            "safety_provider",
            None
        )

        self.safety_guard = safety_guard or LlamaGuard(
            ollama_client=self.ollama_client,
            guard_model=self.model_manager.guard_model,
            cloudflare_client=self.cloudflare_client,
            provider=safety_provider
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

        if provider == "cloudflare":
            if self.cloudflare_client is None:
                raise ConnectionError(
                    "Cloudflare client is unavailable."
                )

            return self.cloudflare_client.generate(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature
            )

        if provider == "groq":
            if self.groq_client is None:
                raise ConnectionError(
                    "Groq client is unavailable."
                )

            return self.groq_client.generate(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature
            )

        if provider == "ollama":
            return self.ollama_client.generate(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature
            )

        raise ValueError(
            f"Unsupported AI provider: {provider}"
        )

    def _check_safety(
        self,
        content,
        stage,
        user_message=None
    ):
        """Screen content using the configured Llama Guard."""

        if stage == "input":
            result = self.safety_guard.check_input(content)

        elif stage == "output":
            result = self.safety_guard.check_output(
                assistant_response=content,
                user_message=user_message
            )

        else:
            raise ValueError(
                f"Unsupported safety screening stage: {stage}"
            )

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
        Screen user input, generate using the preferred provider,
        attempt fallback if needed, and screen the final response.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if system_prompt is None:
            system_prompt = self.get_system_prompt(template_name)

        # -----------------------------------------------
        # STAGE 1: SCREEN USER INPUT
        # -----------------------------------------------

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
                "I'm sorry, but I cannot process this request "
                "because it did not pass ZENTIQ's safety checks."
            )

        # -----------------------------------------------
        # STAGE 2: GET PROVIDER PRIORITY
        # -----------------------------------------------

        try:
            providers = self.model_manager.get_provider_order()

        except Exception as error:
            raise ConnectionError(
                "ZENTIQ could not determine the available "
                "AI provider order."
            ) from error

        if not providers:
            raise ConnectionError(
                "No conversational AI provider is configured."
            )

        # -----------------------------------------------
        # STAGE 3: GENERATE WITH PRIMARY / FALLBACK
        # -----------------------------------------------

        response = None
        active_provider = None
        provider_errors = []

        for provider in providers:

            try:
                model = self.model_manager.select_model(
                    task=task,
                    provider=provider
                )

                candidate = self._generate_with_provider(
                    provider=provider,
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature
                )

                if (
                    not isinstance(candidate, str)
                    or not candidate.strip()
                ):
                    raise ValueError(
                        f"{provider} returned an empty response."
                    )

                response = candidate.strip()
                active_provider = provider
                break

            except Exception as error:
                provider_errors.append(
                    f"{provider}: {error}"
                )

        if response is None:
            error_summary = " | ".join(provider_errors)

            raise ConnectionError(
                "All conversational AI providers failed. "
                f"Details: {error_summary}"
            )

        # -----------------------------------------------
        # STAGE 4: SCREEN GENERATED RESPONSE
        # -----------------------------------------------

        try:
            output_safe = self._check_safety(
                response,
                stage="output",
                user_message=prompt
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

        # -----------------------------------------------
        # STAGE 5: RECORD ACTIVE PROVIDER
        # -----------------------------------------------

        self.model_manager.active_provider = active_provider

        return response
                    
