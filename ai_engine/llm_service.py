"""
ZENTIQ - Hybrid LLM Service

PRODUCTION:
    Primary AI: Cloudflare Gemma 4
    Fallback AI: Groq Cloud
    Safety: Cloudflare Llama Guard 3 8B

LOCAL DEVELOPMENT:
    Primary AI: Ollama Llama 3.2:1B
    Fallback AI: Groq Cloud
    Safety: Local Llama Guard 3:1B

Features:
    - Automatic provider fallback
    - Cloudflare diagnostic logging
    - Centralized input/output safety screening
    - Active provider tracking
    - Fail-closed safety handling

Developed by EverMark Creative Solutions.
"""

from ai_engine.model_manager import ModelManager
from ai_engine.prompt_templates import PromptTemplates
from safety.llama_guard import LlamaGuard


class LLMService:
    """Coordinates AI generation and safety screening."""

    def __init__(
        self,
        client=None,
        model_manager=None,
        groq_client=None,
        ollama_client=None,
        cloudflare_client=None,
        safety_guard=None
    ):
        # Backward compatibility with the previous client argument.
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

        self.safety_guard = safety_guard or LlamaGuard(
            ollama_client=self.ollama_client,
            guard_model=self.model_manager.guard_model,
            cloudflare_client=self.cloudflare_client,
            provider=self.model_manager.safety_provider
        )

        # Track the most recently successful provider.
        self.last_provider = None

    # --------------------------------------------------
    # 1. SYSTEM PROMPTS
    # --------------------------------------------------

    def get_system_prompt(self, template_name="general"):
        """Load the requested ZENTIQ system prompt."""

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

    # --------------------------------------------------
    # 2. AI PROVIDER GENERATION
    # --------------------------------------------------

    def _generate_with_provider(
        self,
        provider,
        prompt,
        model,
        system_prompt,
        temperature
    ):
        """Generate a response using the selected provider."""

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
            if self.ollama_client is None:
                raise ConnectionError(
                    "Ollama client is unavailable."
                )

            return self.ollama_client.generate(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature
            )

        raise ValueError(
            f"Unsupported AI provider: {provider}"
        )

    # --------------------------------------------------
    # 3. CENTRAL SAFETY SCREENING
    # --------------------------------------------------

    def _check_safety(
        self,
        content,
        stage,
        user_message=None
    ):
        """Screen incoming requests or outgoing responses."""

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
                "The safety system returned an invalid result."
            )

        return result.get("safe") is True

    # --------------------------------------------------
    # 4. PUBLIC INPUT SAFETY CHECK
    # --------------------------------------------------

    def check_input_safety(self, content):
        """Public safety check for incoming user messages."""

        return self._check_safety(
            content,
            stage="input"
        )

    # --------------------------------------------------
    # 5. PUBLIC OUTPUT SAFETY CHECK
    # --------------------------------------------------

    def check_output_safety(
        self,
        content,
        user_message=None
    ):
        """Public safety check for outgoing responses."""

        return self._check_safety(
            content,
            stage="output",
            user_message=user_message
        )

    # --------------------------------------------------
    # 6. PROVIDER DIAGNOSTICS
    # --------------------------------------------------

    @staticmethod
    def _log_provider_failure(provider, error):
        """
        Log provider failures without exposing credentials
        or user prompts.
        """

        message = (
            f"[ZENTIQ DIAGNOSTIC] "
            f"{provider} generation failed: "
            f"{type(error).__name__}: {str(error)[:300]}"
        )

        print(message, flush=True)

    # --------------------------------------------------
    # 7. MAIN AI RESPONSE GENERATION
    # --------------------------------------------------

    def generate_response(
        self,
        prompt,
        task="general",
        system_prompt=None,
        temperature=0.3,
        template_name="general"
    ):
        """Generate a response with safety checks and fallback."""

        # Reset provider status for every new request.
        self.model_manager.active_provider = None
        self.last_provider = None

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if system_prompt is None:
            system_prompt = self.get_system_prompt(
                template_name
            )

        # --------------------------------------------------
        # STEP 1: SCREEN THE INCOMING USER MESSAGE
        # --------------------------------------------------

        try:
            input_safe = self.check_input_safety(prompt)

        except Exception as error:
            print(
                "[ZENTIQ SAFETY DIAGNOSTIC] "
                "Input screening failed: "
                f"{type(error).__name__}",
                flush=True
            )

            raise RuntimeError(
                "ZENTIQ could not complete input safety screening. "
                "The request was not processed."
            ) from error

        if not input_safe:
            return (
                "I'm sorry, but I cannot process this request "
                "because it did not pass ZENTIQ's safety checks."
            )

        # --------------------------------------------------
        # STEP 2: GET THE PROVIDER PRIORITY ORDER
        # --------------------------------------------------

        try:
            providers = self.model_manager.get_provider_order()

        except Exception as error:
            raise ConnectionError(
                "ZENTIQ could not determine the AI provider order."
            ) from error

        if not providers:
            raise ConnectionError(
                "No conversational AI provider is configured."
            )

        # --------------------------------------------------
        # STEP 3: GENERATE WITH AUTOMATIC PROVIDER FALLBACK
        # --------------------------------------------------

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

                # Record diagnostic information without
                # logging prompts or authentication details.
                self._log_provider_failure(
                    provider,
                    error
                )

                provider_errors.append(
                    f"{provider}: {type(error).__name__}"
                )

        if response is None:

            error_summary = " | ".join(provider_errors)

            raise ConnectionError(
                "All conversational AI providers failed. "
                f"Provider errors: {error_summary}"
            )

        # --------------------------------------------------
        # STEP 4: SCREEN THE GENERATED AI RESPONSE
        # --------------------------------------------------

        try:
            output_safe = self.check_output_safety(
                response,
                user_message=prompt
            )

        except Exception as error:

            print(
                "[ZENTIQ SAFETY DIAGNOSTIC] "
                "Output screening failed: "
                f"{type(error).__name__}",
                flush=True
            )

            raise RuntimeError(
                "ZENTIQ could not complete response safety screening. "
                "The generated response was not released."
            ) from error

        if not output_safe:
            return (
                "ZENTIQ generated a response that did not pass "
                "the safety checks. The response has been withheld."
            )

        # --------------------------------------------------
        # STEP 5: RECORD THE SUCCESSFUL PROVIDER
        # --------------------------------------------------

        self.model_manager.active_provider = active_provider
        self.last_provider = active_provider

        print(
            "[ZENTIQ PROVIDER STATUS] "
            f"Successful provider: {active_provider}",
            flush=True
        )

        return response
generated a response that did not pass "
                "the safety checks. The response has been withheld."
            )

        # --------------------------------------------------
        # STEP 5: RECORD THE SUCCESSFUL PROVIDER
        # --------------------------------------------------

        self.model_manager.active_provider = active_provider

        return response
