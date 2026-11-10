"""
ZENTIQ - Cloudflare Workers AI Client

Production:
    Conversational AI: Cloudflare Gemma 4
    Safety classification: Cloudflare Llama Guard 3 8B

Features:
    - Secure Cloudflare authentication through Streamlit Secrets
    - Gemma 4 text generation
    - Llama Guard text and structured JSON response support
    - Strict safety-result validation
    - Safe error handling and connection diagnostics

Developed by EverMark Creative Solutions.
"""

import requests
import streamlit as st


class CloudflareClient:
    """Client for Cloudflare Workers AI."""

    BASE_URL = (
        "https://api.cloudflare.com/client/v4/accounts"
    )

    CHAT_MODEL = "@cf/google/gemma-4-26b-a4b-it"

    GUARD_MODEL = "@cf/meta/llama-guard-3-8b"

    def __init__(self):

        self.account_id = str(
            st.secrets.get("CLOUDFLARE_ACCOUNT_ID", "")
        ).strip()

        self.api_token = str(
            st.secrets.get("CLOUDFLARE_API_TOKEN", "")
        ).strip()

        if not self.account_id or not self.api_token:
            raise ValueError(
                "Cloudflare account credentials are not configured."
            )

        self.session = requests.Session()

        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    # --------------------------------------------------
    # 1. CLOUDFARE API REQUEST
    # --------------------------------------------------

    def _run_model(self, model, payload):
        """Execute a Workers AI inference request."""

        url = (
            f"{self.BASE_URL}/"
            f"{self.account_id}/ai/run/{model}"
        )

        try:

            response = self.session.post(
                url,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            try:
                data = response.json()

            except ValueError as error:
                raise RuntimeError(
                    "Cloudflare returned an invalid JSON response."
                ) from error

            if not isinstance(data, dict):
                raise RuntimeError(
                    "Cloudflare returned an invalid response structure."
                )

            if data.get("success") is not True:

                errors = data.get("errors", [])

                safe_errors = []

                if isinstance(errors, list):

                    for item in errors:

                        if isinstance(item, dict):

                            code = item.get("code")
                            message = item.get("message")

                            if code is not None:
                                safe_errors.append(
                                    f"Code {code}"
                                )

                            if isinstance(message, str):
                                safe_errors.append(
                                    message[:200]
                                )

                detail = "; ".join(safe_errors)

                raise RuntimeError(
                    "Cloudflare inference failed."
                    + (
                        f" Details: {detail}"
                        if detail
                        else ""
                    )
                )

            result = data.get("result")

            if not isinstance(result, dict):
                raise RuntimeError(
                    "Cloudflare returned an invalid inference result."
                )

            return result

        except requests.exceptions.Timeout as error:

            raise ConnectionError(
                "Cloudflare inference request timed out."
            ) from error

        except requests.exceptions.HTTPError as error:

            status = (
                error.response.status_code
                if error.response is not None
                else "unknown"
            )

            raise ConnectionError(
                f"Cloudflare HTTP request failed (status {status})."
            ) from error

        except requests.exceptions.RequestException as error:

            raise ConnectionError(
                "Cloudflare network request failed: "
                f"{type(error).__name__}."
            ) from error

    # --------------------------------------------------
    # 2. TEXT RESPONSE EXTRACTION
    # --------------------------------------------------

    @staticmethod
    def _extract_text(result):
        """Extract a normal text-generation response."""

        if not isinstance(result, dict):
            raise ValueError(
                "Cloudflare returned an invalid result."
            )

        response = result.get("response")

        if not isinstance(response, str):
            raise ValueError(
                "Cloudflare returned a non-text AI response."
            )

        response = response.strip()

        if not response:
            raise ValueError(
                "Cloudflare returned an empty AI response."
            )

        return response

    # --------------------------------------------------
    # 3. GEMMA 4 TEXT GENERATION
    # --------------------------------------------------

    def generate(
        self,
        prompt,
        model=None,
        system_prompt=None,
        temperature=0.3
    ):
        """Generate a conversational response using Gemma 4."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "A valid generation prompt is required."
            )

        model = model or self.CHAT_MODEL

        messages = []

        if (
            isinstance(system_prompt, str)
            and system_prompt.strip()
        ):

            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "messages": messages,
            "temperature": float(temperature)
        }

        result = self._run_model(
            model=model,
            payload=payload
        )

        return self._extract_text(result)

    # --------------------------------------------------
    # 4. LLAMA GUARD RESPONSE NORMALIZATION
    # --------------------------------------------------

    @staticmethod
    def _normalize_safety_response(result):
        """
        Normalize Cloudflare Llama Guard responses.

        Supported formats:

        Text:
            safe

        Text:
            unsafe
            S1, S2

        Structured:
            {
                "safe": true
            }

        Structured:
            {
                "safe": false,
                "categories": ["S1", "S2"]
            }

        Returns the text format expected by LlamaGuard.
        """

        if not isinstance(result, dict):
            raise ValueError(
                "Invalid Cloudflare safety result."
            )

        response = result.get("response")

        # Format A: Cloudflare returns classification text.
        if isinstance(response, str):

            response = response.strip()

            if not response:
                raise ValueError(
                    "Cloudflare returned an empty safety classification."
                )

            return response

        # Format B: Cloudflare returns structured JSON.
        if isinstance(response, dict):

            safe = response.get("safe")

            if not isinstance(safe, bool):
                raise ValueError(
                    "Structured safety response has no valid Boolean "
                    "'safe' field."
                )

            categories = response.get("categories", [])

            if not isinstance(categories, list):

                raise ValueError(
                    "Safety categories must be returned as a list."
                )

            if not all(
                isinstance(category, str)
                and category.strip()
                for category in categories
            ):

                raise ValueError(
                    "Safety response contains invalid categories."
                )

            categories = [
                category.strip().upper()
                for category in categories
            ]

            if safe:

                if categories:
                    raise ValueError(
                        "Safe classification unexpectedly contains "
                        "unsafe categories."
                    )

                return "safe"

            if not categories:
                raise ValueError(
                    "Unsafe classification has no safety categories."
                )

            return (
                "unsafe\n"
                + ", ".join(categories)
            )

        raise ValueError(
            "Cloudflare returned an unsupported safety response format."
        )

    # --------------------------------------------------
    # 5. LLAMA GUARD SAFETY CLASSIFICATION
    # --------------------------------------------------

    def check_safety(self, messages):
        """Classify a conversation using Cloudflare Llama Guard 3."""

        if not isinstance(messages, list) or not messages:

            raise ValueError(
                "Valid conversation messages are required."
            )

        for message in messages:

            if not isinstance(message, dict):

                raise ValueError(
                    "Invalid conversation message structure."
                )

            if message.get("role") not in (
                "user",
                "assistant"
            ):

                raise ValueError(
                    "Unsupported conversation message role."
                )

            if (
                not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):

                raise ValueError(
                    "Conversation messages must contain valid text."
                )

        payload = {
            "messages": messages,
            "temperature": 0.0
        }

        result = self._run_model(
            model=self.GUARD_MODEL,
            payload=payload
        )

        return self._normalize_safety_response(result)

    # --------------------------------------------------
    # 6. CONNECTION TESTS
    # --------------------------------------------------

    def check_connection(self):
        """Test the Cloudflare conversational AI endpoint."""

        try:

            result = self._run_model(
                model=self.CHAT_MODEL,
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Reply with the word OK."
                        }
                    ],
                    "temperature": 0.0
                }
            )

            self._extract_text(result)

            return True

        except Exception:
            return False

    def check_safety_connection(self):
        """Test Cloudflare Llama Guard classification."""

        try:

            result = self.check_safety([
                {
                    "role": "user",
                    "content": "Hello."
                }
            ])

            return result.strip().lower() == "safe"

        except Exception:
            return False

    # --------------------------------------------------
    # 7. AVAILABLE MODELS
    # --------------------------------------------------

    def list_models(self):
        """Return the Cloudflare models configured for ZENTIQ."""

        return [
            self.CHAT_MODEL,
            self.GUARD_MODEL
        ]
