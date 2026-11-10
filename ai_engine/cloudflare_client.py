"""
ZENTIQ - Cloudflare Workers AI Client

Provides hosted AI generation and safety classification
through Cloudflare Workers AI.

Production conversational model:
    @cf/google/gemma-4-26b-a4b-it

Production safety model:
    @cf/meta/llama-guard-3-8b

Developed by EverMark Creative Solutions.
"""

import requests
import streamlit as st


class CloudflareClient:
    """Client for communicating with Cloudflare Workers AI."""

    BASE_URL = "https://api.cloudflare.com/client/v4/accounts"

    CHAT_MODEL = "@cf/google/gemma-4-26b-a4b-it"
    GUARD_MODEL = "@cf/meta/llama-guard-3-8b"

    def __init__(self, timeout=120):
        self.timeout = timeout

        try:
            self.account_id = st.secrets[
                "CLOUDFLARE_ACCOUNT_ID"
            ]
            self.api_token = st.secrets[
                "CLOUDFLARE_API_TOKEN"
            ]

        except (KeyError, FileNotFoundError) as error:
            raise ValueError(
                "Cloudflare credentials are missing. "
                "Configure CLOUDFLARE_ACCOUNT_ID and "
                "CLOUDFLARE_API_TOKEN in Streamlit Secrets."
            ) from error

        self.account_id = str(self.account_id).strip()
        self.api_token = str(self.api_token).strip()

        if not self.account_id:
            raise ValueError("Cloudflare Account ID is empty.")

        if not self.api_token:
            raise ValueError("Cloudflare API token is empty.")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    def _run_model(self, model, payload):
        """Execute a Cloudflare Workers AI model request."""

        if not isinstance(model, str) or not model.strip():
            raise ValueError("A valid Cloudflare model ID is required.")

        if not isinstance(payload, dict):
            raise ValueError(
                "Cloudflare request payload must be a dictionary."
            )

        url = (
            f"{self.BASE_URL}/{self.account_id}"
            f"/ai/run/{model}"
        )

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as error:
                raise ValueError(
                    "Cloudflare returned invalid JSON."
                ) from error

            if not isinstance(data, dict):
                raise ValueError(
                    "Cloudflare returned an invalid response structure."
                )

            if data.get("success") is not True:
                raise RuntimeError(
                    f"Cloudflare API request was unsuccessful: "
                    f"{data.get('errors', [])}"
                )

            result = data.get("result")

            if not isinstance(result, dict):
                raise ValueError(
                    "Cloudflare response contains no valid result object."
                )

            return result

        except requests.Timeout as error:
            raise ConnectionError(
                "Cloudflare request timed out."
            ) from error

        except requests.HTTPError as error:
            status = (
                error.response.status_code
                if error.response is not None
                else "unknown"
            )
            raise ConnectionError(
                f"Cloudflare HTTP request failed (status {status})."
            ) from error

        except requests.RequestException as error:
            raise ConnectionError(
                f"Unable to communicate with Cloudflare: {error}"
            ) from error

    @staticmethod
    def _extract_text(result, model_name):
        """Extract and validate generated text."""

        response = result.get("response")

        if not isinstance(response, str):
            raise ValueError(
                f"{model_name} returned an invalid response format."
            )

        response = response.strip()

        if not response:
            raise ValueError(
                f"{model_name} returned an empty response."
            )

        return response

    def generate(
        self,
        prompt,
        model=None,
        system_prompt=None,
        temperature=0.3
    ):
        """Generate a conversational AI response."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise ValueError(
                "Temperature must be a number between 0 and 2."
            )

        model = model or self.CHAT_MODEL

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": str(system_prompt)
            })

        messages.append({
            "role": "user",
            "content": prompt.strip()
        })

        result = self._run_model(
            model=model,
            payload={
                "messages": messages,
                "temperature": float(temperature)
            }
        )

        return self._extract_text(
            result,
            "Cloudflare conversational model"
        )

    def check_safety(self, messages):
        """
        Run Cloudflare Llama Guard 3 8B.

        Returns raw classification text for validation
        by the LlamaGuard safety module.
        """

        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "Safety screening requires a non-empty message list."
            )

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(
                    "Each safety message must be a dictionary."
                )

            if message.get("role") not in (
                "system", "user", "assistant"
            ):
                raise ValueError(
                    "Invalid role in safety message."
                )

            if (
                not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ValueError(
                    "Safety message content must be non-empty text."
                )

        result = self._run_model(
            model=self.GUARD_MODEL,
            payload={
                "messages": messages,
                "temperature": 0.0
            }
        )

        return self._extract_text(
            result,
            "Cloudflare Llama Guard"
        )

    def check_connection(self):
        """Check conversational model availability."""

        try:
            response = self.generate(
                prompt="Reply with exactly the word READY.",
                model=self.CHAT_MODEL,
                temperature=0.0
            )
            return bool(response)

        except Exception:
            return False

    def check_safety_connection(self):
        """Check safety model availability."""

        try:
            response = self.check_safety([
                {
                    "role": "user",
                    "content": "Hello."
                }
            ])
            return bool(response)

        except Exception:
            return False

    def list_models(self):
        """Return configured Cloudflare model identifiers."""

        return [
            self.CHAT_MODEL,
            self.GUARD_MODEL
        ]
