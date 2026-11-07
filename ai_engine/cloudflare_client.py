"""
ZENTIQ - Cloudflare Workers AI Client

Provides hosted AI generation and safety classification
through Cloudflare Workers AI.

Conversational model:
@cf/google/gemma-4-26b-a4b-it

Safety model:
@cf/meta/llama-guard-3-8b

Developed by EverMark Creative Solutions.
"""

import requests
import streamlit as st


class CloudflareClient:
    """Client for Cloudflare Workers AI."""

    BASE_URL = "https://api.cloudflare.com/client/v4/accounts"

    CHAT_MODEL = "@cf/google/gemma-4-26b-a4b-it"
    GUARD_MODEL = "@cf/meta/llama-guard-3-8b"

    def __init__(self, timeout=120):
        self.timeout = timeout

        try:
            self.account_id = st.secrets["CLOUDFLARE_ACCOUNT_ID"]
            self.api_token = st.secrets["CLOUDFLARE_API_TOKEN"]

        except (KeyError, FileNotFoundError) as error:
            raise ValueError(
                "Cloudflare credentials are missing. "
                "Configure CLOUDFLARE_ACCOUNT_ID and "
                "CLOUDFLARE_API_TOKEN in Streamlit Secrets."
            ) from error

        if not str(self.account_id).strip():
            raise ValueError("Cloudflare Account ID is empty.")

        if not str(self.api_token).strip():
            raise ValueError("Cloudflare API token is empty.")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        })

    def _run_model(self, model, payload):
        """Send a request to a Cloudflare Workers AI model."""

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
            data = response.json()

            if not isinstance(data, dict):
                raise ValueError(
                    "Cloudflare returned an invalid response."
                )

            if data.get("success") is False:
                raise RuntimeError(
                    f"Cloudflare API error: {data.get('errors', [])}"
                )

            result = data.get("result")

            if not isinstance(result, dict):
                raise ValueError(
                    "Cloudflare response has no valid result."
                )

            return result

        except requests.RequestException as error:
            raise ConnectionError(
                f"Cloudflare request failed: {error}"
            ) from error

    def check_connection(self):
        """Check whether the conversational model is available."""

        try:
            self.generate(
                prompt="Reply with the word READY.",
                model=self.CHAT_MODEL,
                temperature=0.0
            )
            return True

        except Exception:
            return False

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

        model = model or self.CHAT_MODEL

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        result = self._run_model(
            model=model,
            payload={
                "messages": messages,
                "temperature": temperature
            }
        )

        response = result.get("response")

        if isinstance(response, str) and response.strip():
            return response.strip()

        raise ValueError(
            "Cloudflare returned an empty or invalid AI response."
        )

    def check_safety(self, messages):
        """Run Llama Guard 3 8B on conversation messages."""

        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "Safety screening requires conversation messages."
            )

        result = self._run_model(
            model=self.GUARD_MODEL,
            payload={
                "messages": messages
            }
        )

        response = result.get("response")

        if not isinstance(response, str) or not response.strip():
            raise ValueError(
                "Llama Guard returned an empty classification."
            )

        return response.strip()

    def list_models(self):
        """Return the configured Cloudflare model identifiers."""

        return [
            self.CHAT_MODEL,
            self.GUARD_MODEL
          ]
