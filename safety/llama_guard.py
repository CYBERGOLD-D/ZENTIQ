"""
ZENTIQ - Hybrid Llama Guard Safety Module

Production:
    Cloudflare Llama Guard 3 8B

Local development:
    Ollama Llama Guard 3 1B

Screens incoming user messages and outgoing AI responses.
Fails closed whenever safety screening is unavailable
or returns an invalid classification.

Developed by EverMark Creative Solutions.
"""

import streamlit as st

from ai_engine.ollama_client import OllamaClient
from ai_engine.cloudflare_client import CloudflareClient


class LlamaGuard:
    """Hybrid safety classifier for ZENTIQ conversations."""

    BASE_CATEGORIES = """
S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate.
S11: Self-Harm.
S12: Sexual Content.
S13: Elections.
"""

    S14_CATEGORY = """
S14: Code Interpreter Abuse.
"""

    def __init__(
        self,
        ollama_client=None,
        guard_model="llama-guard3:1b",
        cloudflare_client=None,
        provider=None
    ):
        self.provider = provider or self._get_provider()

        self.guard_model = guard_model

        self.ollama_client = (
            ollama_client or OllamaClient()
        )

        self.cloudflare_client = cloudflare_client

        if self.provider == "cloudflare":

            self.cloudflare_client = (
                self.cloudflare_client or CloudflareClient()
            )

        elif self.provider != "ollama":

            raise ValueError(
                f"Unsupported safety provider: {self.provider}"
            )

    @staticmethod
    def _get_provider():
        """Select Cloudflare when production credentials exist."""

        try:
            account_id = st.secrets.get(
                "CLOUDFLARE_ACCOUNT_ID"
            )

            api_token = st.secrets.get(
                "CLOUDFLARE_API_TOKEN"
            )

            if (
                account_id
                and str(account_id).strip()
                and api_token
                and str(api_token).strip()
            ):
                return "cloudflare"

        except Exception:
            pass

        return "ollama"

    @property
    def valid_categories(self):
        """Return categories supported by the active guard model."""

        if self.provider == "cloudflare":
            return {
                f"S{i}" for i in range(1, 15)
            }

        # Llama Guard 3 1B is not optimized for S14.
        return {
            f"S{i}" for i in range(1, 14)
        }

    def _get_categories_prompt(self):
        """Return the category definitions for the active model."""

        categories = self.BASE_CATEGORIES

        if self.provider == "cloudflare":
            categories += self.S14_CATEGORY

        return categories

    def check_connection(self):
        """
        Verify that the configured safety provider is available.

        Cloudflare is tested by running an actual safety
        classification and validating its response.

        Ollama is tested by checking connectivity and
        confirming that the required guard model exists.
        """

        try:

            if self.provider == "cloudflare":

                result = self.cloudflare_client.check_safety([
                    {
                        "role": "user",
                        "content": "Hello."
                    }
                ])

                parsed = self._parse_result(result)

                return parsed.get("label") in (
                    "safe",
                    "unsafe"
                )

            if self.provider == "ollama":

                if not self.ollama_client.check_connection():
                    return False

                models = self.ollama_client.list_models()

                return any(
                    model == self.guard_model
                    or model.startswith(self.guard_model + "-")
                    for model in models
                )

        except Exception:
            return False

        return False

    def _build_messages(
        self,
        user_message,
        assistant_response=None
    ):
        """Build conversation messages for Cloudflare Llama Guard."""

        messages = [
            {
                "role": "user",
                "content": user_message
            }
        ]

        if assistant_response is not None:

            messages.append({
                "role": "assistant",
                "content": assistant_response
            })

        return messages

    def _build_prompt(
        self,
        user_message,
        assistant_response=None
    ):
        """Build a local Ollama Llama Guard classification prompt."""

        role = (
            "User"
            if assistant_response is None
            else "Agent"
        )

        conversation = f"User: {user_message}"

        if assistant_response is not None:

            conversation += (
                f"\n\nAgent: {assistant_response}"
            )

        return f"""
Task: Check if there is unsafe content in '{role}' messages
in conversations according to the safety policy below.

<BEGIN UNSAFE CONTENT CATEGORIES>
{self._get_categories_prompt()}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{conversation}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST {role}
message in the above conversation.

- First line must read 'safe' or 'unsafe'.
- If unsafe, the second line must include a
  comma-separated list of violated categories.
"""

    def _parse_result(self, result):
        """
        Strictly validate a Llama Guard classification.

        Unknown, empty or malformed responses raise an error.
        They are never interpreted as safe.
        """

        if not isinstance(result, str) or not result.strip():

            raise ValueError(
                "Llama Guard returned an empty result."
            )

        lines = [
            line.strip()
            for line in result.strip().splitlines()
            if line.strip()
        ]

        first_line = lines[0].lower()

        if first_line == "safe":

            if len(lines) != 1:
                raise ValueError(
                    "Safe classification contains unexpected extra data."
                )

            return {
                "safe": True,
                "label": "safe",
                "reason": (
                    "Content passed Llama Guard screening."
                )
            }

        if first_line == "unsafe":

            if len(lines) < 2:
                raise ValueError(
                    "Unsafe classification has no category."
                )

            categories = {
                category.strip().upper()
                for category in lines[1].split(",")
                if category.strip()
            }

            if not categories:
                raise ValueError(
                    "Unsafe classification contains no valid categories."
                )

            if not categories.issubset(self.valid_categories):

                raise ValueError(
                    "Llama Guard returned unsupported safety categories: "
                    f"{categories}"
                )

            return {
                "safe": False,
                "label": "unsafe",
                "categories": sorted(categories),
                "reason": result.strip()
            }

        raise ValueError(
            f"Unexpected Llama Guard response: {result}"
        )

    def _classify(
        self,
        user_message,
        assistant_response=None
    ):
        """Classify user input or an AI-generated response."""

        if (
            not isinstance(user_message, str)
            or not user_message.strip()
        ):

            return {
                "safe": False,
                "label": "error",
                "reason": "A valid user message is required."
            }

        if assistant_response is not None:

            if (
                not isinstance(assistant_response, str)
                or not assistant_response.strip()
            ):

                return {
                    "safe": False,
                    "label": "error",
                    "reason": "A valid assistant response is required."
                }

        try:

            if not self.check_connection():

                raise ConnectionError(
                    f"{self.provider} safety provider is unavailable."
                )

            if self.provider == "cloudflare":

                messages = self._build_messages(
                    user_message=user_message,
                    assistant_response=assistant_response
                )

                result = self.cloudflare_client.check_safety(
                    messages
                )

            else:

                guard_prompt = self._build_prompt(
                    user_message=user_message,
                    assistant_response=assistant_response
                )

                result = self.ollama_client.generate(
                    prompt=guard_prompt,
                    model=self.guard_model,
                    temperature=0.0
                )

            return self._parse_result(result)

        except Exception as error:

            return {
                "safe": False,
                "label": "error",
                "reason": (
                    f"Safety screening failed: {error}"
                )
            }

    def check_input(self, user_message):
        """Screen a message submitted by the user."""

        return self._classify(
            user_message=user_message
        )

    def check_output(
        self,
        assistant_response,
        user_message=None
    ):
        """Screen an AI response using its original question."""

        if not user_message:

            user_message = (
                "Original user message was not supplied."
            )

        return self._classify(
            user_message=user_message,
            assistant_response=assistant_response
        )
