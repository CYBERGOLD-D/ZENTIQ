
"""
ZENTIQ - Llama Guard Safety Module

Uses Llama Guard 3:1B through Ollama to screen
incoming user messages and outgoing AI responses.

Developed by EverMark Creative Solutions.
"""

from ai_engine.ollama_client import OllamaClient


class LlamaGuard:
    """Safety classifier for ZENTIQ conversations."""

    SAFETY_CATEGORIES = """
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

    VALID_CATEGORIES = {
        f"S{i}" for i in range(1, 14)
    }

    def __init__(
        self,
        ollama_client=None,
        guard_model="llama-guard3:1b"
    ):
        self.ollama_client = ollama_client or OllamaClient()
        self.guard_model = guard_model

    def check_connection(self):
        """Check whether Ollama and the guard model are available."""

        try:
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

    def _build_prompt(
        self,
        user_message,
        assistant_response=None
    ):
        """Build a Llama Guard 3 classification prompt."""

        role = "User" if assistant_response is None else "Agent"

        conversation = f"User: {user_message}"

        if assistant_response is not None:
            conversation += (
                f"\n\nAgent: {assistant_response}"
            )

        return f"""
Task: Check if there is unsafe content in '{role}' messages
in conversations according to our safety policy.

<BEGIN UNSAFE CONTENT CATEGORIES>
{self.SAFETY_CATEGORIES}
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

    def _classify(
        self,
        user_message,
        assistant_response=None
    ):
        """Classify a user message or assistant response."""

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

        if not self.check_connection():
            return {
                "safe": False,
                "label": "error",
                "reason": "Llama Guard or its model is unavailable."
            }

        guard_prompt = self._build_prompt(
            user_message=user_message,
            assistant_response=assistant_response
        )

        try:
            result = self.ollama_client.generate(
                prompt=guard_prompt,
                model=self.guard_model,
                temperature=0.0
            )

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
                return {
                    "safe": True,
                    "label": "safe",
                    "reason": "Content passed Llama Guard screening."
                }

            if first_line == "unsafe":

                if len(lines) < 2:
                    raise ValueError(
                        "Unsafe classification has no category."
                    )

                categories = {
                    category.strip().upper()
                    for category in lines[1].split(",")
                }

                if (
                    not categories
                    or not categories.issubset(
                        self.VALID_CATEGORIES
                    )
                ):
                    raise ValueError(
                        "Llama Guard returned invalid safety categories."
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

        except Exception as error:
            return {
                "safe": False,
                "label": "error",
                "reason": f"Safety screening failed: {error}"
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
        """Screen an AI response using its original user question."""

        if not user_message:
            user_message = (
                "Original user message was not supplied."
            )

        return self._classify(
            user_message=user_message,
            assistant_response=assistant_response
        )
