"""
ZENTIQ - LLM Service

Coordinates requests and responses between ZENTIQ
and locally hosted language models.
"""

from ai_engine.ollama_client import OllamaClient
from ai_engine.model_manager import ModelManager
from ai_engine.prompt_templates import PromptTemplates


class LLMService:

    def __init__(self, client=None, model_manager=None):

        self.client = client or OllamaClient()

        self.model_manager = (
            model_manager or ModelManager(self.client)
        )

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
        """Generate a response using the selected language model."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # Retrieve the appropriate system instructions
        if system_prompt is None:
            system_prompt = self.get_system_prompt(template_name)

        # Select the appropriate language model
        model = self.model_manager.select_model(task)

        # Generate the AI response
        response = self.client.generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature
        )

        return response