
"""
ZENTIQ - Groq Client
Handles communication between ZENTIQ and Groq hosted AI models.
"""

import streamlit as st
from groq import Groq


class GroqClient:
    """Client for communicating with Groq Cloud."""

    def __init__(self, timeout=120):
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError) as error:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Please configure it in Streamlit Secrets."
            ) from error

        if not api_key or not str(api_key).strip():
            raise ValueError(
                "GROQ_API_KEY is empty. "
                "Please check your Streamlit Secrets."
            )

        self.client = Groq(
            api_key=api_key,
            timeout=timeout
        )

    def check_connection(self):
        """Check whether Groq API is available."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def list_models(self):
        """Return available Groq models."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]

        except Exception as error:
            raise ConnectionError(
                f"Could not retrieve Groq models: {error}"
            ) from error

    def generate(
        self,
        prompt,
        model="openai/gpt-oss-20b",
        system_prompt=None,
        temperature=0.3
    ):
        """Generate a response using Groq."""

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

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

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )

            if not response.choices:
                raise ValueError(
                    "Groq returned no response choices."
                )

            content = response.choices[0].message.content

            if not content or not content.strip():
                raise ValueError(
                    "Groq returned an empty response."
                )

            return content.strip()

        except Exception as error:
            raise ConnectionError(
                f"Groq generation failed: {error}"
            ) from error
