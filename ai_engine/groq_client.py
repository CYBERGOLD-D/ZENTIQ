
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
        except (KeyError, FileNotFoundError):
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Please configure it in Streamlit Secrets."
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
        model="llama-3.3-70b-versatile",
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

            return (
                response.choices[0].message.content or ""
            ).strip()

        except Exception as error:
            raise ConnectionError(
                f"Groq generation failed: {error}"
            ) from error
              
