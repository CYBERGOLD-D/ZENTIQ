
"""
ZENTIQ - Supabase Database Connection
Handles secure Supabase client creation.
"""

import streamlit as st
from supabase import create_client, Client


def get_supabase_client() -> Client:
    """
    Create a Supabase client using Streamlit secrets.

    Uses the publishable/anon key, never the service-role key.
    """

    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_ANON_KEY"]

    except (KeyError, FileNotFoundError) as error:
        raise ValueError(
            "Supabase configuration is missing. "
            "Check SUPABASE_URL and SUPABASE_ANON_KEY."
        ) from error

    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials cannot be empty.")

    return create_client(
        supabase_url,
        supabase_key
    )
  
