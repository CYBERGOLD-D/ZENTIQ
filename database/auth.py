
"""
ZENTIQ - Supabase Authentication
Handles user registration, login, session restoration and logout.
"""

from supabase import Client


def register_user(
    client: Client,
    email: str,
    password: str
):
    """Register a new user using email and password."""

    email = email.strip().lower()

    if not email or not password:
        raise ValueError("Email and password are required.")

    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")

    response = client.auth.sign_up({
        "email": email,
        "password": password
    })

    return response.user, response.session


def login_user(
    client: Client,
    email: str,
    password: str
):
    """Authenticate an existing user."""

    email = email.strip().lower()

    if not email or not password:
        raise ValueError("Email and password are required.")

    response = client.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

    return response.user, response.session


def restore_user_session(
    client: Client,
    access_token: str,
    refresh_token: str
):
    """Restore and refresh an existing authenticated session."""

    if not access_token or not refresh_token:
        raise ValueError("Valid session tokens are required.")

    response = client.auth.set_session(
        access_token,
        refresh_token
    )

    return response.user, response.session


def get_session_tokens(session):
    """Extract tokens for Streamlit session-state storage."""

    if session is None:
        return None

    return {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token
    }


def logout_user(client: Client):
    """Sign out the currently authenticated user."""

    client.auth.sign_out()
  
