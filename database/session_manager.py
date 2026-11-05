
"""
ZENTIQ - Authentication Session Manager

Manages authenticated Supabase sessions, token refresh,
session restoration and secure logout.
"""

import streamlit as st

from database.dp import get_supabase_client
from database.auth import (
    restore_user_session,
    get_session_tokens
)


SESSION_KEY = "zentiq_supabase_tokens"
USER_ID_KEY = "zentiq_authenticated_user_id"
USER_EMAIL_KEY = "zentiq_authenticated_user_email"


def save_authenticated_session(session):
    """Save authentication tokens in the current Streamlit session."""

    tokens = get_session_tokens(session)

    if not tokens:
        return False

    st.session_state[SESSION_KEY] = tokens

    return True


def clear_local_session():
    """Clear authentication data from the current Streamlit session."""

    for key in (
        SESSION_KEY,
        USER_ID_KEY,
        USER_EMAIL_KEY
    ):
        st.session_state.pop(key, None)


def get_authenticated_client():
    """
    Create a Supabase client and restore the user's session.

    Returns:
        client, user, error_message
    """

    try:
        client = get_supabase_client()

    except Exception as error:
        return None, None, (
            f"Unable to connect to Supabase: {error}"
        )

    tokens = st.session_state.get(SESSION_KEY)

    if not tokens:
        return client, None, None

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")

    if not access_token or not refresh_token:
        clear_local_session()

        return client, None, (
            "Your login session is incomplete. Please sign in again."
        )

    try:
        # Restore the session. Supabase refreshes an expired
        # access token when the refresh token remains valid.
        _, session = restore_user_session(
            client,
            access_token,
            refresh_token
        )

        if session is None:
            raise ValueError("Supabase did not return an active session.")

        # Verify the authenticated user with Supabase.
        user_response = client.auth.get_user()
        user = user_response.user

        if user is None or not user.id:
            raise ValueError("Unable to verify your account.")

        # Store the latest tokens because refresh tokens rotate.
        save_authenticated_session(session)

        st.session_state[USER_ID_KEY] = user.id
        st.session_state[USER_EMAIL_KEY] = user.email or ""

        return client, user, None

    except Exception as error:

        error_message = str(error).lower()

        invalid_session_markers = (
            "invalid refresh token",
            "refresh_token_not_found",
            "refresh token not found",
            "session_not_found",
            "invalid jwt",
            "jwt expired"
        )

        if any(
            marker in error_message
            for marker in invalid_session_markers
        ):
            clear_local_session()

            return client, None, (
                "Your login session has expired or is no longer valid. "
                "Please sign in again."
            )

        # Preserve tokens during temporary network or server errors.
        # This allows the user to retry without losing the session.
        return client, None, (
            "Unable to verify your session right now. "
            "Please check your connection and try again. "
            f"Details: {error}"
        )


def logout_authenticated_user(client):
    """
    Sign out only the current session and clear local credentials.
    """

    logout_error = None

    try:
        if client is not None:
            client.auth.sign_out(
                options={"scope": "local"}
            )

    except Exception as error:
        logout_error = str(error)

    finally:
        clear_local_session()

    if logout_error:
        return False, (
            "Your local session was cleared, but Supabase "
            f"could not confirm logout: {logout_error}"
        )

    return True, "You have been signed out successfully."
      
