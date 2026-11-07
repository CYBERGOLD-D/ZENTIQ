"""
ZENTIQ - Conversation Repository

Handles creating, retrieving, updating and deleting
authenticated users' conversations in Supabase.
"""

from datetime import datetime, timezone
from supabase import Client


def validate_messages(messages):
    """Validate the conversation message structure."""

    if not isinstance(messages, list):
        raise ValueError("Messages must be provided as a list.")

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a dictionary.")

        if message.get("role") not in ("user", "assistant"):
            raise ValueError("Invalid message role.")

        if not isinstance(message.get("content"), str):
            raise ValueError("Message content must be text.")

    return messages


def create_conversation(
    client: Client,
    user_id: str,
    title: str,
    mode: str,
    messages: list
):
    """Create a new conversation for an authenticated user."""

    if not user_id:
        raise ValueError("Authenticated user ID is required.")

    if not title.strip():
        raise ValueError("Conversation title is required.")

    if not mode.strip():
        raise ValueError("Conversation mode is required.")

    validate_messages(messages)

    payload = {
        "user_id": user_id,
        "title": title.strip(),
        "mode": mode.strip(),
        "messages": messages
    }

    response = (
        client.table("conversations")
        .insert(payload)
        .select("*")
        .single()
        .execute()
    )

    return response.data


def get_conversations(
    client: Client,
    user_id: str
):
    """Retrieve all conversations belonging to a user."""

    if not user_id:
        raise ValueError("Authenticated user ID is required.")

    response = (
        client.table("conversations")
        .select("*")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )

    return response.data or []


def get_conversation(
    client: Client,
    user_id: str,
    conversation_id: str
):
    """Retrieve one conversation belonging to a user."""

    if not user_id or not conversation_id:
        raise ValueError("User ID and conversation ID are required.")

    response = (
        client.table("conversations")
        .select("*")
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    return response.data


def update_conversation(
    client: Client,
    user_id: str,
    conversation_id: str,
    messages: list,
    title: str = None
):
    """Update an existing conversation."""

    if not user_id or not conversation_id:
        raise ValueError("User ID and conversation ID are required.")

    validate_messages(messages)

    payload = {
        "messages": messages,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

    if title is not None:
        if not title.strip():
            raise ValueError("Conversation title cannot be empty.")

        payload["title"] = title.strip()

    response = (
        client.table("conversations")
        .update(payload)
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .select("*")
        .maybe_single()
        .execute()
    )

    return response.data


def delete_conversation(
    client: Client,
    user_id: str,
    conversation_id: str
):
    """Delete a conversation belonging to a user."""

    if not user_id or not conversation_id:
        raise ValueError("User ID and conversation ID are required.")

    response = (
        client.table("conversations")
        .delete()
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .execute()
    )

    return response.data or []
  
