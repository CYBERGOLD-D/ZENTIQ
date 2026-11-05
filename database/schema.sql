
"""
ZENTIQ - Database Schema

Defines database table names, expected fields,
conversation modes and message validation rules.
"""

CONVERSATIONS_TABLE = "conversations"

CONVERSATION_COLUMNS = (
    "id",
    "user_id",
    "title",
    "mode",
    "messages",
    "updated_at"
)

SUPPORTED_MODES = (
    "ZENTIQ AI Assistant",
    "Machine Learning Retail Support"
)

VALID_MESSAGE_ROLES = (
    "user",
    "assistant"
)


def validate_message(message):
    """Validate one conversation message."""

    if not isinstance(message, dict):
        raise ValueError("Message must be a dictionary.")

    if message.get("role") not in VALID_MESSAGE_ROLES:
        raise ValueError("Message role must be user or assistant.")

    if not isinstance(message.get("content"), str):
        raise ValueError("Message content must be text.")

    return True


def validate_conversation_messages(messages):
    """Validate a complete conversation message list."""

    if not isinstance(messages, list):
        raise ValueError("Conversation messages must be a list.")

    for message in messages:
        validate_message(message)

    return True


def validate_conversation_mode(mode):
    """Check whether the requested assistant mode is supported."""

    if mode not in SUPPORTED_MODES:
        raise ValueError("Unsupported ZENTIQ conversation mode.")

    return True
