"""
Chat history parsers for multiple platforms.
"""

from .base import BaseChatParser, Conversation, Message
from .chatgpt import ChatGPTParser
from .claude import ClaudeParser
from .chunking import (
    VectorDocument,
    prepare_message_for_vector_db,
    prepare_conversation_for_vector_db
)

__all__ = [
    'BaseChatParser',
    'Conversation',
    'Message',
    'ChatGPTParser',
    'ClaudeParser',
    'VectorDocument',
    'prepare_message_for_vector_db',
    'prepare_conversation_for_vector_db',
]
