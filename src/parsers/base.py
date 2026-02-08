"""
Base parser classes for chat history exports.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

# TODO: What does dataclass mean? 
@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    message_index: int  # Position in conversation


@dataclass
class Conversation:
    """Represents a complete conversation."""
    conversation_id: str
    title: str
    platform: str  # 'chatgpt', 'claude', 'cursor'
    messages: List[Message]
    created_at: datetime
    updated_at: Optional[datetime] = None
    model: Optional[str] = None


class BaseChatParser(ABC):
    """Base class for all chat parsers."""
    
    @abstractmethod
    def find_conversation_by_title(self, file_path: str, title: str) -> Optional[Conversation]:
        """
        Find a specific conversation by title without loading all conversations.
        
        Args:
            file_path: Path to the export file
            title: Title of the conversation to find
            
        Returns:
            Conversation object if found, None otherwise
        """
        pass
    
    @abstractmethod
    def parse_conversation(self, conversation_data: dict) -> Conversation:
        """
        Parse a single conversation from raw data.
        
        Args:
            conversation_data: Raw conversation data dictionary
            
        Returns:
            Parsed Conversation object
        """
        pass
    
    @abstractmethod
    def list_all_conversations(self, file_path: str) -> List[dict]:
        """
        Get a list of all conversations with basic metadata (title, id, date).
        This should be lightweight and not parse all messages.
        
        Args:
            file_path: Path to the export file
            
        Returns:
            List of conversation metadata dictionaries
        """
        pass
