"""
Parser for Claude export format (JSON from browser extensions or official export).

Claude exports can come from:
1. Official export (Settings → Privacy → Export Data) - JSON in ZIP
2. Browser extensions (Claude Conversation Exporter) - Direct JSON

Both have similar structure but may vary slightly.
"""
import json
import sys
from datetime import datetime
from typing import List, Optional
from pathlib import Path

try:
    from .base import BaseChatParser, Conversation, Message
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.parsers.base import BaseChatParser, Conversation, Message


class ClaudeParser(BaseChatParser):
    """Parser for Claude conversation exports."""
    
    def find_conversation_by_title(self, file_path: str, title: str) -> Optional[Conversation]:
        """
        Find a Claude conversation by title/name.
        
        Claude exports may be:
        - Single conversation (one JSON object)
        - Multiple conversations (array of objects)
        - Nested in an export wrapper
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Handle different export formats
        # TODO Skipping for now since current plan is to only use offical json files
        #conversations = self._normalize_export_format(data)
        
        for conv_data in conversations:
            conv_name = conv_data.get('name', conv_data.get('title', ''))
            if conv_name.lower() == title.lower():
                return self.parse_conversation(conv_data)
        
        return None
    
    def list_all_conversations(self, file_path: str) -> List[dict]:
        """List all Claude conversations with basic metadata."""
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        # TODO Uncomment if i decide to support more formats
        #conversations = self._normalize_export_format(data)

        result = []
        for conv in conversations:
            created_at = self._parse_timestamp(conv.get('created_at'))
            updated_at = self._parse_timestamp(conv.get('updated_at'))
            
            result.append({
                'id': conv.get('uuid', conv.get('id', 'unknown')),
                'title': conv.get('name', conv.get('title', 'Untitled')),
                'created_at': created_at,
                'updated_at': updated_at,
                'message_count': len(conv.get('chat_messages', []))
            })
        
        return result
    
    def parse_conversation(self, conversation_data: dict) -> Conversation:
        """
        Parse a Claude conversation from export format.
        
        Expected structure:
        {
            "uuid": "conv-id",
            "name": "Conversation Title",
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T11:45:00Z",
            "model": "claude-3-5-sonnet-20241022",  // may be null
            "chat_messages": [
                {
                    "uuid": "msg-id",
                    "text": "Message content",
                    "sender": "human" | "assistant",
                    "created_at": "2024-01-15T10:30:00Z"
                }
            ]
        }
        """
        conv_id = conversation_data.get('uuid', conversation_data.get('id', 'unknown'))
        title = conversation_data.get('name', conversation_data.get('title', 'Untitled'))
        model = conversation_data.get('model')
        
        chat_messages = conversation_data.get('chat_messages', [])

        messages = []
        for idx, msg_data in enumerate(chat_messages):
            sender = msg_data.get('sender', 'unknown')
            if sender == 'human':
                role = 'user'
            elif sender == 'assistant':
                role = 'assistant'
            else:
                continue
            
            content = msg_data.get('text', '')

            if not content:
                continue
            
            timestamp = self._parse_timestamp(msg_data.get('created_at'))
            
            messages.append(Message(
                role=role,
                content=content,
                timestamp=timestamp,
                message_index=idx
            ))
        
        # Re-index messages after filtering
        for idx, msg in enumerate(messages):
            msg.message_index = idx
        
        created_at = self._parse_timestamp(conversation_data.get('created_at'))
        updated_at = self._parse_timestamp(conversation_data.get('updated_at'))
        
        return Conversation(
            conversation_id=conv_id,
            title=title,
            platform='claude',
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            model=model
        )
    
    def _normalize_export_format(self, data: any) -> List[dict]:
        """
        Normalize different Claude export formats into a list of conversations.
        
        Handles:
        - Single conversation: {...}
        - Array of conversations: [{...}, {...}]
        - Nested export: {"conversations": [{...}]}
        """
        # If it's a single conversation (dict with uuid/name)
        if isinstance(data, dict):
            # Check if it's a wrapper object
            if 'conversations' in data:
                return data['conversations']
            # Check if it has conversation fields
            elif 'uuid' in data or 'name' in data or 'chat_messages' in data:
                return [data]
            else:
                # Unknown format, try to find conversations anywhere
                for key, value in data.items():
                    if isinstance(value, list):
                        return value
                return []
        
        # If it's already an array
        elif isinstance(data, list):
            return data
        
        else:
            return []
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """
        Parse Claude's timestamp format.
        
        Handles:
        - ISO format: "2024-01-15T10:30:00Z"
        - ISO with milliseconds: "2024-01-15T10:30:00.123Z"
        - None/empty
        """
        if not timestamp_str:
            return datetime.now()
        
        try:
            # Remove 'Z' and parse
            timestamp_str = timestamp_str.replace('Z', '+00:00')
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, AttributeError):
            # Fallback to current time if parsing fails
            return datetime.now()
