"""
Parser for ChatGPT export format (conversations.json).
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



class ChatGPTParser(BaseChatParser):
    """Parser for ChatGPT conversations.json format."""
    
    def find_conversation_by_title(self, file_path: str, title: str) -> Optional[Conversation]:
        """
        Find a ChatGPT conversation by title.
        
        Efficiently searches through the conversations array without loading
        all messages until the matching conversation is found.
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        for conv_data in conversations:
            if conv_data.get('title', '').lower() == title.lower():
                return self.parse_conversation(conv_data)
        
        return None
    
    def list_all_conversations(self, file_path: str) -> List[dict]:
        """List all conversations with basic metadata."""
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        return [
            {
                'id': conv.get('id'),
                'title': conv.get('title'),
                'created_at': datetime.fromtimestamp(conv.get('create_time', 0)),
                'updated_at': datetime.fromtimestamp(conv.get('update_time', 0)),
                'message_count': len(conv.get('mapping', {}))
            }
            for conv in conversations
        ]
    
    def parse_conversation(self, conversation_data: dict) -> Conversation:
        """
        Parse a ChatGPT conversation from the export format.
        
        The mapping field contains a tree structure of messages that may
        have branches. We extract messages and sort by timestamp.
        """
        conv_id = conversation_data.get('id', 'unknown')
        title = conversation_data.get('title', 'Untitled')
        mapping = conversation_data.get('mapping', {})

        messages = []
        message_index = 0
        
        for node_id, node in mapping.items():
            if 'message' not in node or not node['message']:
                continue
            
            msg_data = node['message']

            if not msg_data.get('content') or not msg_data['content'].get('parts'):
                continue
            
            role = msg_data.get('author', {}).get('role', 'unknown')

            if role == 'user':
                role = 'user'
            elif role == 'assistant':
                role = 'assistant'
            else:
                continue
            
            # Combine all parts into one message
            content = '\n'.join(msg_data['content']['parts'])
            
            timestamp = datetime.fromtimestamp(msg_data.get('create_time', 0))
            
            messages.append(Message(
                role=role,
                content=content,
                timestamp=timestamp,
                message_index=message_index
            ))
            message_index += 1
        
        messages.sort(key=lambda m: m.timestamp)
        
        # Re-index after sorting
        # TODO
        for idx, msg in enumerate(messages):
            msg.message_index = idx
        
        return Conversation(
            conversation_id=conv_id,
            title=title,
            platform='chatgpt',
            messages=messages,
            created_at=datetime.fromtimestamp(conversation_data.get('create_time', 0)),
            updated_at=datetime.fromtimestamp(conversation_data.get('update_time', 0))
        )