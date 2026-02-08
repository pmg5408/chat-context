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
        
        # Search for conversation by title
        for conv_data in conversations:
            if conv_data.get('title', '').lower() == title.lower():
                return self.parse_conversation(conv_data)
        
        return None
    
    def list_all_conversations(self, file_path: str) -> List[dict]:
        """List all conversations with basic metadata."""
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Return lightweight metadata only
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
        
        # Extract and parse messages
        messages = []
        message_index = 0
        
        for node_id, node in mapping.items():
            if 'message' not in node or not node['message']:
                continue
            
            msg_data = node['message']
            
            # Skip messages without content
            if not msg_data.get('content') or not msg_data['content'].get('parts'):
                continue
            
            role = msg_data.get('author', {}).get('role', 'unknown')
            
            # Map ChatGPT roles to our standard roles
            if role == 'user':
                role = 'user'
            elif role == 'assistant':
                role = 'assistant'
            else:
                # Skip system messages or other roles
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
        
        # Sort messages by timestamp to get chronological order
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


# Example usage and testing
if __name__ == '__main__':
    parser = ChatGPTParser()
    
    # Test with your export file
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python chatgpt.py <path_to_conversations.json> <conversation_title>")
        print("\nExample:")
        print('  python chatgpt.py ~/Downloads/conversations.json "My Conversation Title"')
        sys.exit(1)
    
    file_path = sys.argv[1]
    conversation_title = sys.argv[2]
    
    print(f"Loading ChatGPT export from: {file_path}\n")
    
    # List all conversations
    print("=" * 60)
    print("ALL CONVERSATIONS")
    print("=" * 60)
    conversations = parser.list_all_conversations(file_path)
    print(f"Found {len(conversations)} conversation(s)\n")
    
    for i, conv in enumerate(conversations[:5], 1):
        print(f"{i}. {conv['title']}")
        print(f"   ID: {conv['id']}")
        print(f"   Messages: {conv['message_count']}")
        print(f"   Created: {conv['created_at']}")
        print()
    
    if len(conversations) > 5:
        print(f"... and {len(conversations) - 5} more conversations")
        print()
    
    # Find and display the specific conversation
    if conversations:
        print("\n" + "=" * 60)
        print("DETAILED VIEW - REQUESTED CONVERSATION")
        print("=" * 60)
        
        conv = parser.find_conversation_by_title(file_path, conversation_title)
        
        if conv:
            print(f"Title: {conv.title}")
            print(f"Platform: {conv.platform}")
            print(f"Created: {conv.created_at}")
            print(f"Total Messages: {len(conv.messages)}\n")
            
            print("-" * 60)
            print("MESSAGES (showing first 5)")
            print("-" * 60)
            
            for msg in conv.messages[:5]:
                print(f"\n[{msg.role.upper()}] (Index: {msg.message_index})")
                print(f"Time: {msg.timestamp}")
                
                # Show first 200 characters of content
                content_preview = msg.content[:200]
                if len(msg.content) > 200:
                    content_preview += "..."
                print(f"Content: {content_preview}")
                print("-" * 60)
            
            if len(conv.messages) > 5:
                print(f"\n... and {len(conv.messages) - 5} more messages")
        else:
            print(f"\nConversation with title '{conversation_title}' not found.")
