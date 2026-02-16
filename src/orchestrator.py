"""
Orchestrator for managing conversation ingestion and storage.
"""
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import re

from .parsers.claude import ClaudeParser
from .parsers.chatgpt import ChatGPTParser
#from .embedders.local_embedder import LocalEmbedder
from .embedders.openai_embedder import OpenAIEmbedder
from .storage.chroma_store import ChromaStore
from .config import Config
from .topic_extractor import OpenRouterTopicExtractor


class IngestionOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.parsers = {
            'claude': ClaudeParser(),
            'chatgpt': ChatGPTParser(),
        }

        if config.embedding.provider == 'openai':
            self.embedder = OpenAIEmbedder(config.embedding.openai)
        #else:
            #self.embedder = LocalEmbedder(config.embedding.local)

        # Initialize topic extractor if enabled
        self.topic_extractor = None
        if config.topic_extraction and config.topic_extraction.enabled:
            if config.topic_extraction.provider == 'openrouter':
                self.topic_extractor = OpenRouterTopicExtractor(
                    config.topic_extraction.openrouter
                )
                print(f"  Topic extractor enabled: {config.topic_extraction.openrouter.model}")

        self.chroma_path = config.storage.chroma_db_path
        self.metadata_path = Path(config.storage.collections_metadata)

        self._ensure_metadata_file()
    
    def _ensure_metadata_file(self):
        """Create metadata file if it doesn't exist."""
        if not self.metadata_path.exists():
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, 'w') as f:
                yaml.dump({'conversations': []}, f)
    
    def _load_metadata(self) -> Dict:
        """Load conversations metadata."""
        with open(self.metadata_path) as f:
            return yaml.safe_load(f) or {'conversations': []}
    
    # TODO Separate naming convention for metadata in the metadata.yaml file
    # from the metadata stored in chroma db documents
    def _save_metadata(self, metadata: Dict):
        """Save conversations metadata."""
        with open(self.metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    def _sanitize_collection_name(self, title: str) -> str:
        """Convert conversation title to valid collection name."""
        # Lowercase, replace spaces/special chars with underscores
        sanitized = re.sub(r'[^a-z0-9]+', '_', title.lower())
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length
        return sanitized[:63]  # ChromaDB has 63 char limit
    
    def _get_export_path(self, platform: str) -> Path:
        """Get export file path for platform from config."""
        export_path = getattr(self.config.exports, platform)
        return Path(export_path)
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp to datetime, handling different formats."""
        if not timestamp_str:
            return datetime.now()
        
        # Try ISO 8601 format first (Claude format)
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            pass
        
        # Try Unix timestamp (ChatGPT format)
        try:
            return datetime.fromtimestamp(float(timestamp_str))
        except (ValueError, TypeError):
            return datetime.now()
    
    def add_conversation(self, platform: str, title: str) -> Dict:
        """
        Add a conversation to ChromaDB.
        
        Args:
            platform: 'claude', 'chatgpt', or 'cursor'
            title: Conversation title to search for
            
        Returns:
            Dict with stats: {
                'collection_name': str,
                'message_count': int,
                'conversation_title': str,
                'platform': str
            }
        """
        parser = self.parsers[platform]
        export_path = self._get_export_path(platform)
        
        if not export_path.exists():
            raise FileNotFoundError(f"Export file not found: {export_path}")

        conversation = parser.find_conversation_by_title(str(export_path), title)
        
        if not conversation:
            raise ValueError(f"Conversation '{title}' not found in {platform} export")

        collection_name = self._sanitize_collection_name(title)

        metadata = self._load_metadata()
        existing = [c for c in metadata['conversations'] if c['collection_name'] == collection_name]
        if existing:
            raise ValueError(f"Collection '{collection_name}' already exists. Use 'update' to add new messages.")

        # Extract topics if topic extractor is enabled
        # Using dictionary for O(1) lookups later
        topics_by_index: Dict[int, List[str]] = {}
        if self.topic_extractor:
            print(f"  Extracting topics using SLM...")
            
            # Get texts in order (only non-empty)
            texts = []
            msg_indices = []
            for msg in conversation.messages:
                if msg.content.strip():
                    texts.append(msg.content)
                    msg_indices.append(msg.message_index)
            
            # Extract topics - pass both texts and message_indices
            topics_by_index = self.topic_extractor.extract_topics(texts, msg_indices)
            
            print(f"  Extracted topics for {len(topics_by_index)} messages")

        # Build documents with topics
        documents = []
        for msg in conversation.messages:
            if not msg.content.strip():
                continue
            
            # Get topics for this message
            topics = topics_by_index.get(msg.message_index, [])
            
            documents.append({
                'text': msg.content,
                'metadata': {
                    'conversation_id': conversation.conversation_id,
                    'conversation_title': conversation.title,
                    'platform': conversation.platform,
                    'role': msg.role,
                    'timestamp': msg.timestamp.isoformat(),
                    'message_index': msg.message_index,
                    'total_messages': len(conversation.messages),
                    'topics': ','.join(topics) if topics else '',
                    'is_deprecated': 'false'
                }
            })

        storage = ChromaStore(self.chroma_path, collection_name, self.embedder)
        storage.add_documents(documents)

        # Update collection topics in metadata and print
        if topics_by_index:
            all_topics = set()
            for topics in topics_by_index.values():
                all_topics.update(topics)
            sorted_topics = sorted(all_topics)
            storage.update_collection_topics(sorted_topics)
            print(f"  Topics: {sorted_topics}")

        last_timestamp = max(
            (self._parse_timestamp(doc['metadata']['timestamp']) for doc in documents),
            default=datetime.now()
        )

        metadata['conversations'].append({
            'collection_name': collection_name,
            'original_title': conversation.title,
            'platform': platform,
            'conversation_id': conversation.conversation_id,
            'created_at': datetime.now().isoformat(),
            'message_count': len(documents),
            'last_message_timestamp': last_timestamp.isoformat()
        })
        self._save_metadata(metadata)
        
        return {
            'collection_name': collection_name,
            'message_count': len(documents),
            'conversation_title': conversation.title,
            'platform': platform
        }
    
    def update_conversation(self, platform: str, title: str) -> Dict:
        """
        Update an existing conversation with new messages.
        
        Args:
            platform: 'claude', 'chatgpt', or 'cursor'
            title: Conversation title
            
        Returns:
            Dict with stats: {
                'new_messages': int,
                'total_messages': int,
                'conversation_title': str
            }
        """
        collection_name = self._sanitize_collection_name(title)

        metadata = self._load_metadata()
        conv_meta = None
        for c in metadata['conversations']:
            if c['collection_name'] == collection_name:
                conv_meta = c
                break
        
        if not conv_meta:
            raise ValueError(f"Collection '{collection_name}' not found. Use 'add' first.")

        parser = self.parsers[platform]
        export_path = self._get_export_path(platform)

        conversation = parser.find_conversation_by_title(str(export_path), title)
        
        if not conversation:
            raise ValueError(f"Conversation '{title}' not found in {platform} export")
        
        # Filter messages after last timestamp
        last_timestamp = self._parse_timestamp(conv_meta['last_message_timestamp'])
        
        new_messages = []
        for msg in conversation.messages:
            msg_timestamp = self._parse_timestamp(msg.timestamp.isoformat())
            
            # Only add messages AFTER last timestamp
            if msg_timestamp > last_timestamp and msg.content.strip():
                new_messages.append(msg)
        
        if not new_messages:
            return {
                'new_messages': 0,
                'total_messages': conv_meta['message_count'],
                'conversation_title': conversation.title
            }
        
        # Extract topics for new messages if topic extractor is enabled
        topics_by_index: Dict[int, List[str]] = {}
        if self.topic_extractor and new_messages:
            print(f"  Extracting topics for {len(new_messages)} new messages...")
            
            # Get all existing topics from ChromaDB to pass to SLM
            storage = ChromaStore(self.chroma_path, collection_name, self.embedder)
            existing_topics = storage.get_all_topics()
            print(f"  Existing topics in conversation: {existing_topics}")
            
            # Extract topics for new messages (passing texts, indices, and existing topics)
            texts = [msg.content for msg in new_messages]
            msg_indices = [msg.message_index for msg in new_messages]
            topics_by_index = self.topic_extractor.extract_topics_with_context(
                texts, 
                msg_indices,
                existing_topics
            )
            
            print(f"  Extracted topics for {len(topics_by_index)} new messages")

        # Build documents with topics
        new_documents = []
        for msg in new_messages:
            topics = topics_by_index.get(msg.message_index, [])
            
            new_documents.append({
                'text': msg.content,
                'metadata': {
                    'conversation_id': conversation.conversation_id,
                    'conversation_title': conversation.title,
                    'platform': conversation.platform,
                    'role': msg.role,
                    'timestamp': msg.timestamp.isoformat(),
                    'message_index': msg.message_index,
                    'total_messages': len(conversation.messages),
                    'topics': ','.join(topics) if topics else '',
                    'is_deprecated': 'false'
                }
            })
        
        # Add to existing collection
        storage = ChromaStore(self.chroma_path, collection_name, self.embedder)
        storage.add_documents(new_documents)
        
        # Update collection topics in metadata and print
        if topics_by_index:
            all_topics = set(existing_topics)  # Start with existing topics
            for topics in topics_by_index.values():
                all_topics.update(topics)
            sorted_topics = sorted(all_topics)
            storage.update_collection_topics(sorted_topics)
            print(f"  Topics: {sorted_topics}")
        
        # Update metadata
        new_last_timestamp = max(
            (self._parse_timestamp(doc['metadata']['timestamp']) for doc in new_documents),
            default=last_timestamp
        )
        
        conv_meta['message_count'] += len(new_documents)
        conv_meta['last_message_timestamp'] = new_last_timestamp.isoformat()
        self._save_metadata(metadata)
        
        return {
            'new_messages': len(new_documents),
            'total_messages': conv_meta['message_count'],
            'conversation_title': conversation.title
        }
    
    def remove_conversation(self, title: str) -> Dict:
        """
        Remove a conversation from ChromaDB.
        
        Args:
            title: Conversation title (or collection name)
            
        Returns:
            Dict with stats: {'collection_name': str, 'message_count': int}
        """
        collection_name = self._sanitize_collection_name(title)

        metadata = self._load_metadata()
        conv_meta = None
        conv_index = None
        
        for i, c in enumerate(metadata['conversations']):
            if c['collection_name'] == collection_name or c['original_title'] == title:
                conv_meta = c
                conv_index = i
                break
        
        if not conv_meta:
            raise ValueError(f"Collection '{title}' not found")

        storage = ChromaStore(self.chroma_path, collection_name, self.embedder)
        storage.delete_collection()

        message_count = conv_meta['message_count']
        del metadata['conversations'][conv_index]
        self._save_metadata(metadata)
        
        return {
            'collection_name': collection_name,
            'message_count': message_count
        }
    
    def list_conversations(self) -> List[Dict]:
        """
        List all available conversations.
        
        Returns:
            List of conversation metadata dicts
        """
        metadata = self._load_metadata()
        return metadata['conversations']