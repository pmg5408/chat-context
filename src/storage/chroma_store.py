"""
ChromaDB storage layer for managing conversation collections.
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional

# Document type constants for reliable filtering
DOC_TYPE_MAIN = "main"
DOC_TYPE_TOPIC = "topic"


class ChromaStore:
    """
    Wrapper around ChromaDB for storing and searching conversation messages.
    
    Each conversation gets its own collection.
    """
    
    def __init__(self, chroma_db_path: str, collection_name: str, embedder):
        """
        Initialize ChromaDB client and get/create collection.
        
        Args:
            chroma_db_path: Path to ChromaDB persistent storage
            collection_name: Name of the collection to use
            embedder: Embedder instance ( or OpenAIEmbedder)
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        self.embedder = embedder
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.chroma_db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Conversation: {collection_name}"}
        )
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the collection.
        
        For each message, creates:
        1. One main document (no topic field) - for regular search
        2. One document per topic - for topic filtering
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
                Example: [
                    {
                        'text': 'Message content',
                        'metadata': {
                            'role': 'user',
                            'timestamp': '2024-01-15T10:30:00Z',
                            'topics': 'database,postgresql,backend',
                            ...
                        }
                    }
                ]
        """
        if not documents:
            return

        # Collect all unique texts for embedding (efficiency)
        unique_texts = []
        text_to_embedding = {}
        
        # Build expanded docs:
        # 1. Main doc (no topic) - for regular search
        # 2. Topic docs (one per topic) - for filtering
        expanded_docs = []
        
        for doc in documents:
            text = doc['text']
            metadata = doc['metadata']
            topics_str = metadata.get('topics', '')
            
            # Get or create embedding for this text
            if text not in text_to_embedding:
                unique_texts.append(text)
            
            # Split topics
            topics = []
            if topics_str:
                topics = [t.strip() for t in topics_str.split(',') if t.strip()]
            
            # Create main document - for regular search
            expanded_docs.append({
                'text': text,
                'metadata': {**metadata, 'doc_type': DOC_TYPE_MAIN},
                'message_id': f"{metadata['platform']}_{metadata['conversation_id']}_{metadata['message_index']}"
            })
            
            # Create topic-specific documents (for filtering)
            for topic in topics:
                expanded_docs.append({
                    'text': text,
                    'metadata': {**metadata, 'topic': topic, 'doc_type': DOC_TYPE_TOPIC},
                    'message_id': f"{metadata['platform']}_{metadata['conversation_id']}_{metadata['message_index']}"
                })
        
        # Generate embeddings for unique texts only
        print(f"  Generating embeddings for {len(unique_texts)} unique messages...")
        embeddings_list = self.embedder.embed_texts(unique_texts)
        
        # Map texts to their embeddings
        for text, emb in zip(unique_texts, embeddings_list):
            text_to_embedding[text] = emb
        
        # Build final arrays for ChromaDB
        texts = [d['text'] for d in expanded_docs]
        embeddings = [text_to_embedding[d['text']] for d in expanded_docs]
        metadatas = [d['metadata'] for d in expanded_docs]
        
        # IDs: main doc gets special marker, topic docs include topic
        ids = []
        for d in expanded_docs:
            msg_id = d['message_id']
            topic = d['metadata'].get('topic', '')
            if topic:
                ids.append(f"{msg_id}_{topic}")
            else:
                ids.append(f"{msg_id}_main")

        print(f"  Storing {len(expanded_docs)} documents in ChromaDB...")
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  ✓ Stored {len(expanded_docs)} documents ({len(documents)} messages)")
    
    def search(
        self, 
        query: str, 
        n_results: int = 5, 
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar documents in the collection.
        
        Always returns only "main" documents (doc_type="main"), which represent
        one entry per message. Use search_by_topic() to search topic documents.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional additional metadata filters e.g. {"role": "assistant"}
        
        Returns:
            Dict with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        query_embedding = self.embedder.embed_single(query)

        # Always restrict to main documents (one per message, no duplicates).
        # Use explicit $eq operator for reliable cross-version ChromaDB behaviour.
        doc_type_filter = {"doc_type": {"$eq": DOC_TYPE_MAIN}}

        if filters:
            # Combine with caller-supplied filters using $and
            where_filter = {"$and": [{k: {"$eq": v}} for k, v in filters.items()] + [doc_type_filter]}
        else:
            where_filter = doc_type_filter

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def search_by_topic(
        self, 
        query: str, 
        topic: str,
        n_results: int = 5
    ) -> Dict:
        """
        Search for similar documents filtered by topic.
        
        Returns only topic-specific documents (doc_type="topic") whose topic
        field exactly matches the given topic.
        
        Args:
            query: Search query text
            topic: Topic to filter by (exact match)
            n_results: Number of results to return
        
        Returns:
            Dict with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        query_embedding = self.embedder.embed_single(query)

        # Explicitly target topic documents for the requested topic only.
        where_filter = {
            "$and": [
                {"doc_type": {"$eq": DOC_TYPE_TOPIC}},
                {"topic": {"$eq": topic}}
            ]
        }

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )

        return results
    
    def get_document_count(self) -> int:
        """Get total number of documents in collection."""
        return self.collection.count()
    
    def delete_collection(self):
        """Delete this collection from ChromaDB."""
        self.client.delete_collection(name=self.collection_name)
    
    def update_collection_topics(self, topics: List[str]):
        """
        Update the topics stored in collection metadata.
        
        Note: ChromaDB doesn't support arrays in metadata, so we store as comma-separated string.
        
        Args:
            topics: List of topic strings to store
        """
        current_meta = self.collection.metadata or {}
        # Store as comma-separated string, sorted alphabetically (ChromaDB limitation - no arrays in metadata)
        current_meta["topics"] = ','.join(sorted(topics)) if topics else ''
        current_meta["description"] = current_meta.get("description", f"Conversation: {self.collection_name}")
        
        self.collection.modify(metadata=current_meta)
    
    def get_collection_topics(self) -> List[str]:
        """
        Get topics from collection metadata (efficient - no document scan).
        
        Returns:
            List of topic strings, or empty list if not set
        """
        meta = self.collection.metadata or {}
        topics_str = meta.get("topics", '')
        if topics_str:
            return [t.strip() for t in topics_str.split(',') if t.strip()]
        return []
    
    # Keep for backward compatibility - now prefers collection metadata
    def get_all_topics(self) -> List[str]:
        """
        Get all unique topics from this collection.
        
        First tries collection metadata, falls back to scanning documents.
        
        Returns:
            List of unique topic strings
        """
        # Try collection metadata first (efficient)
        topics = self.get_collection_topics()
        if topics:
            return sorted(topics)
        
        # Fallback: scan all documents
        results = self.collection.get()
        
        topics_set = set()
        for metadata in results.get('metadatas', []):
            topic = metadata.get('topic', '')
            if topic:
                topics_set.add(topic)
        
        return sorted(list(topics_set))
