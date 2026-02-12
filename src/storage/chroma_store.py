"""
ChromaDB storage layer for managing conversation collections.
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional

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
            embedder: Embedder instance (LocalEmbedder or OpenAIEmbedder)
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
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
                Example: [
                    {
                        'text': 'Message content',
                        'metadata': {
                            'role': 'user',
                            'timestamp': '2024-01-15T10:30:00Z',
                            ...
                        }
                    }
                ]
        """
        if not documents:
            return

        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]

        ids = [
            f"{doc['metadata']['platform']}_{doc['metadata']['conversation_id']}_{doc['metadata']['message_index']}"
            for doc in documents
        ]

        print(f"  Generating embeddings for {len(texts)} messages...")
        embeddings = self.embedder.embed_texts(texts)
        
        print(f"  Storing in ChromaDB collection '{self.collection_name}'...")
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  ✓ Stored {len(documents)} documents")
    
    def search(
        self, 
        query: str, 
        n_results: int = 5, 
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar documents in the collection.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional metadata filters
                Example: {"role": "assistant"}
        
        Returns:
            Dict with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        query_embedding = self.embedder.embed_single(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters
        )
        
        return results
    
    def get_document_count(self) -> int:
        """Get total number of documents in collection."""
        return self.collection.count()
    
    def delete_collection(self):
        """Delete this collection from ChromaDB."""
        self.client.delete_collection(name=self.collection_name)
    
    @staticmethod
    def list_all_collections(chroma_db_path: str) -> List[str]:
        """
        List all collection names in the ChromaDB.
        
        Args:
            chroma_db_path: Path to ChromaDB storage
            
        Returns:
            List of collection names
        """
        client = chromadb.PersistentClient(
            path=str(chroma_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        collections = client.list_collections()
        return [col.name for col in collections]