"""
ChromaDB storage layer for managing conversation collections.
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# Document type constants for reliable filtering
DOC_TYPE_MAIN = "main"
DOC_TYPE_TOPIC = "topic"

# MMR defaults
MMR_LAMBDA = 0.7  # Balance: 1.0 = pure relevance, 0.0 = pure diversity
MMR_OVERFETCH_FACTOR = 3  # Fetch 3x candidates for MMR reranking

# Recency clustering defaults
CLUSTER_DISTANCE_THRESHOLD = 0.15  # Messages within this distance are same discussion


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
    
    def _compute_mmr(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        candidate_distances: List[float],
        n_select: int,
        lambda_param: float = MMR_LAMBDA
    ) -> List[int]:
        """
        Compute Maximal Marginal Relevance to select diverse results.
        
        MMR balances relevance to the query against similarity to already-selected
        results, promoting diversity in the returned documents.
        
        Args:
            query_embedding: The query vector
            candidate_embeddings: Embeddings of all candidates
            candidate_distances: ChromaDB distances (lower = more similar)
            n_select: Number of candidates to select
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        
        Returns:
            List of indices of selected candidates (in selection order)
        """
        if len(candidate_embeddings) <= n_select:
            return list(range(len(candidate_embeddings)))
        
        # Convert distances to similarities (ChromaDB uses cosine distance)
        # cosine_similarity = 1 - cosine_distance
        query_similarities = [1 - d for d in candidate_distances]
        
        # Convert embeddings to numpy for efficient computation
        query_vec = np.array(query_embedding)
        candidate_vecs = np.array(candidate_embeddings)
        
        # Normalize for cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        candidate_norms = candidate_vecs / (np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-10)
        
        selected_indices = []
        remaining_indices = set(range(len(candidate_embeddings)))
        
        # First selection: highest similarity to query
        first_idx = max(range(len(query_similarities)), key=lambda i: query_similarities[i])
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Subsequent selections: maximize MMR score
        while len(selected_indices) < n_select and remaining_indices:
            best_idx = None
            best_score = float('-inf')
            
            for idx in remaining_indices:
                # Relevance component: similarity to query
                relevance = query_similarities[idx]
                
                # Diversity component: max similarity to any already-selected
                # Compute cosine similarity between this candidate and all selected
                sims_to_selected = np.dot(candidate_norms[list(selected_indices)], candidate_norms[idx])
                max_sim_to_selected = np.max(sims_to_selected) if len(sims_to_selected) > 0 else 0
                
                # MMR score: lambda * relevance - (1 - lambda) * redundancy
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def _apply_recency_clustering(
        self,
        candidate_embeddings: List[List[float]],
        candidate_metadatas: List[Dict],
        threshold: float = CLUSTER_DISTANCE_THRESHOLD
    ) -> List[int]:
        """
        Cluster similar results and keep only the most recent from each cluster.
        
        When results are very close in embedding space (within threshold), they
        likely represent the same discussion thread. This collapses them to a
        single result — the most recent message in the cluster.
        
        Args:
            candidate_embeddings: Embeddings of candidates
            candidate_metadatas: Metadata including message_index for recency
            threshold: Distance threshold for clustering (lower = tighter clusters)
        
        Returns:
            List of indices to keep (one per cluster, the most recent)
        """
        if not candidate_embeddings:
            return []
        
        n = len(candidate_embeddings)
        if n == 1:
            return [0]
        
        # Convert to numpy for distance computation
        embeddings = np.array(candidate_embeddings)
        norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute pairwise cosine distances
        similarity_matrix = np.dot(norms, norms.T)  # similarity = 1 - distance
        distance_matrix = 1 - similarity_matrix
        
        # Build clusters using union-find
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union results that are within threshold distance
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] < threshold:
                    union(i, j)
        
        # Group by cluster
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        
        # From each cluster, keep the most recent (highest message_index)
        kept_indices = []
        for cluster_indices in clusters.values():
            # Find the one with highest message_index
            best_idx = max(cluster_indices, key=lambda i: candidate_metadatas[i].get('message_index', 0))
            kept_indices.append(best_idx)
        
        # Sort by original order (preserving MMR ranking)
        kept_indices.sort(key=lambda i: i)
        
        return kept_indices
    
    def search(
        self, 
        query: str, 
        n_results: int = 5, 
        filters: Optional[Dict] = None,
        use_mmr: bool = True,
        mmr_lambda: float = MMR_LAMBDA,
        cluster_threshold: float = CLUSTER_DISTANCE_THRESHOLD
    ) -> Dict:
        """
        Search for similar documents in the collection.
        
        Always returns only "main" documents (doc_type="main"), which represent
        one entry per message. Use search_by_topic() to search topic documents.
        
        Applies MMR for diversity, then recency clustering to collapse
        similar messages from the same discussion thread.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional additional metadata filters e.g. {"role": "assistant"}
            use_mmr: If True, apply MMR for diverse results (default: True)
            mmr_lambda: Relevance-diversity balance (1.0=relevance, 0.0=diversity)
            cluster_threshold: Distance threshold for recency clustering (default: 0.15)
        
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

        # Fetch extra candidates for MMR reranking
        fetch_count = n_results * MMR_OVERFETCH_FACTOR if use_mmr else n_results
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_count,
            where=where_filter,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        # Handle empty results
        if not results['documents'] or not results['documents'][0]:
            return results
        
        # Extract results from ChromaDB format (lists of lists)
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]
        embeddings = results['embeddings'][0]
        
        # Step 1: Apply MMR reranking if enabled
        if use_mmr and len(docs) > n_results:
            selected_indices = self._compute_mmr(
                query_embedding=query_embedding,
                candidate_embeddings=embeddings,
                candidate_distances=distances,
                n_select=n_results,
                lambda_param=mmr_lambda
            )
        else:
            selected_indices = list(range(min(len(docs), n_results)))
        
        # Step 2: Apply recency clustering on MMR-selected results
        # This collapses tight clusters into single results (keeping most recent)
        if embeddings and len(selected_indices) > 1:
            # Get embeddings and metadatas for selected results
            selected_embeddings = [embeddings[i] for i in selected_indices]
            selected_metadatas = [metadatas[i] for i in selected_indices]
            
            # Cluster and filter
            kept_relative_indices = self._apply_recency_clustering(
                candidate_embeddings=selected_embeddings,
                candidate_metadatas=selected_metadatas,
                threshold=cluster_threshold
            )
            
            # Map back to original indices
            final_indices = [selected_indices[i] for i in kept_relative_indices]
        else:
            final_indices = selected_indices
        
        # Build final results
        results = {
            'documents': [[docs[i] for i in final_indices]],
            'metadatas': [[metadatas[i] for i in final_indices]],
            'distances': [[distances[i] for i in final_indices]],
            'ids': [[ids[i] for i in final_indices]]
        }
        
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
