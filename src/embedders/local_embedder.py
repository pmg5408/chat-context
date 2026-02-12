from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder

class LocalEmbedder(BaseEmbedder):
    def __init__(self, config):
        self.model_name = config.model_name
        self.device = config.device
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)
    
    def embed_texts(self, texts):
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings.tolist()
    
    def embed_single(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    @property
    def dimension(self):
        return self.model.get_sentence_embedding_dimension()