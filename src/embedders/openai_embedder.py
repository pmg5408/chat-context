from openai import OpenAI
from .base import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, config):
        self.client = OpenAI(api_key=config.api_key)
        self.model = config.model
        self.batch_size = config.batch_size
    
    def embed_texts(self, texts):
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_single(self, text):
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding
    
    @property
    def dimension(self):
        return 1536  # text-embedding-3-small dimension