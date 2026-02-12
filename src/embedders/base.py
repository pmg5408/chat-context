from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts, return list of embeddings."""
        pass
    
    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension size."""
        pass