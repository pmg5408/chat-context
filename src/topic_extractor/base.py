from abc import ABC, abstractmethod
from typing import List


class BaseTopicExtractor(ABC):
    """Abstract base class for topic extraction implementations."""
    
    @abstractmethod
    def extract_topics(self, texts: List[str]) -> List[List[str]]:
        """
        Extract topics from a batch of texts.
        
        Args:
            texts: List of message texts to extract topics from
            
        Returns:
            List of topic lists, one for each input text
        """
        pass
    
    @abstractmethod
    def extract_topics_sync(self, text: str) -> List[str]:
        """
        Extract topics from a single text.
        
        Args:
            text: Message text to extract topics from
            
        Returns:
            List of topics extracted from the text
        """
        pass
