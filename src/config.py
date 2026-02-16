import yaml
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ExportsConfig:
    """Export paths for different platforms."""
    claude: str
    chatgpt: str
    cursor: str = ""


@dataclass
class LocalEmbeddingConfig:
    model_name: str
    device: str


@dataclass
class OpenAIEmbeddingConfig:
    model: str
    api_key: Optional[str]
    batch_size: int


@dataclass
class EmbeddingConfig:
    provider: str  # 'local' or 'openai'
    local: LocalEmbeddingConfig
    openai: OpenAIEmbeddingConfig


@dataclass
class StorageConfig:
    chroma_db_path: str
    collections_metadata: str


@dataclass
class ChunkingConfig:
    max_tokens: int
    overlap: int
    enabled: bool


@dataclass
class IngestionConfig:
    batch_size: int
    skip_empty: bool


@dataclass
class OpenRouterTopicConfig:
    """Configuration for OpenRouter topic extraction."""
    model: str
    api_key: Optional[str]
    batch_size: int
    prompt_template: str


@dataclass
class TopicExtractionConfig:
    """Configuration for topic extraction using SLM."""
    enabled: bool
    provider: str  # 'openrouter' or 'none'
    openrouter: Optional[OpenRouterTopicConfig] = None


@dataclass
class Config:
    exports: ExportsConfig
    embedding: EmbeddingConfig
    storage: StorageConfig
    chunking: ChunkingConfig
    ingestion: IngestionConfig
    topic_extraction: Optional[TopicExtractionConfig] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    exports = ExportsConfig(**data['exports'])

    local_config = LocalEmbeddingConfig(**data['embedding']['local'])
    
    openai_config = OpenAIEmbeddingConfig(
        model=data['embedding']['openai']['model'],
        api_key=os.getenv(data['embedding']['openai']['api_key_env']),
        batch_size=data['embedding']['openai']['batch_size']
    )
    
    embedding = EmbeddingConfig(
        provider=data['embedding']['provider'],
        local=local_config,
        openai=openai_config
    )
    
    # Load topic extraction config only if enabled
    topic_extraction = None
    if data.get('topic_extraction', {}).get('enabled', False):
        topic_data = data['topic_extraction']
        provider = topic_data.get('provider', 'none')
        openrouter_config = None
        
        if provider == 'openrouter' and 'openrouter' in topic_data:
            or_data = topic_data['openrouter']
            openrouter_config = OpenRouterTopicConfig(
                model=or_data['model'],
                api_key=os.getenv(or_data['api_key_env']),
                batch_size=or_data['batch_size'],
                prompt_template=or_data.get('prompt_template', '')
            )
        
        topic_extraction = TopicExtractionConfig(
            enabled=True,
            provider=provider,
            openrouter=openrouter_config
        )
    
    return Config(
        exports=exports,
        embedding=embedding,
        storage=StorageConfig(**data['storage']),
        chunking=ChunkingConfig(**data['chunking']),
        ingestion=IngestionConfig(**data['ingestion']),
        topic_extraction=topic_extraction
    )
