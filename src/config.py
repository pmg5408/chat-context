import yaml
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


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
class Config:
    exports: ExportsConfig
    embedding: EmbeddingConfig
    storage: StorageConfig
    chunking: ChunkingConfig
    ingestion: IngestionConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    print("Loading config")
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
    
    return Config(
        exports=exports,
        embedding=embedding,
        storage=StorageConfig(**data['storage']),
        chunking=ChunkingConfig(**data['chunking']),
        ingestion=IngestionConfig(**data['ingestion'])
    )
