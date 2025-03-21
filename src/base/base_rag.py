from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from llama_index.core import StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM


class BaseRag(ABC):
    def __init__(self, llm: LLM, storage: StorageContext, embed_model: Optional[BaseEmbedding] = None):
        self.llm = llm
        self.storage = storage
        self.embed_model = embed_model

    @abstractmethod
    def ingest(self, urls: List[str]) -> None:
        """Ingest documents from URLs"""
        pass

    @abstractmethod
    def query(self, query: str) -> str:
        """Query the RAG system"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the storage"""
        pass
