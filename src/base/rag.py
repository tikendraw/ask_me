from abc import ABC, abstractmethod
from .logger import Logger

logger = Logger().get_logger()


class BaseRag(ABC):
    
    
    def ingest(self, **kwargs)-> None:
        logger.info(f"Ingesting...")
        self._ingest(**kwargs)
        logger.info(f"Ingestion complete")
    
    @abstractmethod
    def _ingest(self, data)-> None:
        pass

    async def aingest(self, **kwargs)-> None:
        logger.info(f"Async Ingesting...")
        await self._ingest(**kwargs)
        logger.info(f"Async ingestion complete")
    
    @abstractmethod
    async def _aingest(self, data)-> None:
        pass
    
    def query(self, query, n):
        logger.info(f"Querying...")
        return self._query(query, n)

    
    @abstractmethod
    def _query(self, query, n)-> str:
        pass
    
    async def aquery(self, query, n):
        logger.info(f"Async Querying...")
        return await self._query(query, n)
    
    @abstractmethod
    async def _aquery(self, query, n)-> str:
        pass