from typing import List

import html2text
from IPython.display import Markdown, display
from llama_index.core import (
    KnowledgeGraphIndex,
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from llama_index.readers.remote import RemoteReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine
from .base.base_rag import BaseRag
from .base.logger import Logger

logger = Logger().get_logger()

N_DOCS=4
QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)

class WebRAG(BaseRag):
    def __init__(self, user_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_id = user_id
        self.reader = RemoteReader()
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_tables = True
        self.html_converter.ignore_mailto_links = True
        self.html_converter.ignore_emphasis = True
        self.index = None
        logger.info(f"Initialized WebRAG for user {self.user_id}")

    def _clean_html_content(self, text: str) -> str:
        """Clean HTML content using html2text"""
        try:
            return self.html_converter.handle(text)
        except Exception as e:
            logger.error(f"Error cleaning HTML content for user {self.user_id}: {e}")
            return text

    def _get_url_docs(self, url: str) -> List[Document]:
        """Fetch and clean documents from a URL"""
        try:
            docs = self.reader.load_data(url=url)
            for doc in docs:
                cleaned_content = self._clean_html_content(doc.text)
                doc.set_content(cleaned_content)
                # Add user_id to document metadata
                doc.metadata["user_id"] = self.user_id
            return docs
        except Exception as e:
            logger.error(f"Error fetching content from {url} for user {self.user_id}: {e}")
            return []

    def ingest(self, urls: List[str]) -> None:
        """Ingest documents from multiple URLs"""
        all_docs = []
        for url in urls:
            docs = self._get_url_docs(url)
            all_docs.extend(docs)
            logger.info(f"User {self.user_id} ingested {len(docs)} documents from {url}")

        if all_docs:
            self.index = VectorStoreIndex.from_documents(
                all_docs,
                storage_context=self.storage,
                embed_model=self.embed_model,
                transformations=[
                    
                ]
            )
            
            vector_retriever = self.index.as_retriever(similarity_top_k=N_DOCS)

            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.index.docstore, similarity_top_k=N_DOCS
            )
            

            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=N_DOCS,
                num_queries=2,  # set this to 1 to disable query generation
                mode="reciprocal_rerank",
                use_async=True,
                verbose=True,
                query_gen_prompt=QUERY_GEN_PROMPT,  # we could override the query generation prompt here
            )
            from llama_index.core.query_engine import RetrieverQueryEngine

            self.query_engine = RetrieverQueryEngine.from_args(
                retriever,
                # text_qa_template=PromptTemplate(
                #     template="Givent the query answer the question accondingly, if question be discriptive about the answer. Question: {query_str}\nAnswer:",

                # )
                )

            logger.info(f"Created index with {len(all_docs)} documents for user {self.user_id}")
        else:
            logger.warning(f"No documents were ingested for user {self.user_id}")

    async def query(self, query: str) -> str:
        """Query the RAG system"""
        if not self.index:
            raise ValueError(f"No content has been ingested yet for user {self.user_id}")
        
        if not self.query_engine:
            self.query_engine = self.index.as_query_engine()
        response = await self.query_engine.aquery(query)
        return str(response)

    def clear(self) -> None:
        """Clear the storage"""
        self.index = None
        self.storage.clear()
        logger.info(f"Cleared data for user {self.user_id}")
