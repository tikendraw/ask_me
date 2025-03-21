# %%
import os
import sqlite3
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from llama_index.core import (
    ComposableGraph,
    ServiceContext,
    Settings,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_response

# imports
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.litellm import LiteLLM
from llama_index.llms.openai import OpenAI
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore

from base.logger import Logger
from base.rag import BaseRag

load_dotenv("/home/t/atest/.global_env")

logger = Logger().get_logger()
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

# %%

models = ['gemini/gemini-2.0-flash']

llm = LiteLLM(temperature=0, model=models[0])
embed_model = GeminiEmbedding()

# %%
Settings.embed_model = embed_model
Settings.llm = llm

# %%
os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"  # default is "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"  # assumed we have NebulaGraph installed locally

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

# %%

# graph_store = NebulaGraphStore(
#     space_name=space_name,
#     edge_types=edge_types,
#     rel_prop_names=rel_prop_names,
#     tags=tags,
# )
# storage_context = StorageContext.from_defaults(graph_store=graph_store)

# %%




DEFAULT_CHUNK_SIZE=2048
DEFAULT_CHUNK_OVERLAP = 200

class SiteRag(BaseRag):

    def __init__(self, llm:LLM, storage:StorageContext, embed_model:BaseEmbedding=None,
                 chunk_size:int=DEFAULT_CHUNK_SIZE, chunk_overlap:int=DEFAULT_CHUNK_OVERLAP):
        self.reader = None

        self.llm = llm
        
        if embed_model:
            self.embed_model = embed_model
            Settings.embed_model = self.embed_model

        Settings.llm = self.llm
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        self.storage_context=storage

    
    def _ingest(self, dir:Path)->None:
        self.reader =SimpleDirectoryReader(input_dir=dir)
        documents = self.reader.load_data()
        nodes = SentenceSplitter().get_nodes_from_documents(documents)


        self.storage_context.docstore.add_documents(nodes)
        logger.info(f"{len(self.storage_context.docstore.docs)} Ingested.")
        
    
    
    # def _query(self, query:str, n:int)->str:
    #     self.query_engine = self.storage_context.as_query_engine()
    #     list_response = self.query_engine.query("What is a summary of this document?")
        
    #     pass

# %%
import asyncio
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.readers.wikipedia import WikipediaReader

# %%
documents = WikipediaReader().load_data(pages=['avengers (2012)', 'Ironman (2008)'])

# %%
documents[0]

# %%
transformations = [
    
]

# %%
index = VectorStoreIndex.from_documents(
    documents,
    transformations = []
    
    )


# %%

# Create a RAG tool using LlamaIndex
# documents = SimpleDirectoryReader("data").load_data()
query_engine = index.as_query_engine()

# %%
out = query_engine.query("what does  Asgardian Loki  do")

# %%
dir(out)

# %%
out.response

# %%
sn = out.source_nodes[0]

# %%
dir(sn)

# %%
sn.score

# %%
sn

# %%
from llama_index.readers.remote import RemoteReader

# %%
r = RemoteReader()

# %%
p  = r.load_data(url = 'https://en.wikipedia.org/wiki/Nick_Fury_(Marvel_Cinematic_Universe)')

# %%
print(dir(p))

# %%
# h = html2text.HTML2Text()
# h.ignore_links=True
# h.ignore_tables=True
# h.ignore_mailto_links=True
# h.ignore_emphasis=True
# %%
import logging

import html2text

logger = logging.getLogger('notebook')

# %%
def get_url_docs(url):
    r = RemoteReader()
    h = html2text.HTML2Text()
    
    h.ignore_links=True
    h.ignore_tables=True
    h.ignore_mailto_links=True
    h.ignore_emphasis=True
    
    docs  = r.load_data(url = url)
    clean_doc = []
    for i in docs:
        try:
            clean_doc.append(h.handle(i.text))
        except Exception :
            logger.error(f'Failed to get web content for {url}')
            clean_doc.append(None)
        
    for i,j in zip(docs, clean_doc):
        i.set_content(j) 
    
    return docs

# %%
# dir(p[0])

# %%
out = get_url_docs('https://en.wikipedia.org/wiki/Nick_Fury_(Marvel_Cinematic_Universe)')

# %%
# load_dotenv('../../.global_env')

# %%
out

# %%
from functools import cache

from nest_asyncio import apply

apply()

# %%
from functools import lru_cache
from itertools import chain

from llama_index.core.base.base_query_engine import BaseQueryEngine

# A dictionary-based manual cache (because lists are unhashable for lru_cache)
_query_engine_cache = {}

def _hash_urls(urls: list[str]) -> int:
    """Create a hashable key from the list of URLs."""
    return hash(frozenset(urls))  # Using frozenset so order doesn't matter

def get_query_engine(urls: list[str]) -> BaseQueryEngine:
    """Return a cached query engine if available, otherwise create a new one."""
    cache_key = _hash_urls(urls)
    
    if cache_key in _query_engine_cache:
        return _query_engine_cache[cache_key]
    
    documents = list(chain.from_iterable(get_url_docs(url) for url in urls))

    index = VectorStoreIndex(use_async=True, nodes=documents, transformations=[])
    # retriever=index.as_retriever()
    query_engine = index.as_query_engine()
    
    _query_engine_cache[cache_key] = query_engine  # Cache the result
    
    return query_engine

# %%


# %%
from llama_index.core.schema import NodeWithScore


def get_answers(urls:list[str], query:str)->NodeWithScore:
    query_engine = get_query_engine(urls)
    out = query_engine.query(query)
    return out

# %%
urls = ['https://en.wikipedia.org/wiki/Nick_Fury_(Marvel_Cinematic_Universe)', 'https://en.wikipedia.org/wiki/Maria_Hill', 'https://en.wikipedia.org/wiki/Avengers:_Age_of_Ultron']

# %%
documents = list(chain.from_iterable(get_url_docs(url) for url in urls))


# %%



# %%
ii = VectorStoreIndex(nodes=documents[:1], transformations=transformations)

# %%
out = get_answers(urls = urls,
                  query = "what is maria hill 's full name",
                  )

# %%
out.response

# %%
q= get_query_engine(urls)

# %%
out = q.query('who plays maria hill in live actionmoview')

# %%
dir(out)

# %%
out.response

# %%



