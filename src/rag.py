from pydantic_core.core_schema import none_schema
import requests
from bs4 import BeautifulSoup
import sqlite3
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SentenceSplitter

# imports
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.llms.litellm import LiteLLM
import os

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core import SummaryIndex
from llama_index.core import ComposableGraph
from llama_index.llms.openai import OpenAI
from llama_index.core.response.notebook_utils import display_response
from llama_index.core import Settings
from dotenv import load_dotenv
from .base.rag import BaseRag
from .base.logger import Logger
from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore

load_dotenv("/home/t/atest/.global_env")

logger = Logger().get_logger()

class SiteRag(BaseRag):

    ...