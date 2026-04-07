import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

load_dotenv()

from crawler import async_crawl, async_extract
from logger import *


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "qwen3-embedding:0.6b"
CHAT_MODEL = os.getenv("CHAT_MODEL")
INDEX_NAME = os.getenv("INDEX_NAME")

## Configure SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

## define embeddings and vector store
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# vectore_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# define text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)


async def main(url):
    log_header("DOCUMENTATION HELPER - INGESTION PHASE")

    # Crawl or Extract documents from the documentation site
    all_docs = await async_crawl(url)
    # all_docs = await async_extract(url)

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"🔪  Text Splitter: Starting to split {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    all_chunks = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Successfully split into {len(all_chunks)} chunks for {len(all_docs)} documents")


if __name__ == "__main__":
    url = "https://docs.langchain.com/oss/python/langchain/overview"
    # url = "https://python.langchain.com/"
    asyncio.run(main(url))
