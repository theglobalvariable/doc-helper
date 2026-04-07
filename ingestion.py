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

from logger import *

load_dotenv()

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

## define tavily instances
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=5, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main(url):
    log_header("DOCUMENTATION HELPER - INGESTION")

    log_info(
        f"🗺️  TavilyCrawl: Starting to crawl documentation from {url}", Colors.PURPLE
    )

    crawl_result = tavily_crawl.invoke(
        {
            "url": url,
            "max_depth": 5,
            "extract_depth": "advanced",
            "instructions": "Documentation relevant to ai agents, langchain, and langgraph. Focus on extracting code snippets, API references, and best practices. Ignore unrelated content like marketing pages, blogs, or user forums.",
        }
    )

    if crawl_result.get("error"):
        log_error(f"TavilyCrawl: Error crawling {url} - {crawl_result['error']}")
        return

    log_success(
        f"TavilyCrawl: Successfully crawled {len(crawl_result["results"])} documents from {url}"
    )

    all_docs = []
    for crawl_item in crawl_result["results"]:
        page_url = crawl_item["url"]
        content = crawl_item["raw_content"]

        if not content:
            log_warning(f"TavilyCrawl: No content extracted from {page_url}")
            continue

        log_info(f"TavilyCrawl: Successfully crawled {page_url}")

        all_docs.append(Document(page_content=content, metadata={"source": page_url}))


if __name__ == "__main__":
    url = "https://docs.langchain.com/oss/python/langchain/overview"
    # url = "https://python.langchain.com/"
    asyncio.run(main(url))
