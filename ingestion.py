from dotenv import load_dotenv

load_dotenv()

import asyncio

from config import set_ssl_certificates
from crawler import crawl_async, extract_async
from logger import *
from splitter import chunk_documents
from vectorstore import index_documents_async

## Configure SSL context to use certifi's CA bundle
ssl_context = set_ssl_certificates()


async def main(url):
    log_header("DOCUMENTATION HELPER - INGESTION PIPELINE")

    # Crawl or Extract documents from the documentation site
    all_docs = await crawl_async(url)
    # all_docs = await extract_async(url)

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    all_chunks = chunk_documents(all_docs)

    # Index chunks into the vector store
    log_header("VECTOR STORE INDEXING PHASE")
    # await index_documents_async(all_chunks, batch_size=500)
    await index_documents_async(all_chunks, batch_size=500, use_pinecone=False)

    log_header("INGESTION PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline completed successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • URLs mapped: {len(all_docs)}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(all_chunks)}")


if __name__ == "__main__":
    url = "https://docs.langchain.com/oss/python/langchain/overview"
    # url = "https://python.langchain.com/"
    asyncio.run(main(url))
