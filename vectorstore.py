import asyncio
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import get_config
from logger import *

config = get_config()

## define embeddings and vector store
embeddings = OllamaEmbeddings(model=config["EMBEDDING_MODEL"])

vector_store_chroma = Chroma(
    persist_directory="./chroma_db", embedding_function=embeddings
)
vector_store_pinecone = PineconeVectorStore(
    index_name=config["INDEX_NAME"], embedding=embeddings
)


async def index_documents_async(
    documents: List[Document], batch_size: int = 50, use_pinecone: bool = True
):
    log_info(
        f"📥  Vector Store Indexing: Starting to index {len(documents)} documents into {'Pinecone' if use_pinecone else 'Chroma'} with batch size {batch_size}",
        Colors.DARKCYAN,
    )

    vector_store = vector_store_pinecone if use_pinecone else vector_store_chroma

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f"Vector Store Indexing: Split documents into {len(batches)} batches of {batch_size} size for indexing",
    )

    async def index_batch(batch_num: int, batch_docs: List[Document]):
        try:
            await vector_store.aadd_documents(batch_docs)
            log_success(
                f"Vector Store Indexing: Successfully indexed batch {batch_num}/{len(batches)} ({len(batch_docs)} documents)"
            )
            return True
        except Exception as e:
            log_error(
                f"Vector Store Indexing: Error indexing batch {batch_num} - {str(e)}"
            )
            return False

    tasks = [index_batch(idx + 1, batch) for idx, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_batches = sum(1 for result in results if result is True)

    if successful_batches == len(batches):
        log_success(
            f"Vector Store Indexing: All {len(batches)} batches indexed successfully! ({successful_batches}/{len(batches)})"
        )
    else:
        log_warning(
            f"Vector Store Indexing: Indexed {successful_batches}/{len(batches)} batches successfully"
        )
