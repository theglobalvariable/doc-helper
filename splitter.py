from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from logger import *

# define text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)


def chunk_documents(all_docs):
    log_info(
        f"🔪  Text Splitter: Starting to split {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    all_chunks = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Successfully split into {len(all_chunks)} chunks for {len(all_docs)} documents"
    )
    return all_chunks
