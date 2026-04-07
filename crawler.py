import asyncio
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import *

## define tavily instances
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=5, max_pages=1000)
tavily_crawl = TavilyCrawl()

INSTRUCTIONS = "Documentation relevant to ai agents, langchain, and langgraph. Focus on extracting code snippets, API references, and best practices. Ignore unrelated content like marketing pages, blogs, or user forums."


async def async_crawl(url: str) -> list[Document]:
    log_info(
        f"🗺️  TavilyCrawl: Starting to crawl documentation from {url}", Colors.PURPLE
    )

    crawl_result = await tavily_crawl.ainvoke(
        {
            "url": url,
            "max_depth": 5,
            "extract_depth": "advanced",
            "instructions": INSTRUCTIONS,
        }
    )

    if crawl_result.get("error"):
        log_error(f"TavilyCrawl: Error crawling {url} - {crawl_result['error']}")
        return []

    log_success(
        f"TavilyCrawl: Successfully crawled {len(crawl_result["results"])} documents from documentation site"
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

    return all_docs


async def async_map(url: str):
    log_info(f"🗺️  TavilyMap: Starting to map documentation from {url}", Colors.PURPLE)

    map_result = await tavily_map.ainvoke(
        {
            "url": url,
            "instructions": INSTRUCTIONS,
        }
    )

    if map_result.get("error"):
        log_error(f"TavilyMap: Error mapping {url} - {map_result['error']}")
        return []

    log_success(
        f"TavilyMap: Successfully mapped {len(map_result['results'])} URLs from documentation site"
    )

    return list(map_result["results"] or [])


def chunk_urls(urls: list[str], chunk_size: int = 20) -> list[list[str]]:
    return [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]


async def extract_batch(urls: list[str], batch_num: int) -> List[Dict[str, Any]]:
    try:
        log_info(
            f"TavilyExtract: Processing batch {batch_num} extraction for {len(urls)} URLs",
            Colors.BLUE,
        )

        extract_result = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(
            f"TavilyExtract: Completed batch {batch_num} - extracted {len(extract_result.get('results', []))} documents"
        )
        return extract_result
    except Exception as e:
        log_error(
            f"TavilyExtract: Error occurred while processing batch {batch_num} - {str(e)}"
        )
        return []


async def async_extract(url: str) -> list[Document]:
    urls = await async_map(url)

    url_batches = chunk_urls(urls, chunk_size=20)
    log_info(
        f"URL Processing: Split {len(urls)} URLs into {len(url_batches)} batches for extraction",
        Colors.BLUE,
    )

    log_info(
        f"🗺️  TavilyExtract: Starting to extract documentation from {url}",
        Colors.PURPLE,
    )

    log_info(
        f"TavilyExtract: Starting concurrent batch extraction for {len(url_batches)} batches",
        Colors.DARKCYAN,
    )

    tasks = [extract_batch(batch, idx + 1) for idx, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_pages = []
    failed_batches = 0

    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {str(result)}")
            failed_batches += 1
            continue

        assert isinstance(result, dict)
        for extracted_page in result.get("results", []):
            document = Document(
                page_content=extracted_page["raw_content"] or "",
                metadata={"source": extracted_page["url"] or "unknown"},
            )
            all_pages.append(document)

    log_success(
        f"TavilyExtract: Extraction completed! Total pages extracted: {len(all_pages)}"
    )

    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")

    return all_pages
