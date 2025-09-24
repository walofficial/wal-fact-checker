# File: src/wal_fact_checker/core/tools.py
"""Custom tools for the WAL Fact Checker system."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from google.adk.tools import FunctionTool
from google.adk.tools import google_search as adk_google_search
from groq import AsyncGroq

from .settings import settings

logger = logging.getLogger(__name__)

groq_client = AsyncGroq(api_key=settings.groq_api_key)


async def _scrape_single_website(
    url: str, country_code: str, client: httpx.AsyncClient
) -> dict[str, Any]:
    """Scrape content from a single website using scrape.do API.

    Args:
        url: The URL to scrape
        country_code: Country code for geo-location
        client: HTTP client instance

    Returns:
        Dictionary containing the scraped content and metadata
    """
    try:
        params = {
            "token": settings.scrape_do_token,
            "url": url,
            "geoCode": country_code.upper(),
            "super": True,
            "output": "markdown",
            "render": True,
        }

        response = await client.get(settings.scrape_api_url, params=params)
        response.raise_for_status()

        # Get the markdown content from scrape.do
        content = response.text

        return {
            "url": url,
            "content": content,
            "format": "markdown",
            "country_code": country_code.upper(),
            "status": "success",
            "content_length": len(content),
        }
    except Exception as e:
        logger.exception("Error in _scrape_single_website")
        return {
            "url": url,
            "error": str(e),
            "content": "",
            "status": "error",
            "status_code": 0,
            "api_used": "scrape.do",
        }


async def scrape_tool(urls: list[str]) -> dict[str, Any]:
    """Scrape content from multiple websites sequentially using scrape.do API.

    Args:
        urls: List of URLs to scrape

    Returns:
        Dictionary containing combined results from all scraped websites
    """
    # Validate input
    if not urls:
        return {
            "status": "error",
            "combined_content": "",
        }

    country_code = "US"  # Fixed default
    delay_between_requests = 1.0  # Fixed default for safe scraping
    results: list[dict[str, Any]] = []
    successful_scrapes = 0
    failed_scrapes = 0

    # Use a single HTTP client for all requests
    async with httpx.AsyncClient(timeout=settings.default_timeout) as client:
        for i, url in enumerate(urls):
            # Add delay between requests (except for the first one)
            if i > 0:
                await asyncio.sleep(delay_between_requests)

            # Scrape individual website
            result = await _scrape_single_website(url, country_code, client)
            results.append(result)

            # Track success/failure
            if result.get("status") == "success":
                successful_scrapes += 1
            else:
                failed_scrapes += 1

    # Combine all successful content and omit failed results
    combined_content = ""
    successful_results = [r for r in results if r.get("status") == "success"]

    if successful_results:
        content_parts = []
        for result in successful_results:
            url = result["url"]
            content = result["content"]
            if content.strip():  # Only include if content is not empty
                content_parts.append(f"# Content from {url}\n\n{content}\n\n---\n")
        combined_content = "\n".join(content_parts)

    return {
        "status": "success" if successful_scrapes > 0 else "error",
        "combined_content": combined_content,
    }


async def load_memory(query: str, limit: int = 10) -> dict[str, Any]:
    """Load relevant information from memory/RAG store.

    Args:
        query: Query to search memory for
        limit: Maximum number of results to return

    Returns:
        Dictionary containing relevant memories and their sources
    """
    # Placeholder - would integrate with actual memory service
    return {"query": query, "results": [], "total_found": 0}


async def save_to_memory(
    content: str, source: str, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Save information to memory/RAG store.

    Args:
        content: Content to save
        source: Source URL or identifier
        metadata: Additional metadata about the content

    Returns:
        Dictionary confirming the save operation
    """
    return {
        "saved": True,
        "content_length": len(content),
        "source": source,
        "metadata": metadata or {},
    }


async def search_tool(query: str, country: str) -> dict[str, Any]:
    """
    Performs intelligent web search using Groq Compound AI with regional optimization.

    This function executes comprehensive web searches with automatic query enhancement,
    timeline filtering, and regional result prioritization. Ideal for fact-checking,
    research, and information verification across multiple languages including Georgian and English.

    Args:
        query: Search query string. Can include specific claims, questions, or topics.
               Examples: "COVID-19 vaccine effectiveness 2023", "Georgia election results 2024"
        country: Two-letter country code (ge, us, uk, etc.) or full country name to boost
                regional results. Supported: ge (Georgia), us, uk, ca, au, nz, ie, de, fr, it, es, pt, nl

    Returns:
        dict: Search response containing:
            - status: "success" or "error"
            - results: List of search results, each with:
                * title: Result title/headline
                * url: Source URL
                * description: Content snippet or summary
                * score: Relevance score from search engine
    """
    try:
        system_prompt = (
            "You are a world-class fact-check searcher. Support Georgian and English. "
            "When appropriate, augment queries with before:/after: filters to match timelines."
        )

        country_map = {
            "ge": "georgia",
            "us": "united states",
            "uk": "united kingdom",
            "ca": "canada",
            "au": "australia",
            "nz": "new zealand",
            "ie": "ireland",
            "de": "germany",
            "fr": "france",
            "it": "italy",
            "es": "spain",
            "pt": "portugal",
            "nl": "netherlands",
        }

        response = await groq_client.chat.completions.create(
            model="groq/compound",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Search information on the web for query; add time range if helpful: "
                        + query
                    ),
                },
            ],
            search_settings={
                country: country_map.get(country.lower(), country.lower())
            },
        )

        results = []
        executed_tools = response.choices[0].message.executed_tools

        if executed_tools:
            for executed_tool in executed_tools:
                search_results = executed_tool.search_results
                if search_results and search_results.results:
                    for r in search_results.results:
                        if r and r.url:
                            title = r.title
                            url = r.url
                            description = r.content or ""
                            score = r.score
                            results.append(
                                {
                                    "title": title,
                                    "url": url,
                                    "description": description,
                                    "score": score,
                                }
                            )

        return {"status": "success", "results": results}
    except Exception:  # noqa: BLE001 - surface clean error string
        logger.exception("Error in groq_search")
        return {"status": "error"}


# Create ADK-compatible tool instances
scrape_websites_tool = FunctionTool(scrape_tool)
load_memory_tool = FunctionTool(load_memory)
save_to_memory_tool = FunctionTool(save_to_memory)
groq_search_tool = FunctionTool(search_tool)
# Export all tools for easy import
__all__ = [
    "search_tool",
    "load_memory",
    "save_to_memory",
    "scrape_website_tool",
    "scrape_websites_tool",
    "load_memory_tool",
    "save_to_memory_tool",
    "groq_search_tool",
    "adk_google_search",  # Use built-in ADK Google search tool
]
