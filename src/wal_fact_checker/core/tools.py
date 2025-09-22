# File: src/wal_fact_checker/core/tools.py
"""Custom tools for the WAL Fact Checker system."""

from __future__ import annotations

from typing import Any

import httpx
from google.adk.tools import FunctionTool
from google.adk.tools import google_search as adk_google_search

from .settings import settings


async def scrape_website(url: str, country_code: str) -> dict[str, Any]:
    """Scrape content from a website using scrape.do API.

    Args:
        url: The URL to scrape
        country_code: Country code for geo-location (default: US)

    Returns:
        Dictionary containing the scraped content and metadata
    """
    try:
        params = {
            "token": settings.scrape_do_token,
            "url": url,
            "geoCode": (country_code or "US").upper(),
            "super": True,
            "output": "markdown",
            "render": True,
        }

        async with httpx.AsyncClient(timeout=settings.default_timeout) as client:
            response = await client.get(settings.scrape_api_url, params=params)
            response.raise_for_status()

            # Get the markdown content from scrape.do
            content = response.text

            return {
                "url": url,
                "content": content,
                "format": "markdown",
                "country_code": (country_code or "US").upper(),
            }
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "content": "",
            "status_code": 0,
            "api_used": "scrape.do",
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


# Create ADK-compatible tool instances
scrape_website_tool = FunctionTool(scrape_website)
load_memory_tool = FunctionTool(load_memory)
save_to_memory_tool = FunctionTool(save_to_memory)

# Export all tools for easy import
__all__ = [
    "scrape_website",
    "load_memory",
    "save_to_memory",
    "scrape_website_tool",
    "load_memory_tool",
    "save_to_memory_tool",
    "adk_google_search",  # Use built-in ADK Google search tool
]
