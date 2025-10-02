# File: src/wal_fact_checker/agents/research/single_question_research_agent.py
"""Unified research agent that combines decision-making, scraping, and analysis."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import numpy as np
from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool, ToolContext
from numpy._typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from wal_fact_checker.core.settings import settings
from wal_fact_checker.core.tools import groq_search_tool, scrape_websites_tool
from wal_fact_checker.utils.embedding_service import embedding_service

logger = logging.getLogger(__name__)

MAX_NUMBER_OF_SEARCH_TOOL_CALLS = 2
MAX_NUMBER_OF_SCRAPE_TOOL_CALLS = 1
COSINE_SIMILARITY_THRESHOLD = 0.85  # Threshold for considering queries as duplicates

tool_max_calls: dict[str, int] = {
    "search_tool": MAX_NUMBER_OF_SEARCH_TOOL_CALLS,
    "scrape_tool": MAX_NUMBER_OF_SCRAPE_TOOL_CALLS,
}


def create_enforce_query_deduplication_callback(
    cache: dict[str, Any],
) -> Callable[[BaseTool, dict[str, Any], ToolContext], dict[str, Any] | None]:
    async def enforce_query_deduplication(
        tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        """
        Prevent duplicate search queries using embedding similarity.

        Returns error dict if query is too similar to a previous query, None otherwise.
        """
        # Only apply to search_tool
        if tool.name != "search_tool":
            return None

        query = args.get("query", "")
        if not query:
            return None

        try:
            embedding = (
                await embedding_service.generate_embeddings(
                    [query], task_type="SEMANTIC_SIMILARITY"
                )
            )[0]

            queries_key = f"{tool.name}_queries_embeddings"

            previous_embeddings: list[NDArray] = cache.get(queries_key, [])

            if not previous_embeddings:
                cache[queries_key] = [embedding]
                return None

            previous_embeddings_matrix = np.array(previous_embeddings)
            current_embedding_matrix = np.array([embedding])

            similarity_matrix = cosine_similarity(
                current_embedding_matrix, previous_embeddings_matrix
            )

            is_duplicate = False

            for i in range(len(current_embedding_matrix)):
                for j in range(len(previous_embeddings_matrix)):
                    similarity = similarity_matrix[i, j]

                    logger.info(
                        "enforce_query_deduplication: Similarity between queries",
                        extra={
                            "json_fields": {"similarity": similarity},
                        },
                    )

                    if similarity >= COSINE_SIMILARITY_THRESHOLD:
                        is_duplicate = True
                        break

            previous_embeddings.append(embedding)
            cache[queries_key] = previous_embeddings

            if is_duplicate:
                return {
                    "status": "error",
                    "message": "Query is too similar to a previous query",
                }

        except Exception:
            logger.exception(
                "enforce_query_deduplication: Failed to generate embedding, proceeding anyway.",
                extra={
                    "json_fields": {
                        "operation": "enforce_query_deduplication",
                    }
                },
            )
        return None

    return enforce_query_deduplication


def create_enforce_tool_call_limits_callback(
    cache: dict[str, Any],
) -> Callable[[BaseTool, dict[str, Any], ToolContext], dict[str, Any] | None]:
    def enforce_tool_call_limits(
        tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        """Inspects/modifies tool args or skips the tool call."""
        agent_name = tool_context.agent_name
        tool_name = tool.name

        key = f"{agent_name}_{tool_name}_calls"
        number_of_calls = cache.get(key, 0)

        max_number_of_calls = tool_max_calls.get(tool_name, 0)

        if number_of_calls >= max_number_of_calls:
            logger.warning(
                "[Callback] Detected 'BLOCK'. Skipping tool execution.",
                extra={
                    "json_fields": {
                        "agent_name": agent_name,
                        "tool_name": tool_name,
                        "number_of_calls": number_of_calls,
                        "max_number_of_calls": max_number_of_calls,
                    }
                },
            )
            return {"status": "error", "message": f"{tool_name} call limit reached"}

        cache[key] = number_of_calls + 1
        logger.info(
            "[Callback] Tool call limit enforced.",
            extra={
                "json_fields": {
                    "agent_name": agent_name,
                    "tool_name": tool_name,
                    "number_of_calls": number_of_calls + 1,
                }
            },
        )
        return None

    return enforce_tool_call_limits


def compose_before_tool_callbacks(
    callbacks: list[
        Callable[
            [BaseTool, dict[str, Any], ToolContext],
            dict[str, Any] | None | Awaitable[dict[str, Any] | None],
        ]
    ],
) -> Callable[
    [BaseTool, dict[str, Any], ToolContext], Awaitable[dict[str, Any] | None]
]:
    async def combined(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext):
        for cb in callbacks:
            out = cb(tool, args, tool_context)
            if inspect.isawaitable(out):
                out = await out
            if out is not None:
                return out
        return None

    return combined


def create_store_search_urls_callback(
    cache: dict[str, Any],
) -> Callable[
    [BaseTool, dict[str, Any], ToolContext, dict[str, Any]],
    dict[str, Any] | None,
]:
    """
    Create after_tool_callback for search_tool to store URL->query mapping.

    Stores URLs as keys and queries as values for later use by scrape_tool.
    """

    def store_search_urls(
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Store URLs from search results with corresponding query."""
        if tool.name != "search_tool":
            return None

        query = args.get("query", "")
        if not query:
            return None

        results = tool_response.get("results", [])
        if not results:
            return None

        url_to_query_key = "url_to_query_mapping"
        url_to_query = cache.get(url_to_query_key, {})

        for result in results:
            url = result.get("url")
            if url:
                url_to_query[url] = query

        cache[url_to_query_key] = url_to_query

        logger.info(
            "store_search_urls: Stored URL->query mappings",
            extra={
                "json_fields": {
                    "query": query,
                    "urls_count": len([r for r in results if r.get("url")]),
                    "total_mappings": len(url_to_query),
                }
            },
        )

        return None

    return store_search_urls


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_filter_scraped_content_callback(
    cache: dict[str, Any],
    similarity_threshold: float = 0.75,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
) -> Callable[
    [BaseTool, dict[str, Any], ToolContext, dict[str, Any]],
    Awaitable[dict[str, Any] | None],
]:
    """
    Create after_tool_callback for scrape_tool to filter content by embedding similarity.

    Uses stored URL->query mappings to filter scraped chunks by cosine similarity.
    """

    async def filter_scraped_content(
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Filter scraped content chunks by query embedding similarity per URL."""
        if tool.name != "scrape_tool":
            return None

        combined_content = tool_response.get("combined_content", {})
        if not combined_content or not isinstance(combined_content, dict):
            return None

        url_to_query = cache.get("url_to_query_mapping", {})
        if not url_to_query:
            logger.warning(
                "filter_scraped_content: No URL->query mappings found, returning original content"
            )
            return None

        try:
            # Collect all unique queries and generate embeddings in one batch call
            unique_queries = list(
                {query for query in url_to_query.values() if query and query.strip()}
            )

            if not unique_queries:
                logger.warning(
                    "filter_scraped_content: No valid queries found in URL mappings"
                )
                return None

            query_embeddings_list = await embedding_service.generate_embeddings(
                unique_queries, task_type="RETRIEVAL_QUERY"
            )

            query_embeddings_cache: dict[str, list[float]] = dict(
                zip(unique_queries, query_embeddings_list, strict=True)
            )

            logger.info(
                "filter_scraped_content: Generated embeddings for unique queries",
                extra={
                    "json_fields": {
                        "unique_queries_count": len(unique_queries),
                        "total_urls": len(combined_content),
                    }
                },
            )

            filtered_combined_content: dict[str, str] = {}
            total_original_chunks = 0
            total_filtered_chunks = 0

            for url, content in combined_content.items():
                if not content or not content.strip():
                    continue

                query = url_to_query.get(url)
                if not query or not query.strip():
                    logger.warning(
                        "filter_scraped_content: No query found for URL, skipping",
                        extra={"json_fields": {"url": url}},
                    )
                    filtered_combined_content[url] = content
                    continue

                query_embedding = query_embeddings_cache.get(query)
                if query_embedding is None:
                    logger.warning(
                        "filter_scraped_content: No embedding found for query, skipping",
                        extra={"json_fields": {"url": url, "query": query}},
                    )
                    filtered_combined_content[url] = content
                    continue

                chunks = chunk_text(content, chunk_size, chunk_overlap)

                if not chunks:
                    filtered_combined_content[url] = content
                    continue

                total_original_chunks += len(chunks)

                chunk_embeddings = await embedding_service.generate_embeddings(
                    chunks, task_type="RETRIEVAL_DOCUMENT"
                )

                query_embedding_matrix = np.array([query_embedding])
                chunk_embeddings_matrix = np.array(chunk_embeddings)

                similarity_matrix = cosine_similarity(
                    chunk_embeddings_matrix, query_embedding_matrix
                )

                similarities = similarity_matrix.flatten()

                high_similarity_indices = [
                    i
                    for i, sim in enumerate(similarities)
                    if sim >= similarity_threshold
                ]

                if not high_similarity_indices:
                    logger.warning(
                        "filter_scraped_content: No chunks met threshold for URL",
                        extra={
                            "json_fields": {
                                "url": url,
                                "threshold": similarity_threshold,
                                "max_similarity": float(similarities.max()),
                            }
                        },
                    )
                    filtered_combined_content[url] = content
                    continue

                filtered_chunks = [chunks[i] for i in high_similarity_indices]
                total_filtered_chunks += len(filtered_chunks)

                filtered_combined_content[url] = "\n\n".join(filtered_chunks)

                logger.info(
                    "filter_scraped_content: Filtered content by embedding similarity",
                    extra={
                        "json_fields": {
                            "url": url,
                            "filtered_combined_content": filtered_combined_content[url],
                            "length_of_filtered_combined_content": len(
                                filtered_combined_content[url]
                            ),
                            "length_of_original_content": len(content),
                        },
                    },
                )

            logger.info(
                "filter_scraped_content: Filtered content by embedding similarity",
                extra={
                    "json_fields": {
                        "urls_processed": len(combined_content),
                        "original_chunks": total_original_chunks,
                        "filtered_chunks": total_filtered_chunks,
                        "threshold": similarity_threshold,
                    }
                },
            )

            return {
                "status": tool_response.get("status", "success"),
                "combined_content": filtered_combined_content,
            }

        except Exception:
            logger.exception(
                "filter_scraped_content: Error filtering content, returning original"
            )
            return None

    return filter_scraped_content


def compose_after_tool_callbacks(
    callbacks: list[
        Callable[
            [BaseTool, dict[str, Any], ToolContext, dict[str, Any]],
            dict[str, Any] | None | Awaitable[dict[str, Any] | None],
        ]
    ],
) -> Callable[
    [BaseTool, dict[str, Any], ToolContext, dict[str, Any]],
    Awaitable[dict[str, Any] | None],
]:
    """Compose multiple after_tool callbacks in sequence."""

    async def combined(
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict[str, Any],
    ):
        current_response = tool_response
        for cb in callbacks:
            out = cb(tool, args, tool_context, current_response)
            if inspect.isawaitable(out):
                out = await out
            if out is not None:
                current_response = out
        return current_response if current_response != tool_response else None

    return combined


def create_combined_before_tool_callback(
    cache: dict[str, Any],
) -> Callable[[BaseTool, dict[str, Any], ToolContext], dict[str, Any] | None]:
    """
    Chain multiple before-tool callbacks in sequence.

    First checks query deduplication, then enforces tool call limits.
    If any callback returns a dict (error/skip), stop and return it.
    """
    dedup_callback = create_enforce_query_deduplication_callback(cache)
    limit_callback = create_enforce_tool_call_limits_callback(cache)

    return compose_before_tool_callbacks([dedup_callback, limit_callback])


def create_combined_after_tool_callback(
    cache: dict[str, Any],
) -> Callable[
    [BaseTool, dict[str, Any], ToolContext, dict[str, Any]],
    Awaitable[dict[str, Any] | None],
]:
    """
    Chain multiple after-tool callbacks in sequence.

    First stores URL->query mappings from search, then filters scraped content.
    """
    store_urls_callback = create_store_search_urls_callback(cache)
    filter_content_callback = create_filter_scraped_content_callback(cache)

    return compose_after_tool_callbacks([store_urls_callback, filter_content_callback])


def create_single_question_research_agent(question: str, output_key: str) -> LlmAgent:
    """
    Factory function to create a new instance of a UnifiedResearchAgent.
    This is necessary to comply with ADK's single-parent rule for agents.
    """
    current_date = datetime.now().strftime("%B %d, %Y")

    # Create shared cache for callbacks
    callback_cache: dict[str, Any] = {}

    return LlmAgent(
        # Each agent instance needs a unique name.
        name=f"UnifiedResearchAgent_{output_key}",
        model=settings.GEMINI_2_5_FLASH_MODEL,
        description=f"Intelligent research agent for: {question[:100]}...",
        instruction=f"""You are an intelligent research agent. Your task is to thoroughly research
the given question by strategically using search and scraping tools to gather
comprehensive, verifiable evidence.

## CURRENT DATE

**Today's date is: {current_date}**

Use this date when the question asks about "current", "as of today", "latest",
or any time-sensitive information. This is the reference point for all temporal
queries.

## CRITICAL REQUIREMENT - NO TRAINING DATA

**YOU MUST NOT use any factual information from your training data**

- ALL factual claims MUST come from search results and scraped content
- Your training data may ONLY be used for:
  - Logic and reasoning capabilities
  - Understanding language and context
  - Analytical frameworks that don't change over time
  - Planning research strategy
- MANDATORY: Use search_tool for ALL factual research

## YOUR TOOLS

**search_tool (use up to {MAX_NUMBER_OF_SEARCH_TOOL_CALLS} times)**:
- Takes a query string (can be any length - powered by LLM)
- Returns search results with titles, snippets, and URLs
- Use for exploration and finding authoritative sources

**scrape_tool (use up to {MAX_NUMBER_OF_SCRAPE_TOOL_CALLS} time)**:
- Takes a list of URLs (maximum 5 URLs per call)
- Returns full page content from those URLs
- Use only when search snippets are insufficient

## QUESTION DECOMPOSITION STRATEGY (CRITICAL)

The research question is complex and cannot be answered with a single search query.
You MUST decompose it into targeted exploration areas.

DECOMPOSITION APPROACH:
1. **Analyze the question** - identify key components, entities, timeframes, numbers
2. **Break into sub-queries** - create 2-3 focused search queries that together cover the question
3. **Sequential refinement** - use first search to inform second search

DO NOT pass the full question to search_tool - it's too complex!
Instead, create FOCUSED, TARGETED search queries for specific aspects.

### Decomposition Examples:

Question: "Is Alice Kim currently serving as CTO of Acme Corp as of {current_date}?"

Decomposition strategy:
- Search 1: "Alice Kim CTO Acme Corp 2025" (find current status)
- Search 2: "Acme Corp leadership team 2025" (verify from company side)
- OR if Search 1 unclear: "Alice Kim Acme Corp departure resignation" (check if she left)

Question: "When did Tesla officially begin Cybertruck production according to company announcements?"

Decomposition strategy:
- Search 1: "Tesla Cybertruck production start date official announcement"
- Search 2: "Tesla Cybertruck manufacturing 2023 2024" (if first search needs temporal refinement)
- OR: "Tesla Cybertruck factory Austin production timeline" (if need location context)

Question: "What was Acme Corp's exact revenue in Q4 2023 according to official reports?"

Decomposition strategy:
- Search 1: "Acme Corp Q4 2023 revenue earnings report"
- Search 2: "Acme Corp 2023 annual financial results SEC filing" (if need official source)

## MULTI-SEARCH STRATEGY

You have {MAX_NUMBER_OF_SEARCH_TOOL_CALLS} search calls - use them strategically:

**APPROACH A - Broad then Narrow:**
- Search 1: Broad query covering main question
- Search 2: Narrow query based on gaps from Search 1

**APPROACH B - Different Angles:**
- Search 1: Direct entity/event query
- Search 2: Alternative angle (company perspective, time-based, official sources)

**APPROACH C - Temporal Refinement:**
- Search 1: General query with entity names
- Search 2: Add specific timeframe or "latest news" or "as of 2025"

**Choose the approach** based on the question type and what you learn from Search 1.

## SCRAPING DECISION FRAMEWORK

Use scrape_tool ONLY if:
✓ Search snippets mention key facts but lack sufficient detail
✓ You found authoritative sources (official docs, announcements, reports)
✓ Snippets reference specific data but don't show full context
✓ Multiple sources point to same URL as definitive source

DO NOT scrape if:
✗ Search snippets already provide complete answer
✗ URLs are low-quality sources (forums, blogs, aggregators)
✗ No clear authoritative source emerged from search
✗ Search results are contradictory (scraping won't resolve)

**When scraping**: Select up to 5 most authoritative URLs from search results.
Priority: official websites > news organizations > academic > industry publications

## RESEARCH WORKFLOW

### Step 1: Decompose Question
- Analyze the question and identify key components
- Break into {MAX_NUMBER_OF_SEARCH_TOOL_CALLS}-3 focused search-friendly queries
- Plan your search strategy (broad-narrow, multi-angle, or temporal)

### Step 2: Execute Search Calls (up to {MAX_NUMBER_OF_SEARCH_TOOL_CALLS} times)
- **For each search call**:
  - Use search_tool with focused query (NOT the full question)
  - Review results: titles, snippets, URLs, source quality
  - Evaluate what you learned and what gaps remain

- **Between search calls**:
  - Assess if current snippets can answer the question
  - Identify what's still missing or unclear
  - Refine next query based on findings
  - Note any authoritative URLs worth scraping

- **Stop searching if**:
  - You have sufficient information to answer comprehensively
  - You've used all {MAX_NUMBER_OF_SEARCH_TOOL_CALLS} search calls
  - Additional searches unlikely to add value

### Step 3: Scrape (if needed, up to {MAX_NUMBER_OF_SCRAPE_TOOL_CALLS} time)
- **Evaluate scraping value**:
  - Do search snippets lack critical details?
  - Did you find authoritative sources needing full content?
  - Will scraping provide decisive evidence?

- **If scraping**:
  - Select up to 5 most authoritative URLs from search results
  - Use scrape_tool with selected URLs
  - Extract relevant quotes and evidence

### Step 4: Synthesize Answer
- Combine ALL evidence from search calls and scraping (if used)
- Construct comprehensive answer addressing the full question
- Ensure every factual claim is backed by a source
- Note any limitations, conflicts, or uncertainties found

## SOURCE CAPTURE RULES (MANDATORY)

**From search_tool**:
- You MAY use the search snippet verbatim as the citation
- You MUST attach the exact result URL that snippet came from
- Format: {{"url": "https://...", "citation": "snippet text..."}}

**From scrape_tool**:
- You MUST quote verbatim text from the scraped page
- The URL MUST be exactly one of the URLs passed to scrape_tool
- Format: {{"url": "https://exact-scraped-url.com", "citation": "exact quote from page..."}}

**Source quality priority**:
1. Official company/organization websites
2. Government sites, SEC filings, official registries
3. Major news organizations
4. Industry publications and trade journals
5. Academic sources

## OUTPUT FORMAT

Return a JSON object with this exact structure:
{{
    "question": "The original research question (copy exactly)",
    "detailed_answer": "Comprehensive answer addressing the full question with context, evidence, and analysis. Include specific facts, dates, numbers, and sources. If information is conflicting or uncertain, note this explicitly.",
    "sources": [
        {{
            "url": "https://example.com/path-to-exact-page",
            "citation": "Verbatim quote or key datum from that exact URL"
        }}
    ]
}}

## QUALITY REQUIREMENTS

**Comprehensiveness**:
- Answer must fully address the research question
- Include all relevant facts, dates, numbers, names found
- Provide context that helps verify/refute related claims
- Note any qualifiers, conditions, or uncertainties

**Evidence-based**:
- Every factual statement must be sourced
- Cross-reference when multiple sources confirm same fact
- Acknowledge conflicts between sources if found
- State explicitly if information cannot be found

**Source quality**:
- Prioritize authoritative, primary sources
- Include URL and citation for each source
- URL must be the exact page where citation appears
- Citation must be verbatim excerpt (prefer direct quotes)

**Transparency**:
- If answer is incomplete, state what's missing
- If sources conflict, present both perspectives
- If timeframe is critical, note as-of dates from sources (remember: today is {current_date})
- If no reliable information found, state this clearly

## MANDATORY URL-CITATION CONSISTENCY

For every item in "sources" array:
- **url**: Exact page where citation text appears
  - Include full protocol: https://
  - No homepages unless citation is from homepage
  - No search pages or aggregator pages
  - No shortened URLs - use full canonical URL
  - For PDFs, include fragment if available: #page=3

- **citation**: Verbatim excerpt from that exact URL
  - Prefer direct quotes in quotation marks
  - If paraphrasing, it must be close paraphrase of specific content
  - Must be traceable to the specific URL provided
  - Include enough context to be meaningful

- Do not include any source without a specific citation
- Do not cite training data or general knowledge

---

**Research Question**: {question}

Begin by analyzing the question and planning your search decomposition strategy.
Then execute your research workflow systematically.
""",
        tools=[
            groq_search_tool,
            scrape_websites_tool,
        ],
        before_tool_callback=create_combined_before_tool_callback(callback_cache),
        after_tool_callback=create_combined_after_tool_callback(callback_cache),
        output_key=output_key,
    )
