# File: src/wal_fact_checker/agents/research/single_question_research_agent.py
"""Unified research agent that combines decision-making, scraping, and analysis."""

from __future__ import annotations

import logging
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool, ToolContext

from wal_fact_checker.core.settings import settings
from wal_fact_checker.core.tools import groq_search_tool, scrape_websites_tool

logger = logging.getLogger(__name__)

MAX_NUMBER_OF_SEARCH_TOOL_CALLS = 5
MAX_NUMBER_OF_SCRAPE_TOOL_CALLS = 3

tool_max_calls: dict[str, int] = {
    "search_tool": MAX_NUMBER_OF_SEARCH_TOOL_CALLS,
    "scrape_tool": MAX_NUMBER_OF_SCRAPE_TOOL_CALLS,
}


def enforce_tool_call_limits(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """Inspects/modifies tool args or skips the tool call."""
    agent_name = tool_context.agent_name
    tool_name = tool.name

    key = f"{agent_name}_{tool_name}_calls"
    number_of_calls = tool_context.state.get(key, 0)

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

    tool_context.state[key] = number_of_calls + 1
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


def create_single_question_research_agent(question: str, output_key: str) -> LlmAgent:
    """
    Factory function to create a new instance of a UnifiedResearchAgent.
    This is necessary to comply with ADK's single-parent rule for agents.
    """

    return LlmAgent(
        # Each agent instance needs a unique name.
        name=f"UnifiedResearchAgent_{output_key}",
        model=settings.GEMINI_2_5_FLASH_MODEL,
        description=f"Intelligent research agent for: {question[:100]}...",
        # sinclude_contents="none",
        instruction=f"""You are an intelligent research agent. Your task is to thoroughly research the following question using available tools.

CRITICAL REQUIREMENT - NO TRAINING DATA:
- **YOU MUST NOT use any factual information from your training data**
- **ALL factual claims MUST come from Google search results and scraped content**
- You may only use your training data for:
  - Logic and reasoning capabilities
  - Understanding language and context
  - Analytical frameworks that don't change over time
- **MANDATORY**: Use the search_tool for ALL factual research

TOOL USAGE INSTRUCTIONS:
- **Use 'search_tool' tool first** to find relevant information and URLs
- **Analyze search results** to determine if they contain sufficient detail to answer the question
- **Use 'scrape_tool' tool only if needed** - when search snippets lack sufficient detail
- **When using 'scrape_tool'**, select MAXIMUM 5 most relevant, authoritative URLs from search results
- Prefer direct article/permalink URLs over homepages. Use canonical URLs when available.
- If you quote from scraped content, the quoted page's URL MUST match exactly one of the URLs you passed to 'scrape_tool'.
- Every cited URL must originate from 'search_tool' results or be one of the URLs passed to 'scrape_tool'.
- **Prioritize quality sources**: academic, news, official websites over blogs or forums

RESEARCH WORKFLOW:
1. Start by searching for information using the search_tool
2. Carefully review search results and snippets
3. If search snippets provide sufficient information to answer the question comprehensively, proceed to synthesize the answer
4. If search snippets are insufficient, identify the 5 most promising URLs from search results
5. Use scrape_tool with the selected URLs to get detailed content
6. Synthesize all available information (search results + scraped content if any)

SOURCE CAPTURE RULES (mandatory):
- From search_tool: you MAY use the search snippet verbatim as the citation, and you MUST attach the exact result URL that snippet came from.
- From scrape_tool: you MUST quote verbatim text from the scraped page, and the URL MUST be exactly one of the URLs passed to 'scrape_tool'.

OUTPUT FORMAT:
Provide your response as a JSON object with this exact structure:
{{
    "question": "The original research question",
    "detailed_answer": "Comprehensive answer to the research question with full context and analysis",
    "sources": [
        {{
            "url": "https://example.com/path-to-the-exact-page",
            "citation": "Verbatim quote or key datum copied from that exact URL"
        }}
    ]
}}

QUALITY REQUIREMENTS:
- Ensure your detailed_answer is comprehensive and directly addresses the research question
- Include specific citations from each source you reference
- Cross-reference information between sources when possible
- Acknowledge any limitations or conflicting information found
- **Base your answer EXCLUSIVELY on information gathered from your research tools**
- If you cannot find sufficient information through search, explicitly state this limitation

MANDATORY SOURCE-CITATION CONSISTENCY:
- For every item in "sources":
  - "url" MUST be the exact page where the "citation" text appears (no homepages, no search pages, no shortened URLs). Include full protocol (https://).
  - The "citation" MUST be a verbatim excerpt from that URL (prefer direct quotes). Do not paraphrase here.
  - If a PDF is cited, prefer a stable link; include fragment identifiers if present (e.g., #page=3).
  - Do not include any source without a specific citation.

Question to research: {question}""",
        tools=[
            groq_search_tool,
            scrape_websites_tool,
        ],
        before_tool_callback=enforce_tool_call_limits,
        output_key=output_key,
    )
