# File: src/wal_fact_checker/agents/research/single_question_research_agent.py
"""Unified research agent that combines decision-making, scraping, and analysis."""

from __future__ import annotations

from google.adk.agents import LlmAgent

from wal_fact_checker.core.settings import settings
from wal_fact_checker.core.tools import groq_search_tool, scrape_websites_tool


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
- **Prioritize quality sources**: academic, news, official websites over blogs or forums

RESEARCH WORKFLOW:
1. Start by searching for information using the search_tool
2. Carefully review search results and snippets
3. If search snippets provide sufficient information to answer the question comprehensively, proceed to synthesize the answer
4. If search snippets are insufficient, identify the 5 most promising URLs from search results
5. Use scrape_tool with the selected URLs to get detailed content
6. Synthesize all available information (search results + scraped content if any)

OUTPUT FORMAT:
Provide your response as a JSON object with this exact structure:
{{
    "question": "The original research question",
    "detailed_answer": "Comprehensive answer to the research question with full context and analysis",
    "sources": [
        {{
            "url": "https://example.com",
            "citation": "Specific quote or key information from this source"
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

Question to research: {question}""",
        tools=[
            groq_search_tool,
            scrape_websites_tool,
        ],
        output_key=output_key,
    )
