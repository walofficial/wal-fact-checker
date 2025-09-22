# File: src/wal_fact_checker/agents/research/single_question_research_agent.py
"""Autonomous agent for researching a single gap question with intelligent website selection."""

from __future__ import annotations

from typing import Final

from google.adk import Agent
from google.adk.agents import SequentialAgent

from ...core.tools import adk_google_search, scrape_website_tool

MODEL: Final[str] = "gemini-2.0-flash"

# Search agent
search_agent = Agent(
    model=MODEL,
    name="SearchAgent",
    instruction="""
    Find relevant URLs using Google search for the assigned question.
    Focus on finding diverse, authoritative sources that might contain the answer.

    Question:
    {current_question}
    """,
    tools=[adk_google_search],
    output_key="search_results",
)

# Decision agent - NEW: Decides which websites need scraping
decision_agent = Agent(
    model=MODEL,
    name="DecisionAgent",
    instruction="""
    Analyze the search results and decide which websites need to be scraped for more detailed information.
    
    DECISION CRITERIA:
    1. Can the question be answered from search result snippets alone? If YES, mark "scraping_needed: false"
    2. If scraping needed, select MAXIMUM 7 most important websites based on:
       - Relevance to the specific question
       - Authority/credibility of the source
       - Likelihood that full content contains the needed information
       - Avoid duplicate/similar sources
    
    OUTPUT FORMAT:
    {
        "scraping_needed": true/false,
        "reasoning": "Why scraping is/isn't needed",
        "selected_urls": ["url1", "url2", ...], // max 7 URLs
        "preliminary_answer": "Answer if available from snippets only",
        "search_results": "Pass through original search results for analysis"
    }
    
    Question:
    {current_question}
    """,
    output_key="scraping_decision",
)

# Scraping agent - Updated to scrape only selected websites
scrape_agent = Agent(
    model=MODEL,
    name="ScrapeAgent",
    instruction="""
    Scrape content from the selected websites (max 7) identified by the DecisionAgent.
    Focus on extracting relevant information that answers the research question.
    If no URLs were selected for scraping, skip this step.
    
    OUTPUT FORMAT:
    {
        "scraped_content": "Content from scraped websites",
        "scraping_decision": "Pass through the scraping decision",
        "search_results": "Pass through original search results for analysis"
    }
    
    Question:
    {current_question}
    """,
    tools=[scrape_website_tool],
    output_key="research_data",
)

# Analysis agent - Updated to work with both search results and scraped content
analysis_agent = Agent(
    model=MODEL,
    name="AnalysisAgent",
    instruction="""
    You will receive research_data containing:
    - search_results: Original Google search results with snippets
    - scraping_decision: Decision logic and preliminary answers
    - scraped_content: Full content from selected websites (if any)
    
    Synthesize ALL available information to formulate a comprehensive answer:
    1. Use search result snippets for breadth and quick facts
    2. Use scraped website content for detailed information and verification
    3. Cross-reference information between sources
    4. Identify any contradictions or gaps
    
    Output a comprehensive, cited answer with:
    - Direct answer to the research question
    - Confidence score (0.0-1.0) based on source quality and consistency
    - Source citations with specific URLs (both search results and scraped sites)
    - Any limitations or gaps in the information
    
    Question:
    {current_question}
    """,
    output_key="research_answer",
)

single_question_research_agent = SequentialAgent(
    name="SingleQuestionResearchAgent",
    sub_agents=[search_agent, decision_agent, scrape_agent, analysis_agent],
    description="Intelligent research agent that selectively scrapes websites based on need (max 7 sites)",
)
