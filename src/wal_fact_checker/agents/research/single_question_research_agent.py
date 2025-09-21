# File: src/wal_fact_checker/agents/research/single_question_research_agent.py
"""Autonomous agent for researching a single gap question."""

from __future__ import annotations

from google.adk import Agent
from google.adk.agents import SequentialAgent

from ...core.tools import adk_google_search, scrape_website_tool

MODEL = "gemini-2.0-flash"

# Search agent
search_agent = Agent(
    model=MODEL,
    name="SearchAgent",
    instruction="Find relevant URLs using Google search for the assigned question",
    tools=[adk_google_search],
)

# Scraping agent
scrape_agent = Agent(
    model=MODEL,
    name="ScrapeAgent",
    instruction="Extract content from the best URLs found by SearchAgent",
    tools=[scrape_website_tool],
)

# Analysis agent
analysis_agent = Agent(
    model=MODEL,
    name="AnalysisAgent",
    instruction="Analyze scraped content and formulate direct, cited answer with confidence score",
)

single_question_research_agent = SequentialAgent(
    name="SingleQuestionResearchAgent",
    sub_agents=[search_agent, scrape_agent, analysis_agent],
    description="Autonomous specialist for investigating one gap question thoroughly",
)
