# File: src/wal_fact_checker/agents/research/research_orchestrator_agent.py
"""Orchestrator agent that manages parallel research of gap questions."""

from __future__ import annotations

from google.adk import Agent

MODEL = "gemini-2.0-flash"

research_orchestrator_agent = Agent(
    model=MODEL,
    name="ResearchOrchestratorAgent",
    instruction="""
    Manage parallel research of gap questions by:
    1. Reading gap_questions from session state
    2. Creating SingleQuestionResearchAgent instances for each question
    3. Coordinating parallel execution via ParallelAgent
    4. Compiling results into comprehensive_answer_set
    """,
    description="Orchestrates parallel research of identified gaps",
    output_key="comprehensive_answer_set",
)
