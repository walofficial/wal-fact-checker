# File: src/wal_fact_checker/agents/analysis/gap_identification_agent.py
"""Agent for identifying critical gaps and weaknesses in claims."""

from __future__ import annotations

from google.adk import Agent

MODEL = "gemini-2.0-flash"

gap_identification_agent = Agent(
    model=MODEL,
    name="GapIdentificationAgent",
    instruction="""
    Analyze structured claims and generate critical gap questions focusing on:
    
    TEMPORAL DATA: "When was this published?", "Is this person still in this role?"
    QUANTIFIABLE DATA: "What is the exact number?", "What is the source?"
    AMBIGUOUS TERMS: "What does 'significant growth' mean precisely?"
    IMPLICIT ASSUMPTIONS: "Does this assume X is true? We need to verify X first."
    
    Be highly skeptical. Distrust parametric knowledge. Focus on what could be wrong, outdated, or missing.
    """,
    description="Identifies critical gaps and potential weaknesses in claims",
    output_key="gap_questions",
)
