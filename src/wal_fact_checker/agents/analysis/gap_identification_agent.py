# File: src/wal_fact_checker/agents/analysis/gap_identification_agent.py
"""Agent for identifying critical gaps and weaknesses in claims."""

from __future__ import annotations

from google.adk import Agent

from wal_fact_checker.core.models import GapQuestionsOutput
from wal_fact_checker.core.settings import settings

MODEL = settings.GEMINI_2_5_FLASH_MODEL

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
    
    Return a structured JSON response with gap_questions containing id, question, claim_id, 
    question_type (temporal/quantifiable/ambiguous/implicit), and priority (1-3).
    """,
    description="Identifies critical gaps and potential weaknesses in claims",
    output_schema=GapQuestionsOutput,
    output_key="gap_questions",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
