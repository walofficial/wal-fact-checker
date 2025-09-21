# File: src/wal_fact_checker/agents/analysis/claim_structuring_agent.py
"""Agent for structuring free-form text into atomic, verifiable claims."""

from __future__ import annotations

from google.adk import Agent

MODEL = "gemini-2.0-flash"

claim_structuring_agent = Agent(
    model=MODEL,
    name="ClaimStructuringAgent",
    instruction="""
    Transform free-form input text into discrete, atomic, and verifiable claims.
    Each claim should be:
    - Specific and measurable
    - Independently verifiable
    - Clear and unambiguous
    
    Output structured claims with IDs and categories.
    """,
    description="Transforms input text into structured, verifiable claims",
    output_key="structured_claims",
)
