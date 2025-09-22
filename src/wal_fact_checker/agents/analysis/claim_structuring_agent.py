# File: src/wal_fact_checker/agents/analysis/claim_structuring_agent.py
"""Agent for structuring free-form text into atomic, verifiable claims."""

from __future__ import annotations

from google.adk import Agent

from wal_fact_checker.core.models import StructuredClaimsOutput

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
    
    Return a structured JSON response with claims containing id, text, category (optional), 
    and confidence score (0.0-1.0).
    """,
    description="Transforms input text into structured, verifiable claims",
    output_schema=StructuredClaimsOutput,
    output_key="structured_claims",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
