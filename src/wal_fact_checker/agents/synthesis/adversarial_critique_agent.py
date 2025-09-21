# File: src/wal_fact_checker/agents/synthesis/adversarial_critique_agent.py
"""Agent for adversarial critique and refinement of fact-check reports."""

from __future__ import annotations

from google.adk import Agent

from ...core.tools import load_memory_tool

MODEL = "gemini-2.0-flash"

adversarial_critique_agent = Agent(
    model=MODEL,
    name="AdversarialCritiqueAgent",
    instruction="""
    Act as skeptical peer reviewer/red teamer. Challenge the draft_report by checking for:
    
    LOGICAL FALLACIES: "Does the evidence truly support the conclusion?"
    WEAK CITATIONS: "Is the source for this claim authoritative?"  
    CONTRADICTIONS: "Does evidence for Claim 1 contradict evidence for Claim 3?"
    MISSED NUANCE: "Is 'Verified' too strong? Should it be 'Partially True'?"
    
    Rewrite the draft to address criticisms, strengthen arguments, clarify nuance.
    Ensure all conclusions are robustly supported by gathered evidence.
    """,
    description="Adversarial critic that refines and strengthens fact-check reports",
    tools=[load_memory_tool],
)
