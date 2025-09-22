# File: src/wal_fact_checker/agents/synthesis/evidence_synthesizer_agent.py
"""Agent for synthesizing evidence and creating initial fact-check report."""

from __future__ import annotations

from google.adk import Agent

from ...core.models import DraftReportOutput

MODEL = "gemini-2.0-flash"

evidence_synthesizer_agent = Agent(
    model=MODEL,
    name="EvidenceSynthesizerAgent",
    instruction="""
    Act as lead investigator to:
    1. Assemble all evidence from comprehensive_answer_set
    2. Build temporary RAG store using save_to_memory_tool
    3. For each structured_claim, query RAG for supporting/refuting evidence
    4. Write initial draft fact-checking report with verdicts and citations
    
    Return structured JSON with original_text, verdicts (containing claim_id, verdict, 
    confidence, supporting_evidence, refuting_evidence, nuance), methodology, and overall_assessment.
    
    Verdicts: verified, false, partially_true, insufficient_evidencec
    """,
    description="Synthesizes evidence into initial fact-checking report",
    # tools=[save_to_memory_tool, load_memory_tool],
    output_schema=DraftReportOutput,
    output_key="draft_report",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
