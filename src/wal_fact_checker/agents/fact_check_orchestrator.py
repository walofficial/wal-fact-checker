# File: src/wal_fact_checker/agents/fact_check_orchestrator.py
"""Main orchestrator agent for the WAL Fact Checker workflow."""

from __future__ import annotations

from google.adk.agents import SequentialAgent

from .analysis import claim_structuring_agent, gap_identification_agent
from .research import research_orchestrator_agent
from .synthesis import adversarial_critique_agent, evidence_synthesizer_agent

# Stage 1: Analysis & Strategy (Sequential)
analysis_stage = SequentialAgent(
    name="AnalysisStage",
    sub_agents=[
        claim_structuring_agent,  # Input -> structured_claims
        gap_identification_agent,  # structured_claims -> gap_questions
    ],
    description="Analyzes input and creates research strategy",
)

# Stage 2: Parallelized Research (Managed by orchestrator)
research_stage = (
    research_orchestrator_agent  # gap_questions -> comprehensive_answer_set
)

# Stage 3: Synthesis & Verification (Sequential)
synthesis_stage = SequentialAgent(
    name="SynthesisStage",
    sub_agents=[
        evidence_synthesizer_agent,  # comprehensive_answer_set -> draft_report
        adversarial_critique_agent,  # draft_report -> final_report
    ],
    description="Synthesizes evidence and applies adversarial critique",
)

# Main orchestrator: Analysis -> Research -> Synthesis
fact_check_orchestrator = SequentialAgent(
    name="FactCheckOrchestrator",
    sub_agents=[analysis_stage, research_stage, synthesis_stage],
    description="WAL Proactive Gap Analysis & Adversarial Critique Fact Checker",
)
