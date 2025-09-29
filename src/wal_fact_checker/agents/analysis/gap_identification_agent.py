# File: src/wal_fact_checker/agents/analysis/gap_identification_agent.py
"""Agent for identifying critical gaps and weaknesses in claims."""

from __future__ import annotations

from google.adk import Agent

from wal_fact_checker.core.models import GapQuestionsOutput
from wal_fact_checker.core.settings import settings

MODEL = settings.GEMINI_2_5_FLASH_MODEL
MAX_GAP_QUESTIONS: int = 15

gap_identification_agent = Agent(
    model=MODEL,
    name="GapIdentificationAgent",
    instruction=f"""
    From the structured claims, generate critical, falsifiable research questions
    that probe what is missing, outdated, ambiguous, or assumed. Questions must
    be directly answerable via open-web sources (search + page content).

    Four question types (choose one per question):
    - TEMPORAL: validate timing or current status (e.g., "as of today").
    - QUANTIFIABLE: request exact numbers, units, sources, or measurement methods.
    - AMBIGUOUS: pin down vague terms by asking for explicit definitions/criteria.
    - IMPLICIT: surface prerequisite facts that must be true for the claim to hold.

    Requirements:
    - Atomicity: one question probes one precise gap. If it benefits multiple
      claims, assign claim_id to the most impacted claim while ensuring all
      other claims remain covered by the overall set.
    - Traceability: include the exact claim_id; restate entities to avoid
      pronouns.
    - Verifiability: prefer yes/no verification, specific values, definitions, or
      authoritative sources (who/where/when) over open-ended prompts.
    - Time-sensitivity: add a time anchor when relevant (e.g., "as of <today>").
    - No speculation: avoid opinions or hypotheticals without asserted truth.
    - Scope and cap: produce a minimal sufficient set (≤ {
        MAX_GAP_QUESTIONS
    } questions total)
      whose answers together make every claim decidable (true/false). Deduplicate
      across claims and consolidate overlapping needs.
    - Coverage: ensure every claim is covered by ≥ 1 question. If more than {
        MAX_GAP_QUESTIONS
    }
      would be required, prioritize questions that (a) unlock verification for
      multiple claims, (b) resolve blocking preconditions, or (c) target highest
      uncertainty or time-sensitive assertions.
    - Stopping condition: stop when every claim is covered by decisive questions,
      or when you reach {MAX_GAP_QUESTIONS} questions, whichever comes first.
    - Ordering: maintain original claim order; within a claim, list higher-utility
      questions first.
    - Brevity: ≤ 200 characters per question when possible; do not truncate meaning.

    Output format (strict JSON; no extra fields or commentary):
    {{"gap_questions": [
        {{"id": "Q1", "question": "<question?>", "claim_id": "C1", "question_type": "temporal"}}
      ]
    }}
    - IDs: "Q1", "Q2", ... sequential in generation order across all claims.
    - question_type ∈ {{temporal, quantifiable, ambiguous, implicit}}.
    - Quantity: do not exceed {MAX_GAP_QUESTIONS} items in gap_questions.

    Examples:
    Claim C1: "Since 2021, Alice Kim has served as CTO of Acme Corp."
    → {{"id": "Q1", "question": "Is Alice Kim still CTO of Acme Corp as of today?", "claim_id": "C1", "question_type": "temporal"}}

    Claim C2: "GPT-4 supports image inputs."
    → {{"id": "Q2", "question": "What official documentation confirms that GPT-4 supports image inputs?", "claim_id": "C2", "question_type": "implicit"}}
    → {{"id": "Q3", "question": "Which GPT-4 modalities are supported and where is this stated?", "claim_id": "C2", "question_type": "ambiguous"}}
    """,
    description="Identifies critical gaps and potential weaknesses in claims",
    output_schema=GapQuestionsOutput,
    output_key="gap_questions",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
