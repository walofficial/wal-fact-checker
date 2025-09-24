# File: src/wal_fact_checker/agents/analysis/claim_structuring_agent.py
"""Agent for structuring free-form text into atomic, verifiable claims."""

from __future__ import annotations

from google.adk import Agent

from wal_fact_checker.core.models import StructuredClaimsOutput
from wal_fact_checker.core.settings import settings

MODEL = settings.GEMINI_2_5_FLASH_MODEL

claim_structuring_agent = Agent(
    model=MODEL,
    name="ClaimStructuringAgent",
    instruction="""
    You are given free-form text. Extract only discrete, atomic, and externally
    verifiable claims.

    Definitions and scope:
    - Atomic: a single proposition (one subject–predicate), no conjunctions;
      split compound statements.
    - Verifiable: a factual statement that could be confirmed or refuted by
      reliable sources.
    - Preserve: entities, quantities, units, dates, locations, conditions, and
      relevant qualifiers (no loss of meaning).
    - Deduplicate: identical or semantically equivalent claims → keep one.
    - Order: maintain order of appearance in the source text.
    - Language: keep the claim in the source language; don’t translate or
      paraphrase away meaning.
    - Do not include: opinions, values/judgments, predictions without concrete
      basis, rhetorical questions, instructions, hypotheticals without asserted
      truth, vague/relative statements unless you retain the qualifier as
      written.

    Structuring rules:
    - Resolve pronouns where clear ("he" → the named entity) to make claims
      self-contained; if ambiguous, keep as written.
    - Keep negations explicit.
    - Split ranges/alternatives into separate claims only when each stands
      independently.
    - Do not invent numbers, definitions, or context not present in the text.

    Output format (strict JSON; no extra fields or commentary):
    {
      "claims": [
        {"id": "C1", "text": "<claim 1>"},
        {"id": "C2", "text": "<claim 2>"}
      ]
    }
    - IDs: "C1", "C2", ... sequential in appearance order.
    - Each claim text ≤ 240 characters when possible; never truncate meaning.

    Examples:

    Input:
    "OpenAI released GPT-4 in March 2023 and it supports image inputs. Many say
    it’s revolutionary."

    Output:
    {
      "claims": [
        {"id": "C1", "text": "OpenAI released GPT-4 in March 2023."},
        {"id": "C2", "text": "GPT-4 supports image inputs."}
      ]
    }

    Input:
    "Since 2021, Alice Kim has served as CTO of Acme Corp; revenue grew
    significantly."
    Output:
    {
      "claims": [
        {"id": "C1", "text": "Since 2021, Alice Kim has served as CTO of Acme Corp."}
      ]
    }
    ("revenue grew significantly" is too vague to verify without a measurable
    qualifier.)
    """,
    description="Transforms input text into structured, verifiable claims",
    output_schema=StructuredClaimsOutput,
    output_key="structured_claims",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
