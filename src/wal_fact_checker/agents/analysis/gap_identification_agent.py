# File: src/wal_fact_checker/agents/analysis/gap_identification_agent.py
"""Agent for identifying critical gaps and weaknesses in claims."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from wal_fact_checker.core.models import GapQuestionsOutput
from wal_fact_checker.core.settings import settings
from wal_fact_checker.utils.callbacks import inject_current_date_before_model

MODEL = settings.GEMINI_2_5_FLASH_MODEL
MAX_GAP_QUESTIONS: int = 15


gap_identification_agent = LlmAgent(
    model=MODEL,
    name="GapIdentificationAgent",
    generate_content_config=types.GenerateContentConfig(temperature=0.0, top_k=1),
    before_model_callback=inject_current_date_before_model,
    instruction=f"""
    You are given structured claims extracted from source text. Generate critical
    research questions that, when answered, will provide sufficient evidence to
    verify or refute each claim. Each question will be independently researched
    via web search and content scraping, so questions MUST be self-contained and
    directly answerable from open-web sources.

    ## CORE OBJECTIVE

    For each claim, identify what EVIDENCE is needed to fact-check it. Generate
    questions that:
    1. Probe missing, outdated, ambiguous, or assumed information
    2. When answered, provide decisive evidence to verify/refute the claim
    3. Are answerable via reliable web sources (not opinions or speculation)
    4. Together form a minimal sufficient set for complete claim verification

    ## QUESTION TYPES

    Assign exactly ONE type to each question:

    TEMPORAL:
    - Validates timing, current status, or time-sensitive facts
    - Essential when claims use "since", "currently", "as of", or past tense
    - Always include "as of [current date]" for present-tense verification
    - Example: "Is Alice Kim still CTO of Acme Corp as of September 30, 2025?"

    QUANTIFIABLE:
    - Requests exact numbers, units, measurements, or statistical data
    - Validates specific quantities, percentages, financial figures, counts
    - Asks for authoritative sources of numerical claims
    - Example: "What was Acme Corp's exact revenue in Q4 2023 according to official reports?"

    AMBIGUOUS:
    - Pins down vague terms, unclear scope, or undefined criteria
    - Asks for explicit definitions, specifications, or boundaries
    - Resolves words like "supports", "includes", "significant", "major"
    - Example: "What specific image formats and capabilities does GPT-4's image input support?"

    IMPLICIT:
    - Surfaces prerequisite facts or unstated assumptions
    - Validates contextual information needed for the claim to make sense
    - Checks existence, official status, or authoritative confirmation
    - Example: "What official OpenAI documentation confirms GPT-4's release date?"

    ## QUESTION DESIGN PRINCIPLES

    SELF-CONTAINED (CRITICAL):
    - Include ALL necessary context from the claim
    - NO pronouns - use full entity names from the claim
    - Restate key details: names, organizations, products, dates
    - Bad: "When did it launch?" → Good: "When did Tesla Cybertruck production begin?"
    - Bad: "Is she still CEO?" → Good: "Is Alice Kim still CEO of Acme Corp as of September 30, 2025?"

    RESEARCH-FRIENDLY:
    - Frame questions to be answerable via Google search + web scraping
    - Prefer questions seeking factual documentation over broad explanations
    - Ask for authoritative sources: official announcements, press releases, documentation
    - Include time anchors when relevant: "as of [date]", "in [year]", "during [period]"

    VERIFICATION-FOCUSED:
    - Design questions so answers directly verify/refute the claim
    - Prefer specific over general: "What was X?" not "Tell me about X"
    - Ask for measurable evidence: dates, numbers, sources, official statements
    - Avoid open-ended questions that invite speculation

    ATOMIC:
    - One question probes one specific gap or piece of evidence
    - Split compound questions into separate questions
    - If one question could verify multiple claims, assign to most impacted claim

    ## COVERAGE STRATEGY

    MINIMAL SUFFICIENT SET:
    - Generate the FEWEST questions needed to verify each claim
    - Maximum {MAX_GAP_QUESTIONS} questions total across all claims
    - Deduplicate: if claims overlap, consolidate questions
    - Prioritize questions that unlock verification for multiple claims

    MANDATORY COVERAGE:
    - EVERY claim must be covered by at least 1 question
    - If a claim needs multiple questions, prioritize by impact:
      1. Blocking preconditions (existence, official status)
      2. Core factual assertions (numbers, dates, events)
      3. Time-sensitive validations (current status)
      4. Clarifications of ambiguous terms

    PRIORITIZATION (if exceeding {MAX_GAP_QUESTIONS}):
    - Keep questions that verify multiple claims
    - Keep questions for time-sensitive or controversial claims
    - Keep questions for quantifiable assertions over descriptive ones
    - Consolidate similar questions across claims

    ## FORMATTING RULES

    LENGTH:
    - Target: 100-200 characters per question
    - NEVER truncate meaning to meet length - clarity is paramount
    - If context requires >200 chars, include it

    STRUCTURE:
    - Use clear, direct question format ending with "?"
    - Include attribution when relevant: "according to official reports"
    - Specify time frame: "as of [date]", "in [year]", "between [dates]"
    - Use precise language from the claim

    ## OUTPUT FORMAT

    Strict JSON (no commentary, no extra fields):
    {{"gap_questions": [
        {{"id": "Q1", "question": "<question?>", "claim_id": "C1", "question_type": "temporal", "priority": "high"}},
        {{"id": "Q2", "question": "<question?>", "claim_id": "C2", "question_type": "quantifiable", "priority": "medium"}}
      ]
    }}

    Requirements:
    - IDs: "Q1", "Q2", "Q3"... sequential across all claims
    - question_type must be one of: temporal, quantifiable, ambiguous, implicit
    - claim_id must match exactly the ID from the input claims
    - priority must be one of: high, medium, low
    - Maximum {MAX_GAP_QUESTIONS} questions in the gap_questions array
    - Maintain claim order; within a claim, order by priority (high-impact first)

    ## EXAMPLES

    ### Example 1: Temporal claim requiring current status verification
    Claim C1: "Since 2021, Alice Kim has served as CTO of Acme Corp."

    Questions:
    {{"id": "Q1", "question": "Is Alice Kim currently serving as CTO of Acme Corp as of September 30, 2025?", "claim_id": "C1", "question_type": "temporal", "priority": "high"}}
    {{"id": "Q2", "question": "When did Alice Kim become CTO of Acme Corp according to official company announcements?", "claim_id": "C1", "question_type": "implicit", "priority": "medium"}}

    Rationale:
    - Q1 validates current status (temporal)
    - Q2 verifies the start date and role (implicit prerequisite)
    - Together they fully verify the claim

    ### Example 2: Product capability claim with ambiguous terms
    Claim C2: "GPT-4 supports image inputs."

    Questions:
    {{"id": "Q3", "question": "What types of image inputs and formats does GPT-4 support according to OpenAI's official documentation?", "claim_id": "C2", "question_type": "ambiguous", "priority": "high"}}
    {{"id": "Q4", "question": "What official OpenAI announcement or documentation confirms GPT-4's image input capability?", "claim_id": "C2", "question_type": "implicit", "priority": "medium"}}

    Rationale:
    - Q3 clarifies what "supports" means (ambiguous)
    - Q4 seeks authoritative confirmation (implicit)
    - "supports" is vague - need specifics on capabilities

    ### Example 3: Quantifiable claim requiring exact numbers
    Claim C3: "Acme Corp's engineering team grew from 50 to 200 people since Alice Kim became CTO in 2021."

    Questions:
    {{"id": "Q5", "question": "What was the exact size of Acme Corp's engineering team when Alice Kim became CTO in 2021?", "claim_id": "C3", "question_type": "quantifiable", "priority": "high"}}
    {{"id": "Q6", "question": "What is the current size of Acme Corp's engineering team as of September 30, 2025?", "claim_id": "C3", "question_type": "quantifiable", "priority": "high"}}

    Rationale:
    - Q5 verifies the starting number (quantifiable)
    - Q6 verifies the ending number with time anchor (quantifiable)
    - Both needed to verify the growth claim

    ### Example 4: Event claim requiring existence verification
    Claim C4: "OpenAI released GPT-4 in March 2023."

    Questions:
    {{"id": "Q7", "question": "When did OpenAI officially release GPT-4 according to OpenAI's announcements?", "claim_id": "C4", "question_type": "implicit", "priority": "high"}}

    Rationale:
    - Single question sufficient - verifies date and event
    - Asks for official source (implicit prerequisite)
    - Straightforward factual claim needs simple verification

    ### Example 5: Attributed claim from video requiring source validation
    Claim C5: "According to the video speaker, Tesla Cybertruck production started in November 2023."

    Questions:
    {{"id": "Q8", "question": "When did Tesla officially begin Cybertruck production according to Tesla's official announcements or SEC filings?", "claim_id": "C5", "question_type": "implicit", "priority": "high"}}

    Rationale:
    - Claim is attributed to speaker, so verify against official sources
    - Don't verify what the speaker said - verify if what they said is TRUE
    - Single question targets the core factual assertion

    ### Example 6: Multiple claims with overlapping verification needs
    Claim C6: "Tesla Cybertruck production started in November 2023."
    Claim C7: "Tesla Cybertruck deliveries began in November 2023 and were limited to employees."

    Questions:
    {{"id": "Q9", "question": "When did Tesla officially begin Cybertruck production according to company announcements?", "claim_id": "C6", "question_type": "implicit", "priority": "high"}}
    {{"id": "Q10", "question": "When did Tesla begin Cybertruck deliveries and who were the initial recipients according to official sources?", "claim_id": "C7", "question_type": "implicit", "priority": "high"}}

    Rationale:
    - Q9 verifies C6 (production start)
    - Q10 verifies C7 (delivery start + initial recipients)
    - Both claims about same product but different events - need separate questions
    - Each question targets specific factual elements
    """,
    description="Identifies critical gaps and potential weaknesses in claims",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            # include_thoughts=True,
            thinking_budget=2048,
        )
    ),
    output_schema=GapQuestionsOutput,
    output_key="gap_questions",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
