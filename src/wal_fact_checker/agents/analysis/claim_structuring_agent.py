# File: src/wal_fact_checker/agents/analysis/claim_structuring_agent.py
"""Agent for structuring free-form text into atomic, verifiable claims."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from wal_fact_checker.core.models import StructuredClaimsOutput
from wal_fact_checker.core.settings import settings

MODEL = settings.GEMINI_2_5_FLASH_MODEL

claim_structuring_agent = LlmAgent(
    model=MODEL,
    name="ClaimStructuringAgent",
    generate_content_config=types.GenerateContentConfig(temperature=0.0, top_k=1),
    instruction="""
    You are given free-form text (articles, image descriptions, video summaries,
    etc.). Extract ONLY discrete, atomic, and externally verifiable claims. Each
    claim will later be independently researched, so they MUST be self-contained
    and fully understandable without referring back to the source.

    ## CORE DEFINITIONS

    ATOMIC:
    - One factual proposition (single subject-predicate relationship)
    - No compound statements with "and" or "but" — split them
    - No causal chains ("X because Y") unless the causation itself is the verifiable fact

    VERIFIABLE:
    - Must be confirmable or refutable using reliable external sources
    - Must contain sufficient context to be fact-checked independently
    - Exclude: opinions, value judgments, predictions without concrete basis,
      rhetorical questions, vague comparative statements

    SELF-CONTAINED (CRITICAL):
    - Each claim must be fully understandable WITHOUT the source text
    - ALWAYS resolve pronouns and implicit references to named entities
    - Include ALL necessary context: who, what, when, where, under what conditions
    - Bad: "He served as CEO" → Good: "John Smith served as CEO of Acme Corp"
    - Bad: "It supports 50 languages" → Good: "ChatGPT supports 50 languages"
    - Bad: "The company raised $100M" → Good: "Acme Corp raised $100M in Series B funding"

    ## EXTRACTION RULES

    PRESERVE ESSENTIAL CONTEXT:
    - Entities: full names, organizations, products, locations
    - Quantities: numbers + units + what they measure
    - Time anchors: dates, years, timeframes, "as of [date]"
    - Conditions: qualifying phrases that affect verifiability
    - Scope: "in the US", "for the first time", "according to [source]"

    RESOLVE ALL REFERENCES:
    - Pronouns → named entities ("she" → "Dr. Jane Smith")
    - Demonstratives → specific referents ("this feature" → "GPT-4's image input feature")
    - Implicit subjects → explicit subjects (in image descriptions: "wearing a red hat" → "The person in the image is wearing a red hat")
    - Relative terms → absolute terms when possible ("recently" → "in March 2023" if date is given)

    HANDLE AMBIGUITY:
    - Preserve qualifiers that make vague statements verifiable:
      ✓ "Acme Corp claims revenue grew 50% in 2023"
      ✗ "Revenue grew significantly" (too vague)
    - Keep attributed claims with attribution:
      ✓ "According to the CEO, the product launched in May 2023"
      ✗ "The product might have launched in May" (speculation)
    - For image/video descriptions, preserve visual context:
      ✓ "The image shows a red Tesla Model 3 at a charging station"
      ✗ "A car is charging" (loses critical details)

    DEDUPLICATION:
    - Remove semantically identical claims (keep first occurrence)
    - Keep claims that differ in time, quantity, or scope even if similar
    - Example: "X was CEO in 2020" ≠ "X was CEO in 2023" (both kept)

    EXCLUSIONS (DO NOT EXTRACT):
    - Pure opinions without factual grounding ("I think X is great")
    - Value judgments ("X is the best", "Y is revolutionary")
    - Future predictions without attributed source
    - Rhetorical questions or instructions
    - Statements lacking specificity: "many people", "often", "very high" without numbers or measurable criteria

    LENGTH GUIDANCE:
    - Target: 100-240 characters per claim
    - NEVER truncate meaning to meet character limit
    - If context requires >240 chars, include it — self-containment is paramount

    ## OUTPUT FORMAT

    Strict JSON (no commentary, no extra fields):
    {
      "claims": [
        {"id": "C1", "text": "<claim 1>"},
        {"id": "C2", "text": "<claim 2>"}
      ]
    }

    - IDs: "C1", "C2", "C3"... sequential in order of appearance
    - Maintain source text ordering
    - Keep claims in the source language (no translation)

    ## EXAMPLES

    ### Example 1: Article text
    Input:
    "OpenAI released GPT-4 in March 2023 and it supports image inputs. Many say
    it's revolutionary, but the company hasn't disclosed training details."

    Output:
    {
      "claims": [
        {"id": "C1", "text": "OpenAI released GPT-4 in March 2023."},
        {"id": "C2", "text": "GPT-4 supports image inputs."},
        {"id": "C3", "text": "OpenAI has not disclosed GPT-4's training details."}
      ]
    }

    Rationale:
    - "it" → "GPT-4" (resolved pronoun)
    - "Many say it's revolutionary" → excluded (opinion without attribution)
    - "the company" → "OpenAI" (resolved reference)

    ### Example 2: Company description
    Input:
    "Since 2021, Alice Kim has served as CTO of Acme Corp. Under her leadership,
    the engineering team grew from 50 to 200 people. Revenue grew significantly."

    Output:
    {
      "claims": [
        {"id": "C1", "text": "Since 2021, Alice Kim has served as CTO of Acme Corp."},
        {"id": "C2", "text": "Acme Corp's engineering team grew from 50 to 200 people since Alice Kim became CTO in 2021."}
      ]
    }

    Rationale:
    - C1: Kept as-is (self-contained with time anchor)
    - C2: "her leadership" → "Alice Kim became CTO" (resolved + added context)
    - "Revenue grew significantly" → excluded (no measurable qualifier)

    ### Example 3: Image description
    Input:
    "The image shows a protest in New York. Signs visible include 'Climate Action
    Now' and people are wearing green. Estimates suggest 10,000 attendees."

    Output:
    {
      "claims": [
        {"id": "C1", "text": "The image shows a protest in New York with signs reading 'Climate Action Now'."},
        {"id": "C2", "text": "The protest shown in the image had an estimated 10,000 attendees."}
      ]
    }

    Rationale:
    - Made explicit: "people wearing green" → excluded (descriptive detail, not verifiable claim)
    - "Estimates suggest" → "estimated" (kept qualifier showing uncertainty)
    - "The image shows" → kept (grounds visual claims to source medium)

    ### Example 4: Video summary with temporal info
    Input:
    "In the video, the speaker claims that Cybertruck production started in
    November 2023. He mentions deliveries began the same month but were limited to
    employees. Tesla aims for 250K units annually."

    Output:
    {
      "claims": [
        {"id": "C1", "text": "According to the video speaker, Tesla Cybertruck production started in November 2023."},
        {"id": "C2", "text": "According to the video speaker, Tesla Cybertruck deliveries began in November 2023 and were limited to employees."},
        {"id": "C3", "text": "According to the video speaker, Tesla aims for 250,000 Cybertruck units produced annually."}
      ]
    }

    Rationale:
    - "the speaker claims" → "According to the video speaker" (attribution preserved)
    - "He" → removed, kept "the video speaker" (pronoun resolution)
    - "Cybertruck" → "Tesla Cybertruck" (full product name)
    - "the same month" → "November 2023" (resolved temporal reference)
    - "250K" → "250,000" (expanded abbreviation for clarity)

    ### Example 5: Complex statement with attribution
    Input:
    "The CEO announced Q4 2023 revenue of $500M, up 40% YoY. This makes them the
    fastest-growing company in the sector."

    Output:
    {
      "claims": [
        {"id": "C1", "text": "The company's CEO announced Q4 2023 revenue of $500 million."},
        {"id": "C2", "text": "The company's CEO announced Q4 2023 revenue was up 40% year-over-year."}
      ]
    }

    Rationale:
    - Context needed: which CEO, which company? If unclear from full source, use "The company's CEO" with available context
    - "YoY" → "year-over-year" (expanded abbreviation)
    - Split compound statement into two claims
    - "fastest-growing in sector" → excluded (superlative without clear measurement criteria)
    """,
    description="Transforms input text into structured, verifiable claims",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            # include_thoughts=True,
            thinking_budget=1024,
        )
    ),
    output_schema=StructuredClaimsOutput,
    output_key="structured_claims",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
