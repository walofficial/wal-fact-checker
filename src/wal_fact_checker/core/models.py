# File: src/wal_fact_checker/core/models.py
"""Data models for the WAL Fact Checker system."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class AtomicClaim:
    """Represents a single, verifiable claim extracted from input text."""

    id: str
    text: str
    category: str | None = None
    confidence: float = 0.0


@dataclass
class GapQuestion:
    """Represents a critical question to investigate potential weaknesses."""

    id: str
    question: str
    claim_id: str
    question_type: str  # temporal, quantifiable, ambiguous, implicit
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class ResearchResult:
    """Result from investigating a single gap question."""

    question_id: str
    answer: str
    source_urls: list[str]
    confidence_score: float
    evidence_quality: str  # strong, moderate, weak
    timestamp: str


@dataclass
class FactCheckVerdict:
    """Final verdict for a claim after fact-checking."""

    claim_id: str
    verdict: str  # verified, false, partially_true, insufficient_evidence
    confidence: float
    supporting_evidence: list[str]
    refuting_evidence: list[str]
    nuance: str | None = None


@dataclass
class FactCheckReport:
    """Complete fact-checking report."""

    original_text: str
    claims: list[AtomicClaim]
    verdicts: list[FactCheckVerdict]
    methodology: str
    timestamp: str
    overall_assessment: str


# Pydantic schemas for agent outputs
class AtomicClaimOutput(BaseModel):
    """Pydantic schema for structured claim output."""

    id: str = Field(description="Unique identifier for the claim")
    text: str = Field(description="The atomic, verifiable claim text")
    confidence: float = Field(
        description="The model's confidence in the accuracy of the extracted claim, from 0.0 to 1.0"
    )


class StructuredClaimsOutput(BaseModel):
    """Output schema for claim structuring agent."""

    claims: list[AtomicClaimOutput] = Field(
        description="List of structured atomic claims"
    )


class GapQuestionOutput(BaseModel):
    """Pydantic schema for gap question output."""

    id: str = Field(description="Unique identifier for the gap question")
    question: str = Field(description="The critical question to investigate")
    claim_id: str = Field(description="ID of the claim this question relates to")
    question_type: str = Field(
        description="Type: temporal, quantifiable, ambiguous, or implicit"
    )
    priority: str = Field(
        description="Priority of the question: high, medium, or low"
    )


class GapQuestionsOutput(BaseModel):
    """Output schema for gap identification agent."""

    gap_questions: list[GapQuestionOutput] = Field(
        description="List of critical gap questions"
    )


class ReferenceOutput(BaseModel):
    """Pydantic schema for evidence reference."""

    is_supportive: bool = Field(
        description="Whether this reference supports or refutes the claim"
    )
    citation: str = Field(
        description="Specific quote or key information from the source"
    )
    url: str = Field(description="URL of the source")


class SectionItemOutput(BaseModel):
    """Item within a results section for a claim and its argumentation."""

    claim_id: str = Field(description="ID of the referenced claim")
    claim_text: str = Field(description="Text of the referenced claim")
    argumentative_explanation: str = Field(
        description=(
            "Short, high-impact explanation (1-2 sentences, 18-40 words) that "
            "states the finding first, then cites the strongest evidence type "
            "and recency. No hedging, no URLs, no quotes; use concrete specifics "
            "(numbers/dates/entities) and simple syntax."
        )
    )


class FactCheckVerdictOutput(BaseModel):
    """Pydantic schema for fact check verdict."""

    claim_id: str = Field(description="ID of the claim being evaluated")
    claim_text: str = Field(description="The original claim text being evaluated")
    verdict: str = Field(
        description="Verdict: verified, false, partially_true, or insufficient_evidence"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verdict")
    argumentative_explanation: str = Field(
        description="Detailed argumentative explanation of the verdict with reasoning"
    )
    references: list[ReferenceOutput] = Field(
        description="List of references with supportive/refuting indicators"
    )


class EvidenceAdjudicatorOutput(BaseModel):
    """Compact output schema for the main fact-checker agent."""

    verdict: str = Field(
        description=(
            "Overall verdict used for UI gating (show/hide). Choose one: mostly_true, "
            "mostly_false, mixed, unverified. 'Unverified' means no material confirmations "
            "or refutations. 'Material' = important to the core thesis (affects outcomes, "
            "timelines, quantities, official roles/status, key identities, money, safety, "
            "legality). Trivial details (e.g., color, incidental attributes) are non-material. "
            "If any material true/false finding exists, do not use 'unverified'."
        )
    )
    factuality: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall factuality score for the verdict (0.0-1.0)",
    )
    headline_summary_md: str = Field(
        description=(
            "Reader-facing markdown summary for the main page. Up to three lines, "
            "each optional, in this order: 'True — <one sentence>'; 'False — <one "
            "sentence>'; 'Unverified — <one sentence>'. Omit any line with no "
            "notable content. Keep it concise and non-speculative."
        )
    )
    what_was_true: list[SectionItemOutput] = Field(
        description="Claims determined true with brief justifications"
    )
    what_was_false: list[SectionItemOutput] = Field(
        description="Claims determined false with brief justifications"
    )
    what_could_not_be_verified: list[SectionItemOutput] = Field(
        description=(
            "Claims that are unverifiable or lack sufficient evidence, with "
            "brief justifications"
        )
    )
    references: list[ReferenceOutput] = Field(
        description=(
            "Global list of all references cited across the report. Do not invent URLs."
        )
    )


class CritiqueOutput(BaseModel):
    """Output schema for adversarial critique agent."""

    identified_issues: list[str] = Field(
        description="List of identified issues in the draft"
    )
    revised_verdicts: list[FactCheckVerdictOutput] = Field(
        description="Revised verdicts after critique"
    )
    strengthened_methodology: str = Field(
        description="Improved methodology description"
    )
    final_assessment: str = Field(description="Final overall assessment")


class ScrapeInput(BaseModel):
    """Pydantic schema for scrape input."""

    urls: list[str] = Field(description="List of URLs to scrape")


class ScrapeOutput(BaseModel):
    """Pydantic schema for scrape output."""

    combined_content: str = Field(description="Combined content from all scraped URLs")
    status: str = Field(description="Status of the scrape operation")


class TransformationReferenceOutput(BaseModel):
    """Pydantic schema for evidence reference."""

    is_supportive: bool = Field(
        description="Whether this reference supports or refutes the claim"
    )
    key_quote: str = Field(
        description="Specific quote or key information from the source"
    )
    url: str = Field(description="URL of the source")


class TransformationOutput(BaseModel):
    """Output schema for report transformation agent."""

    verdict: str = Field(
        description=(
            "Overall verdict for UI gating (mostly_true, mostly_false, mixed, unverified)"
        )
    )
    factuality: float = Field(description="Factuality score of the transformed report")
    reason: str = Field(description="Detailed explanation of the fact checking")
    reason_summary: str = Field(description="Summary explanation of the fact checking")
    score_justification: str = Field(
        description="Justification of the fact checking score"
    )
    references: list[TransformationReferenceOutput] = Field(
        description="List of references used in the fact checking"
    )
