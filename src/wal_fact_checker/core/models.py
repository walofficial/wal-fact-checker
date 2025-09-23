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
    category: str | None = Field(default=None, description="Category of the claim")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score"
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
    priority: int = Field(
        default=1, ge=1, le=3, description="Priority level (1=high, 2=medium, 3=low)"
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


class DraftReportOutput(BaseModel):
    """Output schema for evidence synthesizer agent."""

    original_text: str = Field(description="Original text being fact-checked")
    verdicts: list[FactCheckVerdictOutput] = Field(description="List of claim verdicts")
    methodology: str = Field(description="Methodology used for fact-checking")
    overall_assessment: str = Field(description="Overall assessment of the text")


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
