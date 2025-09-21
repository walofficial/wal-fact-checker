# File: src/wal_fact_checker/core/models.py
"""Data models for the WAL Fact Checker system."""

from __future__ import annotations

from dataclasses import dataclass


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
