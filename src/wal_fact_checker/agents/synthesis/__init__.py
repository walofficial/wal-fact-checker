# File: src/wal_fact_checker/agents/synthesis/__init__.py
"""Synthesis stage agents for evidence compilation and adversarial critique."""

from .evidence_adjudicator_agent import evidence_adjudicator_agent
from .report_transformation_agent import (
    report_transformation_agent,
    transform_adjudicated_report,
)

__all__ = [
    "evidence_adjudicator_agent",
    "report_transformation_agent",
    "transform_adjudicated_report",
]
