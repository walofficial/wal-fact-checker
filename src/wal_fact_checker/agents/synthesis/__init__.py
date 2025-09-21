# File: src/wal_fact_checker/agents/synthesis/__init__.py
"""Synthesis stage agents for evidence compilation and adversarial critique."""

from .adversarial_critique_agent import adversarial_critique_agent
from .evidence_synthesizer_agent import evidence_synthesizer_agent

__all__ = ["evidence_synthesizer_agent", "adversarial_critique_agent"]
