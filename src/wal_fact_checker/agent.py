# File: src/wal_fact_checker/agent.py
"""Main entry point for the WAL Fact Checker agent."""

from __future__ import annotations

from .agents.fact_check_orchestrator import root_agent

# Export root agent for ADK
__all__ = ["root_agent"]
