# File: agent.py
"""Main agent entry point for ADK."""

from src.wal_fact_checker import fact_check_orchestrator

# ADK requires root_agent to be defined at module level
root_agent = fact_check_orchestrator
