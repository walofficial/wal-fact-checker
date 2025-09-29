# File: src/wal_fact_checker/__init__.py
"""WAL Fact Checker: Proactive Gap Analysis & Adversarial Critique Workflow."""

from .core.logging_config import setup_logging
from .observability.langfuse_tracing import initialize_langfuse_tracing

# Initialize tracing as early as possible so all ADK spans are captured
initialize_langfuse_tracing()
setup_logging()


__version__ = "0.1.0"
__all__ = ["root_agent"]
