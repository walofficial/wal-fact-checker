# File: src/wal_fact_checker/core/__init__.py
"""Core module initialization with logging setup."""

from .logging_config import setup_logging

# Initialize logging when the core module is imported
setup_logging()
