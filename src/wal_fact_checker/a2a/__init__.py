"""A2A server integration for WAL Fact Checker.

Exposes `a2a_app` for ASGI servers to run.
"""

from .app import a2a_app

__all__ = ["a2a_app"]
