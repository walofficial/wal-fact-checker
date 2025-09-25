"""Langfuse initialization and Google ADK OpenTelemetry instrumentation."""

from __future__ import annotations

import logging
import os
from typing import Final

from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from wal_fact_checker.core.settings import settings

logger: Final = logging.getLogger(__name__)


def _set_env_if_missing(key: str, value: str) -> None:
    """Set an environment variable if it's not already set and value is non-empty."""
    if key in os.environ:
        return
    if value:
        os.environ[key] = value


def initialize_langfuse_tracing() -> None:
    """Initialize Langfuse client and instrument Google ADK with OpenTelemetry.

    This reads credentials from `app.core.settings.settings` and sets standard
    Langfuse environment variables if they are not already present. It then
    authenticates the client and instruments Google ADK so that agent/tool and
    model spans are exported to Langfuse.
    """
    try:
        _set_env_if_missing("LANGFUSE_HOST", settings.langfuse_host)
        _set_env_if_missing("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        _set_env_if_missing("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        _set_env_if_missing("LANGFUSE_ENV", settings.langfuse_tracing_environment)

        # Optional: ensure Gemini key is available for ADK examples/tools
        _set_env_if_missing("GOOGLE_API_KEY", settings.google_api_key)

        client = get_client()
        if client.auth_check():
            logger.info("Langfuse client authenticated; enabling ADK instrumentation")
        else:
            logger.warning("Langfuse authentication failed. Check LANGFUSE_* settings.")

        # Idempotent; safe to call multiple times
        GoogleADKInstrumentor().instrument()
    except Exception as exc:
        logger.exception("Failed to initialize Langfuse tracing: %s", exc)
