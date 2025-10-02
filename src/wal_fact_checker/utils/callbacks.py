"""Shared callbacks for agent configuration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)


def inject_current_date_before_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """
    Prepend today's UTC date to the system instruction before model call.

    Args:
        callback_context: Execution context for the callback.
        llm_request: Mutable LLM request that will be sent to the model.

    Returns:
        None to proceed with the (possibly modified) request.
    """
    today_utc = datetime.now(timezone.utc).date().isoformat()
    header_prefix = f"Current date: {today_utc} (UTC).\n\n"

    system_instruction = llm_request.config.system_instruction or types.Content(
        role="system", parts=[]
    )

    if not isinstance(system_instruction, types.Content):
        system_instruction = types.Content(
            role="system", parts=[types.Part(text=str(system_instruction))]
        )

    if not system_instruction.parts:
        system_instruction.parts.append(types.Part(text=""))

    first_text = system_instruction.parts[0].text or ""
    system_instruction.parts[0].text = header_prefix + first_text
    llm_request.config.system_instruction = system_instruction

    logger.info(
        "Injecting current date before model",
        extra={
            "json_fields": {
                "today_utc": today_utc,
                "agent": callback_context.agent_name,
            },
        },
    )
    return None
