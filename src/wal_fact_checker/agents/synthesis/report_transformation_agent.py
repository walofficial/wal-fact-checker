# File: src/wal_fact_checker/agents/synthesis/report_transformation_agent.py
"""Agent for transforming adjudicated reports into custom formats."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from ...core.models import EvidenceAdjudicatorOutput, TransformationOutput

# Configure logging for this module
logger = logging.getLogger(__name__)


def transform_adjudicated_report(
    adjudicated_report: EvidenceAdjudicatorOutput,
) -> TransformationOutput:
    """
    Transform an adjudicated report into the final output format.

    Args:
        adjudicated_report: The adjudicated evidence report to transform

    Returns:
        TransformationOutput: The transformed report with markdown reason
    """
    # Generate markdown string for reason field
    reason_markdown = _generate_reason_markdown(adjudicated_report)

    logger.info(f"Reason markdown: {reason_markdown}")

    # Map easily mappable fields
    return TransformationOutput(
        factuality=adjudicated_report.factuality,  # Use confidence as factuality score
        reason=reason_markdown,
        reason_summary=adjudicated_report.headline_summary_md,
        score_justification=f"Overall verdict: {adjudicated_report.verdict} with confidence {adjudicated_report.factuality:.2f}",
        references=adjudicated_report.references,
    )


def _generate_reason_markdown(adjudicated_report: EvidenceAdjudicatorOutput) -> str:
    """
    Generate markdown string with True, False, and Unverified sections.

    Args:
        adjudicated_report: The adjudicated evidence report

    Returns:
        str: Markdown formatted string with sections and bullet points
    """
    markdown_sections = []

    if adjudicated_report.what_was_true:
        markdown_sections.append("## True")
        for item in adjudicated_report.what_was_true:
            markdown_sections.append(f"- {item.claim_text}")
            markdown_sections.append(f"  {item.argumentative_explanation}")
        markdown_sections.append("")

    if adjudicated_report.what_was_false:
        markdown_sections.append("## False")
        for item in adjudicated_report.what_was_false:
            markdown_sections.append(f"- {item.claim_text}")
            markdown_sections.append(f"  {item.argumentative_explanation}")
        markdown_sections.append("")

    if adjudicated_report.what_could_not_be_verified:
        markdown_sections.append("## Unverified")
        for item in adjudicated_report.what_could_not_be_verified:
            markdown_sections.append(f"- {item.claim_text}")
            markdown_sections.append(f"  {item.argumentative_explanation}")
        markdown_sections.append("")

    return "\n".join(markdown_sections).strip()


class ReportTransformationAgent(BaseAgent):
    """Custom agent that transforms adjudicated reports using pure Python logic."""

    def __init__(self) -> None:
        super().__init__(name="ReportTransformationAgent")
        logger.debug(f"Initialized {self.name}")

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Execute the report transformation logic.

        Retrieves the adjudicated_report from session state, applies the
        transformation, and stores the result back to session state.
        """
        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Starting report transformation"
        )

        # Retrieve the adjudicated report from session state
        adjudicated_report_dict = ctx.session.state.get("adjudicated_report")

        if adjudicated_report_dict is None:
            logger.error(
                f"[{ctx.invocation_id}] {self.name}: No adjudicated_report found in session state"
            )
            return

        # Convert to Pydantic model
        adjudicated_report = EvidenceAdjudicatorOutput(**adjudicated_report_dict)

        # Apply transformation logic
        transformation_result = transform_adjudicated_report(adjudicated_report)
        transformation_result_dict = transformation_result.model_dump()

        # Store result in session state
        ctx.session.state["transformation_result"] = transformation_result_dict

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(
                state_delta={"transformation_result": transformation_result_dict},
            ),
        )

        # Build final response content as JSON and yield as final event
        content = types.Content(
            parts=[
                types.Part(
                    text=transformation_result.model_dump_json(),
                    # function_response=types.FunctionResponse(
                    #     response=transformation_result.model_dump()
                    # ),
                )
            ]
        )

        final_event = Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=content,
            actions=EventActions(escalate=True),
        )

        yield final_event

        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Transformation completed successfully"
        )


# Create the agent instance
report_transformation_agent = ReportTransformationAgent()
