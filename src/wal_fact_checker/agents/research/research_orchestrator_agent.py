# File: src/wal_fact_checker/agents/research/research_orchestrator_agent.py
"""Orchestrator agent that manages parallel research of gap questions."""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Final

from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from wal_fact_checker.core.models import GapQuestionsOutput
from wal_fact_checker.core.settings import settings

from .single_question_research_agent import (
    create_single_question_research_agent,
)

# Configure logging for this module
logger = logging.getLogger(__name__)

MODEL: Final[str] = settings.GEMINI_2_5_FLASH_MODEL


class ResearchOrchestratorAgent(BaseAgent):
    """Run SingleQuestionResearchAgent in parallel for each gap question."""

    def __init__(self) -> None:
        super().__init__(name="ResearchOrchestratorAgent")
        logger.debug(f"Initialized {self.name}")

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        gap_questions_output_dict = ctx.session.state.get("gap_questions") or None
        questions = []

        logger.debug(
            f"[{ctx.invocation_id}] {self.name}: Raw gap_questions output: {gap_questions_output_dict}"
        )

        if gap_questions_output_dict is not None:
            gap_questions_output = GapQuestionsOutput(**gap_questions_output_dict)
            questions = [item.question for item in gap_questions_output.gap_questions]

        if not questions:
            # Handle the case where the previous agent failed to produce output.
            # In a production system, you might yield an error event here.
            print("Error: 'gap_questions' not found in session state.")
            return

        # Step 2: Dynamically create a worker agent for each question.
        worker_agents: list[LlmAgent] = []
        for i, question in enumerate(questions):
            output_key = f"research_answer_{i}"
            worker = create_single_question_research_agent(
                question=question, output_key=output_key
            )
            worker_agents.append(worker)

        if not worker_agents:
            print("No worker agents created, skipping parallel execution.")
            return

        # Step 3: Create a ParallelAgent on the fly with the new workers.
        parallel_workflow = ParallelAgent(
            name="GapQuestionResearchWorkflowParallelAgent",
            sub_agents=worker_agents,
        )

        self.sub_agents.append(parallel_workflow)

        # Step 4: Execute the parallel workflow and yield its events.
        # This is crucial for maintaining observability.
        async for event in parallel_workflow.run_async(ctx):
            yield event

        research_answers: list[dict] = []
        for i, question in enumerate(questions):
            output_key = f"research_answer_{i}"
            research_answer = ctx.session.state.get(output_key)
            if research_answer is None:
                logger.warning(
                    f"[{ctx.invocation_id}] {self.name}: No answer found for key '{output_key}' (question {i})"
                )
                continue
            research_answers.append(research_answer)

        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Retrieved {len(research_answers)} research answers"
        )

        ctx.session.state["research_answers"] = research_answers


research_orchestrator_agent = ResearchOrchestratorAgent()
