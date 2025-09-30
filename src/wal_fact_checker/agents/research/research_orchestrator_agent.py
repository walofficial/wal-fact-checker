# File: src/wal_fact_checker/agents/research/research_orchestrator_agent.py
"""Orchestrator agent that manages parallel research of gap questions."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Final

from google.adk.agents import BaseAgent, ParallelAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

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

    def _create_batch_parallel_agents(
        self, questions: list[str], batch_size: int
    ) -> list[ParallelAgent]:
        """Create ParallelAgent instances for batched question processing."""
        question_batches = [
            questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
        ]

        batch_parallel_agents: list[ParallelAgent] = []

        for batch_idx, batch_questions in enumerate(question_batches):
            batch_worker_agents = [
                create_single_question_research_agent(
                    question=question,
                    output_key=f"research_answer_{batch_idx * batch_size + q_idx}",
                )
                for q_idx, question in enumerate(batch_questions)
            ]

            batch_parallel_agent = ParallelAgent(
                name=f"SingleBatchBatchResearchAgent_{batch_idx}",
                sub_agents=batch_worker_agents,
            )
            batch_parallel_agents.append(batch_parallel_agent)

        return batch_parallel_agents

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
            logger.error(
                f"[{ctx.invocation_id}] {self.name}: No gap questions found in session state"
            )
            return

        # Step 2: Create batched parallel workflow
        batch_size = 5
        batch_parallel_agents = self._create_batch_parallel_agents(
            questions, batch_size
        )

        if not batch_parallel_agents:
            logger.warning(
                f"[{ctx.invocation_id}] {self.name}: No batch agents created"
            )
            return

        # Step 3: Create sequential workflow to orchestrate batches
        sequential_workflow = SequentialAgent(
            name="BatchSequentialResearchAgent",
            sub_agents=batch_parallel_agents,
        )

        self.sub_agents.append(sequential_workflow)

        # Step 4: Execute the sequential workflow and yield its events
        async for event in sequential_workflow.run_async(ctx):
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

        state_update_event = Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta={"research_answers": research_answers}),
        )
        yield state_update_event


research_orchestrator_agent = ResearchOrchestratorAgent()
