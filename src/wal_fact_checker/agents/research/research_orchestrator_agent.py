# File: src/wal_fact_checker/agents/research/research_orchestrator_agent.py
"""Orchestrator agent that manages parallel research of gap questions."""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Final

from google.adk.agents import BaseAgent, ParallelAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from pydantic import Field

from wal_fact_checker.core.models import GapQuestionsOutput

from .single_question_research_agent import single_question_research_agent

# Configure logging for this module
logger = logging.getLogger(__name__)

MODEL: Final[str] = "gemini-2.0-flash"


class _SetQuestionAgent(BaseAgent):
    question: str = Field(
        default="", description="The question to set in the session state"
    )

    def __init__(self, question: str, index: int) -> None:
        super().__init__(name=f"SetQuestion_{index}")
        self.question = question
        logger.debug(f"Initialized {self.name} with question: {question[:100]}...")

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Setting current_question in session state"
        )
        logger.debug(
            f"[{ctx.invocation_id}] {self.name}: Question content: {self.question}"
        )

        ctx.session.state["current_question"] = self.question

        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Successfully set current_question"
        )
        yield Event(author=self.name)


class _CopyAnswerAgent(BaseAgent):
    target_key: str = Field(
        default="", description="The key to copy the answer to in the session state"
    )

    def __init__(self, target_key: str) -> None:
        super().__init__(name=f"CopyAnswer_{target_key}")
        self.target_key = target_key
        logger.debug(f"Initialized {self.name} with target_key: {target_key}")

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Copying research_answer to {self.target_key}"
        )

        answer = ctx.session.state.get("research_answer")
        if answer is None:
            logger.warning(
                f"[{ctx.invocation_id}] {self.name}: No research_answer found in session state"
            )
        else:
            logger.debug(
                f"[{ctx.invocation_id}] {self.name}: Found answer with length: {len(str(answer))}"
            )

        ctx.session.state[self.target_key] = answer

        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Successfully copied answer to {self.target_key}"
        )
        yield Event(author=self.name)


class _AggregateAnswersAgent(BaseAgent):
    questions: list[str] = Field(
        default=[], description="The questions to aggregate answers from"
    )

    def __init__(self, questions: list[str]) -> None:
        super().__init__(name="AggregateAnswers")
        self.questions = questions
        logger.debug(f"Initialized {self.name} with {len(questions)} questions")

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Aggregating answers from {len(self.questions)} questions"
        )

        answers = []
        for i, q in enumerate(self.questions, 1):
            key = f"research_answer_{i}"
            ans = ctx.session.state.get(key)

            if ans is None:
                logger.warning(
                    f"[{ctx.invocation_id}] {self.name}: No answer found for key '{key}' (question {i})"
                )
                continue

            logger.debug(
                f"[{ctx.invocation_id}] {self.name}: Found answer for question {i}: {str(ans)[:100]}..."
            )
            answers.append({"question": q, "answer": ans})

        comprehensive_answer_set = {"answers": answers}
        ctx.session.state["comprehensive_answer_set"] = comprehensive_answer_set

        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Successfully aggregated {len(answers)} answers"
        )
        logger.debug(
            f"[{ctx.invocation_id}] {self.name}: Comprehensive answer set keys: {list(comprehensive_answer_set.keys())}"
        )

        yield Event(author=self.name)


class ResearchOrchestratorAgent(BaseAgent):
    """Run SingleQuestionResearchAgent in parallel for each gap question."""

    def __init__(self) -> None:
        super().__init__(name="ResearchOrchestratorAgent")
        logger.debug(f"Initialized {self.name}")

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Starting research orchestration"
        )

        # Get gap questions from session state (structured output from gap identification agent)
        gap_questions_output_dict = ctx.session.state.get("gap_questions") or None
        logger.debug(
            f"[{ctx.invocation_id}] {self.name}: Raw gap_questions output: {gap_questions_output_dict}"
        )

        if gap_questions_output_dict is not None:
            gap_questions_output = GapQuestionsOutput(**gap_questions_output_dict)
            questions = [item.question for item in gap_questions_output.gap_questions]
        else:
            questions = []

        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Retrieved {len(questions)} gap questions from session state"
        )
        logger.debug(f"[{ctx.invocation_id}] {self.name}: Gap questions: {questions}")

        # Validate questions
        if not isinstance(questions, list) or not questions:
            logger.warning(
                f"[{ctx.invocation_id}] {self.name}: No valid gap questions found, setting empty answer set"
            )
            ctx.session.state["comprehensive_answer_set"] = {"answers": []}
            yield Event(author=self.name, actions=EventActions(escalate=True))
            return

        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Creating {len(questions)} parallel research branches"
        )

        # Create parallel research branches
        branches: list[SequentialAgent] = []
        for i, q in enumerate(questions, 1):
            question_str = str(q).strip()
            logger.debug(
                f"[{ctx.invocation_id}] {self.name}: Creating branch {i} for question: {question_str[:100]}..."
            )

            set_q = _SetQuestionAgent(question=question_str, index=i)
            branch = SequentialAgent(
                name=f"ResearchBranch_{i}",
                sub_agents=[
                    set_q,
                    single_question_research_agent.clone(),
                    _CopyAnswerAgent(target_key=f"research_answer_{i}"),
                ],
                description=f"Per-question research branch for question {i}",
            )
            branches.append(branch)
            logger.debug(
                f"[{ctx.invocation_id}] {self.name}: Created ResearchBranch[{i}] with {len(branch.sub_agents)} sub-agents"
            )

        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Starting parallel execution of {len(branches)} research branches"
        )

        # Execute branches in parallel
        parallel = ParallelAgent(
            name="ParallelResearch",
            sub_agents=branches,
            description="Run SingleQuestionResearchAgent concurrently per question",
        )

        event_count = 0
        async for event in parallel._run_async_impl(ctx):
            event_count += 1
            logger.debug(
                f"[{ctx.invocation_id}] {self.name}: Received event {event_count} from {event.author}"
            )

            yield event

        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Parallel research completed, processed {event_count} events"
        )
        logger.info(f"[{ctx.invocation_id}] {self.name}: Starting answer aggregation")

        # Aggregate into comprehensive_answer_set for downstream stages
        question_strings = []
        for q in questions:
            if isinstance(q, dict):
                question_strings.append(q.get("question", str(q)))
            else:
                question_strings.append(str(q))

        aggregator = _AggregateAnswersAgent(question_strings)
        aggregation_event_count = 0
        async for event in aggregator._run_async_impl(ctx):
            aggregation_event_count += 1
            logger.debug(
                f"[{ctx.invocation_id}] {self.name}: Aggregation event {aggregation_event_count} from {event.author}"
            )

            yield event

        logger.info(
            f"[{ctx.invocation_id}] {self.name}: Research orchestration completed successfully"
        )
        logger.debug(
            f"[{ctx.invocation_id}] {self.name}: Final session state keys: {list(ctx.session.state.keys())}"
        )


research_orchestrator_agent = ResearchOrchestratorAgent()
