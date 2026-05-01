"""A2A Starlette application for the WAL Fact Checker agent.

Builds an A2A-compliant server using the Agent2Agent (A2A) Protocol with a
thin `AgentExecutor` bridge to the Google ADK `Runner` hosting our
`root_agent` orchestrator.
"""

from __future__ import annotations

from typing import Final

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import (
    InMemoryCredentialService,
)
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

from wal_fact_checker.a2a.executor import WalAgentExecutor
from wal_fact_checker.agent import root_agent
from wal_fact_checker.core.settings import settings

RPC_URL: Final[str] = f"http://{settings.host}:{settings.port}/"


skill = AgentSkill(
    id="wal_fact_checker",
    name="WAL Fact Check Orchestrator",
    description=(
        "End-to-end fact checker that structures claims, identifies research "
        "gaps, performs open-web research, adjudicates evidence, and returns "
        "a verified report."
    ),
    input_modes=["text"],
    output_modes=["application/json"],
    tags=["fact-checking", "research", "verification", "reports"],
    examples=[
        "Fact-check: OpenAI released GPT-4 in March 2023 and it supports image inputs.",
        "Analyze and verify claims from this article about climate policy changes.",
    ],
)


public_agent_card = AgentCard(
    name="WAL Fact Checker",
    description=(
        "Web-scale automated fact checker orchestrating analysis → research "
        "→ synthesis."
    ),
    url=RPC_URL,
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["application/json"],
    capabilities=AgentCapabilities(state_transition_history=False, streaming=False),
    skills=[skill],
    supports_authenticated_extended_card=True,
)


async def _create_runner() -> Runner:
    """Create an ADK `Runner` hosting the `root_agent` with in-memory services.

    Returns:
        Runner: Configured ADK runner
    """
    return Runner(
        app_name=root_agent.name or "WAL Fact Checker",
        agent=root_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=InMemoryCredentialService(),
    )


# AgentExecutor bridging ADK↔A2A (custom implementation)
agent_executor = WalAgentExecutor(runner=_create_runner)

# Request handler and task store per A2A Protocol
request_handler = DefaultRequestHandler(
    agent_executor=agent_executor, task_store=InMemoryTaskStore()
)

# Starlette application implementing A2A server endpoints
_server = A2AStarletteApplication(
    agent_card=public_agent_card,
    http_handler=request_handler,
)

# ASGI app for uvicorn/gunicorn
a2a_app = _server.build()
