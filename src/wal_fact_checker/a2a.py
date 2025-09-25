from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from wal_fact_checker.agent import root_agent

skill = AgentSkill(
    id="wal_fact_checker",
    name="WAL Fact Check Orchestrator",
    description=(
        "End-to-end fact checker that structures claims, identifies research gaps, "
        "performs open-web research, adjudicates evidence, and returns a verified report."
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
        "Web-scale automated fact checker orchestrating analysis → research → synthesis."
    ),
    url="http://localhost:8080/",
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["application/json"],
    capabilities=AgentCapabilities(state_transition_history=False, streaming=False),
    skills=[skill],
    supports_authenticated_extended_card=True,
)

a2a_app = to_a2a(root_agent, port=8080, agent_card=public_agent_card)
