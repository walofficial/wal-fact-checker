from google.adk.a2a.utils.agent_to_a2a import to_a2a

from wal_fact_checker.agent import root_agent

a2a_app = to_a2a(root_agent, port=8000)
