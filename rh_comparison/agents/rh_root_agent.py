"""
Rolling Horizon Comparison Framework — Root Agent.

The root agent is the user-facing entry point for the RH comparison mode.
It is registered under the 'optichat_rh' ADK app name so it runs in a
completely separate session namespace from OptiChat's 'optichat' app.

Responsibilities:
  1. Session initialization (via rh_initialize_session before_agent_callback):
       Parse the user-provided JSON config, load and solve both epoch models,
       compute QT1, generate descriptions, and populate all RH_* state keys.
  2. Query classification (root LLM + set_query_type tool):
       Classify each user question into one of the RH query types, including
       direct retrieval requests, and register
       it in state before routing to rh_comparison_agent.
  3. Routing (rh_comparison_agent as AgentTool):
       Delegate the user's question to the comparison expert agent, which
       has the appropriate QT analytics injected into its system prompt.

The root LLM's prompt is dynamically filled by rh_check_llm_request
(before_model_callback) on every call.
"""

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from optichat.llm import gpt_5_mini

from rh_comparison.agents.rh_comparison_agent import create_rh_comparison_agent
from rh_comparison.tools.epoch_tools import set_query_type
from rh_comparison.tools.rh_callback_tool import rh_initialize_session, rh_check_llm_request


def create_rh_root_agent() -> Agent:
    """
    Factory: create and return the rh_root_agent.

    Creates the comparison agent internally — always use this factory rather
    than instantiating Agent directly, to ensure both agents share the same
    callback and tool registrations.
    """
    comparison_agent = create_rh_comparison_agent()

    return Agent(
        name="rh_root_agent",
        model=gpt_5_mini,
        tools=[
            set_query_type,
            AgentTool(comparison_agent),
        ],
        description=(
            "Root agent for the Rolling Horizon Comparison framework. "
            "Classifies user queries and routes them to the comparison expert agent."
        ),
        instruction=(
            "You are the Rolling Horizon Comparison Assistant. "
            "Your full context and routing instructions will be provided by the system."
        ),
        before_agent_callback=rh_initialize_session,
        before_model_callback=rh_check_llm_request,
    )
