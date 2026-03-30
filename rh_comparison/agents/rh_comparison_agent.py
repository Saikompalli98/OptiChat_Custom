"""
Rolling Horizon Comparison Framework — Expert Comparison Agent.

The comparison agent is an LLM expert that answers rolling-horizon questions
about model setup, direct value retrieval, structural changes, solution
differences, backward compatibility, and solution attribution between two
optimization model epochs.

It is always invoked by rh_root_agent via AgentTool, never directly by the user.
The before_model_callback (rh_check_llm_request) rebuilds its system prompt before
every LLM call, injecting the relevant QT analytics and a query-type-specific strategy.

Tools:
  get_epoch_data       — retrieve per-epoch data (params, variables, structure)
  get_comparison_json  — retrieve or compute QT1–QT4 comparison results
"""

from google.adk.agents import Agent

from optichat.llm import gpt_5

from rh_comparison.tools.epoch_tools import get_epoch_data, get_comparison_json
from rh_comparison.tools.rh_callback_tool import rh_check_llm_request


def create_rh_comparison_agent() -> Agent:
    """
    Factory: create and return the rh_comparison_agent.

    The agent's `instruction` is a placeholder — the actual system prompt
    is fully rebuilt by rh_check_llm_request on every LLM call.
    """
    return Agent(
        name="rh_comparison_agent",
        model=gpt_5,
        tools=[get_epoch_data, get_comparison_json],
        description=(
            "Expert agent for analyzing differences between two optimization model epochs. "
            "Handles structural changes, solution diffs, backward compatibility, "
            "and solution attribution analysis."
        ),
        instruction=(
            "You are an expert in rolling horizon optimization model comparison. "
            "Your full context and strategy will be provided by the system."
        ),
        before_model_callback=rh_check_llm_request,
    )
