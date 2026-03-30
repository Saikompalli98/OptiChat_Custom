# rh_comparison/agents package — ADK agent factories and prompts (Phase 4)
#
# Imports are lazy (inside __getattr__) so that importing rh_prompts or other
# submodules in test environments does NOT pull in google.adk.models.lite_llm
# unless the agent factories are explicitly requested.

__all__ = ["create_rh_root_agent", "create_rh_comparison_agent"]


def __getattr__(name: str):
    if name == "create_rh_root_agent":
        from rh_comparison.agents.rh_root_agent import create_rh_root_agent
        return create_rh_root_agent
    if name == "create_rh_comparison_agent":
        from rh_comparison.agents.rh_comparison_agent import create_rh_comparison_agent
        return create_rh_comparison_agent
    raise AttributeError(f"module 'rh_comparison.agents' has no attribute {name!r}")
