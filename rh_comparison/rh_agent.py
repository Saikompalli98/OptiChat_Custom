"""
Rolling Horizon Comparison Framework — Entry Point.

Usage in main.py / app.py:

    from rh_comparison.rh_agent import create_rh_root_agent
    from rh_comparison.config.rh_constants import RH_APP_NAME

    rh_agent = create_rh_root_agent()
    # Register with the ADK runner under RH_APP_NAME ('optichat_rh')

This module re-exports create_rh_root_agent() so callers only need a single import.
"""

from rh_comparison.agents.rh_root_agent import create_rh_root_agent

__all__ = ["create_rh_root_agent"]
