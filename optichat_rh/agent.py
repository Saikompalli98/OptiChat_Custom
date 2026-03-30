"""
optichat_rh — ADK entry point for the Rolling Horizon Comparison framework.

ADK's `adk api_server` discovers apps by scanning subdirectories for agent.py
files that export a `root_agent` variable.  This file mirrors the pattern used
by optichat/agent.py exactly, but registers the RH comparison agent under the
separate 'optichat_rh' app namespace.

Start the full stack with:
    ./run.sh            # starts both adk api_server and streamlit

The ADK server will then serve:
    optichat     at  /apps/optichat/...
    optichat_rh  at  /apps/optichat_rh/...
"""

import os
import logging

# Disable OpenTelemetry to avoid context management issues (same as optichat/agent.py)
os.environ["OTEL_SDK_DISABLED"] = "true"
logging.getLogger("opentelemetry").setLevel(logging.ERROR)

from loguru import logger
logger.add("rh_conversation_logs.txt", rotation="10 MB")

from rh_comparison.rh_agent import create_rh_root_agent

root_agent = create_rh_root_agent()
