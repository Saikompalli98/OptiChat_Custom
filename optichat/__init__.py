# Lazy import — avoids loading the full agent stack (and chromadb / langchain)
# when only optichat sub-modules (e.g. optichat.llm, optichat.config) are needed.
# ADK discovers root_agent by importing optichat.agent directly, so no eager
# import is required here.


def __getattr__(name: str):
    if name == "agent":
        from . import agent as _agent_mod
        return _agent_mod
    raise AttributeError(f"module 'optichat' has no attribute {name!r}")