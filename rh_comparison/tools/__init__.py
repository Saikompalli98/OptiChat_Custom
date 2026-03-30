# rh_comparison/tools package — ADK tool wrappers and session callbacks (Phase 4)
#
# Lazy imports: importing this package does NOT pull in google.adk.models.lite_llm
# unless one of the exported names is explicitly accessed.

__all__ = [
    "set_query_type",
    "get_epoch_data",
    "get_comparison_json",
    "_epoch_store_cache",
    "rh_initialize_session",
    "rh_check_llm_request",
]

_epoch_tools   = ["set_query_type", "get_epoch_data", "get_comparison_json", "_epoch_store_cache"]
_callback_tool = ["rh_initialize_session", "rh_check_llm_request"]


def __getattr__(name: str):
    if name in _epoch_tools:
        from rh_comparison.tools import epoch_tools as _m
        return getattr(_m, name)
    if name in _callback_tool:
        from rh_comparison.tools import rh_callback_tool as _m
        return getattr(_m, name)
    raise AttributeError(f"module 'rh_comparison.tools' has no attribute {name!r}")
