"""
Rolling Horizon Comparison Framework — ADK Tool Definitions.

Tools exposed to ADK agents:
  set_query_type(query_type, tool_context)
      Register the query classification in state before routing to comparison_agent.

  get_epoch_data(epoch_id, component_type, family_filter, tool_context)
      Retrieve per-epoch data (params, variables, structure, summaries) from EpochStore.

  get_comparison_json(analysis_type, family_filter, tool_context)
      Retrieve or compute QT1–QT4 comparison results.

Process-level EpochStore cache:
  _epoch_store_cache: {epoch_id: EpochStore}
  Populated by rh_callback_tool.rh_initialize_session.
  Also used by rh_check_llm_request to access stores during on-demand QT computation.

Token-budget safeguards:
  All tool endpoints that return potentially unbounded data (variables, params,
  structure index sets, variable family comparisons) are capped to prevent
  overwhelming the LLM context window on large models (50k+ variables).
  Each cap includes a _note telling the LLM how to drill down for more.
"""

from __future__ import annotations

from loguru import logger
from google.adk.tools.tool_context import ToolContext

from rh_comparison.config.rh_constants import (
    QueryType,
    RH_EPOCH_A_ID, RH_EPOCH_B_ID,
    RH_QT1_RESULT, RH_QT2_RESULT, RH_QT3_RESULT, RH_QT4_RESULT,
    RH_QUERY_TYPE,
)
from rh_comparison.data_store.epoch_store import EpochStore
from rh_comparison.analytics import (
    compute_structural_diff,
    compute_solution_diff,
    get_variable_family_details,
    assess_backward_compat,
    compute_attribution_analysis,
)


# ---------------------------------------------------------------------------
# Process-level EpochStore cache (populated during rh_initialize_session)
# ---------------------------------------------------------------------------

_epoch_store_cache: dict[str, EpochStore] = {}


# ---------------------------------------------------------------------------
# Token-budget caps — keep tool responses LLM-friendly on large models
# ---------------------------------------------------------------------------

_MAX_BULK_ITEMS = 100           # "variables" / "params" (full-dump calls)
_MAX_FAMILY_ITEMS = 200         # family-scoped drill-down calls
_MAX_INDEX_SET_ELEMENTS = 30    # per-set element lists in "structure"


def _truncate_flat_dict(d: dict, limit: int, hint: str) -> dict:
    """Return first *limit* entries of a flat dict; add _note if truncated."""
    if len(d) <= limit:
        return d
    truncated = dict(list(d.items())[:limit])
    truncated["_note"] = (
        f"Showing {limit} of {len(d)} entries. {hint}"
    )
    return truncated


def _truncate_params(params: dict, limit: int) -> dict:
    """Truncate a nested params dict {family: {idx: val}, ...} to ~limit total entries."""
    # Count total entries across all families
    total = sum(
        len(v) for k, v in params.items()
        if k != "epoch_id" and isinstance(v, dict)
    )
    if total <= limit:
        return params

    result = {}
    if "epoch_id" in params:
        result["epoch_id"] = params["epoch_id"]

    count = 0
    for family, values in params.items():
        if family == "epoch_id" or not isinstance(values, dict):
            continue
        if count >= limit:
            break
        remaining = limit - count
        if len(values) <= remaining:
            result[family] = values
            count += len(values)
        else:
            result[family] = dict(list(values.items())[:remaining])
            count += remaining

    result["_note"] = (
        f"Showing {count} of {total} total parameter entries. "
        "Use component_type='param_family' with family_filter for a specific group."
    )
    return result


def _truncate_structure(struct: dict) -> dict:
    """Cap index-set element lists so large sets don't blow up context."""
    result = dict(struct)
    index_sets = result.get("index_sets", {})
    needs_cap = any(len(v) > _MAX_INDEX_SET_ELEMENTS for v in index_sets.values() if isinstance(v, list))
    if not needs_cap:
        return result

    capped = {}
    for name, members in index_sets.items():
        if not isinstance(members, list) or len(members) <= _MAX_INDEX_SET_ELEMENTS:
            capped[name] = members
        else:
            capped[name] = members[:_MAX_INDEX_SET_ELEMENTS]
            capped[name].append(f"... and {len(members) - _MAX_INDEX_SET_ELEMENTS} more ({len(members)} total)")
    result["index_sets"] = capped
    return result


def _truncate_variable_family_comparison(result: dict, limit: int) -> dict:
    """Cap the 'variables' dict inside a qt2_variable_family result."""
    variables = result.get("variables", {})
    if len(variables) <= limit:
        return result

    total = len(variables)
    result = dict(result)
    result["variables"] = dict(list(variables.items())[:limit])
    result["_note"] = (
        f"Showing {limit} of {total} variables in this family."
    )
    return result


_MAX_QT_LIST_ITEMS = 20       # per-list cap inside QT result dicts
_MAX_QT_FAMILY_ENTRIES = 30   # per-family dict cap in QT by_family breakdowns


def _truncate_list(lst: list, limit: int) -> list:
    """Truncate a list and append a note if needed."""
    if len(lst) <= limit:
        return lst
    return lst[:limit] + [f"... and {len(lst) - limit} more ({len(lst)} total)"]


def _truncate_qt_result(qt: dict, analysis_type: str) -> dict:
    """
    Cap potentially large lists/dicts inside a QT result before returning
    to the LLM via tool call.  Keeps the result token-friendly on large models.

    The pre-injected formatters (_format_qt1, etc.) already apply their own caps,
    but this function protects the TOOL path where the LLM explicitly calls
    get_comparison_json() and receives the raw dict.
    """
    cap = _MAX_QT_LIST_ITEMS
    fam_cap = _MAX_QT_FAMILY_ENTRIES
    result = dict(qt)  # shallow copy to avoid mutating state cache

    if analysis_type == "qt1":
        # Cap parameter_changes.by_family — each family's changed/added/removed lists
        pc = result.get("parameter_changes", {})
        if isinstance(pc, dict):
            pc = dict(pc)
            bf = pc.get("by_family", {})
            if len(bf) > fam_cap:
                truncated_bf = dict(list(bf.items())[:fam_cap])
                truncated_bf["_note"] = (
                    f"Showing {fam_cap} of {len(bf)} families. "
                    "Use get_epoch_data with component_type='param_family' for details."
                )
                pc["by_family"] = truncated_bf
            # top_changes already capped at 20 by structural_diff.py
            result["parameter_changes"] = pc

    elif analysis_type == "qt2":
        # Cap churn_summary (per-family stats)
        cs = result.get("churn_summary", {})
        if len(cs) > fam_cap:
            truncated_cs = dict(list(cs.items())[:fam_cap])
            truncated_cs["_note"] = (
                f"Showing {fam_cap} of {len(cs)} families. "
                "Use get_comparison_json('qt2_variable_family', family_filter=...) for details."
            )
            result["churn_summary"] = truncated_cs
        # Cap binding_changes lists
        bc = result.get("binding_changes", {})
        if isinstance(bc, dict):
            bc = dict(bc)
            for key in ("became_binding", "became_nonbinding",
                        "added_constraints", "removed_constraints"):
                lst = bc.get(key, [])
                if isinstance(lst, list) and len(lst) > cap:
                    bc[key] = _truncate_list(lst, cap)
            result["binding_changes"] = bc

    elif analysis_type == "qt3":
        # Cap IIS rounds, minimal cover, and required slacks
        rounds = result.get("iis_rounds", [])
        if isinstance(rounds, list) and len(rounds) > cap:
            result["iis_rounds"] = rounds[:cap]
        cover = result.get("minimal_cover", [])
        if isinstance(cover, list) and len(cover) > cap:
            result["minimal_cover"] = cover[:cap]
        slacks = result.get("required_slacks", {})
        if isinstance(slacks, dict) and len(slacks) > cap:
            result["required_slacks"] = dict(list(slacks.items())[:cap])
            result["_note"] = (
                f"Showing {cap} of {len(slacks)} slack recommendations."
            )

    elif analysis_type == "qt4":
        # Cap bottleneck_evolution lists
        bev = result.get("bottleneck_evolution", {})
        if isinstance(bev, dict):
            bev = dict(bev)
            for key in ("relieved", "new_bottlenecks", "persistent"):
                lst = bev.get(key, [])
                if isinstance(lst, list) and len(lst) > cap:
                    bev[key] = lst[:cap]
                    bev[f"_n_{key}_total"] = len(lst)
            result["bottleneck_evolution"] = bev
        # Cap binary_flips and continuous_adjustments
        for key in ("binary_flips", "continuous_adjustments"):
            lst = result.get(key, [])
            if isinstance(lst, list) and len(lst) > cap:
                result[key] = lst[:cap]

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_store(epoch_id: str) -> EpochStore | None:
    """Return cached EpochStore or try to load from disk."""
    if epoch_id in _epoch_store_cache:
        return _epoch_store_cache[epoch_id]
    try:
        store = EpochStore.load(epoch_id)
        _epoch_store_cache[epoch_id] = store
        return store
    except FileNotFoundError:
        return None


def _resolve_epoch_id(epoch_id: str, state: dict) -> str:
    """Resolve 'epoch_a' / 'a' / 'epoch_b' / 'b' aliases to the real epoch_id in state."""
    if epoch_id.lower() in ("epoch_a", "a"):
        return state.get(RH_EPOCH_A_ID, epoch_id)
    if epoch_id.lower() in ("epoch_b", "b"):
        return state.get(RH_EPOCH_B_ID, epoch_id)
    return epoch_id


# ---------------------------------------------------------------------------
# Tool: set_query_type
# ---------------------------------------------------------------------------

def set_query_type(query_type: str, tool_context: ToolContext) -> str:
    """
    Register the classified query type in session state before calling comparison_agent.

    Args:
        query_type: One of GENERAL, RETRIEVAL, MODEL_DESCRIPTION,
                    STRUCTURAL_CHANGE, SOLUTION_DIFF, BACKWARD_COMPAT,
                    ATTRIBUTION.

    Returns:
        Confirmation string.
    """
    valid = {
        QueryType.GENERAL,
        QueryType.RETRIEVAL,
        QueryType.MODEL_DESCRIPTION,
        QueryType.STRUCTURAL_CHANGE,
        QueryType.SOLUTION_DIFF,
        QueryType.BACKWARD_COMPAT,
        QueryType.ATTRIBUTION,
    }
    if query_type not in valid:
        logger.warning(f"[RH] Unknown query_type '{query_type}'; defaulting to GENERAL.")
        query_type = QueryType.GENERAL

    tool_context.state[RH_QUERY_TYPE] = query_type
    logger.info(f"[RH] Query type set to: {query_type}")
    return "Routing updated."


# ---------------------------------------------------------------------------
# Tool: get_epoch_data
# ---------------------------------------------------------------------------

def get_epoch_data(
    epoch_id: str,
    component_type: str,
    family_filter: str = "",
    tool_context: ToolContext = None,
) -> dict:
    """
    Retrieve data from an epoch's stored files.

    Args:
        epoch_id:       'epoch_a', 'a', 'epoch_b', 'b', or the actual epoch_id string.
        component_type: One of:
                          'solution_summary' — compact LLM-friendly summary
                          'param_summary'    — parameter family names + counts (no values)
                          'structure'        — model structure (index sets capped at 30 per set)
                          'params'           — mutable parameter values (capped at 100 entries)
                          'variables'        — variable values (capped at 100 entries)
                          'param_family'     — single parameter family values (capped at 200)
                          'variable_family'  — single variable family values (capped at 200)
        family_filter:  Family name prefix for 'param_family' or 'variable_family' types.

    Returns:
        Requested data dict, or {'error': ...} on failure.
    """
    state = tool_context.state
    resolved_id = _resolve_epoch_id(epoch_id, state)
    store = _get_store(resolved_id)

    if store is None:
        return {"error": f"Epoch '{resolved_id}' not found. Has the session been initialized?"}

    try:
        if component_type == "solution_summary":
            return store.get_solution_summary()

        elif component_type == "param_summary":
            return store.get_param_summary()

        elif component_type == "structure":
            return _truncate_structure(store.get_structure())

        elif component_type == "params":
            return _truncate_params(
                store.get_params(),
                _MAX_BULK_ITEMS,
            )

        elif component_type == "variables":
            sol = store.get_solution()
            return _truncate_flat_dict(
                sol.get("variables", {}),
                _MAX_BULK_ITEMS,
                "Use component_type='variable_family' with family_filter for a specific group.",
            )

        elif component_type == "param_family":
            if not family_filter:
                return {"error": "family_filter is required for component_type='param_family'"}
            return _truncate_flat_dict(
                store.get_param_family(family_filter),
                _MAX_FAMILY_ITEMS,
                f"Family '{family_filter}' has more entries than shown.",
            )

        elif component_type == "variable_family":
            if not family_filter:
                return {"error": "family_filter is required for component_type='variable_family'"}
            return _truncate_flat_dict(
                store.get_variable_family(family_filter),
                _MAX_FAMILY_ITEMS,
                f"Family '{family_filter}' has more entries than shown.",
            )

        else:
            return {
                "error": (
                    f"Unknown component_type '{component_type}'. "
                    "Valid values: solution_summary, param_summary, structure, params, "
                    "variables, param_family, variable_family"
                )
            }

    except Exception as exc:
        logger.error(f"[RH] get_epoch_data failed for '{resolved_id}': {exc}")
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: get_comparison_json
# ---------------------------------------------------------------------------

def get_comparison_json(
    analysis_type: str,
    family_filter: str = "",
    tool_context: ToolContext = None,
) -> dict:
    """
    Retrieve or compute a comparison analysis result (QT1–QT4).

    Results are cached in session state after the first computation.
    QT1 is always pre-computed at session init.
    QT2 is pure Python (fast). QT3 requires 1 Gurobi call. QT4 requires 0–1 Gurobi calls.

    Args:
        analysis_type: One of:
                         'qt1'                  — Structural Diff (always available)
                         'qt2'                  — Solution Diff (computed on first call)
                         'qt3'                  — Backward Compatibility (1 solver call)
                         'qt4'                  — Solution Attribution (0–1 solver calls)
                         'qt2_variable_family'  — Variable-level QT2 for one family (capped at 200)
        family_filter: For 'qt2_variable_family': the variable family prefix (e.g., 'x', 'cost').

    Returns:
        QT result dict, or {'error': ...} on failure.
    """
    state = tool_context.state
    epoch_a_id = state.get(RH_EPOCH_A_ID, "")
    epoch_b_id = state.get(RH_EPOCH_B_ID, "")

    store_a = _get_store(epoch_a_id)
    store_b = _get_store(epoch_b_id)

    if store_a is None or store_b is None:
        return {"error": "Epoch stores not available. Session may not be initialized."}

    qt1 = state.get(RH_QT1_RESULT)
    qt2 = state.get(RH_QT2_RESULT)

    try:
        if analysis_type == "qt1":
            if qt1 is None:
                qt1 = compute_structural_diff(store_a, store_b)
                state[RH_QT1_RESULT] = qt1
            return _truncate_qt_result(qt1, "qt1")

        elif analysis_type == "qt2":
            if qt2 is None:
                logger.info("[RH] Computing QT2 on demand...")
                qt2 = compute_solution_diff(store_a, store_b, qt1=qt1)
                state[RH_QT2_RESULT] = qt2
            return _truncate_qt_result(qt2, "qt2")

        elif analysis_type == "qt2_variable_family":
            if not family_filter:
                return {"error": "family_filter is required for analysis_type='qt2_variable_family'"}
            logger.info(f"[RH] Getting variable family detail: '{family_filter}'")
            result = get_variable_family_details(store_a, store_b, family_filter)
            return _truncate_variable_family_comparison(result, _MAX_FAMILY_ITEMS)

        elif analysis_type == "qt3":
            qt3 = state.get(RH_QT3_RESULT)
            if qt3 is None:
                logger.info("[RH] Computing QT3 (backward compat, 1 solver call)...")
                qt3 = assess_backward_compat(store_a, store_b, qt1=qt1, qt2=qt2)
                state[RH_QT3_RESULT] = qt3
            return _truncate_qt_result(qt3, "qt3")

        elif analysis_type == "qt4":
            qt4 = state.get(RH_QT4_RESULT)
            if qt4 is None:
                # QT4 needs QT2 — ensure it's computed
                if qt2 is None:
                    logger.info("[RH] Computing QT2 prerequisite for QT4...")
                    qt2 = compute_solution_diff(store_a, store_b, qt1=qt1)
                    state[RH_QT2_RESULT] = qt2
                logger.info("[RH] Computing QT4 (attribution analysis)...")
                qt4 = compute_attribution_analysis(store_a, store_b, qt1=qt1, qt2=qt2)
                state[RH_QT4_RESULT] = qt4
            return _truncate_qt_result(qt4, "qt4")

        else:
            return {
                "error": (
                    f"Unknown analysis_type '{analysis_type}'. "
                    "Valid values: qt1, qt2, qt3, qt4, qt2_variable_family"
                )
            }

    except Exception as exc:
        logger.error(f"[RH] get_comparison_json('{analysis_type}') failed: {exc}")
        return {"error": str(exc)}
