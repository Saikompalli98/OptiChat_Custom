"""
QT2 — Solution Difference Analysis.

Pure Python, zero solver calls.

Compares variable values, objective, and constraint binding status between
two epochs.  All comparisons are derived from solution.json in EpochStore.
No Pyomo model deserialization is required.

On-demand (not computed eagerly at session init).  The compact result
(churn_summary, top_changes, binding_changes, summary) is saved to disk
and returned for LLM injection.  Full per-variable details for a specific
family are available on demand via ``get_variable_family_details``.
"""

from __future__ import annotations

import ast
from datetime import datetime
from typing import Any

from rh_comparison.data_store.epoch_store import EpochStore
from rh_comparison.config.rh_constants import Thresholds

# Number of top individual variable changes included in the compact summary
_TOP_N = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _family_of(var_name: str) -> str:
    """Extract family prefix from a variable name like 'x[chicken]' → 'x'."""
    return var_name.split("[")[0]


def _safe_float(val: Any) -> float | None:
    """Return float(val) or None if val is missing or not numeric."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if f != f else f   # NaN → None
    except (TypeError, ValueError):
        return None


def _rel_change(value_a: float | None, value_b: float | None) -> float | None:
    """
    Compute (value_b − value_a) / |value_a|.

    Returns None when either value is missing or when value_a ≈ 0 and the
    ratio is undefined (caller should treat as a large change).
    """
    if value_a is None or value_b is None:
        return None
    if abs(value_a) > Thresholds.VAR_CHANGE_TOL:
        return (value_b - value_a) / abs(value_a)
    if abs(value_b - value_a) < Thresholds.VAR_CHANGE_TOL:
        return 0.0
    return None   # 0 → nonzero; ratio undefined


def _parse_name_index(var_name: str) -> Any:
    br = var_name.find("[")
    if br == -1 or not var_name.endswith("]"):
        return None
    raw = var_name[br + 1 : -1]
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _activity_value(value: float | None, is_binary: bool) -> float:
    if value is None:
        return 0.0
    if is_binary:
        return 1.0 if abs(value - 1.0) < Thresholds.BINDING_TOL else 0.0
    return float(value)


def _update_allocation_rollups(
    rollups: dict,
    family_totals: dict,
    family: str,
    var_name: str,
    indexed_over: list[str],
    value_a: float,
    value_b: float,
) -> None:
    family_total = family_totals.setdefault(family, {"value_a": 0.0, "value_b": 0.0})
    family_total["value_a"] += value_a
    family_total["value_b"] += value_b

    if not indexed_over:
        return

    parsed = _parse_name_index(var_name)
    if isinstance(parsed, tuple):
        members = list(parsed)
    elif parsed is not None:
        members = [parsed]
    else:
        return

    family_rollup = rollups.setdefault(family, {})
    for pos, member in enumerate(members[:len(indexed_over)]):
        position_rollup = family_rollup.setdefault(pos, {})
        member_key = str(member)
        values = position_rollup.setdefault(member_key, {"value_a": 0.0, "value_b": 0.0})
        values["value_a"] += value_a
        values["value_b"] += value_b


def _build_allocation_summaries(
    rollups: dict,
    family_totals: dict,
    variable_metadata: dict[str, dict],
    variable_labels: dict[str, str],
    set_labels: dict[str, str],
) -> dict:
    summaries: dict = {}
    for family, positions in rollups.items():
        metadata = variable_metadata.get(family, {})
        indexed_over = metadata.get("indexed_over", [])
        dimensions: list[dict] = []

        for pos, members in positions.items():
            changed_members: list[dict] = []
            for member, values in members.items():
                delta = values["value_b"] - values["value_a"]
                if abs(delta) < Thresholds.VAR_CHANGE_TOL:
                    continue
                changed_members.append({
                    "member": member,
                    "value_a": values["value_a"],
                    "value_b": values["value_b"],
                    "delta": delta,
                })

            if not changed_members:
                continue

            changed_members.sort(key=lambda entry: abs(entry["delta"]), reverse=True)
            increases = [entry for entry in changed_members if entry["delta"] > 0][:10]
            decreases = [entry for entry in changed_members if entry["delta"] < 0][:10]
            set_name = indexed_over[pos] if pos < len(indexed_over) else f"dim_{pos + 1}"
            dimensions.append({
                "position": pos,
                "set_name": set_name,
                "set_label": set_labels.get(set_name, ""),
                "top_increases": increases,
                "top_decreases": decreases,
            })

        if not dimensions:
            continue

        totals = family_totals.get(family, {"value_a": 0.0, "value_b": 0.0})
        summaries[family] = {
            "family_label": variable_labels.get(family, "") or family,
            "family_total_a": totals["value_a"],
            "family_total_b": totals["value_b"],
            "dimensions": dimensions,
        }

    return summaries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_solution_diff(
    store_a: EpochStore,
    store_b: EpochStore,
    qt1:     dict | None = None,
    force:   bool        = False,
) -> dict:
    """
    Compute QT2 — Solution Difference Analysis.

    Pure Python, zero solver calls.  Compares variable values, objective
    value, and constraint binding status between two epochs.

    The ``qt1`` argument is optional structural context (from
    ``compute_structural_diff``).  When provided, a brief
    ``structural_context`` note is added to the result indicating how many
    variable additions / removals are explained by index-set changes.

    The compact result is cached; subsequent calls for the same pair return
    the cached result unless ``force=True``.

    Args:
        store_a: EpochStore for the earlier (reference) epoch.
        store_b: EpochStore for the later (comparison) epoch.
        qt1:     QT1 result dict, or None.
        force:   Recompute even if a cached result exists.

    Returns:
        QT2 result dict — compact, suitable for LLM context injection.
    """
    if not force:
        cached = store_a.load_comparison(store_b.epoch_id, "qt2")
        if cached is not None:
            return cached

    sol_a  = store_a.get_solution()
    sol_b  = store_b.get_solution()
    struct_a = store_a.get_structure()
    struct_b = store_b.get_structure()
    vars_a = sol_a.get("variables",   {})
    vars_b = sol_b.get("variables",   {})
    cons_a = sol_a.get("constraints", {})
    cons_b = sol_b.get("constraints", {})
    obj_a  = sol_a.get("objective",   {})
    obj_b  = sol_b.get("objective",   {})

    # --- Objective change ---
    ov_a = _safe_float(obj_a.get("value"))
    ov_b = _safe_float(obj_b.get("value"))
    obj_delta = (ov_b - ov_a) if (ov_a is not None and ov_b is not None) else None
    obj_rel   = _rel_change(ov_a, ov_b)

    variable_labels = {
        family: (info.get("doc", "") or "").strip()
        for family, info in {
            **struct_a.get("variable_families", {}),
            **struct_b.get("variable_families", {}),
        }.items()
    }
    variable_metadata = {
        family: {
            "doc": (info.get("doc", "") or "").strip(),
            "indexed_over": list(info.get("indexed_over", [])),
            "type": info.get("type", ""),
        }
        for family, info in {
            **struct_a.get("variable_families", {}),
            **struct_b.get("variable_families", {}),
        }.items()
    }
    constraint_labels = {
        **struct_a.get("constraint_docs", {}),
        **struct_b.get("constraint_docs", {}),
    }
    set_labels = {
        **struct_a.get("index_set_docs", {}),
        **struct_b.get("index_set_docs", {}),
    }
    objective_label = (
        (struct_b.get("objective_doc", "") or "").strip()
        or (struct_a.get("objective_doc", "") or "").strip()
    )

    objective_change: dict = {
        "value_a":    ov_a,
        "value_b":    ov_b,
        "delta":      obj_delta,
        "rel_change": obj_rel,
        "sense":      obj_a.get("sense", ""),
        "status_a":   obj_a.get("status", ""),
        "status_b":   obj_b.get("status", ""),
        "label":      objective_label,
    }

    # --- Variable comparison ---
    all_var_names = sorted(set(vars_a) | set(vars_b))

    # churn[family] accumulates per-family stats
    churn: dict = {}
    allocation_rollups: dict = {}
    family_activity_totals: dict = {}
    top_candidates: list[dict] = []
    n_added = n_removed = n_changed = n_unchanged = 0

    for var_name in all_var_names:
        family = _family_of(var_name)
        if family not in churn:
            churn[family] = {
                "count_a":            0,
                "count_b":            0,
                "n_added":            0,
                "n_removed":          0,
                "n_changed":          0,
                "n_unchanged":        0,
                "_sum_abs_delta":     0.0,   # temp; removed before returning
                "binary_activations":  0,
                "binary_deactivations": 0,
            }
        fc = churn[family]

        va = vars_a.get(var_name)
        vb = vars_b.get(var_name)

        val_a = _safe_float(va.get("value")) if va else None
        val_b = _safe_float(vb.get("value")) if vb else None
        is_binary = (
            (va.get("is_binary", False) if va else False)
            or (vb.get("is_binary", False) if vb else False)
            or variable_metadata.get(family, {}).get("type", "") == "binary"
        )
        activity_a = _activity_value(val_a, is_binary)
        activity_b = _activity_value(val_b, is_binary)
        _update_allocation_rollups(
            allocation_rollups,
            family_activity_totals,
            family,
            var_name,
            variable_metadata.get(family, {}).get("indexed_over", []),
            activity_a,
            activity_b,
        )

        if va is not None:
            fc["count_a"] += 1
        if vb is not None:
            fc["count_b"] += 1

        # Variable appears only in B (added)
        if va is None and vb is not None:
            fc["n_added"] += 1
            n_added += 1
            if vb.get("is_binary") and val_b is not None and abs(val_b - 1.0) < Thresholds.BINDING_TOL:
                fc["binary_activations"] += 1
            continue

        # Variable appears only in A (removed)
        if vb is None and va is not None:
            fc["n_removed"] += 1
            n_removed += 1
            continue

        # Both present — compare values
        if val_a is None and val_b is None:
            fc["n_unchanged"] += 1
            n_unchanged += 1
            continue

        abs_delta = abs(val_b - val_a) if (val_a is not None and val_b is not None) else None

        if abs_delta is not None and abs_delta < Thresholds.VAR_CHANGE_TOL:
            fc["n_unchanged"] += 1
            n_unchanged += 1
            continue

        fc["n_changed"] += 1
        n_changed += 1
        if abs_delta is not None:
            fc["_sum_abs_delta"] += abs_delta

        # Binary activation tracking
        if is_binary and val_a is not None and val_b is not None:
            was_on = abs(val_a - 1.0) < Thresholds.BINDING_TOL
            now_on = abs(val_b - 1.0) < Thresholds.BINDING_TOL
            if not was_on and now_on:
                fc["binary_activations"] += 1
            elif was_on and not now_on:
                fc["binary_deactivations"] += 1

        delta = (val_b - val_a) if (val_a is not None and val_b is not None) else None
        top_candidates.append({
            "variable":   var_name,
            "family":     family,
            "family_label": variable_labels.get(family, "") or family,
            "value_a":    val_a,
            "value_b":    val_b,
            "delta":      delta,
            "abs_delta":  abs_delta,
            "rel_change": _rel_change(val_a, val_b),
        })

    # Finalise per-family stats
    churn_summary: dict = {}
    relocations_estimate_total = 0
    for family, fc in churn.items():
        n_ch = fc["n_changed"]
        avg  = fc["_sum_abs_delta"] / n_ch if n_ch > 0 else 0.0
        relocations_estimate = min(fc["binary_activations"], fc["binary_deactivations"])
        relocations_estimate_total += relocations_estimate
        churn_summary[family] = {k: v for k, v in fc.items() if k != "_sum_abs_delta"}
        churn_summary[family]["avg_abs_delta"] = avg
        churn_summary[family]["relocations_estimate"] = relocations_estimate
        churn_summary[family]["family_label"] = variable_labels.get(family, "") or family
        churn_summary[family]["indexed_over"] = variable_metadata.get(family, {}).get("indexed_over", [])
        churn_summary[family]["type"] = variable_metadata.get(family, {}).get("type", "")

    # Top-N individual changes sorted by abs_delta descending
    top_changes = sorted(
        top_candidates,
        key=lambda x: (x["abs_delta"] or 0.0),
        reverse=True,
    )[:_TOP_N]
    allocation_summaries = _build_allocation_summaries(
        allocation_rollups,
        family_activity_totals,
        variable_metadata,
        variable_labels,
        set_labels,
    )

    # --- Constraint binding changes ---
    all_con_names = sorted(set(cons_a) | set(cons_b))
    became_binding:    list = []
    became_nonbinding: list = []
    added_constraints:   list = []
    removed_constraints: list = []

    for con_name in all_con_names:
        ca = cons_a.get(con_name)
        cb = cons_b.get(con_name)
        if ca is None:
            added_constraints.append(con_name)
            continue
        if cb is None:
            removed_constraints.append(con_name)
            continue
        bind_a = ca.get("is_binding")
        bind_b = cb.get("is_binding")
        if bind_a is False and bind_b is True:
            became_binding.append(con_name)
        elif bind_a is True and bind_b is False:
            became_nonbinding.append(con_name)

    binding_changes: dict = {
        "became_binding":     became_binding,
        "became_nonbinding":  became_nonbinding,
        "added_constraints":  added_constraints,
        "removed_constraints": removed_constraints,
        "constraint_labels":  constraint_labels,
    }

    # --- Structural context (optional, from QT1) ---
    structural_context: dict | None = None
    if qt1 is not None:
        n_elem_add = qt1.get("summary", {}).get("n_set_element_additions", 0)
        n_elem_rem = qt1.get("summary", {}).get("n_set_element_removals", 0)
        structural_context = {
            "n_index_element_additions": n_elem_add,
            "n_index_element_removals":  n_elem_rem,
            "note": (
                f"{n_added} variable addition(s) and {n_removed} removal(s) "
                f"expected from {n_elem_add} index element addition(s) and "
                f"{n_elem_rem} removal(s) identified in QT1."
            ),
        }

    # --- Summary ---
    n_comparable = n_changed + n_unchanged
    churn_rate   = round(n_changed / n_comparable, 4) if n_comparable > 0 else 0.0

    summary: dict = {
        "n_total_variables_a":   len(vars_a),
        "n_total_variables_b":   len(vars_b),
        "n_added_variables":     n_added,
        "n_removed_variables":   n_removed,
        "n_changed_variables":   n_changed,
        "n_unchanged_variables": n_unchanged,
        "overall_churn_rate":    churn_rate,
        "objective_delta":       obj_delta,
        "objective_rel_change":  obj_rel,
        "n_became_binding":      len(became_binding),
        "n_became_nonbinding":   len(became_nonbinding),
        "n_relocations_estimate": relocations_estimate_total,
    }

    result: dict = {
        "epoch_a":            store_a.epoch_id,
        "epoch_b":            store_b.epoch_id,
        "computed_at":        datetime.now().isoformat(),
        "objective_change":   objective_change,
        "churn_summary":      churn_summary,
        "allocation_summaries": allocation_summaries,
        "top_changes":        top_changes,
        "binding_changes":    binding_changes,
        "labels": {
            "variables": variable_labels,
            "variable_metadata": variable_metadata,
            "constraints": constraint_labels,
            "sets": set_labels,
            "objective": objective_label,
        },
        "summary":            summary,
    }
    if structural_context is not None:
        result["structural_context"] = structural_context

    store_a.save_comparison(store_b.epoch_id, "qt2", result)
    return result


def get_variable_family_details(
    store_a:       EpochStore,
    store_b:       EpochStore,
    family_prefix: str,
) -> dict:
    """
    Return a full variable-by-variable comparison for a single family.

    Used for Tier-4 drill-down when the user asks about a specific variable
    family (e.g. "how did x (food purchase) variables change?").

    Does NOT require QT2 to have been computed first — reads solution.json
    directly from both stores.

    Args:
        store_a:       EpochStore for the earlier epoch.
        store_b:       EpochStore for the later epoch.
        family_prefix: Variable family name, e.g. 'x', 'cost'.

    Returns:
        {
          "family":    family_prefix,
          "epoch_a":   store_a.epoch_id,
          "epoch_b":   store_b.epoch_id,
          "variables": {
            var_name: {
              "value_a": ..., "value_b": ...,
              "delta": ..., "rel_change": ...,
              "status": "added" | "removed" | "changed" | "unchanged",
            }
          },
          "summary": {"n_added", "n_removed", "n_changed", "n_unchanged"},
        }
    """
    vars_a = store_a.get_variable_family(family_prefix)
    vars_b = store_b.get_variable_family(family_prefix)
    all_names = sorted(set(vars_a) | set(vars_b))

    detail: dict = {}
    n_added = n_removed = n_changed = n_unchanged = 0

    for var_name in all_names:
        va = vars_a.get(var_name)
        vb = vars_b.get(var_name)

        val_a = _safe_float(va["value"]) if va else None
        val_b = _safe_float(vb["value"]) if vb else None

        if va is None:
            status = "added"
            n_added += 1
        elif vb is None:
            status = "removed"
            n_removed += 1
        else:
            abs_delta = (
                abs(val_b - val_a)
                if (val_a is not None and val_b is not None)
                else None
            )
            if abs_delta is not None and abs_delta < Thresholds.VAR_CHANGE_TOL:
                status = "unchanged"
                n_unchanged += 1
            else:
                status = "changed"
                n_changed += 1

        delta = (val_b - val_a) if (val_a is not None and val_b is not None) else None

        detail[var_name] = {
            "value_a":    val_a,
            "value_b":    val_b,
            "delta":      delta,
            "rel_change": _rel_change(val_a, val_b),
            "status":     status,
        }

    return {
        "family":    family_prefix,
        "epoch_a":   store_a.epoch_id,
        "epoch_b":   store_b.epoch_id,
        "variables": detail,
        "summary": {
            "n_added":     n_added,
            "n_removed":   n_removed,
            "n_changed":   n_changed,
            "n_unchanged": n_unchanged,
        },
    }
