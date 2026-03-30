"""
QT4 — Solution Attribution Analysis.

Identifies which parameter shifts between epochs are associated with observed
solution changes, by examining shadow prices (dual values) from incumbent LPs.

Note on language:  This analysis identifies *associations* between parameter
shifts and solution differences.  It does not assert causal relationships.
The term "attribution" describes the structured linkage of observable changes
to the shadow price landscape, not mechanistic causation.

Algorithm:
  Phase 1 — Incumbent LP dual extraction (per epoch, cached in duals.json):
    - Pure LP models: duals already in solution.json (no new solver call).
    - MIP models: fix all binary/integer variables to their MIP optimal values,
      solve the resulting LP, extract constraint shadow prices.  1 Gurobi call
      per MIP epoch; result cached to duals.json so repeated QT4 runs are free.

  Phase 2 — Attribution linkage (pure Python, 0 solver calls):
    - Parameter shifts:   from QT1, filtered to changes >= TRIGGER_THRESHOLD.
    - Binary flips:       variables that toggled between 0 and 1.
    - Bottleneck evolution: constraints that gained, lost, or kept significant
      shadow prices between epochs (classified as relieved / new / persistent).
    - Continuous adjustments: non-binary variable deltas from QT2 top_changes.
    - Binary-to-constraint attribution: for each flip, identify which constraint
      in Epoch A carried the highest shadow price for that variable's family.

  Phase 3 — Narrative synthesis (pure Python, 0 solver calls):
    - Generates a structured 5-field text summary describing the main
      parameter shift, prior binding state, solution adaptation, outcome,
      and new binding state — without asserting causal direction.

Solver calls:  0 for LP epochs (duals already available).
               1 per MIP epoch (incumbent LP, cached after first run).
"""

from __future__ import annotations

from datetime import datetime

import pyomo.environ as pe
from pyomo.opt import SolverFactory
from loguru import logger

from rh_comparison.data_store.epoch_store import EpochStore
from rh_comparison.config.rh_constants import Thresholds

# Shadow price threshold below which a constraint is treated as non-binding.
# Uses the centralized constant from rh_constants.Thresholds.
_DUAL_SIG_TOL = Thresholds.DUAL_SIGNIFICANCE_TOL


# ---------------------------------------------------------------------------
# Phase 1: Incumbent LP dual extraction
# ---------------------------------------------------------------------------

def _compute_incumbent_lp_duals(
    model: pe.ConcreteModel,
    sol:   dict,
) -> dict:
    """
    Fix all binary/integer variables to their MIP optimal values and solve
    the resulting LP to extract constraint shadow prices.

    Returns: {constraint_name: dual_value} for all constraints with a dual.
    """
    vars_sol = sol.get("variables", {})

    # Fix binary and integer variables to their rounded MIP solution values
    for var in model.component_objects(pe.Var, active=True):
        for idx in var:
            v = var[idx]
            if v.is_binary() or v.is_integer():
                name = pe.name(v)
                if name in vars_sol:
                    val = vars_sol[name].get("value")
                    if val is not None:
                        v.fix(int(round(float(val))))

    if not hasattr(model, "dual"):
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)

    solver = SolverFactory("gurobi")
    results = solver.solve(model, tee=False)
    tc = str(results.solver.termination_condition)
    if tc != "optimal":
        logger.warning(f"[QT4] Incumbent LP returned '{tc}'. Shadow prices unavailable.")
        return {}

    duals: dict = {}
    for con in model.component_objects(pe.Constraint, active=True):
        for idx in con:
            name = pe.name(con[idx])
            try:
                raw = model.dual.get(con[idx])
                if raw is not None:
                    duals[name] = float(raw)
            except Exception:
                pass
    return duals


def _get_or_compute_incumbent_lp_duals(store: EpochStore) -> dict:
    """
    Return constraint shadow prices for an epoch via incumbent LP.

    For pure LP epochs:   reads constraint duals from solution.json directly —
                          no new solver call required.
    For MIP epochs:       checks duals.json cache first; computes and caches if
                          missing (1 Gurobi call).
    """
    sol = store.get_solution()

    if not sol.get("is_mip", True):
        # Pure LP — duals are already populated in solution.json
        duals: dict = {}
        for con_name, con_data in sol["constraints"].items():
            d = con_data.get("dual")
            if d is not None:
                duals[con_name] = float(d)
        logger.info(f"[QT4] Epoch '{store.epoch_id}': LP duals read from solution.json "
                    f"({len(duals)} values).")
        return duals

    # MIP — check duals.json cache first
    cached = store.get_duals()
    if cached is not None:
        d = {k: v for k, v in cached.items() if k != "epoch_id"}
        logger.info(f"[QT4] Epoch '{store.epoch_id}': MIP duals loaded from duals.json cache "
                    f"({len(d)} values).")
        return d

    # Not cached — solve incumbent LP
    logger.info(f"[QT4] Epoch '{store.epoch_id}': solving incumbent LP for MIP shadow prices...")
    model = store.get_pyomo_model()
    duals = _compute_incumbent_lp_duals(model, sol)
    store.save_duals(duals)
    logger.info(f"[QT4] Epoch '{store.epoch_id}': incumbent LP duals cached ({len(duals)} values).")
    return duals


# ---------------------------------------------------------------------------
# Phase 2: Attribution linkage helpers
# ---------------------------------------------------------------------------

def _identify_parameter_shifts(qt1: dict) -> list[dict]:
    """
    Return parameter changes from QT1 that exceed the trigger threshold.
    Sorted by absolute relative change (largest first).
    """
    threshold = Thresholds.TRIGGER_THRESHOLD
    top = qt1.get("parameter_changes", {}).get("top_changes", [])
    shifts = []
    for entry in top:
        rel = entry.get("rel_change")
        # None means 0→nonzero (always significant); otherwise check threshold
        if rel is None or abs(rel) >= threshold:
            shifts.append({
                "family":     entry["family"],
                "index":      entry["index"],
                "value_a":    entry["value_a"],
                "value_b":    entry["value_b"],
                "delta":      entry["delta"],
                "rel_change": rel,
                "magnitude":  entry["magnitude"],
            })
    return shifts


def _identify_binary_flips(
    store_a: EpochStore,
    store_b: EpochStore,
) -> list[dict]:
    """
    Identify binary variables that toggled between 0 and 1 between epochs.
    Reads solution.json from both stores to check is_binary + value.
    """
    sol_a  = store_a.get_solution()
    sol_b  = store_b.get_solution()
    vars_a = sol_a.get("variables", {})
    vars_b = sol_b.get("variables", {})

    flips: list[dict] = []
    all_names = sorted(set(vars_a) | set(vars_b))
    tol = Thresholds.BINDING_TOL

    for name in all_names:
        va = vars_a.get(name)
        vb = vars_b.get(name)
        if va is None or vb is None:
            continue
        if not (va.get("is_binary") or vb.get("is_binary")):
            continue

        val_a = va.get("value")
        val_b = vb.get("value")
        if val_a is None or val_b is None:
            continue

        val_a_f, val_b_f = float(val_a), float(val_b)
        was_on = abs(val_a_f - 1.0) < tol
        now_on = abs(val_b_f - 1.0) < tol

        if was_on != now_on:
            flips.append({
                "variable":  name,
                "family":    name.split("[")[0],
                "direction": "0->1" if (not was_on and now_on) else "1->0",
                "value_a":   val_a_f,
                "value_b":   val_b_f,
            })

    return flips


def _categorize_constraints(
    duals_a:     dict,
    duals_b:     dict,
    sig_tol:     float = _DUAL_SIG_TOL,
) -> dict:
    """
    Classify every constraint into one of:
      - relieved:      significant dual in A, non-significant in B
      - new_bottlenecks: non-significant in A, significant in B
      - persistent:    significant in both A and B
      (non-significant in both → ignored)

    A dual is "significant" if |dual| > sig_tol.
    """
    all_cons = sorted(set(duals_a) | set(duals_b))

    relieved:       list[dict] = []
    new_bottlenecks: list[dict] = []
    persistent:     list[dict] = []

    for con_name in all_cons:
        da = duals_a.get(con_name, 0.0)
        db = duals_b.get(con_name, 0.0)
        sig_a = abs(da) > sig_tol
        sig_b = abs(db) > sig_tol

        entry = {"constraint": con_name, "dual_a": da, "dual_b": db}
        if sig_a and not sig_b:
            relieved.append(entry)
        elif not sig_a and sig_b:
            new_bottlenecks.append(entry)
        elif sig_a and sig_b:
            persistent.append(entry)

    # Sort each group by absolute dual magnitude (most significant first)
    for lst in (relieved, new_bottlenecks, persistent):
        lst.sort(key=lambda x: max(abs(x["dual_a"]), abs(x["dual_b"])), reverse=True)

    return {
        "relieved":       relieved,
        "new_bottlenecks": new_bottlenecks,
        "persistent":     persistent,
    }


def _map_flips_to_constraints(
    binary_flips: list[dict],
    duals_a:      dict,
) -> list[dict]:
    """
    For each binary flip, find which constraint in Epoch A carried the largest
    shadow price involving the same index element.

    This linkage is heuristic: if a binary variable x[wheat] flipped, we
    extract the index element ("wheat") and look for the constraint with the
    highest |shadow price| whose name contains that same index element
    (e.g. nb[wheat] or limit[wheat]).  This is more robust than matching by
    variable family prefix, since variable and constraint families rarely
    share the same name.
    """
    enriched: list[dict] = []
    for flip in binary_flips:
        var_name = flip["variable"]
        # Extract the index element: "x[wheat]" → "wheat", "y[A,B]" → "A,B"
        br = var_name.find("[")
        idx_element = var_name[br + 1 : -1] if br != -1 else ""

        top_con = None
        top_dual = 0.0
        if idx_element:
            # Find all constraints whose name contains the same index element
            # e.g. for idx_element="wheat", match "nb[wheat]", "limit[wheat]"
            for con_name, dual_val in duals_a.items():
                if abs(dual_val) <= _DUAL_SIG_TOL:
                    continue
                con_br = con_name.find("[")
                con_idx = con_name[con_br + 1 : -1] if con_br != -1 else ""
                if con_idx == idx_element and abs(dual_val) > abs(top_dual):
                    top_con = con_name
                    top_dual = dual_val

        enriched.append({
            **flip,
            "attributed_constraint": top_con,
            "attributed_dual_a":     top_dual if top_con else None,
        })
    return enriched


def _extract_continuous_adjustments(qt2: dict) -> list[dict]:
    """
    Extract non-binary variable changes from QT2 top_changes.
    Returns list sorted by abs_delta descending.
    """
    adjustments: list[dict] = []
    for entry in qt2.get("top_changes", []):
        # Skip entries that look like binary flips (value close to 0 or 1)
        va = entry.get("value_a")
        vb = entry.get("value_b")
        if va is not None and vb is not None:
            if (abs(va) < 1.5 and abs(vb) < 1.5 and
                    abs(va - round(va)) < 0.1 and abs(vb - round(vb)) < 0.1):
                continue   # likely binary/integer, skip
        adjustments.append({
            "variable":   entry["variable"],
            "family":     entry["family"],
            "value_a":    entry.get("value_a"),
            "value_b":    entry.get("value_b"),
            "delta":      entry.get("delta"),
            "rel_change": entry.get("rel_change"),
        })
    return adjustments[:10]  # top 10 for LLM context


# ---------------------------------------------------------------------------
# Phase 3: Narrative synthesis
# ---------------------------------------------------------------------------

def _synthesize_narrative(
    shifts:                list[dict],
    bottleneck_evolution:  dict,
    binary_flips_enriched: list[dict],
    continuous_adjustments: list[dict],
    label_a:               str = "the earlier period",
    label_b:               str = "the later period",
) -> dict:
    """
    Generate a structured 5-field narrative summary.

    Fields (using non-causal language throughout):
      parameter_shift:     What changed in the input data.
      prior_state:         Which rules were exactly met in the earlier period.
      solution_adaptation: What the plan did differently in the later period.
      outcome:             What the overall result looks like.
      new_state:           Which rules are exactly met in the later period.
    """
    parts: dict = {}

    # parameter_shift
    if shifts:
        top = shifts[0]
        rel_str = (
            f"{top['rel_change']*100:+.1f}%"
            if top["rel_change"] is not None
            else "from zero to a positive value"
        )
        parts["parameter_shift"] = (
            f"The input '{top['family']}[{top['index']}]' shifted "
            f"{rel_str} between {label_a} and {label_b} "
            f"(from {top['value_a']} to {top['value_b']})."
        )
        if len(shifts) > 1:
            other = shifts[1:]
            part2 = "; ".join(
                f"{s['family']}[{s['index']}] {s['rel_change']*100:+.1f}%"
                if s["rel_change"] is not None
                else f"{s['family']}[{s['index']}] (0→nonzero)"
                for s in other[:3]
            )
            parts["parameter_shift"] += f"  Additional shifts: {part2}."
    else:
        parts["parameter_shift"] = "No input shifts above the significance threshold were detected."

    # prior_state
    relieved   = bottleneck_evolution.get("relieved", [])
    persistent = bottleneck_evolution.get("persistent", [])
    binding_in_a = relieved + persistent
    if binding_in_a:
        names = ", ".join(f"'{c['constraint']}'" for c in binding_in_a[:3])
        parts["prior_state"] = (
            f"In {label_a}, the following rule(s) were exactly met "
            f"with significant sensitivity: {names}."
        )
    else:
        parts["prior_state"] = f"No rules with significant sensitivity were identified in {label_a}."

    # solution_adaptation
    if binary_flips_enriched:
        flip_strs = []
        for flip in binary_flips_enriched[:2]:
            d_str = "0→1" if flip["direction"] == "0->1" else "1→0"
            a_con = flip.get("attributed_constraint")
            if a_con:
                flip_strs.append(f"'{flip['variable']}' toggled {d_str} "
                                  f"(associated with the '{a_con}' rule)")
            else:
                flip_strs.append(f"'{flip['variable']}' toggled {d_str}")
        parts["solution_adaptation"] = (
            f"Yes/no decision changes in {label_b}: "
            + "; ".join(flip_strs) + "."
        )
    elif continuous_adjustments:
        top_adj = continuous_adjustments[0]
        rel_str = (
            f" ({top_adj['rel_change']*100:+.1f}%)"
            if top_adj["rel_change"] is not None else ""
        )
        parts["solution_adaptation"] = (
            f"Decision '{top_adj['variable']}' adjusted from "
            f"{top_adj['value_a']} to {top_adj['value_b']}{rel_str}."
        )
        if len(continuous_adjustments) > 1:
            count = len(continuous_adjustments) - 1
            parts["solution_adaptation"] += (
                f"  {count} additional decision(s) also changed."
            )
    else:
        parts["solution_adaptation"] = (
            "No significant decision changes were observed."
        )

    # outcome
    new_bns = bottleneck_evolution.get("new_bottlenecks", [])
    if new_bns or continuous_adjustments:
        obj_note = ""
        if continuous_adjustments:
            deltas = [c["delta"] for c in continuous_adjustments if c["delta"] is not None]
            if deltas:
                avg_d = sum(abs(d) for d in deltas) / len(deltas)
                obj_note = f"  Average absolute change across top adjustments: {avg_d:.4f}."
        parts["outcome"] = (
            f"{len(continuous_adjustments)} decision adjustment(s) observed "
            f"between {label_a} and {label_b}.{obj_note}"
        )
    else:
        parts["outcome"] = "No significant decision changes were detected."

    # new_state
    if new_bns:
        names = ", ".join(f"'{c['constraint']}'" for c in new_bns[:3])
        parts["new_state"] = (
            f"In {label_b}, the following rule(s) became newly exactly met: {names}."
        )
    elif persistent:
        names = ", ".join(f"'{c['constraint']}'" for c in persistent[:3])
        parts["new_state"] = (
            f"The following rule(s) remain exactly met in {label_b}: {names}."
        )
    else:
        parts["new_state"] = f"No newly exactly-met rules were identified in {label_b}."

    return parts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_attribution_analysis(
    store_a: EpochStore,
    store_b: EpochStore,
    qt1:     dict,
    qt2:     dict,
    force:   bool = False,
) -> dict:
    """
    Compute QT4 — Solution Attribution Analysis.

    Associates parameter shifts (from QT1) with solution changes (from QT2)
    via shadow prices extracted from incumbent LPs.  Structures findings into
    parameter shifts, bottleneck evolution, binary flips, continuous
    adjustments, and a narrative summary.

    For pure LP epochs the analysis is free — duals are already available in
    solution.json.  For MIP epochs, at most 1 Gurobi call per epoch is needed
    to compute the incumbent LP, and the result is cached in duals.json.

    Args:
        store_a: EpochStore for the earlier (reference) epoch.
        store_b: EpochStore for the later (comparison) epoch.
        qt1:     QT1 result dict (from compute_structural_diff).
        qt2:     QT2 result dict (from compute_solution_diff).
        force:   Recompute even if a cached result exists.

    Returns:
        QT4 result dict — fully self-contained, suitable for LLM injection.
    """
    if not force:
        cached = store_a.load_comparison(store_b.epoch_id, "qt4")
        if cached is not None:
            return cached

    meta_a = store_a.get_meta()
    meta_b = store_b.get_meta()
    struct_a = store_a.get_structure()
    struct_b = store_b.get_structure()
    parameter_labels = {
        family: (info.get("doc", "") or "").strip()
        for family, info in {
            **struct_a.get("param_families", {}),
            **struct_b.get("param_families", {}),
        }.items()
    }
    variable_labels = {
        family: (info.get("doc", "") or "").strip()
        for family, info in {
            **struct_a.get("variable_families", {}),
            **struct_b.get("variable_families", {}),
        }.items()
    }
    constraint_labels = {
        **struct_a.get("constraint_docs", {}),
        **struct_b.get("constraint_docs", {}),
    }

    # --- Phase 1: Incumbent LP duals ---
    logger.info("[QT4] Extracting incumbent LP duals for both epochs...")
    duals_a = _get_or_compute_incumbent_lp_duals(store_a)
    duals_b = _get_or_compute_incumbent_lp_duals(store_b)
    logger.info(f"[QT4] Duals: A={len(duals_a)}, B={len(duals_b)}")

    # --- Phase 2: Attribution linkage ---
    parameter_shifts       = _identify_parameter_shifts(qt1)
    binary_flips           = _identify_binary_flips(store_a, store_b)
    bottleneck_evolution   = _categorize_constraints(duals_a, duals_b)
    binary_flips_enriched  = _map_flips_to_constraints(
        binary_flips, duals_a
    )
    continuous_adjustments = _extract_continuous_adjustments(qt2)

    # --- Phase 3: Narrative ---
    label_a = meta_a.get("label", meta_a.get("epoch_id", "the earlier period"))
    label_b = meta_b.get("label", meta_b.get("epoch_id", "the later period"))
    narrative = _synthesize_narrative(
        parameter_shifts,
        bottleneck_evolution,
        binary_flips_enriched,
        continuous_adjustments,
        label_a=label_a,
        label_b=label_b,
    )

    # --- Summary ---
    summary: dict = {
        "n_parameter_shifts":       len(parameter_shifts),
        "n_relieved_bottlenecks":   len(bottleneck_evolution["relieved"]),
        "n_new_bottlenecks":        len(bottleneck_evolution["new_bottlenecks"]),
        "n_persistent_bottlenecks": len(bottleneck_evolution["persistent"]),
        "n_binary_flips":           len(binary_flips),
        "n_continuous_adjustments": len(continuous_adjustments),
        "has_binary_decisions":     bool(binary_flips),
        "lp_duals_available_a":     bool(duals_a),
        "lp_duals_available_b":     bool(duals_b),
    }

    result: dict = {
        "epoch_a":                 meta_a["epoch_id"],
        "epoch_b":                 meta_b["epoch_id"],
        "computed_at":             datetime.now().isoformat(),
        "parameter_shifts":        parameter_shifts,
        "bottleneck_evolution":    bottleneck_evolution,
        "binary_flips":            binary_flips_enriched,
        "continuous_adjustments":  continuous_adjustments,
        "narrative":               narrative,
        "labels": {
            "parameters": parameter_labels,
            "variables": variable_labels,
            "constraints": constraint_labels,
        },
        "summary":                 summary,
    }

    store_a.save_comparison(store_b.epoch_id, "qt4", result)
    logger.info(f"[QT4] Done. shifts={len(parameter_shifts)}, "
                f"flips={len(binary_flips)}, "
                f"relieved={len(bottleneck_evolution['relieved'])}, "
                f"new_bns={len(bottleneck_evolution['new_bottlenecks'])}")
    return result
