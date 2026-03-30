"""
QT3 — Backward Compatibility Assessment.

This module follows the paper-aligned interpretation of backward compatibility:
freeze every overlapping decision from the earlier plan onto the later model,
leave genuinely new decisions free, and re-solve the later model.

If the re-solve is infeasible, diagnose that fixed-plan model directly with
solver-backed IIS extraction and slack minimization. We do not evaluate
violations at an artificial "free variables = lower bound" point.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from datetime import datetime

import gurobipy as gp
import pyomo.environ as pe
from loguru import logger
from pyomo.opt import SolverFactory

from rh_comparison.config.rh_constants import Thresholds
from rh_comparison.data_store.epoch_store import EpochStore

_FIX_BLOCK_NAME = "_rh_plan_lock"
_SLACK_BLOCK_NAME = "_rh_cover_relax"


def _safe_value(expr) -> float | None:
    if expr is None:
        return None
    try:
        return float(pe.value(expr))
    except Exception:
        return None


def _strip_fix_constraints(names: list[str]) -> list[str]:
    return [name for name in names if not name.startswith(f"{_FIX_BLOCK_NAME}.")]


def _constraint_name_map(model: pe.ConcreteModel) -> dict[str, pe.ConstraintData]:
    return {
        pe.name(con[idx]): con[idx]
        for con in model.component_objects(pe.Constraint, active=True)
        for idx in con
        if con[idx].active
    }


def _constraint_docs(store_b: EpochStore) -> dict[str, str]:
    return store_b.get_structure().get("constraint_docs", {})


def _business_label(name: str, docs: dict[str, str]) -> str:
    family = name.split("[")[0]
    return docs.get(family, "").strip() or name


def _add_overlap_fix_constraints(
    model_b: pe.ConcreteModel,
    sol_a: dict,
) -> tuple[int, set[str]]:
    """
    Add explicit equality constraints tying Epoch B variables to Epoch A values.

    Using explicit constraints instead of `.fix()` keeps the compatibility logic
    visible to IIS extraction and avoids mixed bound/conflict handling.
    """
    if hasattr(model_b, _FIX_BLOCK_NAME):
        model_b.del_component(getattr(model_b, _FIX_BLOCK_NAME))

    vars_a = sol_a.get("variables", {})
    block = pe.Block()
    setattr(model_b, _FIX_BLOCK_NAME, block)
    block.equalities = pe.ConstraintList()

    n_fixed = 0
    fixed_var_names: set[str] = set()

    for var in model_b.component_objects(pe.Var, active=True):
        for idx in var:
            v = var[idx]
            name = pe.name(v)
            if name not in vars_a:
                continue
            val = vars_a[name].get("value")
            if val is None:
                continue
            block.equalities.add(v == float(val))
            fixed_var_names.add(name)
            n_fixed += 1

    return n_fixed, fixed_var_names


def _solve_model(model: pe.ConcreteModel, *, time_limit: int = 120):
    solver = SolverFactory("gurobi")
    solver.options["TimeLimit"] = time_limit
    return solver.solve(model, tee=False)


def _write_symbolic_lp(model: pe.ConcreteModel) -> str:
    fd, lp_path = tempfile.mkstemp(prefix="rh_qt3_", suffix=".lp")
    os.close(fd)
    model.write(lp_path, io_options={"symbolic_solver_labels": True})
    return lp_path


def _compute_iis_constraints(model: pe.ConcreteModel) -> list[str]:
    """
    Compute a constraint IIS with symbolic names using gurobipy directly.
    """
    lp_path = _write_symbolic_lp(model)
    try:
        gp_model = gp.read(lp_path)
        gp_model.Params.OutputFlag = 0
        gp_model.Params.DualReductions = 0
        gp_model.optimize()

        if gp_model.Status not in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
            return []

        gp_model.computeIIS()
        return sorted({
            constr.ConstrName
            for constr in gp_model.getConstrs()
            if constr.IISConstr
        })
    finally:
        try:
            os.remove(lp_path)
        except OSError:
            pass


def _iterative_iis_rounds(model: pe.ConcreteModel, max_rounds: int = 2) -> list[dict]:
    """
    Compute up to `max_rounds` IIS sets, relaxing each detected business IIS
    before attempting to discover another independent conflict.
    """
    rounds: list[dict] = []
    working_model = model.clone()

    for round_number in range(1, max_rounds + 1):
        iis_names = _compute_iis_constraints(working_model)
        if not iis_names:
            break

        business_constraints = _strip_fix_constraints(iis_names)
        rounds.append({
            "round": round_number,
            "constraints": business_constraints,
            "raw_iis": iis_names,
        })

        if not business_constraints or round_number == max_rounds:
            break

        constraint_map = _constraint_name_map(working_model)
        relaxed_any = False
        for name in business_constraints:
            con = constraint_map.get(name)
            if con is not None and con.active:
                con.deactivate()
                relaxed_any = True
        if not relaxed_any:
            break

        results = _solve_model(working_model, time_limit=60)
        tc = str(results.solver.termination_condition)
        if tc in ("optimal", "feasible", "locallyOptimal"):
            break

    return rounds


def _greedy_minimal_cover(iis_rounds: list[dict]) -> list[str]:
    uncovered = [set(round_info.get("constraints", [])) for round_info in iis_rounds if round_info.get("constraints")]
    cover: list[str] = []

    while uncovered:
        counts = Counter(name for iis_set in uncovered for name in iis_set)
        if not counts:
            break
        best_name = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        cover.append(best_name)
        uncovered = [iis_set for iis_set in uncovered if best_name not in iis_set]

    return cover


def _add_relaxed_constraint(
    block: pe.Block,
    con: pe.ConstraintData,
    slack_var: pe.VarData,
) -> None:
    lb = _safe_value(con.lower)
    ub = _safe_value(con.upper)
    body = con.body

    if lb is not None and ub is not None and abs(lb - ub) <= Thresholds.BINDING_TOL:
        block.relaxed_constraints.add(body <= ub + slack_var)
        block.relaxed_constraints.add(body >= lb - slack_var)
    elif ub is not None:
        block.relaxed_constraints.add(body <= ub + slack_var)
    elif lb is not None:
        block.relaxed_constraints.add(body >= lb - slack_var)


def _solve_required_slacks(
    base_model: pe.ConcreteModel,
    minimal_cover: list[str],
) -> dict[str, dict]:
    if not minimal_cover:
        return {}

    slack_model = base_model.clone()
    constraint_map = _constraint_name_map(slack_model)

    if hasattr(slack_model, _SLACK_BLOCK_NAME):
        slack_model.del_component(getattr(slack_model, _SLACK_BLOCK_NAME))

    block = pe.Block()
    setattr(slack_model, _SLACK_BLOCK_NAME, block)
    block.cover_index = pe.Set(initialize=minimal_cover, ordered=True)
    block.cover_slack = pe.Var(block.cover_index, within=pe.NonNegativeReals)
    block.relaxed_constraints = pe.ConstraintList()

    used_constraints: list[str] = []
    for name in minimal_cover:
        con = constraint_map.get(name)
        if con is None:
            continue
        con.deactivate()
        _add_relaxed_constraint(block, con, block.cover_slack[name])
        used_constraints.append(name)

    for obj in slack_model.component_objects(pe.Objective, active=True):
        obj.deactivate()
    slack_model._rh_cover_objective = pe.Objective(
        expr=sum(block.cover_slack[name] for name in used_constraints),
        sense=pe.minimize,
    )

    results = _solve_model(slack_model, time_limit=120)
    tc = str(results.solver.termination_condition)
    if tc not in ("optimal", "feasible", "locallyOptimal"):
        return {}

    return {
        name: {
            "required_slack": round(float(pe.value(block.cover_slack[name])), 8),
        }
        for name in used_constraints
    }


def _opportunity_gap(forced_obj: float | None, optimal_b: float | None, sense: str) -> tuple[float | None, float | None]:
    if forced_obj is None or optimal_b is None:
        return None, None

    if sense == "maximize":
        gap_abs = max(0.0, optimal_b - forced_obj)
    else:
        gap_abs = max(0.0, forced_obj - optimal_b)

    gap_rel = (gap_abs / abs(optimal_b)) if abs(optimal_b) > 1e-10 else None
    return gap_abs, gap_rel


def _business_summary(
    feasible: bool,
    label_a: str,
    label_b: str,
    gap_rel: float | None = None,
    minimal_cover: list[str] | None = None,
    docs: dict[str, str] | None = None,
) -> str:
    if feasible:
        if gap_rel is None or gap_rel < 1e-4:
            return (
                f"Yes. The plan from {label_a} still works in {label_b} and is "
                "effectively as good as the best plan available in the later run."
            )
        pct = gap_rel * 100
        return (
            f"Yes. The plan from {label_a} still works in {label_b}, but keeping it "
            f"would leave about {pct:.2f}% of the available improvement on the table."
        )

    docs = docs or {}
    cover = minimal_cover or []
    if not cover:
        return (
            f"No. The plan from {label_a} cannot be reused in {label_b}; the current "
            "rules conflict with the older plan."
        )

    top_labels = ", ".join(_business_label(name, docs) for name in cover[:2])
    return (
        f"No. The plan from {label_a} cannot be reused in {label_b}. The main blocking "
        f"rule(s) are {top_labels}."
    )


def assess_backward_compat(
    store_a: EpochStore,
    store_b: EpochStore,
    qt1: dict | None = None,
    qt2: dict | None = None,
    force: bool = False,
) -> dict:
    """
    Compute QT3 — Backward Compatibility Assessment.

    Interpretation used throughout RH mode:
      - Overlapping decisions from the earlier plan are frozen onto the later model.
      - Genuinely new decisions in the later model remain free.
      - Feasibility answers whether the older plan can still be used.
    """
    if not force:
        cached = store_a.load_comparison(store_b.epoch_id, "qt3")
        if cached is not None:
            return cached

    meta_a = store_a.get_meta()
    meta_b = store_b.get_meta()
    sol_a = store_a.get_solution()
    docs = _constraint_docs(store_b)

    logger.info("[QT3] Loading later-epoch model for backward compatibility check...")
    model_b = store_b.get_pyomo_model()

    n_fixed, fixed_var_names = _add_overlap_fix_constraints(model_b, sol_a)
    n_total_b = sum(1 for var in model_b.component_objects(pe.Var, active=True) for _ in var)
    n_free = n_total_b - n_fixed
    logger.info(f"[QT3] Added {n_fixed} overlap-fix constraints; {n_free} later-only variables remain free.")

    results = _solve_model(model_b, time_limit=120)
    tc = str(results.solver.termination_condition)
    feasible = tc in ("optimal", "feasible", "locallyOptimal")

    label_a = meta_a.get("label", meta_a.get("epoch_id", "the earlier plan"))
    label_b = meta_b.get("label", meta_b.get("epoch_id", "the later plan"))
    optimal_b = meta_b.get("objective_value")
    sense = meta_b.get("objective_sense", "minimize")

    if feasible:
        obj_comp = next(model_b.component_data_objects(pe.Objective, active=True), None)
        forced_obj = float(pe.value(obj_comp)) if obj_comp is not None else None
        gap_abs, gap_rel = _opportunity_gap(forced_obj, optimal_b, sense)

        result: dict = {
            "epoch_a": meta_a["epoch_id"],
            "epoch_b": meta_b["epoch_id"],
            "computed_at": datetime.now().isoformat(),
            "feasible": True,
            "termination_condition": tc,
            "n_common_vars_fixed": n_fixed,
            "n_free_vars_in_b": n_free,
            "fixed_variable_names": sorted(fixed_var_names)[:50],
            "forced_objective": forced_obj,
            "optimal_objective_b": optimal_b,
            "objective_sense": sense,
            "gap_abs": gap_abs,
            "gap_rel": gap_rel,
            "iis_rounds": [],
            "minimal_cover": [],
            "minimal_cover_details": [],
            "required_slacks": {},
            "business_summary": _business_summary(True, label_a, label_b, gap_rel=gap_rel),
            "summary": {
                "compatible": True,
                "optimality_gap_pct": round(gap_rel * 100, 4) if gap_rel is not None else None,
            },
        }
    else:
        iis_rounds = _iterative_iis_rounds(model_b, max_rounds=2)
        minimal_cover = _greedy_minimal_cover(iis_rounds)
        required_slacks = _solve_required_slacks(model_b, minimal_cover)

        for name, payload in required_slacks.items():
            payload["description"] = _business_label(name, docs)

        minimal_cover_details = [
            {
                "constraint": name,
                "description": _business_label(name, docs),
            }
            for name in minimal_cover
        ]

        result = {
            "epoch_a": meta_a["epoch_id"],
            "epoch_b": meta_b["epoch_id"],
            "computed_at": datetime.now().isoformat(),
            "feasible": False,
            "termination_condition": tc,
            "n_common_vars_fixed": n_fixed,
            "n_free_vars_in_b": n_free,
            "fixed_variable_names": sorted(fixed_var_names)[:50],
            "forced_objective": None,
            "optimal_objective_b": optimal_b,
            "objective_sense": sense,
            "gap_abs": None,
            "gap_rel": None,
            "iis_rounds": [
                {
                    "round": round_info["round"],
                    "constraints": round_info["constraints"],
                }
                for round_info in iis_rounds
            ],
            "minimal_cover": minimal_cover,
            "minimal_cover_details": minimal_cover_details,
            "required_slacks": required_slacks,
            "business_summary": _business_summary(
                False,
                label_a,
                label_b,
                minimal_cover=minimal_cover,
                docs=docs,
            ),
            "summary": {
                "compatible": False,
                "n_iis_rounds": len(iis_rounds),
                "n_constraints_in_cover": len(minimal_cover),
            },
        }

    store_a.save_comparison(store_b.epoch_id, "qt3", result)
    logger.info(f"[QT3] Result saved: feasible={result['feasible']}")
    return result
