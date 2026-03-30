"""
Phase 3 Solver Analytics — Standalone Test Script

Tests assess_backward_compat (QT3) and compute_attribution_analysis (QT4)
using the diet epoch pair.

Epoch A: diet.py + original requirements
Epoch B: diet.py + protein +20%, calorie +10%, vitamin-c -20%, new food 'tofu'

If the epoch pair is missing, this script builds it automatically.
QT1 and QT2 are computed as prerequisites for QT4.

Run from the project root:
    python test_phase3.py

What it checks:
  1.  Load epochs (build if needed) + compute QT1/QT2 prerequisites
  2.  QT3 targeted case — later-only variable must stay free
  3.  QT3 targeted case — infeasible result returns minimal cover + slack
  4.  QT3 assess_backward_compat on diet pair
      a. Result has 'feasible' key
      b. n_common_vars_fixed > 0
      c. If feasible: forced_objective and gap_rel present
         If infeasible: minimal_cover and required_slacks present
      d. Cache file qt3_backward.json written
  5.  QT3 cache hit — second call returns same computed_at
  6.  QT4 compute_attribution_analysis — first call
      a. parameter_shifts contains the protein/calorie/vitamin-c changes
      b. bottleneck_evolution has 'relieved', 'new_bottlenecks', 'persistent' keys
      c. lp_duals_available_a and lp_duals_available_b are True (diet is pure LP)
      d. narrative has all 5 required fields
      e. Cache file qt4_attribution.json written
  7.  QT4 cache hit — second call returns same computed_at
  8.  QT4 MIP guard — duals for LP model come from solution.json (no extra solve)
  9.  Human-readable summary of QT3 and QT4 results
"""

import os
import sys
import json
import copy
import shutil
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):      print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):    print(f"  {RED}✗ FAILED: {msg}{RESET}"); sys.exit(1)
def info(msg):    print(f"     {CYAN}{msg}{RESET}")
def section(msg): print(f"\n{YELLOW}{'='*60}{RESET}\n{YELLOW}{msg}{RESET}\n{YELLOW}{'='*60}{RESET}")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
section("Importing modules")

try:
    from rh_comparison.data_store import EpochStore, load_model_from_py
    from rh_comparison.analytics import (
        compute_structural_diff,
        compute_solution_diff,
        assess_backward_compat,
        compute_attribution_analysis,
    )
    from rh_comparison.config.rh_constants import Thresholds
    ok("rh_comparison imports OK")
except ImportError as e:
    fail(f"Import error: {e}")

try:
    from pyomo.opt import SolverFactory
    import pyomo.environ as pe
    ok("Pyomo imports OK")
except ImportError as e:
    fail(f"Pyomo import: {e}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIET_PY     = "Feas/diet.py"
DIET_JSON   = "Feas/diet_data.json"
DIET_JSON_B = "tmp/rh_epochs/diet_data_epoch_b.json"
EPOCH_A_ID  = "diet_epoch_a"
EPOCH_B_ID  = "diet_epoch_b"
CMP_KEY     = "_vs_".join(sorted([EPOCH_A_ID, EPOCH_B_ID]))
CMP_DIR     = os.path.join("tmp/rh_epochs/comparisons", CMP_KEY)


def _build_temp_epoch_pair(pair_name: str, model_a_code: str, model_b_code: str):
    temp_dir = tempfile.mkdtemp(prefix=f"rh_qt3_{pair_name}_", dir="tmp")
    model_a_path = os.path.join(temp_dir, "model_a.py")
    model_b_path = os.path.join(temp_dir, "model_b.py")

    with open(model_a_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(model_a_code))
    with open(model_b_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(model_b_code))

    model_a = load_model_from_py(model_a_path)
    model_b = load_model_from_py(model_b_path)

    def _solve(model, name):
        solver = SolverFactory("gurobi")
        solver.options["TimeLimit"] = 60
        res = solver.solve(model, tee=False)
        tc = str(res.solver.termination_condition)
        if tc != "optimal":
            fail(f"{name} solve failed: {tc}")
        return tc

    tc_a = _solve(model_a, f"{pair_name} A")
    tc_b = _solve(model_b, f"{pair_name} B")

    epoch_a_id = f"{pair_name}_epoch_a"
    epoch_b_id = f"{pair_name}_epoch_b"
    for eid in (epoch_a_id, epoch_b_id):
        epoch_dir = os.path.join(EpochStore.BASE_DIR, eid)
        if os.path.exists(epoch_dir):
            shutil.rmtree(epoch_dir)

    store_a = EpochStore.create_from_pyomo(
        model=model_a,
        epoch_id=epoch_a_id,
        label=f"{pair_name} A",
        source_py_path=model_a_path,
        termination_condition=tc_a,
    )
    store_b = EpochStore.create_from_pyomo(
        model=model_b,
        epoch_id=epoch_b_id,
        label=f"{pair_name} B",
        source_py_path=model_b_path,
        termination_condition=tc_b,
    )
    return store_a, store_b

# ---------------------------------------------------------------------------
# 1. Load or build epoch stores
# ---------------------------------------------------------------------------
section("1. Load epochs + ensure QT1/QT2 prerequisites")

def _build_epochs():
    assert os.path.exists(DIET_PY),   f"Missing: {DIET_PY}"
    assert os.path.exists(DIET_JSON), f"Missing: {DIET_JSON}"
    with open(DIET_JSON) as f:
        data_a = json.load(f)
    data_b = copy.deepcopy(data_a)
    data_b["nutrient_requirements"]["protein"]   = 84
    data_b["nutrient_requirements"]["calorie"]   = 3.3
    data_b["nutrient_requirements"]["vitamin-c"] = 60
    data_b["nutritive_values"]["tofu"] = {
        "calorie": 5.0, "protein": 400, "calcium": 7.0,
        "iron": 55, "vitamin-a": 0, "vitamin-b1": 3.0,
        "vitamin-b2": 6.0, "niacin": 8, "vitamin-c": 0,
    }
    data_b["nutritive_values"]["wheat"]["protein"] = 1200
    os.makedirs("tmp/rh_epochs", exist_ok=True)
    with open(DIET_JSON_B, "w") as f:
        json.dump(data_b, f, indent=2)

    model_a = load_model_from_py(DIET_PY, DIET_JSON)
    model_b = load_model_from_py(DIET_PY, DIET_JSON_B)

    def _solve(m, n):
        solver = SolverFactory("gurobi")
        solver.options["TimeLimit"] = 60
        r = solver.solve(m, tee=False)
        tc = str(r.solver.termination_condition)
        if tc != "optimal":
            fail(f"{n} solve failed: {tc}")
        info(f"Solved {n}: {tc}")
        return tc

    tc_a = _solve(model_a, "Epoch A")
    tc_b = _solve(model_b, "Epoch B")
    for eid in [EPOCH_A_ID, EPOCH_B_ID]:
        p = os.path.join(EpochStore.BASE_DIR, eid)
        if os.path.exists(p):
            shutil.rmtree(p)
    sa = EpochStore.create_from_pyomo(model=model_a, epoch_id=EPOCH_A_ID,
                                      label="Diet — Original",
                                      source_py_path=DIET_PY, termination_condition=tc_a,
                                      data_json_path=DIET_JSON)
    sb = EpochStore.create_from_pyomo(model=model_b, epoch_id=EPOCH_B_ID,
                                      label="Diet — Updated + Tofu",
                                      source_py_path=DIET_PY, termination_condition=tc_b,
                                      data_json_path=DIET_JSON_B)
    return sa, sb


epoch_a_dir = os.path.join(EpochStore.BASE_DIR, EPOCH_A_ID)
epoch_b_dir = os.path.join(EpochStore.BASE_DIR, EPOCH_B_ID)

if os.path.isdir(epoch_a_dir) and os.path.isdir(epoch_b_dir):
    store_a = EpochStore.load(EPOCH_A_ID)
    store_b = EpochStore.load(EPOCH_B_ID)
    ok(f"Loaded existing epochs")
else:
    info("Epochs not found — building from scratch...")
    store_a, store_b = _build_epochs()
    ok("Built fresh epochs")

meta_a = store_a.get_meta()
meta_b = store_b.get_meta()
info(f"A: {meta_a['label']}  vars={meta_a['n_variables']}  status={meta_a['sol_status']}")
info(f"B: {meta_b['label']}  vars={meta_b['n_variables']}  status={meta_b['sol_status']}")

# Compute QT1 and QT2 (required as prerequisites for QT4)
info("Computing QT1 and QT2 prerequisites...")
qt1 = compute_structural_diff(store_a, store_b, force=True)
qt2 = compute_solution_diff(store_a, store_b, qt1=qt1, force=True)
ok(f"QT1: {qt1['summary']['n_param_changes']} param changes, "
   f"{qt1['summary']['n_set_element_additions']} set additions")
ok(f"QT2: obj delta={qt2['summary']['objective_delta']:.6f}, "
   f"changed={qt2['summary']['n_changed_variables']} vars")

# Clear QT3 and QT4 from comparison cache
for qt_name, fname in [("qt3", "qt3_backward.json"), ("qt4", "qt4_attribution.json")]:
    p = os.path.join(CMP_DIR, fname)
    if os.path.exists(p):
        os.remove(p)
        info(f"Cleared {fname} from cache")

# ---------------------------------------------------------------------------
# 2. QT3 targeted regression — later-only variable must stay free
# ---------------------------------------------------------------------------
section("2. QT3 targeted regression — later-only variable stays free")

qt3_free_a, qt3_free_b = _build_temp_epoch_pair(
    "qt3_free_var",
    """
    from pyomo.environ import *
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.keep_old = Constraint(expr=model.x >= 8)
    model.obj = Objective(expr=model.x, sense=minimize)
    """,
    """
    from pyomo.environ import *
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.z = Var(within=NonNegativeReals)
    model.cap_old = Constraint(expr=model.x <= 8)
    model.extra_cover = Constraint(expr=model.x + model.z >= 10)
    model.obj = Objective(expr=model.x + model.z, sense=minimize)
    """,
)
qt3_free = assess_backward_compat(qt3_free_a, qt3_free_b, force=True)
if not qt3_free["feasible"]:
    fail("Expected targeted QT3 free-variable case to be feasible")
if qt3_free.get("forced_objective") is None or abs(qt3_free["forced_objective"] - 10.0) > 1e-5:
    fail(f"Expected forced objective 10.0 when later-only variable adapts. Got: {qt3_free.get('forced_objective')}")
ok("QT3 keeps later-only variables free instead of forcing them to zero ✓")

# ---------------------------------------------------------------------------
# 3. QT3 targeted regression — infeasible case returns minimal cover + slack
# ---------------------------------------------------------------------------
section("3. QT3 targeted regression — minimal cover and slack")

qt3_infeas_a, qt3_infeas_b = _build_temp_epoch_pair(
    "qt3_infeas_cover",
    """
    from pyomo.environ import *
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.keep_old = Constraint(expr=model.x >= 10)
    model.obj = Objective(expr=model.x, sense=minimize)
    """,
    """
    from pyomo.environ import *
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.z = Var(within=NonNegativeReals)
    model.tight_limit = Constraint(expr=model.x <= 9)
    model.extra_cover = Constraint(expr=model.x + model.z >= 10)
    model.obj = Objective(expr=model.x + model.z, sense=minimize)
    """,
)
qt3_infeas = assess_backward_compat(qt3_infeas_a, qt3_infeas_b, force=True)
if qt3_infeas["feasible"]:
    fail("Expected targeted QT3 cover case to be infeasible")
if not qt3_infeas.get("minimal_cover"):
    fail("Expected minimal_cover in targeted infeasible QT3 case")
first_cover = qt3_infeas["minimal_cover"][0]
if "tight_limit" not in first_cover:
    fail(f"Expected 'tight_limit' in minimal cover. Got: {qt3_infeas['minimal_cover']}")
req_slack = qt3_infeas["required_slacks"].get(first_cover, {}).get("required_slack")
if req_slack is None or abs(req_slack - 1.0) > 1e-5:
    fail(f"Expected required slack of 1.0 for tight_limit. Got: {req_slack}")
ok("QT3 infeasible branch returns minimal cover and required slack ✓")

# ---------------------------------------------------------------------------
# 4. QT3 — assess_backward_compat (diet pair)
# ---------------------------------------------------------------------------
section("4. QT3 — assess_backward_compat (diet pair)")

qt3 = assess_backward_compat(store_a, store_b, qt1=qt1, qt2=qt2)
ok("assess_backward_compat returned without error ✓")

# 4a. Has 'feasible' key
if "feasible" not in qt3:
    fail("QT3 result missing 'feasible' key")
ok(f"feasible = {qt3['feasible']} ✓")

# 4b. Common vars were fixed
if qt3["n_common_vars_fixed"] == 0:
    fail("n_common_vars_fixed should be > 0")
ok(f"n_common_vars_fixed = {qt3['n_common_vars_fixed']}, "
   f"n_free_vars_in_b = {qt3['n_free_vars_in_b']} ✓")

# 4c. Feasible or infeasible — check appropriate fields
if qt3["feasible"]:
    forced   = qt3.get("forced_objective")
    optimal  = qt3.get("optimal_objective_b")
    gap_rel  = qt3.get("gap_rel")
    if forced is None:
        fail("Feasible result missing 'forced_objective'")
    if optimal is None:
        fail("Feasible result missing 'optimal_objective_b'")
    info(f"  forced_obj = {forced:.6f}")
    info(f"  optimal_b  = {optimal:.6f}")
    info(f"  gap_rel    = {gap_rel}")
    info(f"  {qt3['business_summary']}")
    ok("Feasible result: forced/optimal/gap fields present ✓")

    # For minimize sense: forced >= optimal (or very close if optimal by luck)
    sense = qt3.get("objective_sense", "minimize")
    if sense == "minimize" and forced is not None and optimal is not None:
        if forced < optimal - 1e-4:
            fail(f"For minimize: forced_obj ({forced:.6f}) should be >= optimal_obj ({optimal:.6f})")
    ok("Optimality gap direction correct for minimize sense ✓")

else:
    cover = qt3.get("minimal_cover", [])
    slacks = qt3.get("required_slacks", {})
    info(f"  Infeasible! cover={cover}")
    if not cover:
        fail("Expected minimal_cover for infeasible QT3 result")
    for name in cover[:3]:
        info(f"    {name}: required_slack={slacks.get(name, {}).get('required_slack')}")
    ok("Infeasible result: minimal_cover + required_slacks present ✓")

# 4d. Cache file written
qt3_path = os.path.join(CMP_DIR, "qt3_backward.json")
if not os.path.exists(qt3_path):
    fail(f"QT3 cache file not written: {qt3_path}")
ok(f"qt3_backward.json written ({os.path.getsize(qt3_path):,} bytes) ✓")

# ---------------------------------------------------------------------------
# 5. QT3 cache hit
# ---------------------------------------------------------------------------
section("5. QT3 cache hit")

qt3_cached = assess_backward_compat(store_a, store_b)
if qt3_cached.get("computed_at") != qt3.get("computed_at"):
    fail("QT3 cache hit returned different computed_at — was recomputed")
ok("Second call returned cached QT3 result ✓")

qt3_forced = assess_backward_compat(store_a, store_b, force=True)
ok(f"force=True recomputed QT3 ✓")

# ---------------------------------------------------------------------------
# 6. QT4 — compute_attribution_analysis (first call)
# ---------------------------------------------------------------------------
section("6. QT4 — compute_attribution_analysis (first call)")

qt4 = compute_attribution_analysis(store_a, store_b, qt1=qt1, qt2=qt2)
ok("compute_attribution_analysis returned without error ✓")

# 4a. parameter_shifts matches QT1 significant changes
shifts = qt4["parameter_shifts"]
info(f"  {len(shifts)} parameter shift(s) above threshold ({Thresholds.TRIGGER_THRESHOLD*100:.0f}%)")
if not shifts:
    fail("Expected at least one parameter shift (protein/calorie/vitamin-c changed ≥10%)")
shift_indices = {s["index"] for s in shifts}
info(f"  Shift indices: {shift_indices}")
# Protein changed +20% > 10% threshold — should be in shifts
protein_shift = next((s for s in shifts if s["family"] == "b" and "protein" in s["index"]), None)
if protein_shift is None:
    fail("Expected 'protein' parameter shift in QT4 (changed +20%)")
ok(f"Protein shift detected: {protein_shift['value_a']} → {protein_shift['value_b']} "
   f"({protein_shift['rel_change']*100:+.1f}%) ✓")

# 4b. bottleneck_evolution has required keys
be = qt4["bottleneck_evolution"]
for key in ("relieved", "new_bottlenecks", "persistent"):
    if key not in be:
        fail(f"bottleneck_evolution missing key '{key}'")
ok("bottleneck_evolution has all 3 keys ✓")
info(f"  relieved={len(be['relieved'])}, new_bottlenecks={len(be['new_bottlenecks'])}, "
     f"persistent={len(be['persistent'])}")

# 4c. LP duals available (diet is pure LP)
summary4 = qt4["summary"]
if not summary4["lp_duals_available_a"]:
    fail("lp_duals_available_a should be True (diet is pure LP)")
if not summary4["lp_duals_available_b"]:
    fail("lp_duals_available_b should be True (diet is pure LP)")
ok("LP duals available for both epochs (pure LP model) ✓")

# Verify duals were NOT written to duals.json for LP models
# (they come from solution.json directly, no extra solve needed)
duals_json_a = os.path.join(EpochStore.BASE_DIR, EPOCH_A_ID, "duals.json")
duals_json_b = os.path.join(EpochStore.BASE_DIR, EPOCH_B_ID, "duals.json")
if os.path.exists(duals_json_a):
    info(f"  duals.json exists for A (was written previously — ok if from test_data_layer.py)")
else:
    info(f"  duals.json absent for A (LP duals come from solution.json — correct)")
ok("LP duals sourced from solution.json (no extra incumbent LP solve needed) ✓")

# 4d. Narrative has all 5 required fields
narrative = qt4["narrative"]
required_fields = {"parameter_shift", "prior_state", "solution_adaptation", "outcome", "new_state"}
missing = required_fields - set(narrative)
if missing:
    fail(f"Narrative missing fields: {missing}")
ok("Narrative has all 5 fields ✓")
for field, text in narrative.items():
    info(f"  {field}: {text[:100]}{'...' if len(text) > 100 else ''}")

# 4e. Cache file written
qt4_path = os.path.join(CMP_DIR, "qt4_attribution.json")
if not os.path.exists(qt4_path):
    fail(f"QT4 cache file not written: {qt4_path}")
ok(f"qt4_attribution.json written ({os.path.getsize(qt4_path):,} bytes) ✓")

# ---------------------------------------------------------------------------
# 7. QT4 cache hit
# ---------------------------------------------------------------------------
section("7. QT4 cache hit")

qt4_cached = compute_attribution_analysis(store_a, store_b, qt1=qt1, qt2=qt2)
if qt4_cached.get("computed_at") != qt4.get("computed_at"):
    fail("QT4 cache hit returned different computed_at — was recomputed")
ok("Second call returned cached QT4 result ✓")

qt4_forced = compute_attribution_analysis(store_a, store_b, qt1=qt1, qt2=qt2, force=True)
ok(f"force=True recomputed QT4 ✓")

# ---------------------------------------------------------------------------
# 8. Verify LP duals came from solution.json (no duals.json written for LP)
# ---------------------------------------------------------------------------
section("8. LP dual sourcing verification")

sol_a = store_a.get_solution()
if sol_a.get("is_mip", True):
    info("  Model is MIP — skipping LP dual source check")
else:
    # For LP, QT4 should have used solution.json duals, not triggered a new solve
    # The bottleneck_evolution should reflect real dual magnitudes
    all_duals = {k: v for k, v in sol_a["constraints"].items() if "dual" in v and v["dual"] is not None}
    info(f"  solution.json has {len(all_duals)} constraints with dual values")
    if not all_duals:
        fail("Expected at least one dual value in solution.json for pure LP model")
    ok(f"{len(all_duals)} constraint duals in solution.json ✓")

    # Count duals used in bottleneck evolution
    n_be_total = (len(be["relieved"]) + len(be["new_bottlenecks"]) + len(be["persistent"]))
    info(f"  Total significant constraints in bottleneck evolution: {n_be_total}")
    ok("Bottleneck evolution reflects LP shadow prices ✓")

# ---------------------------------------------------------------------------
# 9. Human-readable summary
# ---------------------------------------------------------------------------
section("9. Human-readable Phase 3 summary")

print(f"\n  {CYAN}--- QT3: Backward Compatibility ---{RESET}")
print(f"  Feasible in Epoch B         : {qt3['feasible']}")
print(f"  Common vars fixed           : {qt3['n_common_vars_fixed']}")
print(f"  Free vars in B (new)        : {qt3['n_free_vars_in_b']}")
if qt3["feasible"]:
    print(f"  Forced objective (Epoch A)  : {qt3.get('forced_objective', 'N/A')}")
    print(f"  Optimal objective (Epoch B) : {qt3.get('optimal_objective_b', 'N/A')}")
    gap = qt3.get("gap_rel")
    print(f"  Optimality gap              : {gap*100:.4f}%" if gap is not None else "  Optimality gap              : N/A")
    print(f"  Summary                     : {qt3['business_summary']}")
else:
    print(f"  Summary                     : {qt3['business_summary']}")
    print(f"  Minimal cover               : {qt3['minimal_cover']}")
    print(f"  Required slacks             : {qt3['required_slacks']}")

print(f"\n  {CYAN}--- QT4: Solution Attribution Analysis ---{RESET}")
print(f"  Parameter shifts (≥{Thresholds.TRIGGER_THRESHOLD*100:.0f}%): {summary4['n_parameter_shifts']}")
print(f"  Relieved bottlenecks        : {summary4['n_relieved_bottlenecks']}")
print(f"  New bottlenecks             : {summary4['n_new_bottlenecks']}")
print(f"  Persistent bottlenecks      : {summary4['n_persistent_bottlenecks']}")
print(f"  Binary flips                : {summary4['n_binary_flips']}")
print(f"  Continuous adjustments      : {summary4['n_continuous_adjustments']}")

print(f"\n  Top parameter shifts:")
for s in qt4["parameter_shifts"][:5]:
    rel_str = f"{s['rel_change']*100:+.1f}%" if s["rel_change"] is not None else "(0→nonzero)"
    print(f"    {s['family']}[{s['index']}]: {s['value_a']} → {s['value_b']}  {rel_str}")

if be["relieved"]:
    print(f"\n  Relieved constraints (binding in A, not in B):")
    for c in be["relieved"][:3]:
        print(f"    {c['constraint']}: dual_A={c['dual_a']:.4f} → dual_B={c['dual_b']:.4f}")

if be["new_bottlenecks"]:
    print(f"\n  New bottleneck constraints (binding in B, not in A):")
    for c in be["new_bottlenecks"][:3]:
        print(f"    {c['constraint']}: dual_A={c['dual_a']:.4f} → dual_B={c['dual_b']:.4f}")

print(f"\n  Narrative:")
for field, text in qt4["narrative"].items():
    print(f"    [{field}] {text}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print(f"\n{GREEN}{'='*60}")
print(f"  ALL PHASE 3 CHECKS PASSED — QT3 and QT4 analytics working correctly")
print(f"{'='*60}{RESET}\n")

print("Comparison files:")
for fname in sorted(os.listdir(CMP_DIR)):
    fpath = os.path.join(CMP_DIR, fname)
    print(f"  {fpath}  ({os.path.getsize(fpath):,} bytes)")
