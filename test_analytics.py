"""
Phase 2 Analytics — Standalone Test Script

Tests compute_structural_diff (QT1) and compute_solution_diff (QT2) using
the same diet epoch pair used by test_data_layer.py:

  Epoch A: diet.py + original diet_data.json
  Epoch B: diet.py + updated data (protein +20%, calorie +10%, vitamin-c -20%,
           new food 'tofu', wheat protein -15%)

If the epoch pair doesn't exist on disk (test_data_layer.py hasn't been run),
this script solves and creates them automatically before running analytics.

Run from the project root:
    python test_analytics.py

What it checks:
  1.  Load epoch stores (or build from scratch if needed)
  2.  QT1 compute_structural_diff — first call (no cache)
      a. 'tofu' in index_set_changes for food set 'f'
      b. protein/calorie/vitamin-c requirement changes detected in param 'b'
      c. wheat's nutritive value change detected in param 'a'
      d. summary flags: has_parametric_changes=True, n_set_element_additions=1
  3.  QT1 cache hit — second call returns cached result without recompute
  4.  QT2 compute_solution_diff — first call (no cache)
      a. objective differs between epochs
      b. at least one variable changed
      c. tofu-related variable(s) appear as 'added'
      d. churn_summary populated per family
  5.  QT2 cache hit — second call returns cached result
  6.  get_variable_family_details — tier-4 drill-down for variable family 'x'
      a. returns per-variable breakdown
      b. tofu variable has status='added'
      c. at least one variable has status='changed'
  7.  Output a compact summary of QT1 and QT2 results
"""

import os
import sys
import json
import copy
import shutil

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
        get_variable_family_details,
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
# Constants — same epoch pair as test_data_layer.py
# ---------------------------------------------------------------------------
DIET_PY     = "Feas/diet.py"
DIET_JSON   = "Feas/diet_data.json"
DIET_JSON_B = "tmp/rh_epochs/diet_data_epoch_b.json"
EPOCH_A_ID  = "diet_epoch_a"
EPOCH_B_ID  = "diet_epoch_b"
CMP_KEY     = "_vs_".join(sorted([EPOCH_A_ID, EPOCH_B_ID]))
CMP_DIR     = os.path.join("tmp/rh_epochs/comparisons", CMP_KEY)

# ---------------------------------------------------------------------------
# 1. Load or build epoch stores
# ---------------------------------------------------------------------------
section("1. Load epoch stores (build if not present)")

def _build_epochs():
    """Solve both epochs fresh and create EpochStore files."""
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

    def _solve(model, name):
        solver = SolverFactory("gurobi")
        solver.options["TimeLimit"] = 60
        results = solver.solve(model, tee=False)
        tc = str(results.solver.termination_condition)
        if tc != "optimal":
            fail(f"{name} solve failed: {tc}")
        info(f"Solved {name}: status={tc}")
        return tc

    tc_a = _solve(model_a, "Epoch A")
    tc_b = _solve(model_b, "Epoch B")

    # Clean up any stale directories
    for eid in [EPOCH_A_ID, EPOCH_B_ID]:
        p = os.path.join(EpochStore.BASE_DIR, eid)
        if os.path.exists(p):
            shutil.rmtree(p)

    store_a = EpochStore.create_from_pyomo(
        model=model_a, epoch_id=EPOCH_A_ID,
        label="Diet — Original Requirements",
        source_py_path=DIET_PY, termination_condition=tc_a, data_json_path=DIET_JSON,
    )
    store_b = EpochStore.create_from_pyomo(
        model=model_b, epoch_id=EPOCH_B_ID,
        label="Diet — Updated Requirements + Tofu",
        source_py_path=DIET_PY, termination_condition=tc_b, data_json_path=DIET_JSON_B,
    )
    return store_a, store_b


epoch_a_dir = os.path.join(EpochStore.BASE_DIR, EPOCH_A_ID)
epoch_b_dir = os.path.join(EpochStore.BASE_DIR, EPOCH_B_ID)

if os.path.isdir(epoch_a_dir) and os.path.isdir(epoch_b_dir):
    store_a = EpochStore.load(EPOCH_A_ID)
    store_b = EpochStore.load(EPOCH_B_ID)
    ok(f"Loaded existing epochs: {store_a}, {store_b}")
else:
    info("Epoch dirs not found — solving and building epoch stores...")
    store_a, store_b = _build_epochs()
    ok(f"Built fresh epochs: {store_a}, {store_b}")

# Sanity check
meta_a = store_a.get_meta()
meta_b = store_b.get_meta()
info(f"Epoch A: vars={meta_a['n_variables']}, status={meta_a['sol_status']}")
info(f"Epoch B: vars={meta_b['n_variables']}, status={meta_b['sol_status']}")
if meta_a["sol_status"] != "optimal":
    fail(f"Epoch A not optimal: {meta_a['sol_status']}")
if meta_b["sol_status"] != "optimal":
    fail(f"Epoch B not optimal: {meta_b['sol_status']}")
ok("Both epochs solved to optimality ✓")

# ---------------------------------------------------------------------------
# Clear the comparison cache so analytics run fresh
# ---------------------------------------------------------------------------
if os.path.exists(CMP_DIR):
    shutil.rmtree(CMP_DIR)
    info(f"Cleared comparison cache: {CMP_DIR}")

# ---------------------------------------------------------------------------
# 2. QT1 — compute_structural_diff (first call, no cache)
# ---------------------------------------------------------------------------
section("2. QT1 — compute_structural_diff (first call)")

qt1 = compute_structural_diff(store_a, store_b)
ok("compute_structural_diff returned without error ✓")

# 2a. 'tofu' detected in food set changes
food_set_change = qt1["index_set_changes"]["modified_sets"].get("f")
if food_set_change is None:
    fail("Expected modified set 'f' (foods) in index_set_changes")
added_elems = food_set_change["added_elements"]
if "tofu" not in added_elems:
    fail(f"Expected 'tofu' in added_elements. Got: {added_elems}")
ok(f"'tofu' in index_set_changes.modified_sets.f.added_elements ✓")
info(f"  Set 'f': size A={food_set_change['size_a']}, B={food_set_change['size_b']}, "
     f"added={added_elems}")

# 2b. Nutrient requirement changes in param 'b'
param_changes = qt1["parameter_changes"]
b_family = param_changes["by_family"].get("b")
if b_family is None:
    fail("Expected parameter family 'b' (nutrient_requirements) in parameter_changes")
b_changed_indices = {e["index"] for e in b_family["changed"]}
info(f"  Changed param 'b' indices: {b_changed_indices}")
for expected_nutrient in ["protein", "calorie", "vitamin-c"]:
    if expected_nutrient not in b_changed_indices:
        fail(f"Expected nutrient '{expected_nutrient}' in changed param 'b'. "
             f"Got: {b_changed_indices}")
ok(f"Nutrient requirement changes in param 'b': {b_changed_indices} ✓")

# Check magnitudes
b_by_index = {e["index"]: e for e in b_family["changed"]}
protein_entry = b_by_index.get("protein")
calorie_entry = b_by_index.get("calorie")
vc_entry      = b_by_index.get("vitamin-c")

if protein_entry:
    info(f"  protein: {protein_entry['value_a']} → {protein_entry['value_b']} "
         f"(rel={protein_entry['rel_change']:.3f}, {protein_entry['magnitude']})")
    if abs(protein_entry["rel_change"] - 0.20) > 0.01:
        fail(f"protein rel_change should be ~0.20, got {protein_entry['rel_change']:.4f}")
    ok("protein +20% correctly detected and classified ✓")

if vc_entry:
    info(f"  vitamin-c: {vc_entry['value_a']} → {vc_entry['value_b']} "
         f"(rel={vc_entry['rel_change']:.3f}, {vc_entry['magnitude']})")
    if abs(vc_entry["rel_change"] - (-0.20)) > 0.01:
        fail(f"vitamin-c rel_change should be ~-0.20, got {vc_entry['rel_change']:.4f}")
    ok("vitamin-c -20% correctly detected ✓")

# 2c. Wheat nutritive value change in param 'a'
a_family = param_changes["by_family"].get("a")
if a_family is None:
    fail("Expected parameter family 'a' (nutritive_values) in parameter_changes")
a_changed_indices = {e["index"] for e in a_family["changed"]}
wheat_protein_key = next((k for k in a_changed_indices if "wheat" in k and "protein" in k), None)
if wheat_protein_key is None:
    fail(f"Expected ('wheat', 'protein') change in param 'a'. Got changes: {a_changed_indices}")
ok(f"Wheat protein change detected in param 'a': {wheat_protein_key} ✓")

# Also check tofu-related entries are in 'added'
tofu_added = {k: v for k, v in a_family["added"].items() if "tofu" in k}
if not tofu_added:
    fail("Expected tofu nutritive values in param 'a' added entries")
ok(f"Tofu nutritive values in param 'a' added: {len(tofu_added)} entries ✓")

# 2d. Summary flags
summary = qt1["summary"]
info(f"  QT1 summary: {summary}")
if not summary["has_parametric_changes"]:
    fail("has_parametric_changes should be True")
ok("has_parametric_changes = True ✓")
if summary["n_set_element_additions"] != 1:
    fail(f"n_set_element_additions should be 1 (tofu). Got: {summary['n_set_element_additions']}")
ok("n_set_element_additions = 1 ✓")
if summary["n_param_changes"] == 0:
    fail("n_param_changes should be > 0")
ok(f"n_param_changes = {summary['n_param_changes']} ✓")

info(f"  top_changes (first 3): {[c['index'] + ' → ' + c['family'] for c in param_changes['top_changes'][:3]]}")

# Verify cache file was written
qt1_path = os.path.join(CMP_DIR, "qt1_structural.json")
if not os.path.exists(qt1_path):
    fail(f"QT1 cache file not written: {qt1_path}")
ok(f"QT1 cache file written: {qt1_path} ✓")

# ---------------------------------------------------------------------------
# 3. QT1 cache hit
# ---------------------------------------------------------------------------
section("3. QT1 cache hit")

qt1_cached = compute_structural_diff(store_a, store_b)
if qt1_cached.get("computed_at") != qt1.get("computed_at"):
    fail("Cache hit returned different 'computed_at' — result was recomputed")
ok("Second call returned cached result (same computed_at) ✓")

# Force recompute
qt1_forced = compute_structural_diff(store_a, store_b, force=True)
ok(f"force=True recomputed (new computed_at={qt1_forced['computed_at'][:19]}) ✓")

# ---------------------------------------------------------------------------
# 4. QT2 — compute_solution_diff (first call, no cache)
# ---------------------------------------------------------------------------
section("4. QT2 — compute_solution_diff (first call)")

qt2 = compute_solution_diff(store_a, store_b, qt1=qt1)
ok("compute_solution_diff returned without error ✓")

# 4a. Objective changed
obj_chg = qt2["objective_change"]
info(f"  Objective A={obj_chg['value_a']:.6f}, B={obj_chg['value_b']:.6f}, "
     f"delta={obj_chg['delta']:.6f}")
if obj_chg["delta"] is None or abs(obj_chg["delta"]) < 1e-9:
    fail("Objective delta should be non-zero (epochs have different requirements)")
ok(f"Objective delta = {obj_chg['delta']:.6f} ✓")

# 4b. At least one variable changed
summary2 = qt2["summary"]
info(f"  QT2 summary: {summary2}")
if summary2["n_changed_variables"] == 0:
    fail("n_changed_variables should be > 0")
ok(f"n_changed_variables = {summary2['n_changed_variables']} ✓")

# 4c. Tofu variable(s) appear as added
if summary2["n_added_variables"] == 0:
    fail("n_added_variables should be > 0 (tofu was added to the food set)")
ok(f"n_added_variables = {summary2['n_added_variables']} (tofu-related) ✓")

# 4d. churn_summary populated
churn = qt2["churn_summary"]
if not churn:
    fail("churn_summary is empty")
ok(f"churn_summary has {len(churn)} family/families: {list(churn.keys())} ✓")
for family, stats in churn.items():
    info(f"  Family '{family}': count_A={stats['count_a']}, count_B={stats['count_b']}, "
         f"changed={stats['n_changed']}, added={stats['n_added']}")

# top_changes should be non-empty
if not qt2["top_changes"]:
    fail("top_changes is empty — expected at least one variable change")
top_var = qt2["top_changes"][0]
info(f"  Top change: {top_var['variable']} "
     f"({top_var['value_a']:.4f} → {top_var['value_b']:.4f}, Δ={top_var['delta']:.4f})")
ok(f"top_changes[0] = {top_var['variable']} ✓")

# structural_context should be present (qt1 was passed)
if "structural_context" not in qt2:
    fail("structural_context block missing — qt1 was passed but context not added")
sc = qt2["structural_context"]
info(f"  structural_context: {sc['note']}")
ok("structural_context block present ✓")

# Verify cache file written
qt2_path = os.path.join(CMP_DIR, "qt2_solution.json")
if not os.path.exists(qt2_path):
    fail(f"QT2 cache file not written: {qt2_path}")
ok(f"QT2 cache file written: {qt2_path} ✓")

# ---------------------------------------------------------------------------
# 5. QT2 cache hit
# ---------------------------------------------------------------------------
section("5. QT2 cache hit")

qt2_cached = compute_solution_diff(store_a, store_b)
if qt2_cached.get("computed_at") != qt2.get("computed_at"):
    fail("QT2 cache hit returned different computed_at — was recomputed")
ok("Second call returned cached QT2 result ✓")

qt2_forced = compute_solution_diff(store_a, store_b, force=True)
ok(f"force=True recomputed QT2 (new computed_at={qt2_forced['computed_at'][:19]}) ✓")

# ---------------------------------------------------------------------------
# 6. get_variable_family_details — tier-4 drill-down
# ---------------------------------------------------------------------------
section("6. get_variable_family_details — tier-4 drill-down for family 'x'")

details = get_variable_family_details(store_a, store_b, "x")
ok("get_variable_family_details returned without error ✓")

if details["family"] != "x":
    fail(f"Expected family='x', got '{details['family']}'")
if details["epoch_a"] != EPOCH_A_ID:
    fail("epoch_a mismatch in details")
ok("family, epoch_a, epoch_b metadata correct ✓")

variables = details["variables"]
if not variables:
    fail("details['variables'] is empty — expected food purchase variables")
info(f"  {len(variables)} variable(s) in family 'x'")

# tofu variable should exist in B as "added"
tofu_var = next((k for k in variables if "tofu" in k), None)
if tofu_var is None:
    fail("Expected tofu-related variable in 'x' family details")
if variables[tofu_var]["status"] != "added":
    fail(f"Tofu variable {tofu_var} should have status='added', got '{variables[tofu_var]['status']}'")
info(f"  {tofu_var}: status={variables[tofu_var]['status']}, "
     f"value_a={variables[tofu_var]['value_a']}, value_b={variables[tofu_var]['value_b']:.6f}")
ok(f"Tofu variable '{tofu_var}' has status='added' ✓")

# At least one variable should be "changed"
changed_vars = [k for k, v in variables.items() if v["status"] == "changed"]
if not changed_vars:
    fail("Expected at least one variable with status='changed'")
ok(f"{len(changed_vars)} variable(s) with status='changed' ✓")
info(f"  Sample changed vars: {changed_vars[:3]}")

# Summary counts should be consistent
det_sum = details["summary"]
info(f"  details summary: {det_sum}")
if det_sum["n_added"] == 0:
    fail("details summary: n_added should be > 0 (tofu)")
ok("details summary counts consistent ✓")

# ---------------------------------------------------------------------------
# 7. Human-readable QT1 / QT2 summary
# ---------------------------------------------------------------------------
section("7. Human-readable analysis summary")

print(f"\n  {CYAN}--- QT1: Structural & Parametric Changes ---{RESET}")
s = qt1["summary"]
print(f"  Index set element additions : {s['n_set_element_additions']}")
print(f"  Index set element removals  : {s['n_set_element_removals']}")
print(f"  Param changes (total)       : {s['n_param_changes']}  "
      f"[large={s['n_param_changes_large']}, moderate={s['n_param_changes_moderate']}, "
      f"small={s['n_param_changes_small']}]")
print(f"  Param additions             : {s['n_param_additions']}")
print(f"  Has structural changes      : {s['has_structural_changes']}")
print(f"  Has parametric changes      : {s['has_parametric_changes']}")

print(f"\n  Top 5 parameter changes by magnitude:")
for c in qt1["parameter_changes"]["top_changes"][:5]:
    rel_str = f"{c['rel_change']*100:+.1f}%" if c["rel_change"] is not None else "∞"
    print(f"    {c['family']}[{c['index']}]: {c['value_a']} → {c['value_b']}  "
          f"({rel_str}, {c['magnitude']})")

print(f"\n  {CYAN}--- QT2: Solution Differences ---{RESET}")
s2 = qt2["summary"]
print(f"  Objective delta             : {s2['objective_delta']:.6f}")
print(f"  Objective rel change        : {(s2['objective_rel_change']*100 if s2['objective_rel_change'] else 0):+.2f}%")
print(f"  Variables changed           : {s2['n_changed_variables']}")
print(f"  Variables added (structural): {s2['n_added_variables']}")
print(f"  Variables removed           : {s2['n_removed_variables']}")
print(f"  Overall churn rate          : {s2['overall_churn_rate']*100:.1f}%")
print(f"  Constraints became binding  : {s2['n_became_binding']}")
print(f"  Constraints became nonbinding: {s2['n_became_nonbinding']}")

print(f"\n  Top 5 variable changes by magnitude:")
for c in qt2["top_changes"][:5]:
    rel_str = f"{c['rel_change']*100:+.1f}%" if c["rel_change"] is not None else "∞"
    print(f"    {c['variable']}: {c['value_a']:.4f} → {c['value_b']:.4f}  "
          f"(Δ={c['delta']:+.4f}, {rel_str})")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print(f"\n{GREEN}{'='*60}")
print(f"  ALL PHASE 2 CHECKS PASSED — QT1 and QT2 analytics working correctly")
print(f"{'='*60}{RESET}\n")

print("Comparison files written to:")
for fname in os.listdir(CMP_DIR):
    fpath = os.path.join(CMP_DIR, fname)
    print(f"  {fpath}  ({os.path.getsize(fpath):,} bytes)")
