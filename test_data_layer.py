"""
Phase 1 Data Layer — Standalone Test Script

Tests EpochStore end-to-end using the SAME model (diet.py) solved twice with
different data — correctly simulating two rolling horizon epochs:

  Epoch A: diet.py + original diet_data.json
  Epoch B: diet.py + updated data (changed nutrient requirements, new food item)
           → same model structure, different parameters, one new index element

Run from the project root:
    python test_data_layer.py

What it checks:
  1.  load_model_from_py (Mode A: .py + .json)
  2.  Gurobi solve for both epochs
  3.  EpochStore.create_from_pyomo → 5 files + pkl + registry
  4.  Lazy-load accessors (get_meta, get_solution, get_params, get_structure)
  5.  get_duals → None before computed; get_description → "" before saved
  6.  Compact LLM summaries (get_solution_summary, get_param_summary)
  7.  Tier-4 drill-down (get_variable_family, get_param_family)
  8.  save_description / get_description round-trip
  9.  save_duals / get_duals round-trip
  10. save_comparison / load_comparison round-trip + None for un-computed QTs
  11. EpochStore.load → reload from disk, all values consistent
  12. get_pyomo_model → pkl round-trip, variable count matches, dual stripped
  13. list_all_epochs → both epochs in registry
  14. Epoch differences visible across both epochs (obj, var count, params)
"""

import os
import sys
import json
import copy

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
    from rh_comparison.config.rh_constants import QueryType, Thresholds
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
# Build Epoch B data — same diet model, updated inputs
# Simulates a rolling horizon re-solve where:
#   - Some nutrient requirements have changed (demand shift)
#   - A new food option "tofu" has been added to the set
# ---------------------------------------------------------------------------
section("Building Epoch A and Epoch B data")

DIET_PY     = "Feas/diet.py"
DIET_JSON   = "Feas/diet_data.json"
DIET_JSON_B = "tmp/rh_epochs/diet_data_epoch_b.json"

assert os.path.exists(DIET_PY),   f"Missing: {DIET_PY}"
assert os.path.exists(DIET_JSON), f"Missing: {DIET_JSON}"

with open(DIET_JSON, "r") as f:
    data_a = json.load(f)

# Epoch B: same structure, updated parameters + one new food item
data_b = copy.deepcopy(data_a)

# 1. Update some nutrient requirements (simulate forecast update)
data_b["nutrient_requirements"]["protein"]   = 84    # was 70  (+20%)
data_b["nutrient_requirements"]["calorie"]   = 3.3   # was 3   (+10%)
data_b["nutrient_requirements"]["vitamin-c"] = 60    # was 75  (-20%)
info("Epoch B: protein req +20%, calorie +10%, vitamin-c -20%")

# 2. Add a new food option not present in Epoch A (set expansion)
data_b["nutritive_values"]["tofu"] = {
    "calorie": 5.0, "protein": 400, "calcium": 7.0,
    "iron": 55, "vitamin-a": 0, "vitamin-b1": 3.0,
    "vitamin-b2": 6.0, "niacin": 8, "vitamin-c": 0,
}
info("Epoch B: new food 'tofu' added to food set")

# 3. Change a nutritive value for an existing food (wheat protein drops)
data_b["nutritive_values"]["wheat"]["protein"] = 1200  # was 1411 (-15%)
info("Epoch B: wheat protein content -15%")

os.makedirs("tmp/rh_epochs", exist_ok=True)
with open(DIET_JSON_B, "w") as f:
    json.dump(data_b, f, indent=2)
ok(f"Epoch B data written to {DIET_JSON_B}")

foods_a   = set(data_a["nutritive_values"].keys())
foods_b   = set(data_b["nutritive_values"].keys())
new_foods = foods_b - foods_a
ok(f"Epoch A: {len(foods_a)} foods | Epoch B: {len(foods_b)} foods | new: {new_foods}")

# ---------------------------------------------------------------------------
# 1. load_model_from_py — same model file, two data configs
# ---------------------------------------------------------------------------
section("1. load_model_from_py — Mode A (same model, two data configs)")

model_a = load_model_from_py(DIET_PY, DIET_JSON)
ok(f"Epoch A model loaded: {type(model_a).__name__}")

model_b = load_model_from_py(DIET_PY, DIET_JSON_B)
ok(f"Epoch B model loaded: {type(model_b).__name__}")

n_foods_a = len(list(model_a.f))
n_foods_b = len(list(model_b.f))
ok(f"Epoch A: {n_foods_a} foods | Epoch B: {n_foods_b} foods (+{n_foods_b - n_foods_a} new)")
if n_foods_b != n_foods_a + 1:
    fail(f"Expected exactly 1 new food in Epoch B. Got A={n_foods_a}, B={n_foods_b}")

# ---------------------------------------------------------------------------
# 2. Solve both epochs
# ---------------------------------------------------------------------------
section("2. Solve both epochs with Gurobi")

def solve(model, name):
    solver = SolverFactory("gurobi")
    solver.options["TimeLimit"] = 60
    results = solver.solve(model, tee=False)
    tc  = str(results.solver.termination_condition)
    obj = pe.value(next(model.component_data_objects(pe.Objective, active=True)))
    info(f"{name}: status={tc}, obj={obj:.6f}")
    if tc != "optimal":
        fail(f"{name} did not solve optimally: {tc}")
    return tc

tc_a = solve(model_a, "Epoch A")
tc_b = solve(model_b, "Epoch B")
ok("Both epochs solved to optimality")

# ---------------------------------------------------------------------------
# 3. EpochStore.create_from_pyomo
# ---------------------------------------------------------------------------
section("3. EpochStore.create_from_pyomo")

import shutil
EPOCH_A_ID = "diet_epoch_a"
EPOCH_B_ID = "diet_epoch_b"
cmp_key    = "_vs_".join(sorted([EPOCH_A_ID, EPOCH_B_ID]))

for eid in [EPOCH_A_ID, EPOCH_B_ID]:
    p = os.path.join(EpochStore.BASE_DIR, eid)
    if os.path.exists(p):
        shutil.rmtree(p)
cmp_dir = os.path.join("tmp/rh_epochs/comparisons", cmp_key)
if os.path.exists(cmp_dir):
    shutil.rmtree(cmp_dir)

store_a = EpochStore.create_from_pyomo(
    model=model_a, epoch_id=EPOCH_A_ID,
    label="Diet — Original Requirements",
    source_py_path=DIET_PY, termination_condition=tc_a, data_json_path=DIET_JSON,
)
ok(f"store_a: {store_a}")

store_b = EpochStore.create_from_pyomo(
    model=model_b, epoch_id=EPOCH_B_ID,
    label="Diet — Updated Requirements + Tofu",
    source_py_path=DIET_PY, termination_condition=tc_b, data_json_path=DIET_JSON_B,
)
ok(f"store_b: {store_b}")

for eid in [EPOCH_A_ID, EPOCH_B_ID]:
    epoch_dir = os.path.join(EpochStore.BASE_DIR, eid)
    for fname in ["solution.json", "params.json", "structure.json", "model.pkl", "meta.json"]:
        fpath = os.path.join(epoch_dir, fname)
        if not os.path.exists(fpath):
            fail(f"Missing: {fpath}")
        info(f"{eid}/{fname}  ({os.path.getsize(fpath):,} bytes)")
    ok(f"All 5 files written for {eid}")

# ---------------------------------------------------------------------------
# 4. Lazy-load accessors (fresh store — no in-process cache)
# ---------------------------------------------------------------------------
section("4. Lazy-load accessors")

fresh_a = EpochStore.load(EPOCH_A_ID)
fresh_b = EpochStore.load(EPOCH_B_ID)

meta_a = fresh_a.get_meta()
meta_b = fresh_b.get_meta()
ok(f"get_meta() A: vars={meta_a['n_variables']}, cons={meta_a['n_constraints']}, "
   f"status='{meta_a['sol_status']}'")
ok(f"get_meta() B: vars={meta_b['n_variables']}, cons={meta_b['n_constraints']}, "
   f"status='{meta_b['sol_status']}'")

if meta_b["n_variables"] <= meta_a["n_variables"]:
    fail(f"Epoch B should have more vars (tofu). A={meta_a['n_variables']}, B={meta_b['n_variables']}")
ok(f"Epoch B has {meta_b['n_variables'] - meta_a['n_variables']} more variable(s) ✓")

sol_a = fresh_a.get_solution()
sol_b = fresh_b.get_solution()
ok(f"get_solution() A: obj={sol_a['objective']}, #vars={len(sol_a['variables'])}")
ok(f"get_solution() B: obj={sol_b['objective']}, #vars={len(sol_b['variables'])}")

# diet.py is a pure LP (no binary/integer vars) — duals should be populated
is_mip_a = sol_a.get("is_mip", True)
if is_mip_a:
    info("Epoch A detected as MIP — dual field omitted from constraints (correct for MIPs)")
else:
    info("Epoch A detected as pure LP — checking shadow prices are populated...")
    binding_with_duals = {
        k: v["dual"] for k, v in sol_a["constraints"].items()
        if v.get("is_binding") and v.get("dual") is not None
    }
    if not binding_with_duals:
        fail("Pure LP model: expected binding constraints to have non-null dual values")
    info(f"  Binding constraints with shadow prices: {binding_with_duals}")
    ok("LP shadow prices populated in solution.json ✓")

params_a = fresh_a.get_params()
struct_a  = fresh_a.get_structure()
struct_b  = fresh_b.get_structure()
param_families = [k for k in params_a if k != "epoch_id"]
ok(f"get_params() A: {len(param_families)} mutable param families")
ok(f"get_structure() A: {len(struct_a['index_sets'])} index sets, "
   f"{len(struct_a['variable_families'])} var families")

# 'tofu' in B's food set, not A's
foods_struct_a = struct_a["index_sets"].get("f", [])
foods_struct_b = struct_b["index_sets"].get("f", [])
if "tofu" in foods_struct_a:
    fail("'tofu' should NOT be in Epoch A food set")
if "tofu" not in foods_struct_b:
    fail("'tofu' should be in Epoch B food set")
ok("'tofu' present in Epoch B index set, absent in Epoch A ✓")

if fresh_a.get_duals() is not None:
    fail("get_duals() should be None before computed")
ok("get_duals() → None ✓")
if fresh_a.get_description() != "":
    fail("get_description() should be '' before saved")
ok("get_description() → '' ✓")

# ---------------------------------------------------------------------------
# 5. Compact LLM summaries
# ---------------------------------------------------------------------------
section("5. Compact LLM summaries")

sum_a = fresh_a.get_solution_summary(top_n=5)
sum_b = fresh_b.get_solution_summary(top_n=5)
ok(f"Epoch A summary: vars={sum_a['n_variables']}, "
   f"binding={sum_a['binding_constraint_families']}")
info(f"  objective: {sum_a['objective']}")
ok(f"Epoch B summary: vars={sum_b['n_variables']}, "
   f"binding={sum_b['binding_constraint_families']}")
info(f"  objective: {sum_b['objective']}")
ok(f"Epoch A param summary: {fresh_a.get_param_summary()}")
ok(f"Epoch B param summary: {fresh_b.get_param_summary()}")

# ---------------------------------------------------------------------------
# 6. Tier-4 drill-down
# ---------------------------------------------------------------------------
section("6. Tier-4 drill-down")

if struct_a["variable_families"]:
    fam = next(iter(struct_a["variable_families"]))
    vf_a = fresh_a.get_variable_family(fam)
    vf_b = fresh_b.get_variable_family(fam)
    ok(f"get_variable_family('{fam}'): A={len(vf_a)}, B={len(vf_b)}")
    if vf_a:
        k = next(iter(vf_a))
        info(f"  sample A: {k} → {vf_a[k]}")

if param_families:
    pf_a = fresh_a.get_param_family(param_families[0])
    pf_b = fresh_b.get_param_family(param_families[0])
    ok(f"get_param_family('{param_families[0]}'): A={len(pf_a)}, B={len(pf_b)}")

# ---------------------------------------------------------------------------
# 7. save_description / get_description
# ---------------------------------------------------------------------------
section("7. save_description / get_description")

desc_a = "Diet model Epoch A: original requirements, 20 food items."
desc_b = "Diet model Epoch B: updated requirements (protein +20%), new food 'tofu'."
fresh_a.save_description(desc_a)
fresh_b.save_description(desc_b)
if fresh_a.get_description() != desc_a: fail("Epoch A description mismatch")
if fresh_b.get_description() != desc_b: fail("Epoch B description mismatch")
ok("save_description / get_description round-trip ✓")

# ---------------------------------------------------------------------------
# 8. save_duals / get_duals
# ---------------------------------------------------------------------------
section("8. save_duals / get_duals")

mock_duals = {"calorie_con": 0.12, "protein_con": 0.45, "iron_con": 0.0}
fresh_a.save_duals(mock_duals)
ret = fresh_a.get_duals()
for k, v in mock_duals.items():
    if ret.get(k) != v:
        fail(f"Dual mismatch: {k}")
ok("save_duals / get_duals round-trip ✓")

# ---------------------------------------------------------------------------
# 9. save_comparison / load_comparison
# ---------------------------------------------------------------------------
section("9. Comparison cache: save_comparison / load_comparison")

for qt in ["qt1", "qt2", "qt3", "qt4"]:
    if fresh_a.load_comparison(EPOCH_B_ID, qt) is not None:
        fail(f"'{qt}' should be None before save")
ok("All 4 QTs return None before save ✓")

mock_qt1 = {
    "epoch_a": EPOCH_A_ID, "epoch_b": EPOCH_B_ID,
    "index_set_changes": {"f": {"new": list(new_foods), "removed": []}},
    "parameter_changes": {"nutrient_requirements": {"changed_count": 3}},
    "summary": {"n_param_families_changed": 1, "n_total_param_changes": 3},
}
fresh_a.save_comparison(EPOCH_B_ID, "qt1", mock_qt1)
ret_qt1 = fresh_a.load_comparison(EPOCH_B_ID, "qt1")
if ret_qt1 is None: fail("load_comparison returned None after save")
if ret_qt1["summary"]["n_total_param_changes"] != 3: fail("QT1 data mismatch")
ok("save_comparison / load_comparison round-trip ✓ (qt1)")

for qt in ["qt2", "qt3", "qt4"]:
    if fresh_a.load_comparison(EPOCH_B_ID, qt) is not None:
        fail(f"'{qt}' should still be None")
ok("qt2/qt3/qt4 still None (on-demand caching correct) ✓")

cmp_path = os.path.join("tmp/rh_epochs/comparisons", cmp_key, "qt1_structural.json")
if not os.path.exists(cmp_path):
    fail(f"qt1_structural.json not found: {cmp_path}")
ok(f"Comparison file at correct path ✓")

# ---------------------------------------------------------------------------
# 10. EpochStore.load — reload from disk
# ---------------------------------------------------------------------------
section("10. EpochStore.load (reload, cache cleared)")

rel = EpochStore.load(EPOCH_A_ID)
if rel.get_meta()["label"] != meta_a["label"]: fail("Meta mismatch after reload")
if len(rel.get_solution()["variables"]) != len(sol_a["variables"]): fail("Var count mismatch")
if rel.get_description() != desc_a: fail("Description mismatch")
if rel.get_duals() is None: fail("Duals should exist after reload")
if rel.load_comparison(EPOCH_B_ID, "qt1") is None: fail("QT1 should exist after reload")
ok("All accessors consistent after EpochStore.load() ✓")

# ---------------------------------------------------------------------------
# 11. get_pyomo_model — pkl round-trip
# ---------------------------------------------------------------------------
section("11. get_pyomo_model (pkl round-trip)")

for store, eid, meta in [(fresh_a, EPOCH_A_ID, meta_a), (fresh_b, EPOCH_B_ID, meta_b)]:
    restored = store.get_pyomo_model()
    if not isinstance(restored, pe.ConcreteModel):
        fail(f"Expected ConcreteModel for {eid}")
    n = sum(1 for v in restored.component_objects(pe.Var, active=True) for _ in v)
    if n != meta["n_variables"]:
        fail(f"{eid}: pkl has {n} vars, meta says {meta['n_variables']}")
    if hasattr(restored, "dual"):
        fail(f"{eid}: dual suffix not stripped")
    ok(f"{eid}: {n} vars match meta, dual stripped ✓")

n_a = meta_a["n_variables"]
n_b = meta_b["n_variables"]
ok(f"Epoch A pkl: {n_a} vars | Epoch B pkl: {n_b} vars (diff = {n_b - n_a})")

# ---------------------------------------------------------------------------
# 12. list_all_epochs — registry
# ---------------------------------------------------------------------------
section("12. list_all_epochs (registry)")

registry = EpochStore.list_all_epochs()
if EPOCH_A_ID not in registry: fail(f"'{EPOCH_A_ID}' not in registry")
if EPOCH_B_ID not in registry: fail(f"'{EPOCH_B_ID}' not in registry")
ok("Both epochs in registry ✓")
for eid, em in registry.items():
    info(f"{eid}: label='{em['label']}', status='{em['sol_status']}', "
         f"vars={em['n_variables']}, cons={em['n_constraints']}")

# ---------------------------------------------------------------------------
# 13. Structural & solution differences visible from stored data
# ---------------------------------------------------------------------------
section("13. Epoch differences visible from stored data")

obj_a = sol_a["objective"]["value"]
obj_b = sol_b["objective"]["value"]
info(f"Objective A = {obj_a:.6f}")
info(f"Objective B = {obj_b:.6f}  (change = {abs(obj_b-obj_a):.6f})")
if abs(obj_a - obj_b) < 1e-9:
    fail("Objectives are identical — epoch B changes had no effect")
ok("Objectives differ between epochs ✓")

if len(sol_b["variables"]) <= len(sol_a["variables"]):
    fail("Epoch B should have more variables (tofu added)")
ok(f"Variable count: A={len(sol_a['variables'])}, B={len(sol_b['variables'])} ✓")

req_a = fresh_a.get_param_family("nutrient_requirements")
req_b = fresh_b.get_param_family("nutrient_requirements")
if req_a and req_b:
    changed = {k for k in req_a if req_a.get(k) != req_b.get(k)}
    info(f"Changed nutrient requirements: {changed}")
    if not changed:
        fail("No changed requirements found — param data not stored correctly")
    ok(f"Changed requirements correctly stored: {changed} ✓")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print(f"\n{GREEN}{'='*60}")
print(f"  ALL 13 CHECKS PASSED — Phase 1 Data Layer working correctly")
print(f"{'='*60}{RESET}\n")

print("Files written to tmp/rh_epochs/:")
for root, dirs, files in os.walk("tmp/rh_epochs"):
    for fname in files:
        fpath = os.path.join(root, fname)
        print(f"  {fpath}  ({os.path.getsize(fpath):,} bytes)")
