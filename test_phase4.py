"""
Phase 4 + 5 Test Script — Agent Layer & UI Integration.

Run with the project venv (requires pyomo, litellm, google-adk[litellm], Gurobi):
    python test_phase4.py

Every test block is SELF-CONTAINED: imports happen inside the with-check block,
so a failed import in one section never cascades to break another.

Sections:
  1. Import chain — each module imported and verified independently
  2. rh_prompts — prompt builder for all 7 query types
  3. epoch_tools — tool signatures and QueryType validation
  4. rh_callback_tool — internal helpers (_try_parse_json, _SESSION_INIT_MESSAGE)
  5. Agent instantiation — create_rh_root_agent(), create_rh_comparison_agent()
  6. Syntax checks — all Phase 4+5 files (no imports needed)
  7. End-to-end smoke — real diet model: load, solve, QT1, prompts
"""

import json
import os
import sys
import py_compile
import time

# ---------------------------------------------------------------------------
# Minimal test harness — suppress exceptions so every block always runs
# ---------------------------------------------------------------------------
failures = []

def check(label):
    class _ctx:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb):
            if exc_type:
                failures.append(f"✗  {label}: {exc}")
                print(f"✗  {label}: {exc}")
                return True   # suppress — next block still runs
            print(f"✓  {label}")
    return _ctx()


# ===========================================================================
# 1. Import Chain — each block is independent
# ===========================================================================
print("\n=== 1. Import Chain ===")

with check("rh_constants — state keys and QueryType"):
    from rh_comparison.config.rh_constants import (
        QueryType, RH_APP_NAME, RH_PERSISTENT_STATES, RH_TEMPORARY_STATES,
        RH_SESSION_INITIALIZED, RH_EPOCH_A_ID, RH_EPOCH_B_ID,
        RH_EPOCH_A_META, RH_EPOCH_B_META, RH_DESCRIPTION_A, RH_DESCRIPTION_B,
        RH_QT1_RESULT, RH_QT2_RESULT, RH_QT3_RESULT, RH_QT4_RESULT, RH_QUERY_TYPE,
    )
    assert RH_APP_NAME == "optichat_rh"

with check("rh_prompts — build_root_agent_prompt, build_comparison_agent_prompt"):
    from rh_comparison.agents.rh_prompts import (
        build_root_agent_prompt, build_comparison_agent_prompt,
        _format_qt1, _format_qt2, _format_qt3, _format_qt4,
    )

with check("epoch_tools — set_query_type, get_epoch_data, get_comparison_json"):
    from rh_comparison.tools.epoch_tools import (
        set_query_type, get_epoch_data, get_comparison_json,
        _epoch_store_cache, _resolve_epoch_id,
    )

with check("rh_callback_tool — callbacks and internal helpers"):
    from rh_comparison.tools.rh_callback_tool import (
        rh_initialize_session, rh_check_llm_request,
        _try_parse_json, _generate_epoch_description,
        _build_epoch_store, _SESSION_INIT_MESSAGE,
    )

with check("rh_comparison_agent — create_rh_comparison_agent"):
    from rh_comparison.agents.rh_comparison_agent import create_rh_comparison_agent

with check("rh_root_agent — create_rh_root_agent"):
    from rh_comparison.agents.rh_root_agent import create_rh_root_agent

with check("rh_agent entry point re-exports create_rh_root_agent"):
    from rh_comparison.rh_agent import create_rh_root_agent as _rh_entry
    from rh_comparison.agents.rh_root_agent import create_rh_root_agent as _cra_direct
    assert _rh_entry is _cra_direct

with check("optichat_rh.agent — root_agent exported"):
    import optichat_rh.agent as _rh_mod
    assert hasattr(_rh_mod, "root_agent")

with check("rh_comparison.tools __init__ — lazy exports accessible"):
    import rh_comparison.tools as _t
    for _sym in ("set_query_type", "get_epoch_data", "get_comparison_json",
                 "rh_initialize_session", "rh_check_llm_request"):
        assert hasattr(_t, _sym), f"Missing: {_sym}"

with check("rh_comparison.agents __init__ — lazy exports accessible"):
    import rh_comparison.agents as _a
    assert hasattr(_a, "create_rh_root_agent")
    assert hasattr(_a, "create_rh_comparison_agent")


# ===========================================================================
# 2. rh_prompts — all 7 query types, formatters
# ===========================================================================
print("\n=== 2. rh_prompts ===")

with check("Root prompt (no session) — contains 'awaiting configuration'"):
    from rh_comparison.agents.rh_prompts import build_root_agent_prompt
    from rh_comparison.config.rh_constants import RH_PERSISTENT_STATES, RH_TEMPORARY_STATES
    _s = dict(RH_PERSISTENT_STATES, **RH_TEMPORARY_STATES)
    p = build_root_agent_prompt(_s)
    assert "awaiting configuration" in p and len(p) > 500

with check("Root prompt (initialized) — contains 'ready'"):
    from rh_comparison.agents.rh_prompts import build_root_agent_prompt
    from rh_comparison.config.rh_constants import (
        RH_PERSISTENT_STATES, RH_TEMPORARY_STATES,
        RH_SESSION_INITIALIZED, RH_EPOCH_A_META, RH_EPOCH_B_META,
    )
    _s = dict(RH_PERSISTENT_STATES, **RH_TEMPORARY_STATES)
    _s[RH_SESSION_INITIALIZED] = True
    _s[RH_EPOCH_A_META] = {"label": "A", "objective_value": 100.0,
                            "objective_sense": "minimize", "n_variables": 50,
                            "n_constraints": 30, "n_params": 10, "sol_status": "optimal"}
    _s[RH_EPOCH_B_META] = {"label": "B", "objective_value": 95.0,
                            "objective_sense": "minimize", "n_variables": 52,
                            "n_constraints": 31, "n_params": 10, "sol_status": "optimal"}
    assert "ready" in build_root_agent_prompt(_s)

with check("All 7 comparison prompts build without error (length > 200)"):
    from rh_comparison.agents.rh_prompts import build_comparison_agent_prompt
    from rh_comparison.config.rh_constants import (
        QueryType, RH_PERSISTENT_STATES, RH_TEMPORARY_STATES,
    )
    _s = dict(RH_PERSISTENT_STATES, **RH_TEMPORARY_STATES)
    for _qt in (QueryType.GENERAL, QueryType.RETRIEVAL, QueryType.MODEL_DESCRIPTION,
                QueryType.STRUCTURAL_CHANGE, QueryType.SOLUTION_DIFF,
                QueryType.BACKWARD_COMPAT, QueryType.ATTRIBUTION):
        assert len(build_comparison_agent_prompt(_s, _qt)) > 200

with check("Comparison prompt includes question-shape guidance"):
    from rh_comparison.agents.rh_prompts import build_comparison_agent_prompt
    from rh_comparison.config.rh_constants import RH_PERSISTENT_STATES, RH_TEMPORARY_STATES, QueryType
    _s = dict(RH_PERSISTENT_STATES, **RH_TEMPORARY_STATES)
    prompt = build_comparison_agent_prompt(_s, QueryType.GENERAL, "What changed overall?")
    assert "Detected focus: broad" in prompt
    assert "Context depth:" in prompt
    assert "Style example" in prompt

with check("Unknown query type falls back gracefully"):
    from rh_comparison.agents.rh_prompts import build_comparison_agent_prompt
    from rh_comparison.config.rh_constants import RH_PERSISTENT_STATES, RH_TEMPORARY_STATES
    _s = dict(RH_PERSISTENT_STATES, **RH_TEMPORARY_STATES)
    assert len(build_comparison_agent_prompt(_s, "UNKNOWN")) > 200

with check("All 4 QT formatters handle None without raising"):
    from rh_comparison.agents.rh_prompts import _format_qt1, _format_qt2, _format_qt3, _format_qt4
    for _fn in (_format_qt1, _format_qt2, _format_qt3, _format_qt4):
        assert "(Not yet computed)" in _fn(None)

with check("QT formatters handle real-shaped sample dicts"):
    from rh_comparison.agents.rh_prompts import _format_qt1, _format_qt2, _format_qt3, _format_qt4
    assert "demand" in _format_qt1({
        "summary": {
            "n_set_element_additions": 1,
            "n_set_element_removals": 0,
            "n_param_changes": 1,
            "n_constraint_additions": 0,
            "n_constraint_removals": 0,
        },
        "parameter_changes": {"by_family": {
            "demand": {
                "family_label": "Demand for each period",
                "added": {},
                "removed": {},
                "changed": [
                    {"index": "1", "value_a": 100, "value_b": 115, "rel_change": 0.15}
                ],
            }
        }},
        "labels": {"parameters": {"demand": "Demand for each period"}},
    }).lower()
    assert "95" in _format_qt2({
        "objective_change": {"value_a": 100.0, "value_b": 95.0, "delta": -5.0, "rel_change": -0.05},
        "summary": {"n_changed_variables": 5, "overall_churn_rate": 0.1,
                    "n_added_variables": 0, "n_removed_variables": 0,
                    "n_became_binding": 0, "n_became_nonbinding": 0},
        "binding_changes": {}, "top_changes": [],
    })
    assert "21.00%" in _format_qt3({
        "feasible": True, "termination_condition": "optimal",
        "n_common_vars_fixed": 10, "n_free_vars_in_b": 2,
        "forced_objective": 97.0, "optimal_objective_b": 80.0,
        "gap_rel": 0.21, "business_summary": "Yes. Still usable.",
    })
    assert "appeared in the later run" in _format_qt4({
        "parameter_shifts": [],
        "bottleneck_evolution": {"relieved": [], "new_bottlenecks": [], "persistent": []},
        "binary_flips": [{"variable": "open[1]", "direction": "0->1"}],
        "narrative": {"prior_state": "A", "parameter_shift": "B",
                      "solution_adaptation": "C", "outcome": "D", "new_state": "E"},
    })


# ===========================================================================
# 3. epoch_tools — self-contained
# ===========================================================================
print("\n=== 3. epoch_tools ===")

with check("_resolve_epoch_id — 'a'/'epoch_a'/'b'/'epoch_b' aliases and passthrough"):
    from rh_comparison.tools.epoch_tools import _resolve_epoch_id
    from rh_comparison.config.rh_constants import RH_EPOCH_A_ID, RH_EPOCH_B_ID
    _st = {RH_EPOCH_A_ID: "epoch_a_abc", RH_EPOCH_B_ID: "epoch_b_abc"}
    assert _resolve_epoch_id("a",         _st) == "epoch_a_abc"
    assert _resolve_epoch_id("epoch_a",   _st) == "epoch_a_abc"
    assert _resolve_epoch_id("b",         _st) == "epoch_b_abc"
    assert _resolve_epoch_id("epoch_b",   _st) == "epoch_b_abc"
    assert _resolve_epoch_id("custom_id", _st) == "custom_id"

with check("set_query_type — invalid type defaults to GENERAL"):
    from rh_comparison.tools.epoch_tools import set_query_type
    from rh_comparison.config.rh_constants import QueryType, RH_QUERY_TYPE
    class _MC: state = {}
    _c = _MC()
    result = set_query_type("BOGUS", _c)
    assert _c.state[RH_QUERY_TYPE] == QueryType.GENERAL
    assert "Routing updated" in result

with check("set_query_type — all 7 valid types register correctly"):
    from rh_comparison.tools.epoch_tools import set_query_type
    from rh_comparison.config.rh_constants import QueryType, RH_QUERY_TYPE
    class _MC2: state = {}
    _c2 = _MC2()
    for _qt in (QueryType.GENERAL, QueryType.RETRIEVAL, QueryType.MODEL_DESCRIPTION,
                QueryType.STRUCTURAL_CHANGE, QueryType.SOLUTION_DIFF,
                QueryType.BACKWARD_COMPAT, QueryType.ATTRIBUTION):
        set_query_type(_qt, _c2)
        assert _c2.state[RH_QUERY_TYPE] == _qt


# ===========================================================================
# 4. rh_callback_tool — internal helpers, self-contained
# ===========================================================================
print("\n=== 4. rh_callback_tool internal helpers ===")

with check("_try_parse_json — valid JSON str"):
    from rh_comparison.tools.rh_callback_tool import _try_parse_json
    assert _try_parse_json('{"epoch_a": {}, "epoch_b": {}}') == {"epoch_a": {}, "epoch_b": {}}

with check("_try_parse_json — valid JSON bytes"):
    from rh_comparison.tools.rh_callback_tool import _try_parse_json
    assert _try_parse_json(b'{"a": 1}') == {"a": 1}

with check("_try_parse_json — invalid str returns None"):
    from rh_comparison.tools.rh_callback_tool import _try_parse_json
    assert _try_parse_json("not json") is None

with check("_try_parse_json — JSON list (non-object) returns None"):
    from rh_comparison.tools.rh_callback_tool import _try_parse_json
    assert _try_parse_json("[1, 2, 3]") is None

with check("_try_parse_json — None returns None"):
    from rh_comparison.tools.rh_callback_tool import _try_parse_json
    assert _try_parse_json(None) is None

with check("_SESSION_INIT_MESSAGE contains required config field names"):
    from rh_comparison.tools.rh_callback_tool import _SESSION_INIT_MESSAGE
    assert "epoch_a"    in _SESSION_INIT_MESSAGE
    assert "model_path" in _SESSION_INIT_MESSAGE
    assert "label"      in _SESSION_INIT_MESSAGE

with check("_generate_epoch_description — scalar components do not show UnindexedComponent_set"):
    import pyomo.environ as pyo
    from rh_comparison.data_store.epoch_store import _extract_structure
    from rh_comparison.tools.rh_callback_tool import _generate_epoch_description

    class _FakeStore:
        def __init__(self, structure, meta):
            self._structure = structure
            self._meta = meta

        def get_structure(self):
            return self._structure

        def get_meta(self):
            return self._meta

    _m = pyo.ConcreteModel(doc="Simple cost-tracking model.")
    _m.cost = pyo.Var(doc="total food bill (dollars)")
    _m.obj = pyo.Objective(expr=_m.cost, sense=pyo.minimize, doc="minimize total food cost")
    _store = _FakeStore(
        _extract_structure(_m),
        {"objective_sense": "minimize"},
    )
    _desc = _generate_epoch_description(_store)
    assert "`cost`" in _desc
    assert "UnindexedComponent_set" not in _desc


# ===========================================================================
# 5. Agent instantiation — requires litellm + google-adk[litellm]
# ===========================================================================
print("\n=== 5. Agent Instantiation ===")

with check("create_rh_comparison_agent() — correct name"):
    from rh_comparison.agents.rh_comparison_agent import create_rh_comparison_agent
    from google.adk.agents import Agent
    _a = create_rh_comparison_agent()
    assert isinstance(_a, Agent)
    assert _a.name == "rh_comparison_agent"

with check("create_rh_root_agent() — correct name, 2 tools"):
    from rh_comparison.agents.rh_root_agent import create_rh_root_agent
    from google.adk.agents import Agent
    _a = create_rh_root_agent()
    assert isinstance(_a, Agent)
    assert _a.name == "rh_root_agent"
    assert len(_a.tools) == 2   # set_query_type + AgentTool(comparison_agent)

with check("optichat_rh.agent.root_agent — Agent named rh_root_agent"):
    import optichat_rh.agent as _rh_mod
    from google.adk.agents import Agent
    assert isinstance(_rh_mod.root_agent, Agent)
    assert _rh_mod.root_agent.name == "rh_root_agent"


# ===========================================================================
# 6. Syntax checks — py_compile only, no imports needed
# ===========================================================================
print("\n=== 6. Syntax Checks ===")

for _fp in [
    "app.py",
    "optichat_rh/__init__.py",
    "optichat_rh/agent.py",
    "rh_comparison/rh_agent.py",
    "rh_comparison/agents/__init__.py",
    "rh_comparison/agents/rh_prompts.py",
    "rh_comparison/agents/rh_comparison_agent.py",
    "rh_comparison/agents/rh_root_agent.py",
    "rh_comparison/tools/__init__.py",
    "rh_comparison/tools/epoch_tools.py",
    "rh_comparison/tools/rh_callback_tool.py",
]:
    with check(f"syntax OK: {_fp}"):
        py_compile.compile(_fp, doraise=True)


# ===========================================================================
# 7. End-to-end smoke — requires Gurobi + diet model + data file
# ===========================================================================
print("\n=== 7. End-to-End Smoke Test (requires Gurobi) ===")

_MODEL = os.path.abspath("Feas/diet.py")
_DATA  = os.path.abspath("Feas/diet_data.json")

if not (os.path.exists(_MODEL) and os.path.exists(_DATA)):
    print(f"  SKIP  diet files missing "
          f"(diet.py={os.path.exists(_MODEL)}, diet_data.json={os.path.exists(_DATA)})")
else:
    _ts = int(time.time())
    _EID_A = f"test_ph4_a_{_ts}"
    _EID_B = f"test_ph4_b_{_ts}"
    _store_a = None
    _store_b = None

    with check("_build_epoch_store: load + solve diet epoch_a"):
        from rh_comparison.tools.rh_callback_tool import _build_epoch_store
        _store_a = _build_epoch_store({
            "epoch_id":   _EID_A,
            "label":      "Diet Baseline",
            "model_path": _MODEL,
            "data_path":  _DATA,
        })
        _meta = _store_a.get_meta()
        assert _meta["sol_status"] == "optimal"
        print(f"       vars={_meta['n_variables']}, cons={_meta['n_constraints']}, "
              f"obj={_meta['objective_value']:.4f}")

    with check("_build_epoch_store: reuse existing epoch from disk"):
        from rh_comparison.tools.rh_callback_tool import _build_epoch_store
        _s2 = _build_epoch_store({
            "epoch_id": _EID_A, "label": "Diet Baseline",
            "model_path": _MODEL, "data_path": _DATA,
        })
        assert _s2.epoch_id == _EID_A

    with check("_build_epoch_store: load + solve diet epoch_b"):
        from rh_comparison.tools.rh_callback_tool import _build_epoch_store
        _store_b = _build_epoch_store({
            "epoch_id":   _EID_B,
            "label":      "Diet Updated",
            "model_path": _MODEL,
            "data_path":  _DATA,
        })
        assert _store_b.get_meta()["sol_status"] == "optimal"

    with check("_generate_epoch_description — non-empty structured text"):
        from rh_comparison.tools.rh_callback_tool import _generate_epoch_description
        if _store_a:
            _desc = _generate_epoch_description(_store_a)
            assert len(_desc) > 100
            assert all(section in _desc for section in ("**Sets**", "**Parameters**", "**Variables**", "**Constraints**", "**Objective**"))
            print(f"       Preview: {_desc[:120].replace(chr(10), ' ')}")

    with check("QT1 computable from real diet stores"):
        from rh_comparison.analytics import compute_structural_diff
        if _store_a and _store_b:
            _qt1 = compute_structural_diff(_store_a, _store_b)
            assert "summary" in _qt1
            print(f"       QT1 summary: {_qt1['summary']}")

    with check("QT2 computable from real diet stores"):
        from rh_comparison.analytics import compute_solution_diff, compute_structural_diff
        if _store_a and _store_b:
            _qt1 = compute_structural_diff(_store_a, _store_b)
            _qt2 = compute_solution_diff(_store_a, _store_b, qt1=_qt1)
            assert "objective_change" in _qt2
            print(f"       obj delta={_qt2['objective_change'].get('delta')}")

    with check("build_comparison_agent_prompt injects real epoch data"):
        from rh_comparison.agents.rh_prompts import (
            build_root_agent_prompt, build_comparison_agent_prompt,
        )
        from rh_comparison.analytics import compute_structural_diff, compute_solution_diff
        from rh_comparison.tools.rh_callback_tool import _generate_epoch_description
        from rh_comparison.config.rh_constants import (
            QueryType, RH_PERSISTENT_STATES, RH_TEMPORARY_STATES,
            RH_SESSION_INITIALIZED, RH_EPOCH_A_ID, RH_EPOCH_B_ID,
            RH_EPOCH_A_META, RH_EPOCH_B_META, RH_DESCRIPTION_A, RH_DESCRIPTION_B,
            RH_QT1_RESULT, RH_QT2_RESULT,
        )
        if _store_a and _store_b:
            _qt1 = compute_structural_diff(_store_a, _store_b)
            _qt2 = compute_solution_diff(_store_a, _store_b, qt1=_qt1)
            _st = dict(RH_PERSISTENT_STATES, **RH_TEMPORARY_STATES)
            _st[RH_SESSION_INITIALIZED] = True
            _st[RH_EPOCH_A_ID]    = _store_a.epoch_id
            _st[RH_EPOCH_B_ID]    = _store_b.epoch_id
            _st[RH_EPOCH_A_META]  = _store_a.get_meta()
            _st[RH_EPOCH_B_META]  = _store_b.get_meta()
            _st[RH_DESCRIPTION_A] = _generate_epoch_description(_store_a)
            _st[RH_DESCRIPTION_B] = _generate_epoch_description(_store_b)
            _st[RH_QT1_RESULT]    = _qt1
            _st[RH_QT2_RESULT]    = _qt2
            for _qt in (QueryType.GENERAL, QueryType.STRUCTURAL_CHANGE, QueryType.SOLUTION_DIFF):
                assert len(build_comparison_agent_prompt(_st, _qt)) > 300
            assert "ready" in build_root_agent_prompt(_st)


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
if failures:
    print(f"FAILED — {len(failures)} check(s) failed:\n")
    for _f in failures:
        print(f"  {_f}")
    sys.exit(1)
else:
    print("All Phase 4 + 5 checks passed.")
