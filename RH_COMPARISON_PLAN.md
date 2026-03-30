# Rolling Horizon Comparison Framework — Implementation Plan

## Implementation Notes

- The current RH implementation generates `description.txt` directly from stored model structure; it does not call OptiChat's `illustrator_agent`.
- QT3 should be read as a solver-backed compatibility check with IIS/minimal-cover/slack analysis on the fixed-plan model. Any older notes about evaluating later-only variables at lower bounds are obsolete.

## 0. Core Design Principles

1. **Zero interference with OptiChat** — all new code lives under `rh_comparison/`. The existing `optichat/` directory is never modified.
2. **Reuse, don't duplicate** — borrow `solve_model`, `extract_model_info`, `save_model_object`, `restore_model_object`, and `illustrator_agent` from OptiChat. Never copy them.
3. **Separate ADK agent tree** — the RH framework has its own root agent registered alongside the existing `root_agent`. Both run under the same `adk api_server` but as different apps or sessions.
4. **Efficient by design** — cheap analytics (QT1) are pre-computed eagerly at session init. Expensive analytics (QT2, QT3, QT4) are on-demand and cached. The LLM always receives targeted, pre-computed JSON — not raw model dumps.
5. **UI-level toggle only** — `app.py` gets a single mode radio button. All routing below it is handled by separate code paths that converge at the ADK `/run` endpoint.
6. **Stateless context injection** — every LLM call in RH mode receives the full relevant context (epoch metadata, descriptions, applicable QT result) injected fresh into the system prompt. We never rely on conversation history being preserved or available. Prompts are fully self-contained.
7. **Large-model-first design** — for models with tens of thousands of variables and constraints, the LLM always receives compact, pre-aggregated summaries (family-level counts, top-N changes, aggregate stats). Full detailed data lives on disk and is retrieved only when a user asks about a specific component. No component is listed individually unless explicitly requested.

---

## 1. OptiChat Illustrator Capability — Confirmed & Reused

**OptiChat already has full model description capability. We reuse it directly.**

The `illustrator_agent` in `optichat/sub_agents/illustrator/agent.py` uses the `get_model_info_for_description()` tool (`optichat/tools/illustrator_tool.py`) which:

- Loads the Pyomo model from `.pkl` and calls `pyomo2json` from `extractor.py` for a rich structured JSON
- Falls back to `format_hierarchical_components()` which groups all variables, constraints, and parameters by **family name** with element counts — e.g., `demand[…]` (count=240) instead of listing 240 individual entries. This makes it inherently efficient for large models.
- `format_pattern_based_summary()` provides an alternative grouped markdown format
- Generated descriptions are saved to `tmp/model_objects/generated_papers/{model}_description.txt`
- For infeasible models, the `diagnose_if_infeasible` before-agent callback automatically adds an infeasibility analysis section

**How we reuse it:** At init, we call the `illustrator_agent` once per epoch (same as OptiChat's `NEED_SYNTHETIC_PAPER` flow). The RH framework reads the saved `description.txt` files and injects them into every relevant LLM prompt.

---

## 2. What We Are Building (Paper → Code Map)

| Paper Section | What It Computes | Solver Calls | When Run |
|---|---|---|---|
| QT1: Structural & Parametric Changes | Period sets, index sets, param deltas, constraint template diff | 0 | Eagerly at init |
| QT2: Solution Differences | Objective delta, churn metrics, per-variable deltas | 0 | **On user demand** |
| QT3: Backward Compatibility Assessment | Fixed-var solve → gap OR IIS → minimal cover → slack min | 3–4 | On user demand |
| QT4: Solution Attribution Analysis | 4-solve MIP+incumbent LP, dual extraction, attribution chain | 4 | On user demand |
| Model Descriptions | Natural language description of each epoch + diff summary | 0 (LLM only) | Eagerly at init |
| Feasibility Report | Whether each epoch solved optimally or is infeasible | 0 (already solved) | Eagerly at init |

---

## 3. Directory Layout

```
OptiChat_Custom/
├── optichat/                          ← UNCHANGED
├── app.py                             ← MINIMAL CHANGE: mode radio button only
│
└── rh_comparison/
    ├── __init__.py
    ├── rh_agent.py                    ← Entry point: create_rh_root_agent()
    │
    ├── config/
    │   └── rh_constants.py            ← State keys, QT names, thresholds
    │
    ├── data_store/
    │   └── epoch_store.py             ← EpochStore: load/save/index epoch data
    │
    ├── analytics/
    │   ├── structural_diff.py         ← Algorithm 1 (QT1): pure Python, 0 solves
    │   ├── solution_diff.py           ← Algorithm 2 (QT2): pure Python, 0 solves
    │   ├── backward_compat.py         ← Algorithm 3 (QT3): 3–4 solver calls
    │   └── attribution_analysis.py    ← Algorithm 4 (QT4): 4 solver calls
    │
    ├── agents/
    │   ├── rh_root_agent.py           ← root_agent for RH mode
    │   ├── rh_comparison_agent.py     ← expert agent for comparison queries
    │   └── rh_prompts.py              ← prompts for all query classes
    │
    └── tools/
        ├── epoch_tools.py             ← ADK tools: get_epoch_data, get_comparison_json
        └── rh_callback_tool.py        ← Callbacks: session init, dynamic prompt injection
```

---

## 4. Data Structures

### 4.1 On-Disk: `tmp/rh_epochs/`

```
tmp/rh_epochs/
├── registry.json                      ← {epoch_id: EpochMeta}
│
├── {epoch_id}/
│   ├── model.pkl                      ← Pyomo model (cloudpickle, reuses OptiChat format)
│   ├── solution.json                  ← Flat variable+constraint solution snapshot
│   ├── params.json                    ← Compact {param_name: {index_key: value}} dict
│   ├── structure.json                 ← Temporal sets, index sets, constraint templates (family-grouped)
│   ├── description.txt                ← Natural language description (from illustrator_agent)
│   └── duals.json                     ← Dual values from incumbent LP (written on demand)
│
└── comparisons/
    └── {epoch_a}_vs_{epoch_b}/
        ├── qt1_structural.json        ← QT1 output (written at init, cached)
        ├── qt2_solution.json          ← QT2 output (on-demand, cached after first run)
        ├── qt3_backward.json          ← QT3 output (on-demand, cached after first run)
        └── qt4_attribution.json       ← QT4 output (on-demand, cached after first run)
```

### 4.2 `EpochMeta` (registry entry)
```json
{
  "epoch_id": "epoch_may_2025",
  "label": "May 2025",
  "model_name": "pricing_model",
  "loaded_at": "2025-05-01T10:00:00",
  "sol_status": "optimal",
  "objective_value": 98450.3,
  "objective_sense": "maximize",
  "time_periods": [1, 2, 3, 4, 5, 6, 7],
  "index_sets": {"products": ["A","B","C"], "stores": [1,2,3]},
  "n_variables": 420,
  "n_constraints": 380,
  "n_params": 96,
  "model_pkl_path": "tmp/rh_epochs/epoch_may_2025/model.pkl",
  "source_py_path": "tmp/rh_epochs/epoch_may_2025/model.py",
  "description_path": "tmp/rh_epochs/epoch_may_2025/description.txt"
}
```

### 4.3 `solution.json` (flat, all variables)
```json
{
  "epoch_id": "epoch_may_2025",
  "objective": {"value": 98450.3, "sense": "maximize", "status": "optimal"},
  "variables": {
    "price[1,A]": {"value": 12.5, "is_binary": false, "is_integer": false},
    "promote[1,A]": {"value": 1.0, "is_binary": true},
    "budget[A]": {"value": 5000.0, "is_binary": false}
  },
  "constraints": {
    "budget_con[A]": {"is_binding": true, "slack": 0.0, "dual": null},
    "demand_con[1,A]": {"is_binding": false, "slack": 45.2, "dual": null}
  }
}
```

For large models (50k+ variables), all values are stored here but the LLM is never given this raw file. Instead, compact family-level summaries are derived from it on demand.

### 4.4 `params.json` (compact, param-name keyed)
```json
{
  "demand": {"(1,'A')": 100.0, "(2,'A')": 150.0, "(1,'B')": 80.0},
  "capacity": {"('A',)": 500.0, "('B',)": 350.0},
  "price_coef": {"(1,'A')": 2.5, "(1,'B')": 3.0}
}
```

### 4.5 `structure.json` (family-grouped, efficient for large models)

All variables and constraints are represented at the **family** level, not as individual components. A model with 50,000 variable instances across 30 families is represented with 30 entries, not 50,000.

```json
{
  "time_periods": [1, 2, 3, 4, 5, 6, 7],
  "index_sets": {
    "products": ["A", "B", "C"],
    "stores": [1, 2, 3]
  },
  "constraint_templates": {
    "budget_con": "sum(spend[t,i] for t in T, i in I) <= budget[i]",
    "demand_con": "sales[t,i] <= demand[t,i] * promote[t,i]",
    "capacity_con": "sum(sales[t,i] for i in I) <= capacity[t]"
  },
  "variable_families": {
    "price": {"indexed_over": ["T","I"], "type": "continuous", "count": 21},
    "promote": {"indexed_over": ["T","I"], "type": "binary", "count": 21},
    "budget": {"indexed_over": ["I"], "type": "continuous", "count": 3}
  },
  "param_families": {
    "demand": {"indexed_over": ["T","I"], "count": 21},
    "capacity": {"indexed_over": ["I"], "count": 3}
  }
}
```

### 4.6 QT1 Output JSON (`qt1_structural.json`)
```json
{
  "epoch_a": "epoch_may_2025",
  "epoch_b": "epoch_jun_2025",
  "temporal_changes": {
    "T_overlap": [1,2,3,4,5],
    "T_new": [6,7,8,9],
    "T_removed": [],
    "pattern": "expansion"
  },
  "index_set_changes": {
    "products": {"overlap": ["A","B"], "new": ["C"], "removed": []},
    "stores": {"overlap": [1,2,3], "new": [], "removed": []}
  },
  "parameter_changes": {
    "demand": {
      "changed_count": 8,
      "total_overlap_count": 15,
      "pct_changed": 0.533,
      "changes": [
        {"index": "(1,'A')", "delta": 20.0, "pct_change": 0.20, "direction": "increase", "magnitude": "moderate"},
        {"index": "(2,'B')", "delta": -10.0, "pct_change": -0.067, "direction": "decrease", "magnitude": "small"}
      ]
    }
  },
  "constraint_structure_changes": {
    "added_templates": ["promo_budget_con"],
    "removed_templates": []
  },
  "summary": {
    "n_param_families_changed": 2,
    "n_total_param_changes": 11,
    "n_new_constraints": 1,
    "n_removed_constraints": 0,
    "n_new_time_periods": 4,
    "n_new_index_elements": 1
  }
}
```

The `summary` sub-object is what the LLM receives by default for large models. Full `parameter_changes` details are on disk and retrieved on specific request.

### 4.7 QT2 Output JSON (`qt2_solution.json`)
```json
{
  "epoch_a": "epoch_may_2025",
  "epoch_b": "epoch_jun_2025",
  "objective": {
    "A": 98450.3, "B": 112340.7,
    "delta_abs": 13890.4, "delta_rel": 0.141,
    "interpretation": "14.1% improvement in epoch B"
  },
  "churn_summary": {
    "promote": {"rate": 0.33, "changed_count": 4, "total_overlap": 12, "type": "binary"},
    "price": {"rate": 0.08, "changed_count": 1, "total_overlap": 12, "type": "continuous"}
  },
  "top_changes": [
    {"variable": "promote[2,A]", "A": 0, "B": 1, "delta": 1, "type": "binary", "direction": "activated"},
    {"variable": "price[1,B]", "A": 12.5, "B": 15.0, "delta": 2.5, "delta_rel": 0.20}
  ],
  "new_period_count": 4,
  "new_option_count": 1,
  "full_variable_changes_on_disk": true
}
```

`top_changes` contains the N most significant changes by absolute value. All individual variable deltas are stored on disk in the same file under `variable_changes` (not shown here for brevity) and retrieved when a user asks about a specific variable or family.

### 4.8 QT3 Output JSON (`qt3_backward.json`)
```json
{
  "epoch_a": "epoch_may_2025",
  "epoch_b": "epoch_jun_2025",
  "feasible": false,
  "iis_rounds": [
    {"round": 1, "constraints": ["promo_budget_con[A]", "demand_con[2,A]"]},
    {"round": 2, "constraints": ["capacity_con[3]"]}
  ],
  "minimal_cover": ["promo_budget_con[A]", "capacity_con[3]"],
  "required_slacks": {
    "promo_budget_con[A]": {"slack": 150.0, "current_rhs": 3000.0, "adjusted_rhs": 3150.0, "governing_param": "promo_budget[A]"},
    "capacity_con[3]": {"slack": 25.0, "current_rhs": 500.0, "adjusted_rhs": 525.0, "governing_param": "capacity[3]"}
  }
}
```

### 4.9 QT4 Output JSON (`qt4_attribution.json`)
```json
{
  "epoch_a": "epoch_may_2025",
  "epoch_b": "epoch_jun_2025",
  "parameter_triggers": [
    {"param": "demand", "index": "(2,'A')", "delta": 20.0, "pct_change": 0.20, "magnitude": "moderate"}
  ],
  "bottleneck_evolution": {
    "relieved": [{"constraint": "demand_con[2,A]", "dual_A": 45.2, "dual_B": 0.0}],
    "new_bottlenecks": [{"constraint": "promo_budget_con[A]", "dual_A": 0.0, "dual_B": 28.3}],
    "persistent": []
  },
  "binary_flips": [
    {"variable": "promote[2,A]", "direction": "0->1", "resistance": 120.5,
     "attributed_constraint": "demand_con[2,A]",
     "interpretation": "Activated to meet increased demand at store 2 for product A"}
  ],
  "continuous_adjustments": [
    {"variable": "price[2,A]", "delta": 2.5, "linked_constraint": "demand_con[2,A]"}
  ],
  "narrative": {
    "trigger": "Demand for product A at store 2 increased by 20% (from 100 to 120 units).",
    "stress": "In epoch A, the demand constraint for store 2 was binding with shadow price 45.2.",
    "action": "Promotion for product A at store 2 was activated (flip 0→1) overcoming resistance of 120.5.",
    "execution": "Price for product A at store 2 increased by $2.50, exploiting the relaxed demand constraint.",
    "consequence": "The promotion budget constraint for product A became a new binding constraint with shadow price 28.3."
  }
}
```

---

## 5. `epoch_store.py` — Core Data Layer

```python
class EpochStore:
    """
    Thin wrapper around tmp/rh_epochs/{epoch_id}/.
    Handles load/save for all epoch data files.
    Never holds Pyomo models in memory — loads on demand.
    """
    def __init__(self, epoch_id: str): ...

    # Build from scratch (called at session init)
    # Supports BOTH upload modes:
    #   Mode A: source_py_path + data_json_path (two separate files)
    #   Mode B: source_py_path only (data embedded in the .py file)
    @classmethod
    def create_from_pyomo(cls, model, version_name: str, label: str,
                          source_py_path: str,
                          data_json_path: str = None) -> "EpochStore": ...

    # Load from existing disk files (lazy, called on demand)
    @classmethod
    def load(cls, epoch_id: str) -> "EpochStore": ...

    # Core accessors (load from disk on first call, cache in-process)
    def get_solution(self) -> dict: ...         # Loads solution.json
    def get_params(self) -> dict: ...           # Loads params.json
    def get_structure(self) -> dict: ...        # Loads structure.json
    def get_pyomo_model(self): ...              # Restores .pkl (expensive, used sparingly for QT3/QT4)
    def get_duals(self) -> dict: ...            # Loads duals.json (written by attribution_analysis)
    def get_description(self) -> str: ...       # Reads description.txt (from illustrator_agent)

    # Write methods
    def save_duals(self, duals: dict): ...
    def save_description(self, text: str): ...

    # Cache comparison results
    def save_comparison(self, other_id: str, qt: str, data: dict): ...
    def load_comparison(self, other_id: str, qt: str) -> dict | None: ...  # None = not yet computed

    # Large model summary helpers
    def get_solution_summary(self, top_n: int = 10) -> dict: ...
    # Returns: {objective, n_variables, n_constraints, top_N_active_binary_families,
    #           binding_constraint_families, non-default-value param families}
    # Used for LLM context — never exposes all 50k rows at once
```

**Key efficiency decision:** `get_params()` and `get_solution()` are pure JSON reads — no Pyomo deserialization. `get_pyomo_model()` is only called for QT3/QT4 solver operations. `get_solution_summary()` gives the LLM a compact picture of a large solution without listing individual components.

---

## 6. Large Model Strategy (Design for Scale)

For models with 50k+ variables and constraints, the framework uses a **tiered access pattern**:

| Tier | What's Stored | Used By | Size |
|---|---|---|---|
| Tier 1: Meta | `registry.json` EpochMeta | All LLM prompts, always | ~1KB |
| Tier 2: Descriptions | `description.txt` (from illustrator) | LLM prompts for QT1/model questions | ~2–5KB |
| Tier 3: Compact QT summaries | QT1 `summary` block, QT2 `churn_summary` + `top_changes` | LLM prompts for QT queries | ~3–8KB |
| Tier 4: Full QT detail | Full `parameter_changes`, `variable_changes` in QT JSON | Retrieved on specific component request | ~50KB–5MB |
| Tier 5: Raw solution/params | `solution.json`, `params.json` | Analytics functions only; never sent to LLM | Up to 100MB |
| Tier 6: Pyomo model | `.pkl` | QT3/QT4 solver operations only | Up to 1GB |

**How the LLM gets context for large models:**

- `structure.json` always groups by family (e.g., `price[T,I]` count=50000, not 50000 rows)
- `illustrator_agent` uses `format_hierarchical_components()` which is already family-grouped
- QT1 always reports aggregate counts first (`n_param_families_changed`, `n_total_param_changes`) then lists individual changes ranked by magnitude
- QT2 always reports `churn_summary` (family-level rates) first, then `top_changes` (top-N individual)
- The `rh_check_llm_request` callback injects only the appropriate tier for the current query type
- When a user asks "show me all changes to demand[2,*]", the tool returns the filtered subset from disk — not all changes at once

**Principle:** Aggregate stats tell the LLM the shape of the problem. Individual details are fetched on explicit request. This works for any model size without changing the architecture.

---

## 7. `analytics/structural_diff.py` — Algorithm 1 (QT1)

```python
def compute_structural_diff(store_a: EpochStore, store_b: EpochStore) -> dict:
    """
    Pure Python. 0 solver calls. O(n_params).
    Uses only store_a.get_structure(), store_b.get_structure(),
    store_a.get_params(), store_b.get_params().
    Returns QT1 JSON dict (see Section 4.6).
    """
    struct_a = store_a.get_structure()
    struct_b = store_b.get_structure()
    params_a = store_a.get_params()
    params_b = store_b.get_params()

    # 1. Temporal comparison: T_overlap, T_new, T_removed, pattern
    T_a = set(struct_a["time_periods"])
    T_b = set(struct_b["time_periods"])

    # 2. Index set comparison (per set: overlap, new, removed)

    # 3. Parameter delta computation (only on overlapping indices)
    #    Magnitude thresholds: small < 10%, moderate 10–30%, large > 30%
    #    Stores full per-index changes in output, plus aggregate summary block

    # 4. Constraint template diff (set difference on template names)

    # 5. Compute summary block for compact LLM context

    return qt1_json
```

---

## 8. `analytics/solution_diff.py` — Algorithm 2 (QT2)

```python
def compute_solution_diff(store_a: EpochStore, store_b: EpochStore,
                          qt1: dict, tol: float = 1e-6) -> dict:
    """
    Pure Python. 0 solver calls. O(n_vars).
    Uses store_a/b.get_solution() only — no Pyomo model needed.
    Computes full variable delta table; stores on disk; returns compact summary.
    """
    sol_a = store_a.get_solution()
    sol_b = store_b.get_solution()

    # 1. Objective comparison
    # 2. Churn metrics by family (aggregate first, per-var details stored in output JSON)
    # 3. top_changes: top-N by |delta_rel| for LLM context
    # 4. New period decisions, new option decisions, removed decisions (counts + examples)

    return qt2_json  # includes full variable_changes on disk, compact summary in top-level keys


def get_variable_family_details(store_a: EpochStore, store_b: EpochStore,
                                 qt2_path: str, family_prefix: str) -> dict:
    """
    On-demand accessor: returns full per-variable changes for a specific family.
    Called by the epoch_tools ADK tool when a user asks about a specific variable.
    Reads the qt2_solution.json from disk and filters to the requested family.
    """
```

---

## 9. `analytics/backward_compat.py` — Algorithm 3 (QT3)

```python
def assess_backward_compat(store_a: EpochStore, store_b: EpochStore,
                           qt1: dict, qt2: dict) -> dict:
    """
    3–4 solver calls using Gurobi.
    Reuses OptiChat's solve infrastructure.
    """
    from optichat.tools.shortcut_functions import solve_model  # reuse
    from optichat.tools.extract_tool import extract_model_info  # reuse

    model_b = store_b.get_pyomo_model()
    sol_a = store_a.get_solution()

    # 1. Fix overlapping variables to solution A values
    _fix_overlapping_vars(model_b, sol_a, qt1)

    # 2. Solve fixed-variable model B
    result = _solve_fixed(model_b)

    if result.feasible:
        gap = (qt2["objective"]["B"] - result.obj) / abs(qt2["objective"]["B"])
        return {"feasible": True, "gap": gap, "cost": qt2["objective"]["B"] - result.obj}

    # 3. Iterative IIS detection (up to 2 rounds)
    iis_sets = _iterative_iis(model_b)

    # 4. Minimal cover heuristic (greedy, O(k*m))
    cover = _minimal_cover_heuristic(iis_sets)

    # 5. Slack minimization LP
    slacks = _slack_minimization(model_b, cover)

    return {"feasible": False, "iis_rounds": iis_sets,
            "minimal_cover": cover, "required_slacks": slacks}
```

---

## 10. `analytics/attribution_analysis.py` — Algorithm 4 (QT4: Solution Attribution Analysis)

This analysis identifies which parameter changes drove specific solution changes between epochs, by extracting shadow prices from incumbent LPs (fixing all integer variables and re-solving as pure LP).

```python
def compute_attribution_analysis(store_a: EpochStore, store_b: EpochStore,
                                  qt1: dict, qt2: dict) -> dict:
    """
    4 solver calls total: 2 MIP (already done at init), 2 new incumbent LPs.
    Attributes solution changes to specific parameter triggers via shadow prices.
    """
    # Phase 1: Incumbent LP dual extraction (cached per epoch)
    duals_a = _get_or_compute_incumbent_lp(store_a)   # checks duals.json cache first
    duals_b = _get_or_compute_incumbent_lp(store_b)

    # Phase 2: Attribution linking
    triggers = _identify_parameter_triggers(qt1, threshold=0.10)
    binary_flips = _identify_binary_flips(qt2)
    bottleneck_evolution = _categorize_constraints(duals_a, duals_b, threshold_pct=0.10)
    attribution_links = _map_flips_to_constraints(binary_flips, duals_a, store_a.get_structure())
    continuous_adjustments = _quantify_continuous_adjustments(qt2, bottleneck_evolution)

    # Phase 3: Narrative synthesis (template-based, fills 5-component structure)
    narrative = _synthesize_narrative(triggers, bottleneck_evolution, binary_flips,
                                       continuous_adjustments, qt1)

    return {"parameter_triggers": triggers,
            "bottleneck_evolution": bottleneck_evolution,
            "binary_flips": binary_flips,
            "continuous_adjustments": continuous_adjustments,
            "narrative": narrative}


def _get_or_compute_incumbent_lp(store: EpochStore) -> dict:
    """Check duals.json cache before running the LP solve."""
    if duals := store.get_duals():
        return duals
    duals = _solve_incumbent_lp(store.get_pyomo_model(), store.get_solution())
    store.save_duals(duals)
    return duals
```

---

## 11. Agent Architecture

### `rh_agent.py` — Entry Point
```python
def create_rh_root_agent() -> Agent:
    """Parallel to optichat/agent.py — creates the RH comparison agent tree."""
    rh_comparison_agent = create_rh_comparison_agent()
    from optichat.sub_agents.illustrator.agent import create_illustrator_agent
    illustrator_agent = create_illustrator_agent()  # Reuse OptiChat's illustrator directly

    return Agent(
        name="rh_root_agent",
        model=gpt_5_mini,
        tools=[AgentTool(rh_comparison_agent), AgentTool(illustrator_agent)],
        instruction=RH_ROOT_PROMPT,
        before_agent_callback=rh_initialize_session,
        before_model_callback=rh_check_llm_request,
        after_tool_callback=rh_handle_illustrator_response
    )
```

### `rh_comparison_agent.py`
```python
def create_rh_comparison_agent() -> Agent:
    return Agent(
        name="rh_comparison_agent",
        model=gpt_5,
        tools=[
            get_epoch_data,          # Get epoch metadata/component values from EpochStore
            get_comparison_json,     # Return pre-computed QT JSON or trigger compute + return
        ],
        instruction=RH_COMPARISON_BASE_PROMPT,
        before_model_callback=rh_check_llm_request  # Injects full context per call
    )
```

### ADK App Registration
```python
# rh_comparison/ gets its own app name: "optichat_rh"
# optichat/ keeps its app name: "optichat"
# Both registered in the same adk api_server process
```

---

## 12. Query Classification (RH Mode)

The `rh_root_agent` classifies every user query into one of:

| Tag | Maps To | Pre-computed? |
|---|---|---|
| `[MODEL_DESCRIPTION]` | Epoch descriptions from illustrator + QT1 summary | Yes (at init) |
| `[STRUCTURAL_CHANGE]` | QT1 JSON | Yes (at init) |
| `[SOLUTION_DIFF]` | QT2 JSON | On-demand + cached |
| `[BACKWARD_COMPAT]` | QT3 JSON | On-demand + cached |
| `[ATTRIBUTION]` | QT4 JSON | On-demand + cached |
| `[GENERAL]` | Direct answer from root | N/A |

---

## 13. Stateless Context Injection (RH Callback)

Every LLM call in RH mode is **fully self-contained**. We never assume conversation history is available. The `rh_check_llm_request` callback (in `rh_callback_tool.py`) rebuilds the complete system prompt on every LLM call:

```python
def rh_check_llm_request(callback_context, llm_request):
    """
    Runs before EVERY LLM call in RH mode.
    Always injects: epoch metadata for both epochs, feasibility status,
    epoch descriptions, and the relevant QT result for the current query type.
    Never relies on conversation history.
    """
    epoch_a_meta = callback_context.state.get(RH_EPOCH_A_META, {})
    epoch_b_meta = callback_context.state.get(RH_EPOCH_B_META, {})
    desc_a = callback_context.state.get(RH_DESCRIPTION_A, "")
    desc_b = callback_context.state.get(RH_DESCRIPTION_B, "")
    query_type = callback_context.state.get(RH_QUERY_TYPE, "GENERAL")

    # Select the appropriate QT context for this query
    qt_context = None
    if query_type == "STRUCTURAL_CHANGE":
        qt_context = callback_context.state.get(RH_QT1_RESULT)
    elif query_type == "SOLUTION_DIFF":
        qt_context = callback_context.state.get(RH_QT2_RESULT)
        if qt_context is None:
            qt_context = _trigger_qt2(callback_context)  # compute now, cache
    elif query_type == "BACKWARD_COMPAT":
        qt_context = callback_context.state.get(RH_QT3_RESULT)
        if qt_context is None:
            qt_context = _trigger_qt3(callback_context)
    elif query_type == "ATTRIBUTION":
        qt_context = callback_context.state.get(RH_QT4_RESULT)
        if qt_context is None:
            qt_context = _trigger_qt4(callback_context)

    # Build fully self-contained prompt (no history assumed)
    new_prompt = get_rh_comparison_prompt(
        query_type=query_type,
        epoch_a_meta=epoch_a_meta,
        epoch_b_meta=epoch_b_meta,
        description_a=desc_a,
        description_b=desc_b,
        qt_context=qt_context   # Compact summary for large models (see Section 6)
    )
    llm_request.config.system_instruction = new_prompt
```

**Context always includes:**
- Both epochs' metadata (name, feasibility status, objective value, variable/constraint counts)
- Both epochs' natural language descriptions (from illustrator_agent)
- The relevant QT result in compact form (aggregate stats + top-N changes)
- Instructions for how to fetch additional detail via tools if needed

---

## 14. Session State for RH Mode

```python
# rh_comparison/config/rh_constants.py

RH_SESSION_INITIALIZED = "RH_SESSION_INITIALIZED"
RH_EPOCH_A_META = "RH_EPOCH_A_META"       # EpochMeta dict
RH_EPOCH_B_META = "RH_EPOCH_B_META"
RH_EPOCH_A_ID = "RH_EPOCH_A_ID"
RH_EPOCH_B_ID = "RH_EPOCH_B_ID"

RH_QT1_RESULT = "RH_QT1_RESULT"           # Computed eagerly at init
RH_QT2_RESULT = "RH_QT2_RESULT"           # Computed on first demand, then cached
RH_QT3_RESULT = "RH_QT3_RESULT"           # Computed on first demand, then cached
RH_QT4_RESULT = "RH_QT4_RESULT"           # Computed on first demand, then cached

RH_DESCRIPTION_A = "RH_DESCRIPTION_A"     # Epoch A natural language description
RH_DESCRIPTION_B = "RH_DESCRIPTION_B"     # Epoch B natural language description
RH_QUERY_TYPE = "RH_QUERY_TYPE"           # Current query classification

RH_PERSISTENT_STATES = {
    RH_SESSION_INITIALIZED: False,
    RH_EPOCH_A_META: {}, RH_EPOCH_B_META: {},
    RH_EPOCH_A_ID: "", RH_EPOCH_B_ID: "",
    RH_QT1_RESULT: None,   # populated at init
    RH_QT2_RESULT: None,   # populated on first demand
    RH_QT3_RESULT: None,   # populated on first demand
    RH_QT4_RESULT: None,   # populated on first demand
    RH_DESCRIPTION_A: "", RH_DESCRIPTION_B: "",
}

RH_TEMPORARY_STATES = {
    RH_QUERY_TYPE: "GENERAL",
}
```

---

## 15. Config Format (Two Upload Modes)

The user uploads a single JSON config. Two formats are supported:

### Mode A — Model file + separate data file
Use this when the `.py` model file is parameterized and data is provided externally as JSON.

```json
{
  "upload_mode": "separate",
  "epoch_a": {
    "label": "May 2025",
    "model_py": "Pricing/may_model.py",
    "data_json": "Pricing/may_data.json"
  },
  "epoch_b": {
    "label": "June 2025",
    "model_py": "Pricing/jun_model.py",
    "data_json": "Pricing/jun_data.json"
  }
}
```

### Mode B — Model file with data embedded
Use this when each `.py` file already contains its own data and is fully self-contained.

```json
{
  "upload_mode": "embedded",
  "epoch_a": {
    "label": "May 2025",
    "model_py": "Pricing/may_model_with_data.py"
  },
  "epoch_b": {
    "label": "June 2025",
    "model_py": "Pricing/jun_model_with_data.py"
  }
}
```

Both modes produce the same `EpochStore` on disk. The `upload_mode` field determines how `_load_model_from_py()` is called in the init pipeline:

```python
def _load_model_from_py(model_py: str, data_json: str = None) -> ConcreteModel:
    if data_json:
        # Load model structure from .py, load data from JSON, instantiate ConcreteModel
        ...
    else:
        # Execute .py directly — data is embedded, returns a ConcreteModel
        ...
```

---

## 16. Streamlit UI Changes (`app.py`)

**Total change to `app.py`: ~30 lines.** A mode selector is added at the top of the sidebar. Everything else is conditional:

```python
# --- Mode Selector ---
mode = st.sidebar.radio("🔀 Mode",
                         ["Standard OptiChat", "Rolling Horizon Comparison"],
                         index=0)

APP_NAME = "optichat" if mode == "Standard OptiChat" else "optichat_rh"

if mode == "Rolling Horizon Comparison":
    st.sidebar.markdown("### Upload RH Config")
    rh_config = st.sidebar.file_uploader("RH Config JSON", type=["json"])
    # Shows two upload slots for model .py files (and optionally data .json files)
    # depending on upload_mode in config
    st.sidebar.markdown("---")
    # Example query hints shown below chat input:
    # "What changed structurally between the two epochs?"
    # "Are both models feasible?"
    # "Why is the solution different?"
    # "Can epoch A's solution work in epoch B's model?"
    ...
else:
    # Existing OptiChat UI code unchanged
    ...
```

---

## 17. Initialization Flow (RH Mode)

When a user uploads the RH config JSON and model files, `rh_initialize_session` runs **once**:

1. **Load & Solve Epoch A**
   - `_load_model_from_py(model_py, data_json)` → Pyomo `ConcreteModel`
   - `solve_model(model, ver, models_dict, ...)` — reuses OptiChat's solver wrapper
   - `EpochStore.create_from_pyomo(model, ...)` → writes `model.pkl`, `solution.json`, `params.json`, `structure.json`

2. **Load & Solve Epoch B** — same as above

3. **Feasibility Report** — immediately available from `sol_status` in both EpochMeta entries. If either epoch is infeasible, the init message reports this clearly.

4. **Pre-compute QT1** (structural + parametric differences)
   - `compute_structural_diff(store_a, store_b)` → writes `qt1_structural.json` → cache in `RH_QT1_RESULT`
   - This captures all parameter changes and structural differences between the two models

5. **Trigger Illustrator for both epochs**
   - Reuses OptiChat's `illustrator_agent` → writes `description.txt` for each epoch
   - Descriptions are saved to `RH_DESCRIPTION_A` / `RH_DESCRIPTION_B` in session state

6. **Generate Welcome Message** — the system synthesizes and displays:
   - Feasibility status of both epochs (optimal/infeasible)
   - Natural language summaries of each model (from descriptions)
   - Structural differences: new/removed constraint templates, changed index sets, new time periods
   - Parameter differences: which parameter families changed, how many, direction/magnitude summary
   - This is the answer to QT1 and the model description query, delivered automatically

**What is NOT computed at init (on-demand only):**
- QT2 (solution differences) — triggered when user asks about solution changes
- QT3 (backward compatibility assessment) — triggered when user asks about applying old solution
- QT4 (solution attribution analysis) — triggered when user asks why specific decisions changed

Steps 1–4 complete synchronously before any chat. Steps 5–6 are triggered as the first internal "message" (mirroring OptiChat's `NEED_SYNTHETIC_PAPER` flow). Step 6 is the welcome message shown in the chat.

---

## 18. Tool Signatures (ADK Tools)

```python
def get_epoch_data(epoch_id: str, component_type: str, pattern: str,
                   tool_context: ToolContext) -> dict:
    """
    Fast lookup of epoch component values.
    Mirrors OptiChat's get_model_components but works on EpochStore.
    Supports: component_type in ['variable', 'constraint', 'parameter', 'summary']
    Pattern supports wildcard (*) and substring matching, same as OptiChat.
    Returns compact family summary by default; use pattern for specific components.
    """

def get_comparison_json(query_type: str, detail_level: str,
                        family_filter: str, tool_context: ToolContext) -> dict:
    """
    Returns the QT result for the given query type.
    For QT2/QT3/QT4: triggers computation if not cached, returns result.

    query_type: one of 'qt1', 'qt2', 'qt3', 'qt4'
    detail_level: 'summary' (default) or 'full' — controls how much detail is returned
    family_filter: optional, e.g. 'demand' to return only changes to the demand family
    """
```

---

## 19. Implementation Phases

### Phase 1 — Data Layer (Foundation)
- `rh_comparison/config/rh_constants.py`
- `rh_comparison/data_store/epoch_store.py`
- `EpochStore.create_from_pyomo()` with both upload modes, `.load()`, all get/save methods
- `get_solution_summary()` for large model compact representation
- Unit tests with small Pyomo models

### Phase 2 — Pure Analytics (QT1 + QT2)
- `rh_comparison/analytics/structural_diff.py` — includes compact summary block in output
- `rh_comparison/analytics/solution_diff.py` — includes top_changes + churn_summary in output
- These must work standalone with no agents or ADK
- Validate JSON outputs against Appendix A.1 and A.2 from the paper

### Phase 3 — Solver Analytics (QT3 + QT4)
- `rh_comparison/analytics/backward_compat.py`
- `rh_comparison/analytics/attribution_analysis.py`
- Integration test using models from `Feas/` with a manually constructed epoch pair

### Phase 4 — Agent Layer
- `rh_comparison/agents/rh_prompts.py` — prompts for all query classes; every prompt is fully self-contained with context injected
- `rh_comparison/tools/epoch_tools.py` — ADK tool wrappers with tiered detail levels
- `rh_comparison/tools/rh_callback_tool.py` — init callback + stateless context injection per call
- `rh_comparison/agents/rh_comparison_agent.py`
- `rh_comparison/agents/rh_root_agent.py`
- `rh_comparison/rh_agent.py`

### Phase 5 — UI Integration
- Add mode selector to `app.py` (minimal, ~30 lines)
- Support both config upload modes (separate vs. embedded)
- Register `optichat_rh` app in ADK server
- End-to-end test with pricing model pair

---

## 20. Key Non-Interference Guarantees

| Concern | How We Handle It |
|---|---|
| OptiChat's `MODELS_DICTIONARY` | RH framework uses its own `EpochStore` and separate state keys |
| OptiChat's callbacks | RH framework defines its own callbacks; never monkey-patches OptiChat's |
| OptiChat's session state | Separate ADK app (`optichat_rh`) = completely separate session namespace |
| OptiChat's `tmp/model_objects/` | RH framework writes to `tmp/rh_epochs/` exclusively |
| Future OptiChat additions | As long as they stay in `optichat/`, zero impact on `rh_comparison/` |
| ADK agent naming | `rh_root_agent`, `rh_comparison_agent` — no name collisions with OptiChat agents |
| `illustrator_agent` reuse | Called via `AgentTool()` — no modification, no copy. OptiChat can update it freely |

---

## 21. Efficiency Summary

| Operation | Cost | Strategy |
|---|---|---|
| QT1 structural diff | O(n_params), 0 solves | Eager at init, pure JSON; family-grouped |
| QT2 solution diff | O(n_vars), 0 solves | On-demand; compact summary + top-N to LLM |
| QT3 backward compat | 3–4 Gurobi calls | On-demand + disk cache |
| QT4 attribution analysis | 4 Gurobi calls (2 reused from init) | On-demand + disk cache |
| Incumbent LP | <1s (all integers fixed) | Cached to `duals.json` per epoch |
| LLM context | Compact summaries + descriptions, not raw model | Tiered access; Tier 1–3 for LLM, Tier 4–6 on specific request |
| Pyomo model load | Only for QT3/QT4 solver ops | `get_pyomo_model()` lazy |
| `get_epoch_data` lookups | Pure JSON reads | No Pyomo deserialization |
| Model descriptions | Reuses OptiChat illustrator | `format_hierarchical_components` is family-grouped; scales to any model size |
