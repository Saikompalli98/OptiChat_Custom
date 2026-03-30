"""
EpochStore — Core Data Layer for the Rolling Horizon Comparison Framework.

Responsibilities:
  - Build and persist all per-epoch data files from a solved Pyomo model
  - Provide lazy-loaded, cached accessors for every data file
  - Manage the shared registry (registry.json)
  - Cache and retrieve comparison results (QT1–QT4 JSONs)
  - Produce compact LLM-facing summaries from large solution data

Directory layout it manages:
  tmp/rh_epochs/
  ├── registry.json
  ├── {epoch_id}/
  │   ├── model.pkl
  │   ├── solution.json
  │   ├── params.json
  │   ├── structure.json
  │   ├── description.txt
  │   └── duals.json          (written on demand by attribution_analysis)
  └── comparisons/
      └── {epoch_a}_vs_{epoch_b}/
          ├── qt1_structural.json
          ├── qt2_solution.json
          ├── qt3_backward.json
          └── qt4_attribution.json

Never modifies anything under optichat/ or tmp/model_objects/.
"""

from __future__ import annotations

import json
import os
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any

import cloudpickle
import pyomo.environ as pe
from loguru import logger
from pyomo.core.expr.visitor import identify_variables

from rh_comparison.config.rh_constants import (
    RH_TMP_ROOT,
    RH_REGISTRY_PATH,
    RH_COMPARISONS_DIR,
    QT1_FILENAME, QT2_FILENAME, QT3_FILENAME, QT4_FILENAME,
    Thresholds, UploadMode,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_json(path: str, data: Any) -> None:
    _make_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for types that aren't natively serializable."""
    if isinstance(obj, (set, frozenset)):
        return sorted(list(obj), key=str)
    if hasattr(obj, "item"):          # numpy scalars
        return obj.item()
    return str(obj)


def _idx_to_str(idx: Any) -> str:
    """Convert a Pyomo index (scalar or tuple) to a stable string key."""
    if idx is None:
        return "None"
    return str(idx)


def _to_serializable(val: Any) -> Any:
    """Recursively convert a value to JSON-safe types."""
    if isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_serializable(v) for v in val]
    if isinstance(val, (set, frozenset)):
        return sorted([_to_serializable(v) for v in val], key=str)
    if hasattr(val, "item"):          # numpy scalar
        return val.item()
    if isinstance(val, float) and (val != val):   # NaN
        return None
    return val


# ---------------------------------------------------------------------------
# Pyomo model extraction helpers (called only by create_from_pyomo)
# ---------------------------------------------------------------------------

def _is_pure_lp(model: pe.ConcreteModel) -> bool:
    """Return True if the model has no binary or integer variables (pure LP)."""
    for var in model.component_objects(pe.Var, active=True):
        for idx in var:
            if var[idx].is_binary() or var[idx].is_integer():
                return False
    return True


def _attach_and_resolve_for_duals(model: pe.ConcreteModel) -> pe.ConcreteModel:
    """
    Attach a dual suffix and re-solve the model as an LP to populate dual values.
    Only called for pure LP models. The re-solve is fast because LPs are cheap.
    Returns the same model object with dual values populated.
    """
    from pyomo.opt import SolverFactory
    if not hasattr(model, "dual"):
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
    solver = SolverFactory("gurobi")
    solver.solve(model, tee=False)
    return model


def _extract_solution(model: pe.ConcreteModel, termination_condition: str) -> dict:
    """
    Build solution.json content from a solved Pyomo model.

    Dual values in solution.json:
      - Pure LP model:  populated (shadow prices from the solve)
      - MIP model:      field omitted entirely — duals are not meaningful for MIPs
                        and should not be confused with incumbent LP duals (duals.json)

    For infeasible or unknown solves, variable values and binding status are
    recorded as None so the file is always structurally complete.
    """
    is_optimal = (str(termination_condition) == "optimal")
    eps        = Thresholds.BINDING_TOL
    is_lp      = is_optimal and _is_pure_lp(model)

    # For LP models: ensure dual suffix is attached and values are available.
    # If the model was solved without a dual suffix (common), do a quick re-solve.
    if is_lp:
        has_dual_values = (
            hasattr(model, "dual") and
            any(True for _ in model.dual)   # suffix is non-empty
        )
        if not has_dual_values:
            logger.info("[EpochStore] Pure LP detected — re-solving with dual suffix to extract shadow prices")
            _attach_and_resolve_for_duals(model)

    # --- Objective ---
    obj_comp = next(model.component_data_objects(pe.Objective, active=True), None)
    if obj_comp is not None:
        sense = "maximize" if obj_comp.sense == pe.maximize else "minimize"
        try:
            obj_value = float(pe.value(obj_comp)) if is_optimal else None
        except Exception:
            obj_value = None
    else:
        sense    = "unknown"
        obj_value = None

    # --- Variables ---
    variables: dict = {}
    for var in model.component_objects(pe.Var, active=True):
        for idx in var:
            name = pe.name(var[idx])
            try:
                value = float(var[idx].value) if var[idx].value is not None else None
            except Exception:
                value = None
            variables[name] = {
                "value":      value,
                "is_binary":  bool(var[idx].is_binary()),
                "is_integer": bool(var[idx].is_integer()),
            }

    # --- Constraints ---
    constraints: dict = {}
    for con in model.component_objects(pe.Constraint, active=True):
        for idx in con:
            name      = pe.name(con[idx])
            is_binding = None
            slack_val  = None

            if is_optimal:
                try:
                    ls = con[idx].lslack()
                    us = con[idx].uslack()
                    ls = abs(ls) if ls is not None else float("inf")
                    us = abs(us) if us is not None else float("inf")
                    slack_val  = float(min(ls, us))
                    is_binding = slack_val < eps
                except Exception:
                    pass

            entry: dict = {"is_binding": is_binding, "slack": slack_val}

            # Only include dual for LP models — MIP duals are meaningless here.
            # For MIP shadow prices use duals.json (incumbent LP, computed by QT4).
            if is_lp and hasattr(model, "dual"):
                try:
                    raw = model.dual.get(con[idx])
                    entry["dual"] = float(raw) if raw is not None else None
                except Exception:
                    entry["dual"] = None

            constraints[name] = entry

    return {
        "objective":   {"value": obj_value, "sense": sense, "status": str(termination_condition)},
        "is_mip":      not is_lp,
        "variables":   variables,
        "constraints": constraints,
    }


def _extract_params(model: pe.ConcreteModel) -> dict:
    """
    Build params.json content from a Pyomo model.

    Returns: {family_name: {index_str: value}}
    Includes both mutable and immutable parameters because most uploaded models
    load business data as immutable Params. RH comparison needs those values
    for model explanation, retrieval, and cross-run change analysis.
    """
    params: dict = {}
    for param in model.component_objects(pe.Param, active=True):
        family: dict = {}
        for idx in param:
            try:
                val = param[idx].value
                if val is not None:
                    family[_idx_to_str(idx)] = float(val)
            except Exception:
                pass
        if family:
            params[param.name] = family
    return params


def _get_index_set_name(index_set: Any) -> list[str]:
    """
    Attempt to resolve the names of the sets that make up a variable/param's
    index set. Falls back to [] on any failure — this is cosmetic metadata only.
    """
    def _is_user_visible_set(name: str) -> bool:
        if not name:
            return False
        lowered = name.strip().lower()
        return lowered not in {"unindexedcomponent_set", "none"}

    if index_set is None:
        return []
    try:
        # Cross-product set (SetProduct, AbstractCrossProductSet, etc.)
        if hasattr(index_set, "_sets"):
            return [
                s.name for s in index_set._sets
                if hasattr(s, "name") and _is_user_visible_set(s.name)
            ]
        if hasattr(index_set, "set_tuple"):
            return [
                s.name for s in index_set.set_tuple
                if hasattr(s, "name") and _is_user_visible_set(s.name)
            ]
        # Simple named set
        if hasattr(index_set, "name") and _is_user_visible_set(index_set.name):
            return [index_set.name]
    except Exception:
        pass
    return []


def _extract_structure(model: pe.ConcreteModel) -> dict:
    """
    Build structure.json content: family-level schema of the model.

    For large models this replaces O(n) component listings with O(families)
    entries. A model with 50k variables across 20 families becomes 20 entries.
    """
    model_doc = (getattr(model, "doc", "") or "").strip()

    objective_doc = ""
    objective_name = ""
    objective_expr = ""
    objective_variable_families: list[str] = []
    for obj in model.component_objects(pe.Objective, active=True):
        objective_doc = (getattr(obj, "doc", "") or "").strip()
        objective_name = obj.name
        try:
            objective_expr = str(obj.expr)
        except Exception:
            objective_expr = ""
        try:
            objective_variable_families = sorted({
                pe.name(v).split("[")[0]
                for v in identify_variables(obj.expr, include_fixed=False)
            })
        except Exception:
            objective_variable_families = []
        break

    # --- Index sets ---
    index_sets: dict     = {}
    index_set_docs: dict = {}
    for s in model.component_objects(pe.Set, active=True):
        name = s.name
        if name.startswith("_"):        # skip Pyomo internal sets
            continue
        try:
            raw = list(s.data()) if hasattr(s, "data") else list(s)
            members = []
            for m in raw:
                if isinstance(m, (list, tuple)):
                    members.append([_to_serializable(x) for x in m])
                else:
                    members.append(_to_serializable(m))
            index_sets[name] = members
        except Exception:
            index_sets[name] = []
        index_set_docs[name] = (getattr(s, "doc", "") or "").strip()

    # --- Variable families ---
    variable_families: dict = {}
    for var in model.component_objects(pe.Var, active=True):
        indices = list(var)
        first   = var[indices[0]] if indices else None
        vtype   = (
            "binary"   if (first is not None and first.is_binary()) else
            "integer"  if (first is not None and first.is_integer()) else
            "continuous"
        )
        try:
            indexed_over = _get_index_set_name(var.index_set())
        except Exception:
            indexed_over = []

        variable_families[var.name] = {
            "indexed_over": indexed_over,
            "type":         vtype,
            "count":        len(indices),
            "doc":          (getattr(var, "doc", "") or "").strip(),
        }

    # --- Parameter families ---
    param_families: dict = {}
    for param in model.component_objects(pe.Param, active=True):
        indices = list(param)
        try:
            indexed_over = _get_index_set_name(param.index_set())
        except Exception:
            indexed_over = []

        param_families[param.name] = {
            "indexed_over": indexed_over,
            "count":        len(indices),
            "doc":          (getattr(param, "doc", "") or "").strip(),
            "mutable":      bool(param.mutable),
        }

    # --- Constraint templates + docs ---
    # Store representative expression of the first element per family.
    # This is informational; not used in analytics computations.
    constraint_templates: dict = {}
    constraint_docs: dict      = {}
    constraint_families: dict  = {}
    for con in model.component_objects(pe.Constraint, active=True):
        indices = list(con)
        if not indices:
            continue
        try:
            template = str(con[indices[0]].expr)
        except Exception:
            template = ""
        constraint_templates[con.name] = template
        constraint_docs[con.name]      = (getattr(con, "doc", "") or "").strip()
        try:
            indexed_over = _get_index_set_name(con.index_set())
        except Exception:
            indexed_over = []
        constraint_families[con.name] = {
            "indexed_over": indexed_over,
            "count": len(indices),
            "doc": (getattr(con, "doc", "") or "").strip(),
        }

    return {
        "model_doc":             model_doc,
        "objective_doc":         objective_doc,
        "objective_name":        objective_name,
        "objective_expr":        objective_expr,
        "objective_variable_families": objective_variable_families,
        "index_sets":            index_sets,
        "index_set_docs":        index_set_docs,
        "variable_families":     variable_families,
        "param_families":        param_families,
        "constraint_templates":  constraint_templates,
        "constraint_docs":       constraint_docs,
        "constraint_families":   constraint_families,
    }


def _build_epoch_meta(
    epoch_id:             str,
    label:                str,
    sol:                  dict,
    struct:               dict,
    params:               dict,
    epoch_dir:            str,
    source_py_path:       str,
    data_json_path:       str | None,
    termination_condition: str,
) -> dict:
    n_params = sum(len(v) for v in params.values())

    return {
        "epoch_id":           epoch_id,
        "label":              label,
        "loaded_at":          datetime.now().isoformat(),
        "sol_status":         str(termination_condition),
        "objective_value":    sol["objective"].get("value"),
        "objective_sense":    sol["objective"].get("sense"),
        "n_variables":        len(sol["variables"]),
        "n_constraints":      len(sol["constraints"]),
        "n_params":           n_params,
        "index_sets":         {k: len(v) for k, v in struct["index_sets"].items()},
        "model_pkl_path":     os.path.join(epoch_dir, "model.pkl"),
        "source_py_path":     source_py_path or "",
        "data_json_path":     data_json_path or "",
        "description_path":   os.path.join(epoch_dir, "description.txt"),
    }


# ---------------------------------------------------------------------------
# Model loader (used by the init pipeline, not EpochStore itself)
# ---------------------------------------------------------------------------

def load_model_from_py(
    model_py_path: str,
    data_json_path: str | None = None,
) -> pe.ConcreteModel:
    """
    Load a Pyomo ConcreteModel from a .py file.

    Mirrors OptiChat's ``_load_model_from_py`` pattern exactly.
    The module is intentionally NOT registered in sys.modules so that
    cloudpickle serialises full function bytecode into the .pkl rather than
    a lightweight module reference (which would break on server restart).

    Mode A (data_json_path provided):
        Loads data from the JSON file and injects it as the ``data`` variable
        into the module namespace before executing the file.  This matches
        the standard model file pattern::

            data = globals().get("data", {})   # reads the injected dict
            model = ConcreteModel()
            ...

    Mode B (data_json_path is None):
        Executes the .py file directly.  The file must produce a module-level
        ``model`` variable (data already embedded inside the file).

    Raises:
        FileNotFoundError: if the .py or .json file does not exist.
        ValueError: if ``model`` is not found or is not a ConcreteModel.
    """
    model_py_path = os.path.abspath(model_py_path)
    if not os.path.exists(model_py_path):
        raise FileNotFoundError(f"Model file not found: {model_py_path}")

    # Use a fixed module name — do NOT add to sys.modules (see docstring)
    spec   = importlib.util.spec_from_file_location("rh_loaded_model", model_py_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

    if data_json_path is not None:
        # --- Mode A: inject data dict before executing the module ---
        data_json_path = os.path.abspath(data_json_path)
        if not os.path.exists(data_json_path):
            raise FileNotFoundError(f"Data JSON file not found: {data_json_path}")
        with open(data_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Inject as 'data' so model files can use: data = globals().get("data", {})
        module.__dict__["data"] = data

    spec.loader.exec_module(module)  # type: ignore[union-attr]

    model = getattr(module, "model", None)
    if model is None:
        raise ValueError(
            f"No module-level 'model' variable found in {model_py_path}. "
            "The .py file must define a Pyomo ConcreteModel named 'model'."
        )
    if not isinstance(model, pe.ConcreteModel):
        raise ValueError(
            f"Expected a Pyomo ConcreteModel, got {type(model).__name__} "
            f"from {model_py_path}."
        )

    return model


# ---------------------------------------------------------------------------
# EpochStore
# ---------------------------------------------------------------------------

class EpochStore:
    """
    Thin wrapper around ``tmp/rh_epochs/{epoch_id}/``.

    Follows a strict lazy-load + in-process cache pattern:
      - JSON files are read from disk on the first access, then cached.
      - The Pyomo model (.pkl) is **never** loaded unless explicitly requested
        via ``get_pyomo_model()``. It is only needed for QT3/QT4 solver calls.
      - ``create_from_pyomo()`` is the only place that writes data.
    """

    BASE_DIR: str = RH_TMP_ROOT

    def __init__(self, epoch_id: str) -> None:
        self.epoch_id  = epoch_id
        self.epoch_dir = os.path.join(self.BASE_DIR, epoch_id)

        # In-process cache — None = not yet loaded
        self._meta_cache:        dict | None = None
        self._solution_cache:    dict | None = None
        self._params_cache:      dict | None = None
        self._structure_cache:   dict | None = None
        self._duals_cache:       dict | None = None
        self._description_cache: str  | None = None

    # ------------------------------------------------------------------
    # Internal path helpers
    # ------------------------------------------------------------------

    def _path(self, filename: str) -> str:
        return os.path.join(self.epoch_dir, filename)

    def _comparison_dir(self, other_id: str) -> str:
        key = "_vs_".join(sorted([self.epoch_id, other_id]))
        return os.path.join(RH_COMPARISONS_DIR, key)

    def _qt_filename(self, qt: str) -> str:
        return {
            "qt1": QT1_FILENAME,
            "qt2": QT2_FILENAME,
            "qt3": QT3_FILENAME,
            "qt4": QT4_FILENAME,
        }[qt.lower()]

    # ------------------------------------------------------------------
    # Factory: build from a solved Pyomo model
    # ------------------------------------------------------------------

    @classmethod
    def create_from_pyomo(
        cls,
        model:                 pe.ConcreteModel,
        epoch_id:              str,
        label:                 str,
        source_py_path:        str        = "",
        termination_condition: str        = "optimal",
        data_json_path:        str | None = None,
    ) -> "EpochStore":
        """
        Build all on-disk files for a new epoch from a solved Pyomo model.

        This is the only method that writes files to disk. It should be called
        once per epoch during session initialisation, right after solve_model().

        Args:
            model:                  Solved Pyomo ConcreteModel.
            epoch_id:               Unique identifier string (e.g. 'epoch_a').
            label:                  Human-readable label (e.g. 'May 2025').
            source_py_path:         Path to the .py file the model was loaded from.
            termination_condition:  Solver termination condition string.
                                    Pass str(results.solver.termination_condition).
            data_json_path:         Path to the data .json file (Mode A only).

        Returns:
            EpochStore instance with all caches pre-populated.
        """
        store = cls(epoch_id)
        _make_dir(store.epoch_dir)

        logger.info(f"[EpochStore] Building epoch '{epoch_id}' (label='{label}')...")

        # 1. Extract data from the Pyomo model
        sol    = _extract_solution(model, termination_condition)
        params = _extract_params(model)
        struct = _extract_structure(model)

        # 2. Write data files
        sol_path    = store._path("solution.json")
        params_path = store._path("params.json")
        struct_path = store._path("structure.json")

        sol_data    = {"epoch_id": epoch_id, **sol}
        params_data = {"epoch_id": epoch_id, **params}

        _write_json(sol_path,    sol_data)
        _write_json(params_path, params_data)
        _write_json(struct_path, struct)

        logger.info(f"[EpochStore] Wrote solution.json, params.json, structure.json for '{epoch_id}'")

        # 3. Save Pyomo model to pkl
        pkl_path = store._path("model.pkl")
        with open(pkl_path, "wb") as f:
            cloudpickle.dump(model, f)
        logger.info(f"[EpochStore] Saved model.pkl for '{epoch_id}'")

        # 4. Build and write EpochMeta
        meta = _build_epoch_meta(
            epoch_id=epoch_id,
            label=label,
            sol=sol,
            struct=struct,
            params=params,
            epoch_dir=store.epoch_dir,
            source_py_path=source_py_path,
            data_json_path=data_json_path,
            termination_condition=termination_condition,
        )
        meta_path = store._path("meta.json")
        _write_json(meta_path, meta)

        # 5. Update shared registry
        store._update_registry(meta)
        logger.info(f"[EpochStore] Registry updated for '{epoch_id}'")

        # 6. Pre-populate in-process cache
        store._meta_cache      = meta
        store._solution_cache  = sol_data
        store._params_cache    = params_data
        store._structure_cache = struct

        logger.info(f"[EpochStore] Epoch '{epoch_id}' ready. "
                    f"vars={meta['n_variables']}, cons={meta['n_constraints']}, "
                    f"status={meta['sol_status']}")
        return store

    # ------------------------------------------------------------------
    # Factory: load from existing disk files
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, epoch_id: str) -> "EpochStore":
        """
        Load an EpochStore from existing on-disk files.

        Does not read any files eagerly — all files are loaded lazily on
        first access. Call this when you need a store for a previously-built
        epoch (e.g. when resuming a session).

        Raises:
            FileNotFoundError: if the epoch directory does not exist.
        """
        store = cls(epoch_id)
        if not os.path.isdir(store.epoch_dir):
            raise FileNotFoundError(
                f"Epoch directory not found: {store.epoch_dir}. "
                f"Has create_from_pyomo() been called for '{epoch_id}'?"
            )
        return store

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _update_registry(self, meta: dict) -> None:
        registry = self._load_registry()
        registry[self.epoch_id] = meta
        _write_json(RH_REGISTRY_PATH, registry)

    @staticmethod
    def _load_registry() -> dict:
        if os.path.exists(RH_REGISTRY_PATH):
            try:
                return _read_json(RH_REGISTRY_PATH)
            except Exception as e:
                logger.warning(f"[EpochStore] Failed to read registry: {e}. Starting fresh.")
        return {}

    @staticmethod
    def list_all_epochs() -> dict:
        """Return the full registry as a dict of {epoch_id: EpochMeta}."""
        return EpochStore._load_registry()

    # ------------------------------------------------------------------
    # Core accessors (lazy-loaded, in-process cached)
    # ------------------------------------------------------------------

    def get_meta(self) -> dict:
        """Return EpochMeta dict. Loads meta.json on first call."""
        if self._meta_cache is None:
            self._meta_cache = _read_json(self._path("meta.json"))
        return self._meta_cache

    def get_solution(self) -> dict:
        """
        Return the full solution dict (all variables + constraints + objective).
        Loads solution.json from disk on first call, then caches in process.
        Never triggers Pyomo model deserialization.
        """
        if self._solution_cache is None:
            path = self._path("solution.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"solution.json not found for epoch '{self.epoch_id}'. "
                    "Call create_from_pyomo() first."
                )
            self._solution_cache = _read_json(path)
        return self._solution_cache

    def get_params(self) -> dict:
        """
        Return the full params dict {family_name: {index_str: value}}.
        Loads params.json from disk on first call, then caches in process.
        Never triggers Pyomo model deserialization.
        """
        if self._params_cache is None:
            path = self._path("params.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"params.json not found for epoch '{self.epoch_id}'."
                )
            self._params_cache = _read_json(path)
        return self._params_cache

    def get_structure(self) -> dict:
        """
        Return the structure dict (index_sets, variable_families, etc.).
        Loads structure.json from disk on first call, then caches in process.
        """
        if self._structure_cache is None:
            path = self._path("structure.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"structure.json not found for epoch '{self.epoch_id}'."
                )
            self._structure_cache = _read_json(path)
        return self._structure_cache

    def get_duals(self) -> dict | None:
        """
        Return dual values from the incumbent LP, or None if not yet computed.
        Loads duals.json if it exists; returns None if file is absent.
        (duals.json is written by attribution_analysis.py on demand.)
        """
        if self._duals_cache is None:
            path = self._path("duals.json")
            if not os.path.exists(path):
                return None
            self._duals_cache = _read_json(path)
        return self._duals_cache

    def get_description(self) -> str:
        """
        Return the natural language description stored for this epoch.
        In the current RH implementation this text is generated internally from
        structure.json, not by OptiChat's illustrator agent.
        Returns empty string if description.txt has not been written yet.
        """
        if self._description_cache is None:
            path = self._path("description.txt")
            if not os.path.exists(path):
                return ""
            with open(path, "r", encoding="utf-8") as f:
                self._description_cache = f.read()
        return self._description_cache

    def get_pyomo_model(self) -> pe.ConcreteModel:
        """
        Restore and return the Pyomo model from model.pkl.

        This is intentionally NOT cached in process (pkl files can be large).
        It is called only by QT3/QT4 solver operations.

        Returns a freshly deserialized model each time it is called so that
        solver modifications (variable fixing, etc.) don't bleed between calls.
        """
        pkl_path = self._path("model.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"model.pkl not found for epoch '{self.epoch_id}'. "
                "Call create_from_pyomo() first."
            )
        with open(pkl_path, "rb") as f:
            model = cloudpickle.load(f)

        # Remove dual suffix if present — analytics code adds its own
        if hasattr(model, "dual"):
            model.del_component(model.dual)

        return model

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def save_duals(self, duals: dict) -> None:
        """
        Persist incumbent LP dual values for this epoch.

        These are NOT the same as the constraint duals in solution.json.

        solution.json duals: from the original solve (LP models only, always
            available at init, shadow prices for the original problem).

        duals.json (this file): from a SEPARATE incumbent LP solve done during
            QT4 (Solution Attribution Analysis). Computed by fixing all integer/
            binary variables to their optimal MIP values and re-solving the
            relaxed LP. This gives well-defined shadow prices for any model,
            including MIPs, and is used to attribute solution changes to
            specific parameter triggers between epochs.

        Written by attribution_analysis.py on first QT4 run, then cached here
        so subsequent QT4 calls on the same epoch skip the re-solve.
        """
        path = self._path("duals.json")
        data = {"epoch_id": self.epoch_id, **duals}
        _write_json(path, data)
        self._duals_cache = data
        logger.info(f"[EpochStore] Saved duals.json for '{self.epoch_id}'")

    def save_description(self, text: str) -> None:
        """
        Persist the natural language description for this epoch.
        RH currently writes this from its own initialization callback.
        """
        path = self._path("description.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self._description_cache = text
        logger.info(f"[EpochStore] Saved description.txt for '{self.epoch_id}'")

    # ------------------------------------------------------------------
    # Comparison result cache
    # ------------------------------------------------------------------

    def save_comparison(self, other_id: str, qt: str, data: dict) -> None:
        """
        Write a QT result JSON to the comparison cache directory.

        Args:
            other_id: The other epoch's epoch_id.
            qt:       One of 'qt1', 'qt2', 'qt3', 'qt4'.
            data:     The QT result dict to persist.
        """
        cmp_dir  = self._comparison_dir(other_id)
        filename = self._qt_filename(qt)
        path     = os.path.join(cmp_dir, filename)
        _make_dir(cmp_dir)
        _write_json(path, data)
        logger.info(f"[EpochStore] Saved {filename} for "
                    f"'{self.epoch_id}' vs '{other_id}'")

    def load_comparison(self, other_id: str, qt: str) -> dict | None:
        """
        Load a cached QT result from disk, or return None if not yet computed.

        Args:
            other_id: The other epoch's epoch_id.
            qt:       One of 'qt1', 'qt2', 'qt3', 'qt4'.

        Returns:
            Parsed JSON dict, or None if the file does not exist yet.
        """
        cmp_dir  = self._comparison_dir(other_id)
        filename = self._qt_filename(qt)
        path     = os.path.join(cmp_dir, filename)
        if not os.path.exists(path):
            return None
        return _read_json(path)

    # ------------------------------------------------------------------
    # Compact LLM-facing summary (Tier 3 in the tiered access pattern)
    # ------------------------------------------------------------------

    def get_solution_summary(self, top_n: int = 10) -> dict:
        """
        Return a compact, LLM-facing summary of the epoch's solution.

        For large models (50k+ vars) this avoids dumping all variable values.
        Always O(n_vars) to compute once; result is derived from the cached
        solution dict.

        Returns:
            {
              epoch_id, label, objective,
              n_variables, n_constraints,
              variable_family_stats: {family: {count, n_binary, n_active_binary, n_integer}},
              binding_constraint_families: [sorted list],
              top_active_binary_decisions: [{family, activated_count, total_count}],
            }
        """
        sol  = self.get_solution()
        meta = self.get_meta()

        # --- Variable family aggregation ---
        family_stats: dict = {}
        for var_name, var_data in sol["variables"].items():
            family = var_name.split("[")[0]
            if family not in family_stats:
                family_stats[family] = {
                    "count":           0,
                    "n_binary":        0,
                    "n_active_binary": 0,
                    "n_integer":       0,
                }
            fs = family_stats[family]
            fs["count"] += 1
            val = var_data.get("value")
            if var_data.get("is_binary"):
                fs["n_binary"] += 1
                if val is not None and abs(val - 1.0) < Thresholds.BINDING_TOL:
                    fs["n_active_binary"] += 1
            elif var_data.get("is_integer"):
                fs["n_integer"] += 1

        # --- Binding constraint families ---
        binding_families: set = set()
        for con_name, con_data in sol["constraints"].items():
            if con_data.get("is_binding"):
                binding_families.add(con_name.split("[")[0])

        # --- Top active binary families (by activation rate) ---
        binary_families = [
            {
                "family":          f,
                "activated_count": s["n_active_binary"],
                "total_count":     s["n_binary"],
                "activation_rate": (
                    round(s["n_active_binary"] / s["n_binary"], 3) if s["n_binary"] > 0 else 0.0
                ),
            }
            for f, s in family_stats.items()
            if s["n_binary"] > 0
        ]
        binary_families.sort(key=lambda x: x["activation_rate"], reverse=True)
        top_active_binary = binary_families[:top_n]

        return {
            "epoch_id":                   self.epoch_id,
            "label":                      meta.get("label", ""),
            "objective":                  sol["objective"],
            "n_variables":                len(sol["variables"]),
            "n_constraints":              len(sol["constraints"]),
            "variable_family_stats":      family_stats,
            "binding_constraint_families": sorted(list(binding_families)),
            "top_active_binary_decisions": top_active_binary,
        }

    def get_param_summary(self) -> dict:
        """
        Return a compact summary of parameter families — names + counts.
        Does not expose individual parameter values.

        Used for LLM context injection when answering model-level questions.
        """
        params = self.get_params()
        return {
            family: {"count": len(vals)}
            for family, vals in params.items()
            if family != "epoch_id"
        }

    def get_variable_family(self, family_prefix: str) -> dict:
        """
        Return all variable entries for a specific family from solution.json.

        Used when a user drills into a specific variable family (Tier 4 access).

        Args:
            family_prefix: e.g. 'price', 'promote', 'demand'

        Returns:
            {var_name: {value, is_binary, is_integer}} filtered to the family.
        """
        sol = self.get_solution()
        prefix = family_prefix.rstrip("[")
        return {
            k: v
            for k, v in sol["variables"].items()
            if k.split("[")[0] == prefix
        }

    def get_param_family(self, family_name: str) -> dict:
        """
        Return all parameter entries for a specific family from params.json.

        Used when a user asks about specific parameter values (Tier 4 access).

        Args:
            family_name: e.g. 'demand', 'capacity'

        Returns:
            {index_str: value} for the requested family, or {} if not found.
        """
        params = self.get_params()
        return params.get(family_name, {})

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        meta = self._meta_cache or {}
        label  = meta.get("label", "?")
        status = meta.get("sol_status", "?")
        return f"EpochStore(id={self.epoch_id!r}, label={label!r}, status={status!r})"
