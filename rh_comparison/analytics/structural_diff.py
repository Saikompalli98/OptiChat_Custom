"""
QT1 — Structural and Parametric Change Analysis.

Pure Python, zero solver calls.

Compares index sets, mutable parameter values, variable families, and
constraint templates between two epochs.  All comparisons are derived from
the pre-extracted JSON artefacts in EpochStore (structure.json, params.json).
No Pyomo model deserialization is required.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rh_comparison.data_store.epoch_store import EpochStore
from rh_comparison.config.rh_constants import Thresholds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_magnitude(rel_change: float | None) -> str:
    """
    Classify relative change into 'small', 'moderate', or 'large'.

    rel_change = None means value_a ≈ 0 and value_b ≠ 0 (ratio undefined).
    This is treated as 'large' — any non-zero shift from zero is significant.
    """
    if rel_change is None:
        return "large"
    a = abs(rel_change)
    if a < Thresholds.MAGNITUDE_SMALL:
        return "small"
    if a < Thresholds.MAGNITUDE_MODERATE:
        return "moderate"
    return "large"


def _rel_sort_key(rel_change: float | None) -> float:
    """Sort key for descending magnitude ranking. None (0→nonzero) ranks first."""
    return abs(rel_change) if rel_change is not None else 1e9


def _elem_key(elem: Any) -> str:
    """Normalize a set element (scalar or list-tuple) to a stable string key."""
    return str(tuple(elem)) if isinstance(elem, list) else str(elem)


# ---------------------------------------------------------------------------
# Per-section comparison functions
# ---------------------------------------------------------------------------

def _compare_index_sets(struct_a: dict, struct_b: dict) -> dict:
    """
    Compare index set membership between two epochs.

    Reports sets that are entirely new in B, entirely removed in B, or that
    gained / lost individual elements.  Element lists are capped at 50 items
    per set to keep the result suitable for LLM context injection.
    """
    sets_a = struct_a.get("index_sets", {})
    sets_b = struct_b.get("index_sets", {})
    all_names = sorted(set(sets_a) | set(sets_b))

    added_sets   = [n for n in all_names if n not in sets_a]
    removed_sets = [n for n in all_names if n not in sets_b]

    modified_sets: dict = {}
    for name in all_names:
        if name not in sets_a or name not in sets_b:
            continue
        mem_a = {_elem_key(x) for x in sets_a[name]}
        mem_b = {_elem_key(x) for x in sets_b[name]}
        added_elems   = sorted(mem_b - mem_a)
        removed_elems = sorted(mem_a - mem_b)
        if added_elems or removed_elems:
            modified_sets[name] = {
                "size_a":          len(sets_a[name]),
                "size_b":          len(sets_b[name]),
                "size_delta":      len(sets_b[name]) - len(sets_a[name]),
                "added_elements":  added_elems[:50],
                "removed_elements": removed_elems[:50],
                "n_added":         len(added_elems),
                "n_removed":       len(removed_elems),
            }

    return {
        "added_sets":    added_sets,
        "removed_sets":  removed_sets,
        "modified_sets": modified_sets,
    }


def _compare_variable_families(struct_a: dict, struct_b: dict) -> dict:
    """
    Compare variable family schemas between epochs.

    Tracks families that are entirely new, removed, have a different element
    count (indicating index-set driven size change), or changed variable type.
    """
    vf_a = struct_a.get("variable_families", {})
    vf_b = struct_b.get("variable_families", {})
    all_families = sorted(set(vf_a) | set(vf_b))

    added_families   = [f for f in all_families if f not in vf_a]
    removed_families = [f for f in all_families if f not in vf_b]

    count_changes: dict = {}
    type_changes:  dict = {}
    for family in all_families:
        if family not in vf_a or family not in vf_b:
            continue
        fa, fb = vf_a[family], vf_b[family]
        count_a = fa.get("count", 0)
        count_b = fb.get("count", 0)
        if count_a != count_b:
            count_changes[family] = {
                "count_a": count_a,
                "count_b": count_b,
                "delta":   count_b - count_a,
            }
        type_a, type_b = fa.get("type", ""), fb.get("type", "")
        if type_a != type_b:
            type_changes[family] = {"type_a": type_a, "type_b": type_b}

    return {
        "added_families":   added_families,
        "removed_families": removed_families,
        "count_changes":    count_changes,
        "type_changes":     type_changes,
    }


def _compare_constraint_templates(struct_a: dict, struct_b: dict) -> dict:
    """
    Compare constraint family presence and representative templates.

    A template change (same family name, different expression string) can
    indicate structural modification of the constraint logic between epochs.
    """
    ct_a = struct_a.get("constraint_templates", {})
    ct_b = struct_b.get("constraint_templates", {})
    all_families = sorted(set(ct_a) | set(ct_b))

    added_families   = [f for f in all_families if f not in ct_a]
    removed_families = [f for f in all_families if f not in ct_b]

    template_changes: dict = {}
    for family in all_families:
        if family not in ct_a or family not in ct_b:
            continue
        if ct_a[family] != ct_b[family]:
            template_changes[family] = {
                "template_a": ct_a[family],
                "template_b": ct_b[family],
            }

    return {
        "added_families":   added_families,
        "removed_families": removed_families,
        "template_changes": template_changes,
    }


def _compare_parameters(params_a: dict, params_b: dict) -> dict:
    """
    Compare mutable parameter values between epochs.

    For each (family, index) triple:
      - Added:   present in B but not A
      - Removed: present in A but not B
      - Changed: present in both, value differs beyond PARAM_CHANGE_TOL

    Relative change = (value_b - value_a) / |value_a|.
    When value_a ≈ 0 the relative change is reported as None (undefined ratio)
    but the entry is still classified as 'large' and included.

    Returns a flat top-20 list ranked by absolute relative change for quick
    LLM consumption, alongside per-family breakdowns.
    """
    # params.json contains an "epoch_id" key — skip it
    all_families = sorted((set(params_a) | set(params_b)) - {"epoch_id"})

    by_family:  dict       = {}
    all_changes: list[dict] = []
    n_changed_total = n_added_total = n_removed_total = 0

    for family in all_families:
        fa = params_a.get(family, {})
        fb = params_b.get(family, {})
        all_indices = sorted(set(fa) | set(fb))

        added:   dict = {}
        removed: dict = {}
        changed: list = []

        for idx in all_indices:
            va = fa.get(idx)
            vb = fb.get(idx)

            if va is None and vb is not None:
                try:
                    added[idx] = float(vb)
                except (TypeError, ValueError):
                    added[idx] = vb
                n_added_total += 1
                continue

            if vb is None and va is not None:
                try:
                    removed[idx] = float(va)
                except (TypeError, ValueError):
                    removed[idx] = va
                n_removed_total += 1
                continue

            # Both present — check whether value changed
            try:
                va_f, vb_f = float(va), float(vb)
            except (TypeError, ValueError):
                continue

            abs_delta = abs(vb_f - va_f)
            if abs_delta < Thresholds.PARAM_CHANGE_TOL:
                continue  # numerically identical

            if abs(va_f) > Thresholds.PARAM_CHANGE_TOL:
                rel: float | None = (vb_f - va_f) / abs(va_f)
            else:
                rel = None  # 0 → nonzero; treat as largest change

            entry: dict = {
                "family":     family,
                "index":      idx,
                "value_a":    va_f,
                "value_b":    vb_f,
                "delta":      vb_f - va_f,
                "rel_change": rel,
                "magnitude":  _classify_magnitude(rel),
            }
            changed.append(entry)
            all_changes.append(entry)
            n_changed_total += 1

        if added or removed or changed:
            by_family[family] = {
                "added":   added,
                "removed": removed,
                "changed": sorted(changed, key=lambda x: _rel_sort_key(x["rel_change"]), reverse=True),
            }

    all_changes_ranked = sorted(all_changes, key=lambda x: _rel_sort_key(x["rel_change"]), reverse=True)
    n_large    = sum(1 for c in all_changes if c["magnitude"] == "large")
    n_moderate = sum(1 for c in all_changes if c["magnitude"] == "moderate")
    n_small    = sum(1 for c in all_changes if c["magnitude"] == "small")

    return {
        "by_family":          by_family,
        "top_changes":        all_changes_ranked[:20],
        "n_changed_total":    n_changed_total,
        "n_added_total":      n_added_total,
        "n_removed_total":    n_removed_total,
        "n_changes_large":    n_large,
        "n_changes_moderate": n_moderate,
        "n_changes_small":    n_small,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_structural_diff(
    store_a: EpochStore,
    store_b: EpochStore,
    force:   bool = False,
) -> dict:
    """
    Compute QT1 — Structural and Parametric Change Analysis.

    Pure Python, zero solver calls.  Compares structure.json and params.json
    from both epochs to identify:
      - Index set changes (new/removed sets, added/removed elements)
      - Variable family changes (new/removed families, count shifts)
      - Constraint family changes (new/removed families, template changes)
      - Mutable parameter changes (ranked by absolute relative change)

    The result is written to the EpochStore comparison cache on first
    computation.  Subsequent calls for the same pair return the cached result
    from disk unless ``force=True`` is passed.

    Args:
        store_a: EpochStore for the earlier (reference) epoch.
        store_b: EpochStore for the later (comparison) epoch.
        force:   Recompute even if a cached result exists.

    Returns:
        QT1 result dict — fully self-contained, suitable for LLM injection.
    """
    if not force:
        cached = store_a.load_comparison(store_b.epoch_id, "qt1")
        if cached is not None:
            return cached

    struct_a = store_a.get_structure()
    struct_b = store_b.get_structure()
    params_a = store_a.get_params()
    params_b = store_b.get_params()

    index_set_changes       = _compare_index_sets(struct_a, struct_b)
    variable_family_changes = _compare_variable_families(struct_a, struct_b)
    constraint_changes      = _compare_constraint_templates(struct_a, struct_b)
    parameter_changes       = _compare_parameters(params_a, params_b)

    set_labels = {
        **struct_a.get("index_set_docs", {}),
        **struct_b.get("index_set_docs", {}),
    }
    variable_labels = {
        family: (info.get("doc", "") or "").strip()
        for family, info in {
            **struct_a.get("variable_families", {}),
            **struct_b.get("variable_families", {}),
        }.items()
    }
    parameter_labels = {
        family: (info.get("doc", "") or "").strip()
        for family, info in {
            **struct_a.get("param_families", {}),
            **struct_b.get("param_families", {}),
        }.items()
    }
    constraint_labels = {
        **struct_a.get("constraint_docs", {}),
        **struct_b.get("constraint_docs", {}),
    }

    for set_name, payload in index_set_changes.get("modified_sets", {}).items():
        payload["label"] = (set_labels.get(set_name, "") or "").strip() or set_name

    for family in index_set_changes.get("added_sets", []):
        set_labels.setdefault(family, family)
    for family in index_set_changes.get("removed_sets", []):
        set_labels.setdefault(family, family)

    for change in parameter_changes.get("top_changes", []):
        family = change.get("family", "")
        change["family_label"] = parameter_labels.get(family, "") or family

    for family, payload in parameter_changes.get("by_family", {}).items():
        payload["family_label"] = parameter_labels.get(family, "") or family

    for family in variable_family_changes.get("added_families", []):
        variable_labels.setdefault(family, family)
    for family in variable_family_changes.get("removed_families", []):
        variable_labels.setdefault(family, family)
    for family, payload in variable_family_changes.get("count_changes", {}).items():
        payload["family_label"] = variable_labels.get(family, "") or family
    for family, payload in variable_family_changes.get("type_changes", {}).items():
        payload["family_label"] = variable_labels.get(family, "") or family

    for family in constraint_changes.get("added_families", []):
        constraint_labels.setdefault(family, family)
    for family in constraint_changes.get("removed_families", []):
        constraint_labels.setdefault(family, family)
    for family, payload in constraint_changes.get("template_changes", {}).items():
        payload["family_label"] = constraint_labels.get(family, "") or family

    # --- Summary ---
    has_structural = bool(
        index_set_changes["added_sets"]
        or index_set_changes["removed_sets"]
        or variable_family_changes["added_families"]
        or variable_family_changes["removed_families"]
        or constraint_changes["added_families"]
        or constraint_changes["removed_families"]
    )

    n_elem_add = sum(s["n_added"]   for s in index_set_changes["modified_sets"].values())
    n_elem_rem = sum(s["n_removed"] for s in index_set_changes["modified_sets"].values())

    summary: dict = {
        "n_set_additions":               len(index_set_changes["added_sets"]),
        "n_set_removals":                len(index_set_changes["removed_sets"]),
        "n_set_element_additions":       n_elem_add,
        "n_set_element_removals":        n_elem_rem,
        "n_variable_family_additions":   len(variable_family_changes["added_families"]),
        "n_variable_family_removals":    len(variable_family_changes["removed_families"]),
        "n_constraint_family_additions": len(constraint_changes["added_families"]),
        "n_constraint_family_removals":  len(constraint_changes["removed_families"]),
        "n_constraint_additions":        len(constraint_changes["added_families"]),
        "n_constraint_removals":         len(constraint_changes["removed_families"]),
        "n_param_changes":               parameter_changes["n_changed_total"],
        "n_param_additions":             parameter_changes["n_added_total"],
        "n_param_removals":              parameter_changes["n_removed_total"],
        "n_param_changes_large":         parameter_changes["n_changes_large"],
        "n_param_changes_moderate":      parameter_changes["n_changes_moderate"],
        "n_param_changes_small":         parameter_changes["n_changes_small"],
        "has_structural_changes":        has_structural,
        "has_parametric_changes":        bool(
            parameter_changes["n_changed_total"]
            + parameter_changes["n_added_total"]
            + parameter_changes["n_removed_total"]
        ),
    }

    result: dict = {
        "epoch_a":                      store_a.epoch_id,
        "epoch_b":                      store_b.epoch_id,
        "computed_at":                  datetime.now().isoformat(),
        "index_set_changes":            index_set_changes,
        "variable_family_changes":      variable_family_changes,
        "constraint_structure_changes": constraint_changes,
        "parameter_changes":            parameter_changes,
        "labels": {
            "sets": set_labels,
            "variables": variable_labels,
            "parameters": parameter_labels,
            "constraints": constraint_labels,
        },
        "summary":                      summary,
    }

    store_a.save_comparison(store_b.epoch_id, "qt1", result)
    return result
