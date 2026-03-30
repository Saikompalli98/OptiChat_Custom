"""
Rolling Horizon Comparison Framework — ADK Callback Implementations.

Three callbacks registered on the RH agents:

  rh_initialize_session(callback_context) → Content | None
      before_agent_callback on the root agent.
      On first invocation: parses a JSON config from the user message, loads and solves
      both epoch models, creates EpochStores, computes QT1, generates text descriptions,
      and populates all RH_* session-state keys.
      On subsequent invocations (session already initialized): restores the module-level
      EpochStore cache from disk if needed (e.g., after a server restart).

  rh_check_llm_request(callback_context, llm_request) → LlmResponse | None
      before_model_callback on both agents.
      Rebuilds the system instruction before every LLM call so the prompt is always
      self-contained (stateless injection pattern — mirrors OptiChat's check_llm_request).
      For rh_root_agent: fills ROOT_AGENT_PROMPT_TEMPLATE with session state.
      For rh_comparison_agent: builds the full comparison prompt (meta + descriptions +
        QT data) for the current RH_QUERY_TYPE; computes missing QT results on demand.

  rh_handle_illustrator_response is not implemented in Phase 4 (descriptions are
  generated directly from EpochStore structure data without invoking illustrator_agent).

JSON Config Format (passed as inline_data bytes or a text JSON block):
  {
    "epoch_a": {
      "epoch_id":    "epoch_a",           // optional; defaults to "epoch_a"
      "label":       "Baseline (April)",   // human-readable label
      "model_path":  "/abs/path/model.py", // required
      "data_path":   null                  // optional data JSON path
    },
    "epoch_b": {
      "epoch_id":    "epoch_b",
      "label":       "Updated (May)",
      "model_path":  "/abs/path/model_b.py",
      "data_path":   null
    }
  }
"""

from __future__ import annotations

import ast
import json
import re
import time
from typing import Any
from google.genai import types as genai_types


from loguru import logger
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest

from rh_comparison.config.rh_constants import (
    QueryType,
    RH_SESSION_INITIALIZED,
    RH_EPOCH_A_ID, RH_EPOCH_B_ID,
    RH_EPOCH_A_META, RH_EPOCH_B_META,
    RH_DESCRIPTION_A, RH_DESCRIPTION_B,
    RH_QT1_RESULT, RH_QT2_RESULT, RH_QT3_RESULT, RH_QT4_RESULT,
    RH_QUERY_TYPE,
    RH_PERSISTENT_STATES, RH_TEMPORARY_STATES,
)
from rh_comparison.data_store.epoch_store import EpochStore, load_model_from_py
from rh_comparison.analytics import (
    compute_structural_diff,
    compute_solution_diff,
    assess_backward_compat,
    compute_attribution_analysis,
)
from rh_comparison.agents.rh_prompts import (
    build_root_agent_prompt,
    build_comparison_agent_prompt,
)
from rh_comparison.tools.epoch_tools import _epoch_store_cache


# ---------------------------------------------------------------------------
# Internal helpers — session config parsing
# ---------------------------------------------------------------------------

_SESSION_INIT_MESSAGE = """
The Rolling Horizon Comparison session is not yet configured.

Please provide a JSON configuration with your two epoch models. You can pass it:
  • As plain text in your message (a JSON object starting with '{')
  • As a file upload (JSON bytes)

Required format:
{
  "epoch_a": {
    "epoch_id":   "epoch_a",
    "label":      "Baseline (period name)",
    "model_path": "/absolute/path/to/model_a.py",
    "data_path":  null
  },
  "epoch_b": {
    "epoch_id":   "epoch_b",
    "label":      "Updated (period name)",
    "model_path": "/absolute/path/to/model_b.py",
    "data_path":  null
  }
}
""".strip()


def _try_parse_json(raw: Any) -> dict | None:
    """Attempt to parse a JSON object from raw bytes or str. Returns dict or None."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return None
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw.startswith("{"):
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _extract_rh_config(callback_context: CallbackContext) -> dict | None:
    """
    Scan the current user message for an RH JSON config.

    Checks in order:
      1. inline_data bytes in any Part (uploaded file)
      2. text content in any Part (JSON pasted inline)

    Returns the parsed dict if it contains 'epoch_a' and 'epoch_b' keys, else None.
    """
    user_content = callback_context.user_content
    if user_content is None:
        return None

    candidates: list[Any] = []

    for part in user_content.parts:
        # Inline binary upload
        if hasattr(part, "inline_data") and part.inline_data is not None:
            data = part.inline_data
            raw = getattr(data, "data", None) or (data if isinstance(data, (bytes, bytearray)) else None)
            if raw is not None:
                candidates.append(raw)

        # Plain text
        if hasattr(part, "text") and part.text:
            candidates.append(part.text)

    for candidate in candidates:
        cfg = _try_parse_json(candidate)
        if cfg and "epoch_a" in cfg and "epoch_b" in cfg:
            return cfg

    return None


def _extract_user_question_text(callback_context: CallbackContext) -> str:
    user_content = callback_context.user_content
    if user_content is None:
        return ""

    parts: list[str] = []
    for part in user_content.parts:
        if hasattr(part, "text") and part.text:
            text = part.text.strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Internal helpers — epoch construction
# ---------------------------------------------------------------------------

def _build_epoch_store(epoch_cfg: dict) -> EpochStore:
    """
    Load and solve a single epoch, creating its EpochStore.

    If an EpochStore for this epoch_id already exists on disk (from a previous
    session or restart), it is reused without re-solving.

    Args:
        epoch_cfg: Dict with keys: epoch_id, label, model_path, data_path (opt).

    Returns:
        Populated EpochStore.
    """
    epoch_id   = epoch_cfg.get("epoch_id") or f"rh_epoch_{int(time.time())}"
    label      = epoch_cfg.get("label", epoch_id)
    model_path = epoch_cfg["model_path"]
    data_path  = epoch_cfg.get("data_path")

    # Reuse existing epoch if already on disk
    try:
        existing = EpochStore.load(epoch_id)
        logger.info(f"[RH] Reusing existing epoch '{epoch_id}' from disk.")
        return existing
    except FileNotFoundError:
        pass

    logger.info(f"[RH] Loading model for epoch '{epoch_id}' from: {model_path}")
    model = load_model_from_py(model_path, data_path)

    # Solve
    from pyomo.opt import SolverFactory
    solver  = SolverFactory("gurobi")
    results = solver.solve(model, tee=False)
    tc      = str(results.solver.termination_condition)
    logger.info(f"[RH] Epoch '{epoch_id}' solve status: {tc}")

    return EpochStore.create_from_pyomo(
        model=model,
        epoch_id=epoch_id,
        label=label,
        source_py_path=model_path,
        termination_condition=tc,
        data_json_path=data_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers — initialization response (static, no LLM)
# ---------------------------------------------------------------------------

def _guess_set_description(name: str, members: list) -> str:
    lname = name.lower()
    if lname in {"t", "time", "times", "period", "periods"} or all(isinstance(x, (int, float)) for x in members[:5]):
        return "The planning periods or stages covered by the model."
    sample_text = " ".join(str(x).lower() for x in members[:6])
    if any(token in sample_text for token in ("protein", "calorie", "vitamin", "iron", "calcium")):
        return "The nutrients or requirement categories that the plan must satisfy."
    if any(token in sample_text for token in ("flyer", "coupon", "sampling", "vehicle")):
        return "The available actions or campaign options the plan can choose from."
    return f"The set of items indexed by `{name}` in the model."


def _as_sentence(text: str, fallback: str = "") -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        cleaned = fallback
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _natural_join(items: list[str]) -> str:
    values = [str(item) for item in items if str(item)]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _natural_sort_key(value: Any) -> tuple:
    parsed = _parse_index(value)
    if isinstance(parsed, (int, float)):
        return (0, float(parsed))
    return (1, str(parsed).lower())


def _ordered_unique(items: list[str]) -> list[str]:
    seen = {str(item) for item in items if str(item)}
    return sorted(seen, key=_natural_sort_key)


def _parse_index(index: Any) -> Any:
    if isinstance(index, str):
        try:
            return ast.literal_eval(index)
        except (ValueError, SyntaxError):
            return index
    return index


def _is_internal_label(name: str, label: str) -> bool:
    text = f"{name} {label}".lower()
    return (
        name.lower() in {"d", "dpos", "exp_r", "lnb"}
        or "internal performance" in text
        or "internal uplift" in text
        or "approximation" in text
        or "geometric grid" in text
        or "logarithm" in text
    )


def _set_subject(label: str) -> str:
    text = " ".join(label.strip().split()).rstrip(".")
    lower = text.lower()
    if lower.startswith("set of "):
        text = text[7:]
    if not text:
        return "items"
    return text[0].lower() + text[1:]


def _summary_label(label: str) -> str:
    text = " ".join((label or "").strip().split()).rstrip(".")
    if text.lower().startswith("set of "):
        text = text[7:]
    if " (" in text:
        text = text.split(" (", 1)[0]
    for marker in (" representing ", " indicating ", " used to "):
        if marker in text.lower():
            parts = text.split(marker, 1)
            text = parts[0]
            break
    return text.rstrip(".")


def _index_context(label: str, indices: list[str]) -> str:
    joined = _natural_join(_ordered_unique(indices))
    return f"for {joined}"


def _describe_parameter_family_change(label: str, increased: list[str], decreased: list[str]) -> list[str]:
    label = _summary_label(label)
    lines: list[str] = []

    if increased:
        lines.append(f"- Higher {label.lower()} {_index_context(label, increased)}.")
    if decreased:
        lines.append(f"- Lower {label.lower()} {_index_context(label, decreased)}.")
    return lines


def _collect_dimension_members(indices: list[str], position: int) -> list[str]:
    members: list[str] = []
    for index in indices:
        parsed = _parse_index(index)
        if isinstance(parsed, tuple) and len(parsed) > position:
            members.append(str(parsed[position]))
    return sorted(set(members))


def _describe_multidimensional_parameter_change(
    family: str,
    label: str,
    added: list[str],
    removed: list[str],
    changed: list[dict],
) -> list[str]:
    if not changed and not added and not removed:
        return []

    lines: list[str] = []
    if added:
        lines.append(f"- New { _summary_label(label).lower() } entries {_index_context(label, added[:6])}.")
    if removed:
        lines.append(f"- Removed { _summary_label(label).lower() } entries {_index_context(label, removed[:6])}.")
    increased_keys = [str(entry.get("index", "?")) for entry in changed if (entry.get("value_b") or 0) > (entry.get("value_a") or 0)]
    decreased_keys = [str(entry.get("index", "?")) for entry in changed if (entry.get("value_b") or 0) < (entry.get("value_a") or 0)]
    if increased_keys:
        lines.append(f"- Higher { _summary_label(label).lower() } for combinations such as {_natural_join(_ordered_unique(increased_keys)[:4])}.")
    if decreased_keys:
        lines.append(f"- Lower { _summary_label(label).lower() } for combinations such as {_natural_join(_ordered_unique(decreased_keys)[:4])}.")
    return lines


def _summarize_parameter_family(
    family: str,
    family_payload: dict,
    param_docs: dict[str, str],
) -> list[str]:
    label = _summary_label((family_payload.get("family_label", "") or param_docs.get(family, "") or family).rstrip("."))
    if not label or _is_internal_label(family, label):
        return []

    added = [str(idx) for idx in family_payload.get("added", {}).keys()]
    removed = [str(idx) for idx in family_payload.get("removed", {}).keys()]
    changed = family_payload.get("changed", [])

    sample_index = None
    if changed:
        sample_index = changed[0].get("index")
    elif added:
        sample_index = added[0]
    elif removed:
        sample_index = removed[0]

    parsed = _parse_index(sample_index)
    if isinstance(parsed, tuple):
        return _describe_multidimensional_parameter_change(family, label, added, removed, changed)

    increased = _ordered_unique([str(entry.get("index", "?")) for entry in changed if (entry.get("value_b") or 0) > (entry.get("value_a") or 0)])
    decreased = _ordered_unique([str(entry.get("index", "?")) for entry in changed if (entry.get("value_b") or 0) < (entry.get("value_a") or 0)])
    lines = _describe_parameter_family_change(label, increased, decreased)

    if added:
        lines.append(f"- New {label.lower()} entries {_index_context(label, added[:6])}.")
    if removed:
        lines.append(f"- Removed {label.lower()} entries {_index_context(label, removed[:6])}.")
    return lines


def _describe_set_change(set_name: str, payload: dict, label: str) -> str | None:
    if _is_internal_label(set_name, label):
        return None

    added = [str(x) for x in payload.get("added_elements", [])[:4]]
    removed = [str(x) for x in payload.get("removed_elements", [])[:4]]
    subject = _summary_label(_set_subject(label))
    added = _ordered_unique(added)
    removed = _ordered_unique(removed)

    if added and removed:
        return (
            f"- Updated {subject}: "
            f"added {_natural_join(added)} and removed {_natural_join(removed)}."
        )
    if added:
        return f"- Added {subject}: {_natural_join(added)}."
    if removed:
        return f"- Removed {subject}: {_natural_join(removed)}."
    return None


def _summarize_variable_family_changes(variable_changes: dict, variable_docs: dict[str, str]) -> list[str]:
    lines: list[str] = []

    for family in variable_changes.get("added_families", []):
        label = _summary_label((variable_docs.get(family, "") or family).rstrip("."))
        if not _is_internal_label(family, label):
            lines.append(f"- New decision area: {label}.")

    for family in variable_changes.get("removed_families", []):
        label = _summary_label((variable_docs.get(family, "") or family).rstrip("."))
        if not _is_internal_label(family, label):
            lines.append(f"- Removed decision area: {label}.")

    return lines


def _summarize_parameter_changes(by_family: dict, param_docs: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for family, payload in by_family.items():
        lines.extend(_summarize_parameter_family(family, payload, param_docs))
    return lines


def _looks_like_sentence(text: str) -> bool:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return False
    if cleaned[-1] in ".!?":
        return True
    return bool(re.match(
        r"^(this|the|a|an|maximize|minimize|exactly|consistency|slot capacity|vehicle usage|profit weight|expected uplift|binary|set of)\b",
        cleaned,
        flags=re.IGNORECASE,
    )) or (cleaned[:1].isupper() and " " in cleaned)


def _singularize_phrase(text: str) -> str:
    singular = " ".join((text or "").strip().split()).lower()
    if singular.startswith("set of "):
        singular = singular[7:]
    replacements = (
        ("time periods", "time period"),
        ("planning periods", "planning period"),
        ("vehicles", "vehicle"),
        ("foods", "food"),
        ("nutrients", "nutrient"),
        ("categories", "category"),
        ("periods", "period"),
    )
    for before, after in replacements:
        singular = singular.replace(before, after)
    if " in which " in singular:
        singular = singular.split(" in which ", 1)[0]
    if " that " in singular:
        singular = singular.split(" that ", 1)[0]
    if singular.endswith("ies"):
        singular = singular[:-3] + "y"
    elif singular.endswith("s") and not singular.endswith(("ss", "us", "ics")):
        singular = singular[:-1]
    return singular.strip() or "item"


def _set_item_label(set_name: str, set_docs: dict[str, str]) -> str:
    raw = _summary_label((set_docs.get(set_name, "") or set_name).rstrip("."))
    if raw.lower() == set_name.lower() and len(set_name) <= 2:
        return "item"
    return _singularize_phrase(raw)


def _component_ref(name: str, indexed_over: list[str]) -> str:
    visible_dims = _visible_index_dims(indexed_over)
    if not visible_dims:
        return name
    return f"{name}[{', '.join(visible_dims)}]"


def _visible_index_dims(indexed_over: list[str]) -> list[str]:
    visible: list[str] = []
    for set_name in indexed_over or []:
        cleaned = str(set_name or "").strip()
        if not cleaned:
            continue
        if cleaned.lower() == "unindexedcomponent_set":
            continue
        visible.append(cleaned)
    return visible


def _humanize_doc_tokens(text: str, indexed_over: list[str], set_docs: dict[str, str]) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return ""
    humanized = cleaned
    for set_name in _visible_index_dims(indexed_over):
        item = _set_item_label(set_name, set_docs)
        if not item:
            continue
        humanized = re.sub(
            rf"\b{re.escape(item)}\s+{re.escape(set_name)}\b",
            f"each {item}",
            humanized,
            flags=re.IGNORECASE,
        )
        humanized = re.sub(
            rf"\b{re.escape(set_name)}\b",
            f"each {item}",
            humanized,
            flags=re.IGNORECASE,
        )
    return humanized


def _lead_sentence(
    kind: str,
    doc: str,
    indexed_over: list[str],
    set_docs: dict[str, str],
    fallback: str,
) -> str:
    cleaned = _humanize_doc_tokens(doc, indexed_over, set_docs).rstrip(".")
    if not cleaned:
        cleaned = fallback.rstrip(".")
    if kind == "objective":
        if _looks_like_sentence(cleaned):
            return _as_sentence(cleaned)
        return _as_sentence(f"The objective is to {cleaned}")
    if _looks_like_sentence(cleaned):
        return _as_sentence(cleaned)
    prefix = {
        "set": "The set of ",
        "parameter": "This input represents ",
        "variable": "This decision represents ",
        "constraint": "This rule represents ",
    }.get(kind, "")
    return _as_sentence(prefix + cleaned)


def _scope_sentence(kind: str, indexed_over: list[str], set_docs: dict[str, str]) -> str:
    dims = [_set_item_label(set_name, set_docs) for set_name in _visible_index_dims(indexed_over)]
    dims = [dim for dim in dims if dim and dim != "item"]
    if not dims:
        return ""
    if len(dims) == 1:
        target = f"each {dims[0]}"
    elif len(dims) == 2:
        target = f"each {dims[0]} and {dims[1]} combination"
    else:
        target = "each combination of " + _natural_join(dims)
    prefix = {
        "parameter": "This input is specified for ",
        "variable": "This decision is defined for ",
        "constraint": "This rule is enforced for ",
    }.get(kind, "")
    return _as_sentence(prefix + target) if prefix else ""


def _objective_sentence(struct: dict, meta: dict, set_docs: dict[str, str]) -> str:
    objective_doc = (struct.get("objective_doc", "") or "").strip()
    sense = meta.get("objective_sense", "minimize")
    generic_docs = {"objective", "objective function", "obj"}
    if objective_doc and objective_doc.strip().lower() not in generic_docs:
        return _lead_sentence("objective", objective_doc, [], set_docs, f"{sense} the overall business outcome")

    for family in struct.get("objective_variable_families", []) or []:
        info = struct.get("variable_families", {}).get(family, {})
        doc = (info.get("doc", "") or "").strip()
        if not doc:
            continue
        phrase = _humanize_doc_tokens(doc, info.get("indexed_over", []), set_docs).rstrip(".")
        if phrase:
            phrase = re.sub(
                r"^(binary variable indicating whether|binary decision selecting|variable indicating whether|decision selecting|decision|variable)\s+",
                "",
                phrase,
                flags=re.IGNORECASE,
            )
            return _as_sentence(f"The objective is to {sense} {phrase}")

    return _as_sentence(f"The objective is to {sense} the overall business outcome")


def _objective_clause(struct: dict, meta: dict, set_docs: dict[str, str]) -> str:
    sentence = _objective_sentence(struct, meta, set_docs).rstrip(".")
    lower = sentence.lower()
    for prefix in ("the objective is to ", "the goal is to "):
        if lower.startswith(prefix):
            return sentence[len(prefix):]
    return sentence


def _summary_phrases(families: dict[str, dict], set_docs: dict[str, str], *, limit: int = 2) -> list[str]:
    phrases: list[str] = []
    for name, info in families.items():
        doc = _humanize_doc_tokens(info.get("doc", ""), info.get("indexed_over", []), set_docs)
        phrase = _summary_label(doc or name).lower()
        if not phrase or _is_internal_label(name, phrase):
            continue
        phrases.append(phrase)
        if len(phrases) >= limit:
            break
    return phrases


def _build_intro(store: EpochStore) -> str:
    meta = store.get_meta()
    struct = store.get_structure()
    model_doc = _as_sentence(struct.get("model_doc", ""))
    set_docs = struct.get("index_set_docs", {})
    objective_sentence = _objective_sentence(struct, meta, set_docs)
    objective_clause = _objective_clause(struct, meta, set_docs)
    decision_docs = _summary_phrases(struct.get("variable_families", {}), set_docs, limit=2)
    input_docs = _summary_phrases(struct.get("param_families", {}), set_docs, limit=2)
    rule_docs = _summary_phrases(struct.get("constraint_families", {}), set_docs, limit=2)

    if model_doc and len(model_doc) >= 120:
        if objective_sentence.lower() not in model_doc.lower() and objective_sentence.lower() != "the objective is to minimize the overall business outcome.":
            return f"{model_doc} {objective_sentence}".strip()
        return model_doc

    lead = "This model represents a planning problem"
    if objective_clause:
        lead += f" where the goal is to {objective_clause}"
    intro_parts = [_as_sentence(lead)]
    if input_docs:
        intro_parts.append(f"It uses input data such as { _natural_join(input_docs) }.")
    if decision_docs:
        intro_parts.append(f"It makes decisions about { _natural_join(decision_docs) }.")
    if rule_docs:
        intro_parts.append(f"The plan must satisfy business rules such as { _natural_join(rule_docs) }.")
    intro_parts.append(
        "By choosing these decision values, the model constructs a plan that balances the available inputs against the stated business goal."
    )
    return " ".join(intro_parts).strip()


def _render_set_entry(name: str, members: list, doc: str) -> str:
    description = _lead_sentence("set", doc, [], {}, _guess_set_description(name, members if isinstance(members, list) else []))
    return f"- `{name}`: {description}"


def _render_component_entry(kind: str, name: str, info: dict, set_docs: dict[str, str]) -> str:
    indexed_over = _visible_index_dims(info.get("indexed_over", []))
    ref = _component_ref(name, indexed_over)
    fallback = {
        "parameter": f"a business input associated with {', '.join(indexed_over)}" if indexed_over else f"business input {name}",
        "variable": f"a decision defined over {', '.join(indexed_over)}" if indexed_over else f"decision {name}",
        "constraint": f"a business rule defined over {', '.join(indexed_over)}" if indexed_over else f"business rule {name}",
    }[kind]
    lead = _lead_sentence(kind, info.get("doc", ""), indexed_over, set_docs, fallback)
    scope = _scope_sentence(kind, indexed_over, set_docs)
    text = lead
    if scope and scope.lower() not in lead.lower():
        text = f"{lead} {scope}"
    return f"- `{ref}`: {text}"


def _profile_limit(profile: str, *, detailed: int, summary: int) -> int:
    return detailed if profile == "detailed" else summary


def _sample_items(items: list[str], profile: str, *, detailed: int = 6, summary: int = 3) -> list[str]:
    ordered = _ordered_unique(items)
    return ordered[:_profile_limit(profile, detailed=detailed, summary=summary)]


def _sample_phrase(items: list[str], profile: str) -> str:
    values = _ordered_unique(items)
    if not values:
        return ""
    numeric = True
    parsed_numbers: list[int] = []
    for item in values:
        try:
            number = float(str(_parse_index(item)))
        except (TypeError, ValueError):
            numeric = False
            break
        if abs(number - round(number)) > 1e-9:
            numeric = False
            break
        parsed_numbers.append(int(round(number)))

    if numeric and parsed_numbers:
        ranges: list[str] = []
        start = end = parsed_numbers[0]
        for number in parsed_numbers[1:]:
            if number == end + 1:
                end = number
                continue
            if start == end:
                ranges.append(f"{start}")
            elif end == start + 1:
                ranges.extend([f"{start}", f"{end}"])
            else:
                ranges.append(f"{start} through {end}")
            start = end = number
        if start == end:
            ranges.append(f"{start}")
        elif end == start + 1:
            ranges.extend([f"{start}", f"{end}"])
        else:
            ranges.append(f"{start} through {end}")
        shown = _sample_items(ranges, profile, detailed=4, summary=3)
        if len(ranges) > len(shown):
            return _natural_join(shown) + f", and {len(ranges) - len(shown)} other ranges"
        return _natural_join(shown)

    shown = _sample_items(values, profile)
    if len(values) > len(shown):
        return _natural_join(shown) + f", and {len(values) - len(shown)} other items"
    return _natural_join(shown)


def _render_init_set_change(set_name: str, payload: dict, label: str, profile: str) -> str | None:
    if _is_internal_label(set_name, label):
        return None
    subject = _summary_label(_set_subject(label))
    added = [str(x) for x in payload.get("added_elements", [])]
    removed = [str(x) for x in payload.get("removed_elements", [])]
    if added and removed:
        return f"- {subject.capitalize()} changed: added {_sample_phrase(added, profile)} and removed {_sample_phrase(removed, profile)}."
    if added:
        return f"- {subject.capitalize()} now include {_sample_phrase(added, profile)}."
    if removed:
        return f"- {subject.capitalize()} no longer include {_sample_phrase(removed, profile)}."
    return None


def _render_init_parameter_family_summary(
    family: str,
    family_payload: dict,
    param_docs: dict[str, str],
    profile: str,
) -> list[str]:
    label = _summary_label((family_payload.get("family_label", "") or param_docs.get(family, "") or family).rstrip("."))
    if not label or _is_internal_label(family, label):
        return []

    added = [str(idx) for idx in family_payload.get("added", {}).keys()]
    removed = [str(idx) for idx in family_payload.get("removed", {}).keys()]
    changed = family_payload.get("changed", [])
    increased = _ordered_unique([str(entry.get("index", "?")) for entry in changed if (entry.get("value_b") or 0) > (entry.get("value_a") or 0)])
    decreased = _ordered_unique([str(entry.get("index", "?")) for entry in changed if (entry.get("value_b") or 0) < (entry.get("value_a") or 0)])

    parts: list[str] = []
    if increased:
        if len(increased) <= _profile_limit(profile, detailed=6, summary=3):
            parts.append(f"increased for {_sample_phrase(increased, profile)}")
        else:
            parts.append(f"increased across many items, including {_sample_phrase(increased, profile)}")
    if decreased:
        if len(decreased) <= _profile_limit(profile, detailed=6, summary=3):
            parts.append(f"decreased for {_sample_phrase(decreased, profile)}")
        else:
            parts.append(f"decreased across many items, including {_sample_phrase(decreased, profile)}")
    if added:
        parts.append(f"now covers {_sample_phrase(added, profile)} as well")
    if removed:
        parts.append(f"no longer covers {_sample_phrase(removed, profile)}")

    if not parts:
        return []
    if profile == "summary":
        return [f"- {label}: " + "; ".join(parts[:2]) + "."]
    return [f"- {label} " + "; ".join(parts) + "."]


def _visible_qt1_family_count(
    qt1: dict,
    set_docs: dict[str, str],
    param_docs: dict[str, str],
    variable_docs: dict[str, str],
    constraint_docs: dict[str, str],
) -> int:
    count = 0
    for set_name, payload in qt1.get("index_set_changes", {}).get("modified_sets", {}).items():
        label = (payload.get("label", "") or set_docs.get(set_name, "") or set_name).rstrip(".")
        if not _is_internal_label(set_name, label):
            count += 1
    count += sum(
        1 for name in qt1.get("index_set_changes", {}).get("added_sets", [])
        if not _is_internal_label(name, set_docs.get(name, "") or name)
    )
    count += sum(
        1 for name in qt1.get("index_set_changes", {}).get("removed_sets", [])
        if not _is_internal_label(name, set_docs.get(name, "") or name)
    )
    count += sum(
        1 for family in qt1.get("parameter_changes", {}).get("by_family", {})
        if not _is_internal_label(family, param_docs.get(family, "") or family)
    )
    count += sum(
        1 for family in qt1.get("variable_family_changes", {}).get("added_families", [])
        if not _is_internal_label(family, variable_docs.get(family, "") or family)
    )
    count += sum(
        1 for family in qt1.get("variable_family_changes", {}).get("removed_families", [])
        if not _is_internal_label(family, variable_docs.get(family, "") or family)
    )
    count += sum(
        1 for family in qt1.get("constraint_structure_changes", {}).get("added_families", [])
        if not _is_internal_label(family, constraint_docs.get(family, "") or family)
    )
    count += sum(
        1 for family in qt1.get("constraint_structure_changes", {}).get("removed_families", [])
        if not _is_internal_label(family, constraint_docs.get(family, "") or family)
    )
    return count


def _non_header_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip() and not line.startswith("**"))


def _build_model_overview(store: EpochStore, profile: str = "detailed") -> str:
    struct = store.get_structure()
    meta = store.get_meta()
    set_docs = struct.get("index_set_docs", {})
    objective_sentence = _objective_sentence(struct, meta, set_docs)

    index_sets = struct.get("index_sets", {})
    var_fams = struct.get("variable_families", {})
    par_fams = struct.get("param_families", {})
    con_fams = struct.get("constraint_families", {})
    if not con_fams and struct.get("constraint_docs"):
        con_fams = {
            name: {"indexed_over": [], "count": 0, "doc": doc}
            for name, doc in struct.get("constraint_docs", {}).items()
        }

    lines = [_build_intro(store)]

    if index_sets:
        lines.extend(["", "**Sets**"])
    for name, members in list(index_sets.items())[:_profile_limit(profile, detailed=5, summary=4)]:
        lines.append(_render_set_entry(name, members, (set_docs.get(name, "") or "").strip()))

    if par_fams:
        lines.extend(["", "**Parameters**"])
    for name, info in list(par_fams.items())[:_profile_limit(profile, detailed=6, summary=4)]:
        lines.append(_render_component_entry("parameter", name, info, set_docs))

    if var_fams:
        lines.extend(["", "**Variables**"])
    for name, info in list(var_fams.items())[:_profile_limit(profile, detailed=6, summary=4)]:
        lines.append(_render_component_entry("variable", name, info, set_docs))

    if con_fams:
        lines.extend(["", "**Constraints**"])
    shown_constraints = 0
    for name, info in con_fams.items():
        lines.append(_render_component_entry("constraint", name, info, set_docs))
        shown_constraints += 1
        if shown_constraints >= _profile_limit(profile, detailed=6, summary=4):
            break

    lines.extend([
        "",
        "**Objective**",
        f"- `{struct.get('objective_name', 'objective')}`: {objective_sentence}",
    ])
    return "\n".join(lines)


def _build_change_overview(store_a: EpochStore, store_b: EpochStore, qt1: dict, profile: str = "detailed") -> str:
    label_a = store_a.get_meta().get("label", "the earlier plan")
    label_b = store_b.get_meta().get("label", "the later plan")
    struct_a = store_a.get_structure()
    struct_b = store_b.get_structure()
    set_docs = {**struct_a.get("index_set_docs", {}), **struct_b.get("index_set_docs", {})}
    param_docs = {
        **{k: (v.get("doc", "") or "").strip() for k, v in struct_a.get("param_families", {}).items()},
        **{k: (v.get("doc", "") or "").strip() for k, v in struct_b.get("param_families", {}).items()},
    }
    variable_docs = {
        **{k: (v.get("doc", "") or "").strip() for k, v in struct_a.get("variable_families", {}).items()},
        **{k: (v.get("doc", "") or "").strip() for k, v in struct_b.get("variable_families", {}).items()},
    }
    constraint_docs = {
        **struct_a.get("constraint_docs", {}),
        **struct_b.get("constraint_docs", {}),
    }

    lines = [f"**What Changed Between {label_a} and {label_b}**"]
    modified_sets = qt1.get("index_set_changes", {}).get("modified_sets", {})
    if modified_sets:
        for set_name, payload in modified_sets.items():
            label = (payload.get("label", "") or set_docs.get(set_name, "") or f"`{set_name}`").rstrip(".")
            line = _render_init_set_change(set_name, payload, label, profile)
            if line:
                lines.append(line)

    variable_changes = qt1.get("variable_family_changes", {})
    for family in variable_changes.get("added_families", []):
        label = _summary_label((variable_docs.get(family, "") or family).rstrip("."))
        if not _is_internal_label(family, label):
            lines.append(f"- New decision area: {label}.")
    for family in variable_changes.get("removed_families", []):
        label = _summary_label((variable_docs.get(family, "") or family).rstrip("."))
        if not _is_internal_label(family, label):
            lines.append(f"- Removed decision area: {label}.")

    all_param_families = qt1.get("parameter_changes", {}).get("by_family", {})
    for family, payload in all_param_families.items():
        lines.extend(_render_init_parameter_family_summary(family, payload, param_docs, profile))

    constraint_changes = qt1.get("constraint_structure_changes", {})
    added_cons = constraint_changes.get("added_families", [])
    removed_cons = constraint_changes.get("removed_families", [])
    if added_cons:
        named = [(constraint_docs.get(name, "") or name).rstrip(".") for name in added_cons]
        lines.append("- New business rules were added: " + _natural_join(named) + ".")
    if removed_cons:
        named = [(constraint_docs.get(name, "") or name).rstrip(".") for name in removed_cons]
        lines.append("- Some business rules were removed: " + _natural_join(named) + ".")

    if len(lines) == 1:
        lines.append("- The later run keeps the same overall model shape and mainly reflects updated business data.")

    return "\n".join(lines)

def _format_init_response(
    store_a: EpochStore,
    store_b: EpochStore,
    qt1: dict,
    desc_a: str,
    desc_b: str,
) -> str:
    """
    Build a concise manager-friendly initialization summary.

    The user should first understand the underlying planning model, then the
    high-level ways the two planning runs differ.
    """
    struct_a = store_a.get_structure()
    struct_b = store_b.get_structure()
    set_docs = {**struct_a.get("index_set_docs", {}), **struct_b.get("index_set_docs", {})}
    param_docs = {
        **{k: (v.get("doc", "") or "").strip() for k, v in struct_a.get("param_families", {}).items()},
        **{k: (v.get("doc", "") or "").strip() for k, v in struct_b.get("param_families", {}).items()},
    }
    variable_docs = {
        **{k: (v.get("doc", "") or "").strip() for k, v in struct_a.get("variable_families", {}).items()},
        **{k: (v.get("doc", "") or "").strip() for k, v in struct_b.get("variable_families", {}).items()},
    }
    constraint_docs = {
        **struct_a.get("constraint_docs", {}),
        **struct_b.get("constraint_docs", {}),
    }
    model_detailed = _build_model_overview(store_a, profile="detailed")
    change_detailed = _build_change_overview(store_a, store_b, qt1, profile="detailed")
    family_count = _visible_qt1_family_count(qt1, set_docs, param_docs, variable_docs, constraint_docs)
    detail_rows = _non_header_line_count(change_detailed)
    rendered_chars = len(model_detailed) + len(change_detailed)
    profile = (
        "detailed"
        if family_count <= 8 and detail_rows <= 20 and rendered_chars <= 2500
        else "summary"
    )
    return (
        _build_model_overview(store_a, profile=profile)
        + "\n\n"
        + _build_change_overview(store_a, store_b, qt1, profile=profile)
    )


# ---------------------------------------------------------------------------
# Internal helpers — description generation
# ---------------------------------------------------------------------------

def _generate_epoch_description(store: EpochStore) -> str:
    """
    Build a reusable business-facing model explanation for prompt injection.
    """
    return _build_model_overview(store)


# ---------------------------------------------------------------------------
# Internal helpers — restore stores after server restart
# ---------------------------------------------------------------------------

def _restore_stores_from_state(state: dict) -> None:
    """
    Reload EpochStore objects from disk into the module-level cache.

    Called when the session is already initialized (RH_SESSION_INITIALIZED=True)
    but the process-level cache is empty (server restart scenario).
    """
    for key in (RH_EPOCH_A_ID, RH_EPOCH_B_ID):
        eid = state.get(key, "")
        if eid and eid not in _epoch_store_cache:
            try:
                _epoch_store_cache[eid] = EpochStore.load(eid)
                logger.info(f"[RH] Restored EpochStore '{eid}' from disk.")
            except FileNotFoundError:
                logger.warning(f"[RH] Could not restore EpochStore '{eid}' — data missing.")


# ---------------------------------------------------------------------------
# Internal helpers — on-demand QT computation for prompt injection
# ---------------------------------------------------------------------------

def _ensure_qt_for_query_type(state: dict, query_type: str) -> None:
    """
    Ensure the QT results needed for the given query_type are in state.

    QT2 is always computed (it is fast, pure Python, and needed by QT4).
    QT3 and QT4 are computed only when their query type is active.

    All results are cached via on-disk EpochStore.save_comparison() after first compute,
    so subsequent calls return the cached value immediately.

    Errors are caught and logged so that a failing QT computation does not crash
    the entire LLM callback — the formatter will simply show "(Not yet computed)".
    """
    epoch_a_id = state.get(RH_EPOCH_A_ID, "")
    epoch_b_id = state.get(RH_EPOCH_B_ID, "")
    store_a    = _epoch_store_cache.get(epoch_a_id)
    store_b    = _epoch_store_cache.get(epoch_b_id)

    if store_a is None or store_b is None:
        return  # Cannot compute — stores not loaded yet

    qt1 = state.get(RH_QT1_RESULT)

    # QT2 is needed by broad comparison and answer synthesis, but not for
    # pure model-description, structural-only, or retrieval-only questions.
    needs_qt2 = query_type not in (
        QueryType.RETRIEVAL,
        QueryType.MODEL_DESCRIPTION,
        QueryType.STRUCTURAL_CHANGE,
    )
    if needs_qt2 and state.get(RH_QT2_RESULT) is None:
        try:
            logger.info("[RH] Computing QT2...")
            qt2 = compute_solution_diff(store_a, store_b, qt1=qt1)
            state[RH_QT2_RESULT] = qt2
        except Exception as exc:
            logger.error(f"[RH] QT2 computation failed: {exc}")

    qt2 = state.get(RH_QT2_RESULT)

    if query_type == QueryType.BACKWARD_COMPAT and state.get(RH_QT3_RESULT) is None:
        try:
            logger.info("[RH] Computing QT3 (1 solver call)...")
            qt3 = assess_backward_compat(store_a, store_b, qt1=qt1, qt2=qt2)
            state[RH_QT3_RESULT] = qt3
        except Exception as exc:
            logger.error(f"[RH] QT3 computation failed: {exc}")

    if query_type == QueryType.ATTRIBUTION and state.get(RH_QT4_RESULT) is None:
        try:
            logger.info("[RH] Computing QT4...")
            qt4 = compute_attribution_analysis(store_a, store_b, qt1=qt1, qt2=qt2)
            state[RH_QT4_RESULT] = qt4
        except Exception as exc:
            logger.error(f"[RH] QT4 computation failed: {exc}")


# ---------------------------------------------------------------------------
# Public Callbacks
# ---------------------------------------------------------------------------

def rh_initialize_session(callback_context: CallbackContext) -> types.Content | None:
    """
    before_agent_callback for rh_root_agent.

    On the FIRST invocation for a session:
      1. Initializes all RH_* state keys to their defaults.
      2. Scans the user message for a JSON config.
      3. Loads, solves, and stores both epoch models.
      4. Computes QT1 (structural diff) eagerly.
      5. Generates compact text descriptions for each epoch.
      6. Populates all persistent state keys.
      7. Returns None so the root agent proceeds normally.

    If no config is found, returns an instructional Content message.
    On subsequent invocations (already initialized): restores the EpochStore
    process cache if needed, then returns None.
    """
    state = callback_context.state

    # --- Already initialized ---
    if state.get(RH_SESSION_INITIALIZED):
        _restore_stores_from_state(state)
        return None

    # --- First-time init: set defaults ---
    for key, val in RH_PERSISTENT_STATES.items():
        if key not in state:
            state[key] = val
    for key, val in RH_TEMPORARY_STATES.items():
        if key not in state:
            state[key] = val

    # --- Parse config ---
    cfg = _extract_rh_config(callback_context)
    if cfg is None:
        return types.Content(
            role="model",
            parts=[types.Part(text=_SESSION_INIT_MESSAGE)],
        )

    # --- Build epoch stores ---
    try:
        ea_cfg = cfg["epoch_a"]
        eb_cfg = cfg["epoch_b"]

        store_a = _build_epoch_store(ea_cfg)
        store_b = _build_epoch_store(eb_cfg)

        # Register in process cache
        _epoch_store_cache[store_a.epoch_id] = store_a
        _epoch_store_cache[store_b.epoch_id] = store_b

        # QT1 — computed eagerly (pure Python, fast)
        logger.info("[RH] Computing QT1 (structural diff)...")
        qt1 = compute_structural_diff(store_a, store_b)

        # Descriptions — generated from structure.json (no solver, no illustrator)
        desc_a = _generate_epoch_description(store_a)
        desc_b = _generate_epoch_description(store_b)
        store_a.save_description(desc_a)
        store_b.save_description(desc_b)

        # Populate state
        state[RH_SESSION_INITIALIZED] = True
        state[RH_EPOCH_A_ID]          = store_a.epoch_id
        state[RH_EPOCH_B_ID]          = store_b.epoch_id
        state[RH_EPOCH_A_META]        = store_a.get_meta()
        state[RH_EPOCH_B_META]        = store_b.get_meta()
        state[RH_DESCRIPTION_A]       = desc_a
        state[RH_DESCRIPTION_B]       = desc_b
        state[RH_QT1_RESULT]          = qt1

        logger.info(
            f"[RH] Session initialized: '{store_a.epoch_id}' vs '{store_b.epoch_id}' | "
            f"QT1 computed, descriptions generated."
        )
        # Return a structured Content (model descriptions + structural diff).
        # This short-circuits the root agent — no LLM call on init, so the
        # response is guaranteed to be factual, jargon-free, and free of
        # solution-level inference.
        init_text = _format_init_response(store_a, store_b, qt1, desc_a, desc_b)
        return types.Content(role="model", parts=[types.Part(text=init_text)])

    except KeyError as exc:
        msg = f"Invalid config — missing required field: {exc}. Expected 'model_path' in epoch_a/epoch_b."
        logger.error(f"[RH] {msg}")
        return types.Content(role="model", parts=[types.Part(text=msg)])

    except Exception as exc:
        msg = f"Session initialization failed: {exc}"
        logger.error(f"[RH] {msg}")
        return types.Content(role="model", parts=[types.Part(text=msg)])


def rh_check_llm_request(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """
    before_model_callback on both rh_root_agent and rh_comparison_agent.

    Rebuilds the system instruction on EVERY LLM call so that the prompt is
    fully self-contained (stateless injection — no reliance on conversation history).

    For rh_root_agent:
        Fills ROOT_AGENT_PROMPT_TEMPLATE with current epoch metadata and session status.

    For rh_comparison_agent:
        1. Reads RH_QUERY_TYPE from state (set by root agent via set_query_type()).
        2. Ensures the required QT result is in state (computes on demand if needed).
        3. Builds and injects the full comparison prompt via build_comparison_agent_prompt().

    Returns None to let the LLM call proceed normally.
    """
    state      = callback_context.state
    agent_name = callback_context.agent_name

    if agent_name == "rh_root_agent":
        system_prompt = build_root_agent_prompt(state)
        if llm_request.config is None:
            llm_request.config = genai_types.GenerateContentConfig()
        llm_request.config.system_instruction = system_prompt

    elif agent_name == "rh_comparison_agent":
        query_type = state.get(RH_QUERY_TYPE, QueryType.GENERAL)
        user_question = _extract_user_question_text(callback_context)

        # Ensure needed QT results are in state before building the prompt
        _ensure_qt_for_query_type(state, query_type)

        system_prompt = build_comparison_agent_prompt(state, query_type, user_question=user_question)
        if llm_request.config is None:
            llm_request.config = genai_types.GenerateContentConfig()
        llm_request.config.system_instruction = system_prompt

    return None
