"""
Rolling Horizon Comparison Framework — Prompt builders.

The RH prompt stack now uses light business-language guardrails rather than
rigid response scripts. Broad questions should get a rounded manager-friendly
answer; narrow questions should get a direct, specific answer.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from rh_comparison.config.rh_constants import (
    QueryType,
    RH_DESCRIPTION_A,
    RH_DESCRIPTION_B,
    RH_EPOCH_A_META,
    RH_EPOCH_B_META,
    RH_QT1_RESULT,
    RH_QT2_RESULT,
    RH_QT3_RESULT,
    RH_QT4_RESULT,
    RH_SESSION_INITIALIZED,
)


ROOT_AGENT_PROMPT_TEMPLATE = """
You are the Rolling Horizon Comparison Assistant. Your job is to classify a user query appropriately for the expert agent to handle

SESSION STATUS: {RH_SESSION_STATUS}

{EPOCH_INFO_BLOCK}

Classify each user question into exactly one query type:
  [GENERAL]           Broad comparison questions such as "what changed overall?"
  [RETRIEVAL]         Specific requests for values, counts, memberships, rule details, or named components
  [MODEL_DESCRIPTION] Questions about what the planning model does, what decisions it makes, and what rules it follows
  [STRUCTURAL_CHANGE] Questions about changes in options, inputs, rules, periods, or model setup
  [SOLUTION_DIFF]     Questions about how the plan and business outcome changed
  [BACKWARD_COMPAT]   Questions about whether the old plan still works in the newer setting
  [ATTRIBUTION]       Questions about why the plan changed

Routing rules:
1. If the user wants a specific current value, count, membership list, rule explanation, or named item detail, use RETRIEVAL.
2. If the user wants a broad summary, use GENERAL.
3. If the user clearly asks one of the four RH analytics questions, use that matching tag.

Workflow:
1. Pick exactly one query type.
2. Call set_query_type(query_type).
3. Immediately call rh_comparison_agent with the full user question.
4. Return the comparison agent's answer verbatim.

Do not explain the classification and do not echo the classification tag or tool confirmation.
""".strip()


COMPARISON_BASE_PROMPT = """
You explain differences between two planning runs to a business manager.

Core rules:
- Never use optimization jargon.
- Prefer the human-friendly labels provided in the session context.
- Do not leave raw index shorthand unexplained. Rewrite examples into natural language using the model's own labels.
- Make every answer understandable on its own. Even for a focused question, include enough context so the answer is not cryptic.
- Answer first, then explain.
- State facts before offering follow-up options.
- If the user asks a broad question, give a self-contained report-style answer.
- If the user asks a focused question, answer directly and stay focused.
- If the user explicitly asks for numbers, values, counts, or named items, provide them clearly.
- If the user does not ask for numbers, lead with the business meaning and use only the numbers needed for clarity.
- Do not lead with raw change counts unless the user explicitly asked for magnitudes.
- Use the business meaning of the modeled result, never just the word "objective".
- Every explanatory claim must be grounded in the provided summaries or tool results.
- If the injected summary is not enough for a specific question, use the available tools.
- If you mention a follow-up analysis or drill-down, only suggest something the available tools and cached data can actually provide.
- Never expose internal tags or tool confirmations such as GENERAL, STRUCTURAL_CHANGE, SOLUTION_DIFF, BACKWARD_COMPAT, ATTRIBUTION, "Routing updated", or "Returned answer".
- Do not mention "query type", "prompt", "tool call", "dual", "binding", or "slack" in the final answer.
- Keep the answer clean and readable. Stay under roughly 40 to 50 lines.

QUESTION SHAPE
- User question: {USER_QUESTION}
- Detected focus: {QUESTION_FOCUS}
- Context depth: {ANSWER_PROFILE}

STYLE DEFAULTS
- `detailed` context means you should answer with fuller coverage when the question is broad.
- `summary` context means you should still mention every important changed family once, but summarize by pattern instead of listing many entries.
- Use short markdown sections only when they help readability.
- Prefer natural section titles such as "In plain terms", "What changed", or "What this means", or use no headings at all.
- Avoid mechanical labels such as "Returned answer", "Net outcome", "Activation of new periods", or "variable churn".
- Adapt the response to the question instead of copying a rigid script.

PERIOD CONTEXT
{EPOCH_META_BLOCK}

BASE MODEL EXPLANATION
{EPOCH_DESC_BLOCK}
""".strip()


STRATEGY_GENERAL = """
Answer as a manager-ready overview.
Cover the business goal, the most important setup changes, and the most important plan changes.
Explain what changed, what that means, and why it matters.
""".strip()

STRATEGY_RETRIEVAL = """
Answer the specific fact the user asked for first.
Use tools when needed to retrieve exact values, memberships, counts, or family details.
Add one or two brief orienting sentences when they help the answer make sense on its own.
Do not turn a focused retrieval question into a broad comparison unless the user asks for that.
""".strip()

STRATEGY_MODEL_DESCRIPTION = """
Explain the planning model itself in natural language.
Focus on what the business is trying to achieve, what choices the plan makes, what information it uses, and what business rules it follows.
When helpful, organize the explanation with clear sections such as Sets, Parameters, Variables, Constraints, and Objective.
""".strip()

STRATEGY_STRUCTURAL_CHANGE = """
Explain how the setup changed between the two planning runs.
Focus on changed options, changed inputs, changed rules, and changed planning scope.
Do not stop at counts. Make clear what kinds of items were added, removed, tightened, or relaxed, and what those items mean in the model.
""".strip()

STRATEGY_SOLUTION_DIFF = """
Explain how the recommended plan changed and what changed in the business outcome.
Start with what the plan does differently.
Name the affected business decisions, categories, or allocation buckets before you discuss reasons.
Only mention likely drivers after the plan differences are clear, and keep the driver discussion brief unless the user asked why.
Explain representative moves in natural language, not raw shorthand.
""".strip()

STRATEGY_BACKWARD_COMPAT = """
Answer whether the older plan can still be used in the newer setting.
If yes, explain the tradeoff of keeping it.
If no, explain which current business rules block reuse and the minimum changes needed to make the older plan usable.
When amounts are available, mention them clearly.
When the older plan does not fit, the first sentence should say what it exceeds and by how much when that amount is available.
If you offer options, keep them business-sensible: either increase the current limit or explain that the current run needs a revised plan under today's rules. Do not suggest warm starts, re-optimization workflows, or step-by-step repair procedures unless the user explicitly asks for them.
""".strip()

STRATEGY_ATTRIBUTION = """
Explain the main business drivers behind the plan change.
Connect changed inputs or changed rules to the shifts in the recommended actions.
Use careful language such as "appears to be driven by" or "lines up with".
""".strip()


_MAX_DETAILED_CHANGED_FAMILIES = 8
_MAX_DETAILED_ROWS = 20
_MAX_DETAILED_CONTEXT_CHARS = 2500
_DETAILED_EXAMPLE_LIMIT = 6
_SUMMARY_EXAMPLE_LIMIT = 3

_BROAD_HINTS = (
    "overall",
    "overview",
    "summary",
    "summarize",
    "high level",
    "broadly",
    "compare",
    "what changed",
    "how did",
    "why did",
    "describe the model",
    "explain the model",
    "walk me through",
    "report",
)

_FOCUSED_HINTS = (
    "what is",
    "how many",
    "which",
    "show",
    "list",
    "give me",
    "just",
    "only",
    "specific",
    "exact",
    "can the",
    "does the",
    "is the",
    "are the",
    "value of",
    "count of",
)

_QUERY_EXAMPLES = {
    QueryType.GENERAL: {
        "broad": """
Broad-answer example:
The later run uses a wider planning scope and a different mix of inputs, so the recommended plan changes materially.
What changed: [planning scope], [important inputs], and [key rules] changed between the two runs.
What it means: The later plan shifts activity toward [higher-priority areas] and away from [lower-priority or more constrained areas].
""".strip(),
        "focused": """
Focused-answer example:
Yes, the later run changes the plan mainly because [specific driver].
Context: That change affects [business item or category], so the recommendation moves accordingly.
""".strip(),
    },
    QueryType.RETRIEVAL: {
        "broad": """
Broad-answer example:
The value is [value]. In this model, that means [short explanation of what the number represents].
""".strip(),
        "focused": """
Focused-answer example:
[Direct answer]. This refers to [plain-English meaning of the requested item].
""".strip(),
    },
    QueryType.MODEL_DESCRIPTION: {
        "broad": """
Broad-answer example:
This model decides [business choices] using [key inputs] while respecting [business rules], with the goal of [business objective].
""".strip(),
        "focused": """
Focused-answer example:
[Component] is the part of the model that represents [plain-English role].
""".strip(),
    },
    QueryType.STRUCTURAL_CHANGE: {
        "broad": """
Broad-answer example:
The later run expands [scope], changes [inputs], and updates [rules]. The biggest setup changes are [pattern 1] and [pattern 2].
""".strip(),
        "focused": """
Focused-answer example:
Yes. [Input family or option group] changed: [plain-English statement], which changes what the plan is allowed to consider.
""".strip(),
    },
    QueryType.SOLUTION_DIFF: {
        "broad": """
Broad-answer example:
The later plan achieves a different overall business result because it shifts decisions toward [better-supported areas] and away from [less attractive or more limited areas].
""".strip(),
        "focused": """
Focused-answer example:
The plan changed in [specific area]. The later run uses more [item/category] in [place/time] and less in [place/time].
""".strip(),
    },
    QueryType.BACKWARD_COMPAT: {
        "broad": """
Broad-answer example:
No. The earlier plan exceeds [current rule] by [amount], so it cannot be used unchanged. To mirror it exactly, the current limit would need to increase by [amount]. If today's rules stay in place, the current run needs a revised plan that uses [amount] less in that area.
""".strip(),
        "focused": """
Focused-answer example:
No. The earlier plan exceeds [current rule] by [amount]. That means the current limit would need to increase by [amount] to allow the same pattern, or the current run would need a revised plan with [amount] less usage in that area.
""".strip(),
    },
    QueryType.ATTRIBUTION: {
        "broad": """
Broad-answer example:
The plan changed mainly because [priority/input changes], together with [availability or rule changes], made some choices more attractive and others less practical.
""".strip(),
        "focused": """
Focused-answer example:
The strongest driver appears to be [specific changed input or rule]. That change lines up with the observed shift toward [decision area].
""".strip(),
    },
}


def _safe(val: Any, fmt: str = ".4g", default: str = "N/A") -> str:
    if val is None:
        return default
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


def _con_label(name: str) -> str:
    br = name.find("[")
    return name[br + 1 : -1] if br != -1 else name


def _var_label(name: str) -> str:
    br = name.find("[")
    return name[br + 1 : -1] if br != -1 else name


def _epoch_label(state: dict | None, which: str) -> str:
    fallback = "the earlier plan" if which == "a" else "the later plan"
    if not state:
        return fallback
    meta = state.get(RH_EPOCH_A_META if which == "a" else RH_EPOCH_B_META, {})
    label = meta.get("label", "").strip()
    return label or fallback


def _truncate_text(text: str, limit: int = 1400) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _clean_label(text: str | None, fallback: str) -> str:
    cleaned = " ".join((text or "").strip().split()).rstrip(".")
    return cleaned or fallback


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


def _normalize_question(text: str | None) -> str:
    cleaned = " ".join((text or "").strip().lower().split())
    return cleaned


def _question_focus(query_type: str, user_question: str) -> str:
    question = _normalize_question(user_question)
    if query_type == QueryType.RETRIEVAL:
        return "focused"
    if any(hint in question for hint in _BROAD_HINTS):
        return "broad"
    if any(hint in question for hint in _FOCUSED_HINTS):
        return "focused"
    if query_type == QueryType.GENERAL:
        return "broad"
    if query_type in (QueryType.BACKWARD_COMPAT, QueryType.RETRIEVAL):
        return "focused"
    return "focused" if len(question.split()) <= 12 else "broad"


def _query_subfocus(query_type: str, user_question: str) -> str:
    question = _normalize_question(user_question)
    if query_type == QueryType.STRUCTURAL_CHANGE:
        if any(term in question for term in ("structure", "structural", "formulation", "model form")):
            return "formulation"
        if any(term in question for term in ("input", "data", "parameter", "scenario", "assumption")):
            return "input_data"
        return "setup"
    if query_type == QueryType.SOLUTION_DIFF:
        if any(term in question for term in ("why", "reason", "driver", "because", "cause", "attribution")):
            return "drivers"
        return "plan_changes"
    if query_type == QueryType.BACKWARD_COMPAT:
        if any(term in question for term in ("how much", "by how much", "what should change", "what needs to change")):
            return "change_amounts"
        return "reuse_check"
    return ""


def _profile_limit(profile: str, *, detailed: int, summary: int) -> int:
    return detailed if profile == "detailed" else summary


def _sample_items(items: list[str], profile: str, *, detailed: int = _DETAILED_EXAMPLE_LIMIT, summary: int = _SUMMARY_EXAMPLE_LIMIT) -> list[str]:
    ordered = _ordered_unique(items)
    limit = _profile_limit(profile, detailed=detailed, summary=summary)
    return ordered[:limit]


def _compact_numeric_series(items: list[str], profile: str) -> str:
    values = _ordered_unique(items)
    parsed: list[int] = []
    for item in values:
        try:
            number = float(str(_parse_index(item)))
        except (TypeError, ValueError):
            return _natural_join(_sample_items(values, profile))
        if abs(number - round(number)) > 1e-9:
            return _natural_join(_sample_items(values, profile))
        parsed.append(int(round(number)))

    if not parsed:
        return ""

    ranges: list[str] = []
    start = end = parsed[0]
    for number in parsed[1:]:
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


def _sample_phrase(items: list[str], profile: str) -> str:
    values = _ordered_unique(items)
    if not values:
        return ""
    if all(re.fullmatch(r"-?\d+(\.0+)?", str(item)) for item in values):
        return _compact_numeric_series(values, profile)
    shown = _sample_items(values, profile)
    if len(values) > len(shown):
        return _natural_join(shown) + f", and {len(values) - len(shown)} other items"
    return _natural_join(shown)


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
    text = _clean_label(label, "items")
    lower = text.lower()
    if lower.startswith("set of "):
        text = text[7:]
    return text[0].lower() + text[1:] if text else "items"


def _summary_label(label: str) -> str:
    text = _clean_label(label, "")
    if text.lower().startswith("set of "):
        text = text[7:]
    if " (" in text:
        text = text.split(" (", 1)[0]
    for marker in (" representing ", " indicating ", " used to "):
        if marker in text.lower():
            text = text.split(marker, 1)[0]
            break
    return text.rstrip(".")


def _index_context(label: str, indices: list[str]) -> str:
    joined = _natural_join(_ordered_unique(indices))
    return f"for {joined}"


def _describe_parameter_family_change(label: str, increased: list[str], decreased: list[str]) -> list[str]:
    label = _summary_label(label)
    lines: list[str] = []

    if increased:
        lines.append(f"  Higher {label.lower()} {_index_context(label, increased)}.")
    if decreased:
        lines.append(f"  Lower {label.lower()} {_index_context(label, decreased)}.")
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
        lines.append(f"  New {_summary_label(label).lower()} entries {_index_context(label, added[:6])}.")
    if removed:
        lines.append(f"  Removed {_summary_label(label).lower()} entries {_index_context(label, removed[:6])}.")
    increased_keys = [str(entry.get("index", "?")) for entry in changed if (entry.get("value_b") or 0) > (entry.get("value_a") or 0)]
    decreased_keys = [str(entry.get("index", "?")) for entry in changed if (entry.get("value_b") or 0) < (entry.get("value_a") or 0)]
    if increased_keys:
        lines.append(
            f"  Higher {_summary_label(label).lower()} for combinations such as "
            f"{_natural_join(_ordered_unique(increased_keys)[:4])}."
        )
    if decreased_keys:
        lines.append(
            f"  Lower {_summary_label(label).lower()} for combinations such as "
            f"{_natural_join(_ordered_unique(decreased_keys)[:4])}."
        )
    return lines


def _summarize_parameter_family(family: str, family_payload: dict, fallback_labels: dict[str, str]) -> list[str]:
    label = _summary_label(_clean_label(family_payload.get("family_label"), fallback_labels.get(family, family)))
    if _is_internal_label(family, label):
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
        lines.append(f"  New {label.lower()} entries {_index_context(label, added[:6])}.")
    if removed:
        lines.append(f"  Removed {label.lower()} entries {_index_context(label, removed[:6])}.")
    return lines


def _describe_set_change(name: str, payload: dict, label: str) -> str | None:
    if _is_internal_label(name, label):
        return None

    added = [str(x) for x in payload.get("added_elements", [])[:4]]
    removed = [str(x) for x in payload.get("removed_elements", [])[:4]]
    subject = _summary_label(_set_subject(label))
    added = _ordered_unique(added)
    removed = _ordered_unique(removed)

    if added and removed:
        return (
            f"  Updated {subject}: "
            f"added {_natural_join(added)} and removed {_natural_join(removed)}."
        )
    if added:
        return f"  Added {subject}: {_natural_join(added)}."
    if removed:
        return f"  Removed {subject}: {_natural_join(removed)}."
    return None


def _summarize_variable_family_changes(variable_changes: dict, variable_labels: dict[str, str]) -> list[str]:
    lines: list[str] = []

    for family in variable_changes.get("added_families", []):
        label = _summary_label(_clean_label(variable_labels.get(family), family))
        if not _is_internal_label(family, label):
            lines.append(f"  New decision area: {label}.")

    for family in variable_changes.get("removed_families", []):
        label = _summary_label(_clean_label(variable_labels.get(family), family))
        if not _is_internal_label(family, label):
            lines.append(f"  Removed decision area: {label}.")

    return lines


def _group_parameter_shift_lines(by_family: dict, fallback_labels: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for family, payload in by_family.items():
        lines.extend(_summarize_parameter_family(family, payload, fallback_labels))
    return lines


def _index_phrase(index: Any) -> str:
    text = str(index)
    return text if text else "the relevant item"


def _format_value_shift(label: str, index: Any, value_a: Any, value_b: Any) -> str:
    return f"{label} for {_index_phrase(index)}: {_safe(value_a)} -> {_safe(value_b)}"


def _parse_name_index(name: str) -> Any:
    br = name.find("[")
    if br == -1 or not name.endswith("]"):
        return None
    raw = name[br + 1 : -1]
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _dimension_label(set_name: str, set_labels: dict[str, str]) -> str:
    label = _clean_label(set_labels.get(set_name), set_name).lower()
    if label.startswith("set of "):
        label = label[7:]
    if "time period" in label or "planning period" in label or "period" in label:
        return "time period"
    if "vehicle" in label:
        return "vehicle"
    if "food" in label:
        return "food"
    if "nutrient" in label:
        return "nutrient"
    words = label.split()
    return " ".join(words[:2]) if words else set_name


def _decision_family_label(label: str) -> str:
    cleaned = _clean_label(label, "decision")
    lower = cleaned.lower()
    prefixes = (
        "binary variable indicating whether ",
        "binary decision selecting ",
        "binary variable ",
        "binary decision ",
        "variable indicating whether ",
        "decision selecting ",
        "decision ",
        "variable ",
    )
    for prefix in prefixes:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    cleaned = " ".join(re.sub(r"\b[a-z]\b", "", cleaned).split())
    cleaned = cleaned.replace(" is assigned to ", " assignment to ")
    return cleaned or "decision"


def _format_domain_location(var_name: str, indexed_over: list[str], set_labels: dict[str, str]) -> str:
    parsed = _parse_name_index(var_name)
    if isinstance(parsed, tuple):
        pieces: list[str] = []
        for pos, value in enumerate(parsed):
            set_name = indexed_over[pos] if pos < len(indexed_over) else f"dim_{pos + 1}"
            dim = _dimension_label(set_name, set_labels)
            if pos == 0:
                pieces.append(f"{value} {dim}")
            else:
                pieces.append(f"at {dim} {value}")
        if not pieces:
            return str(parsed)
        if len(pieces) == 1:
            return pieces[0]
        return pieces[0] + " " + " ".join(pieces[1:])
    if parsed is not None:
        return str(parsed)
    return _var_label(var_name)


def _objective_phrase(label: str | None) -> str:
    cleaned = _clean_label(label, "overall business outcome")
    lower = cleaned.lower()
    for prefix in ("maximize ", "minimize "):
        if lower.startswith(prefix):
            return cleaned[len(prefix):]
    return cleaned


def _family_activity_term(label: str, vtype: str) -> str:
    lower = _decision_family_label(label).lower()
    if "whether" in lower:
        return "recommended choices"
    if vtype == "binary":
        return "binary choices"
    return "decision levels"


def _family_change_strength(changed: int, added: int, removed: int) -> str:
    total = changed + added + removed
    if total >= 15:
        return "changed materially"
    if total >= 5:
        return "changed noticeably"
    return "changed"


def _describe_direction(label: str, direction: str, items: list[str], profile: str) -> str | None:
    if not items:
        return None
    label_text = _summary_label(label).lower()
    item_text = _sample_phrase(items, profile)
    limit = _profile_limit(profile, detailed=6, summary=3)
    if len(_ordered_unique(items)) <= limit:
        verb = "increased" if direction == "up" else "decreased"
        return f"{label_text.capitalize()} {verb} for {item_text}."
    verb = "increased" if direction == "up" else "decreased"
    return f"{label_text.capitalize()} {verb} across many items, including {item_text}."


def _describe_added_removed_entries(label: str, direction: str, items: list[str], profile: str) -> str | None:
    if not items:
        return None
    label_text = _summary_label(label).lower()
    item_text = _sample_phrase(items, profile)
    limit = _profile_limit(profile, detailed=6, summary=3)
    if direction == "added":
        if len(_ordered_unique(items)) <= limit:
            return f"{label_text.capitalize()} now covers {item_text} as well."
        return f"{label_text.capitalize()} now covers multiple new items, including {item_text}."
    if len(_ordered_unique(items)) <= limit:
        return f"{label_text.capitalize()} no longer covers {item_text}."
    return f"{label_text.capitalize()} no longer covers multiple items, including {item_text}."


def _constraint_label(name: str, labels: dict[str, str]) -> str:
    return _clean_label(labels.get(name), name)


def _result_delta_phrase(value_a: Any, value_b: Any) -> str:
    a_val = value_a
    b_val = value_b
    try:
        a_num = float(a_val)
        b_num = float(b_val)
    except (TypeError, ValueError):
        return f"changed from {_safe(a_val)} to {_safe(b_val)}"
    if abs(b_num - a_num) < 1e-12:
        return f"stayed at {_safe(a_num)}"
    direction = "rose" if b_num > a_num else "fell"
    return f"{direction} from {_safe(a_num)} to {_safe(b_num)}"


def _required_change_phrase(description: str, amount: Any) -> str:
    desc = _clean_label(description, "This rule")
    lower = desc.lower()
    qty = _safe(amount)
    if any(token in lower for token in ("budget", "capacity", "cap", "limit", "maximum")):
        return f"{desc} would need to increase by about {qty}."
    if any(token in lower for token in ("minimum", "at least", "lower bound")):
        return f"{desc} would need to decrease by about {qty}."
    return f"{desc} would need to change by about {qty}."


def _revised_current_plan_phrase(amount: Any) -> str:
    qty = _safe(amount)
    return f"the current run would need a revised plan with about {qty} less usage in that area"


def _blocking_overage_phrase(description: str, amount: Any) -> str:
    desc = _clean_label(description, "the current rule")
    qty = _safe(amount)
    lower = desc.lower()
    if any(token in lower for token in ("budget", "capacity", "cap", "limit", "maximum")):
        return f"exceeds {desc.lower()} by about {qty}"
    return f"would violate {desc.lower()} by about {qty}"


def _render_set_change_sentence(set_name: str, payload: dict, label: str, profile: str) -> str | None:
    if _is_internal_label(set_name, label):
        return None
    subject = _summary_label(_set_subject(label))
    added = _ordered_unique([str(x) for x in payload.get("added_elements", [])])
    removed = _ordered_unique([str(x) for x in payload.get("removed_elements", [])])
    added_text = _sample_phrase(added, profile)
    removed_text = _sample_phrase(removed, profile)
    if added and removed:
        return f"  {subject.capitalize()} changed: added {added_text} and removed {removed_text}."
    if added:
        return f"  {subject.capitalize()} now include {added_text}."
    if removed:
        return f"  {subject.capitalize()} no longer include {removed_text}."
    return None


def _render_parameter_family_summary(
    family: str,
    family_payload: dict,
    fallback_labels: dict[str, str],
    profile: str,
) -> list[str]:
    label = _summary_label(_clean_label(family_payload.get("family_label"), fallback_labels.get(family, family)))
    if _is_internal_label(family, label):
        return []

    added = [str(idx) for idx in family_payload.get("added", {}).keys()]
    removed = [str(idx) for idx in family_payload.get("removed", {}).keys()]
    changed = family_payload.get("changed", [])
    increased = _ordered_unique([str(entry.get("index", "?")) for entry in changed if (entry.get("value_b") or 0) > (entry.get("value_a") or 0)])
    decreased = _ordered_unique([str(entry.get("index", "?")) for entry in changed if (entry.get("value_b") or 0) < (entry.get("value_a") or 0)])

    sample_index = None
    if changed:
        sample_index = changed[0].get("index")
    elif added:
        sample_index = added[0]
    elif removed:
        sample_index = removed[0]

    parts: list[str] = []
    if isinstance(_parse_index(sample_index), tuple):
        if increased:
            parts.append(_describe_direction(label, "up", increased, profile))
        if decreased:
            parts.append(_describe_direction(label, "down", decreased, profile))
        if added:
            parts.append(_describe_added_removed_entries(label, "added", added, profile))
        if removed:
            parts.append(_describe_added_removed_entries(label, "removed", removed, profile))
    else:
        if increased:
            parts.append(_describe_direction(label, "up", increased, profile))
        if decreased:
            parts.append(_describe_direction(label, "down", decreased, profile))
        if added:
            parts.append(_describe_added_removed_entries(label, "added", added, profile))
        if removed:
            parts.append(_describe_added_removed_entries(label, "removed", removed, profile))

    parts = [part for part in parts if part]
    if profile == "summary" and parts:
        return [f"  {_summary_label(label)}: " + " ".join(part[0].lower() + part[1:] for part in parts[:2])]
    return [f"  {part}" for part in parts]


def _solution_family_summary(
    family: str,
    family_summary: dict,
    allocation_summary: dict | None,
    variable_labels: dict[str, str],
) -> str | None:
    label = _decision_family_label(family_summary.get("family_label") or variable_labels.get(family, family))
    if _is_internal_label(family, label):
        return None
    changed = family_summary.get("n_changed", 0)
    added = family_summary.get("n_added", 0)
    removed = family_summary.get("n_removed", 0)
    strength = _family_change_strength(changed, added, removed)
    parts = [f"recommendations in this area {strength}"]
    if added:
        parts.append("the later run also introduces choices that did not exist before")
    if removed:
        parts.append("some earlier-only choices are no longer available")
    if allocation_summary is not None:
        total_a = allocation_summary.get("family_total_a", 0.0)
        total_b = allocation_summary.get("family_total_b", 0.0)
        if abs(total_b - total_a) >= 1e-9:
            parts.append(f"overall activity moved from {_safe(total_a)} to {_safe(total_b)}")
    return f"  Decisions about {label.lower()}: " + "; ".join(parts) + "."


def _summarize_solution_family_examples(
    top_changes: list[dict],
    variable_metadata: dict[str, dict],
    set_labels: dict[str, str],
    profile: str,
) -> list[str]:
    grouped: dict[str, dict[str, list[str]]] = {}
    order: list[str] = []

    for change in top_changes:
        family = change.get("family", "")
        meta = variable_metadata.get(family, {})
        label = _clean_label(change.get("family_label"), meta.get("doc", "") or family)
        if _is_internal_label(family, label):
            continue
        if family not in grouped:
            grouped[family] = {"up": [], "down": []}
            order.append(family)
        location = _format_domain_location(
            change.get("variable", "?"),
            meta.get("indexed_over", []),
            set_labels,
        )
        direction = "up" if (change.get("delta") or 0) > 0 else "down"
        grouped[family][direction].append(location)

    family_limit = _profile_limit(profile, detailed=2, summary=1)
    example_limit = _profile_limit(profile, detailed=4, summary=2)
    lines: list[str] = []
    for family in order[:family_limit]:
        meta = variable_metadata.get(family, {})
        label = _decision_family_label(meta.get("doc", "") or family)
        increased = _sample_items(grouped[family]["up"], profile, detailed=example_limit, summary=example_limit)
        decreased = _sample_items(grouped[family]["down"], profile, detailed=example_limit, summary=example_limit)
        if increased and decreased:
            lines.append(
                f"  Representative examples for decisions about {label.lower()}: more activity in {_natural_join(increased)} and less in {_natural_join(decreased)}."
            )
        elif increased:
            lines.append(
                f"  Representative examples for decisions about {label.lower()}: more activity in {_natural_join(increased)}."
            )
        elif decreased:
            lines.append(
                f"  Representative examples for decisions about {label.lower()}: less activity in {_natural_join(decreased)}."
            )
    return lines


def _dimension_shift_sentence(dimension_summary: dict, set_labels: dict[str, str], profile: str) -> str | None:
    set_name = dimension_summary.get("set_name", "")
    dim_label = _dimension_label(set_name, set_labels)
    increased = [str(entry.get("member", "?")) for entry in dimension_summary.get("top_increases", [])]
    decreased = [str(entry.get("member", "?")) for entry in dimension_summary.get("top_decreases", [])]
    up_text = _sample_phrase(increased, profile)
    down_text = _sample_phrase(decreased, profile)

    if dim_label == "time period":
        up_phrase = f"more activity in time periods {up_text}" if up_text else ""
        down_phrase = f"less activity in time periods {down_text}" if down_text else ""
    elif dim_label in {"vehicle", "food", "nutrient"}:
        up_phrase = f"more use of {up_text}" if up_text else ""
        down_phrase = f"less use of {down_text}" if down_text else ""
    else:
        up_phrase = f"more activity in {dim_label} values {up_text}" if up_text else ""
        down_phrase = f"less activity in {dim_label} values {down_text}" if down_text else ""

    if increased and decreased:
        return f"  By {dim_label}, the later plan shows {up_phrase} and {down_phrase}."
    if increased:
        return f"  By {dim_label}, the later plan shows {up_phrase}."
    if decreased:
        return f"  By {dim_label}, the later plan shows {down_phrase}."
    return None


def _format_qt1(
    qt1: dict | None,
    state: dict | None = None,
    profile: str = "detailed",
    subfocus: str = "setup",
) -> str:
    if not qt1:
        return "  (Not yet computed)"

    label_a = _epoch_label(state, "a")
    label_b = _epoch_label(state, "b")
    by_family = qt1.get("parameter_changes", {}).get("by_family", {})
    modified_sets = qt1.get("index_set_changes", {}).get("modified_sets", {})
    added_sets = qt1.get("index_set_changes", {}).get("added_sets", [])
    removed_sets = qt1.get("index_set_changes", {}).get("removed_sets", [])
    labels = qt1.get("labels", {})
    set_labels = labels.get("sets", {})
    variable_labels = labels.get("variables", {})
    parameter_labels = labels.get("parameters", {})
    constraint_labels = labels.get("constraints", {})
    variable_changes = qt1.get("variable_family_changes", {})
    constraint_changes = qt1.get("constraint_structure_changes", {})

    if subfocus == "formulation":
        structural_lines = [f"  Formulation check from {label_a} to {label_b}:"]
        added_sets = [
            _clean_label(set_labels.get(name), name)
            for name in added_sets
            if not _is_internal_label(name, _clean_label(set_labels.get(name), name))
        ]
        removed_sets = [
            _clean_label(set_labels.get(name), name)
            for name in removed_sets
            if not _is_internal_label(name, _clean_label(set_labels.get(name), name))
        ]
        added_vars = [
            _summary_label(_clean_label(variable_labels.get(name), name))
            for name in variable_changes.get("added_families", [])
            if not _is_internal_label(name, _clean_label(variable_labels.get(name), name))
        ]
        removed_vars = [
            _summary_label(_clean_label(variable_labels.get(name), name))
            for name in variable_changes.get("removed_families", [])
            if not _is_internal_label(name, _clean_label(variable_labels.get(name), name))
        ]
        added_rules = [
            _constraint_label(name, constraint_labels)
            for name in constraint_changes.get("added_families", [])
            if not _is_internal_label(name, _constraint_label(name, constraint_labels))
        ]
        removed_rules = [
            _constraint_label(name, constraint_labels)
            for name in constraint_changes.get("removed_families", [])
            if not _is_internal_label(name, _constraint_label(name, constraint_labels))
        ]

        if not any((added_sets, removed_sets, added_vars, removed_vars, added_rules, removed_rules)):
            structural_lines.append(
                "  No meaningful formulation change is detected. The same types of sets, decisions, and business rules are still in place."
            )
            structural_lines.append(
                "  The differences are in planning scope and input data rather than in the model formulation itself."
            )
            return "\n".join(structural_lines)

        if added_sets:
            structural_lines.append("  New set families: " + _natural_join(added_sets) + ".")
        if removed_sets:
            structural_lines.append("  Removed set families: " + _natural_join(removed_sets) + ".")
        if added_vars:
            structural_lines.append("  New decision families: " + _natural_join(added_vars) + ".")
        if removed_vars:
            structural_lines.append("  Removed decision families: " + _natural_join(removed_vars) + ".")
        if added_rules:
            structural_lines.append("  New business rule families: " + _natural_join(added_rules) + ".")
        if removed_rules:
            structural_lines.append("  Removed business rule families: " + _natural_join(removed_rules) + ".")
        return "\n".join(structural_lines)

    lines = [f"  Setup changes from {label_a} to {label_b}:"]
    for set_name, payload in modified_sets.items():
        set_label = _clean_label(payload.get("label") or set_labels.get(set_name), set_name)
        line = _render_set_change_sentence(set_name, payload, set_label, profile)
        if line:
            lines.append(line)
    if added_sets:
        visible_added_sets = [
            _clean_label(set_labels.get(name), name)
            for name in added_sets
            if not _is_internal_label(name, _clean_label(set_labels.get(name), name))
        ]
        if visible_added_sets:
            lines.append("  New business categories were introduced: " + _natural_join(visible_added_sets) + ".")
    if removed_sets:
        visible_removed_sets = [
            _clean_label(set_labels.get(name), name)
            for name in removed_sets
            if not _is_internal_label(name, _clean_label(set_labels.get(name), name))
        ]
        if visible_removed_sets:
            lines.append("  Some business categories were removed: " + _natural_join(visible_removed_sets) + ".")

    variable_lines = []
    for family in variable_changes.get("added_families", []):
        label = _summary_label(_clean_label(variable_labels.get(family), family))
        if not _is_internal_label(family, label):
            variable_lines.append(f"  New decision area: {label}.")
    for family in variable_changes.get("removed_families", []):
        label = _summary_label(_clean_label(variable_labels.get(family), family))
        if not _is_internal_label(family, label):
            variable_lines.append(f"  Removed decision area: {label}.")
    lines.extend(variable_lines)

    grouped_param_lines: list[str] = []
    for family, payload in by_family.items():
        grouped_param_lines.extend(_render_parameter_family_summary(family, payload, parameter_labels, profile))
    if grouped_param_lines:
        lines.append("  Main input shifts:")
        lines.extend(grouped_param_lines)
    added_rules = constraint_changes.get("added_families", [])
    removed_rules = constraint_changes.get("removed_families", [])
    if added_rules:
        lines.append(
            "  New operating rules: "
            + _natural_join([_constraint_label(name, constraint_labels) for name in added_rules])
            + "."
        )
    if removed_rules:
        lines.append(
            "  Removed operating rules: "
            + _natural_join([_constraint_label(name, constraint_labels) for name in removed_rules])
            + "."
        )
    return "\n".join(lines)


def _format_qt2(qt2: dict | None, state: dict | None = None, profile: str = "detailed") -> str:
    if not qt2:
        return "  (Not yet computed)"

    label_a = _epoch_label(state, "a")
    label_b = _epoch_label(state, "b")
    objective = qt2.get("objective_change", {})
    summary = qt2.get("summary", {})
    top_changes = qt2.get("top_changes", [])[:5]
    labels = qt2.get("labels", {})
    variable_labels = labels.get("variables", {})
    variable_metadata = labels.get("variable_metadata", {})
    set_labels = labels.get("sets", {})
    allocation_summaries = qt2.get("allocation_summaries", {})
    objective_label = _objective_phrase(objective.get("label") or labels.get("objective"))
    lines = [
        f"  Business result ({objective_label}) {_result_delta_phrase(objective.get('value_a'), objective.get('value_b'))} between {label_a} and {label_b}.",
    ]
    relocations = summary.get("n_relocations_estimate", 0)
    if relocations:
        lines.append(
            f"  Across the overlapping items and periods, about {relocations} placements were reallocated."
        )
    for family, family_summary in qt2.get("churn_summary", {}).items():
        allocation_summary = allocation_summaries.get(family)
        line = _solution_family_summary(family, family_summary, allocation_summary, variable_labels)
        if line:
            lines.append(line)
        if allocation_summary:
            dim_limit = _profile_limit(profile, detailed=2, summary=1)
            for dimension_summary in allocation_summary.get("dimensions", [])[:dim_limit]:
                dim_line = _dimension_shift_sentence(dimension_summary, set_labels, profile)
                if dim_line:
                    lines.append(dim_line)
    if top_changes:
        lines.append("  Representative plan shifts:")
        lines.extend(_summarize_solution_family_examples(top_changes, variable_metadata, set_labels, profile))
    return "\n".join(lines)


def _format_qt3(qt3: dict | None, state: dict | None = None, profile: str = "detailed") -> str:
    if not qt3:
        return "  (Not yet computed)"

    if qt3.get("feasible"):
        lines = [f"  {qt3.get('business_summary', 'N/A')}"]
        if qt3.get("forced_objective") is not None:
            lines.append(
                f"  If you keep the earlier plan, the business result would be {_safe(qt3.get('forced_objective'))}, "
                f"compared with {_safe(qt3.get('optimal_objective_b'))} from the best later-run plan."
            )
        if qt3.get("gap_rel") is not None:
            lines.append(f"  The remaining opportunity gap would be about {qt3.get('gap_rel') * 100:.2f}%.")
    else:
        minimal_cover = qt3.get("minimal_cover", [])[:5]
        minimal_cover_details = qt3.get("minimal_cover_details", [])[:5]
        required_slacks = qt3.get("required_slacks", {})
        first_blocker = None
        if minimal_cover:
            first_name = minimal_cover[0]
            if first_name in required_slacks:
                first_blocker = required_slacks[first_name]
                first_blocker.setdefault("description", _con_label(first_name))
        if first_blocker is not None:
            lines = [
                "  No. The earlier plan "
                + _blocking_overage_phrase(
                    first_blocker.get("description"),
                    first_blocker.get("required_slack"),
                )
                + ".",
            ]
        else:
            lines = [f"  {qt3.get('business_summary', 'N/A')}"]
        if minimal_cover_details:
            lines.append("  Main blocking rules:")
            for entry in minimal_cover_details:
                lines.append(f"    - {_clean_label(entry.get('description'), _con_label(entry.get('constraint', '?')))}")
        elif minimal_cover:
            lines.append("  Main blocking rules:")
            for name in minimal_cover:
                lines.append(f"    - {_con_label(name)}")
        if required_slacks:
            lines.append("  What this means for the current run:")
            if minimal_cover:
                first_name = minimal_cover[0]
                if first_name in required_slacks:
                    first_desc = required_slacks[first_name].get("description")
                    first_qty = required_slacks[first_name].get("required_slack")
                    lines.append("    - To allow the same pattern, " + _required_change_phrase(first_desc, first_qty))
                    lines.append("    - If today's rules stay in place, " + _revised_current_plan_phrase(first_qty) + ".")
                for name in minimal_cover[1:]:
                    if name in required_slacks:
                        lines.append("    - " + _required_change_phrase(
                            required_slacks[name].get("description"),
                            required_slacks[name].get("required_slack"),
                        ))
    return "\n".join(lines)


def _format_qt4(qt4: dict | None, state: dict | None = None, profile: str = "detailed") -> str:
    if not qt4:
        return "  (Not yet computed)"

    lines: list[str] = []
    shifts = qt4.get("parameter_shifts", [])[:5]
    flips = qt4.get("binary_flips", [])[:5]
    adjustments = qt4.get("continuous_adjustments", [])[:5]
    bottlenecks = qt4.get("bottleneck_evolution", {}).get("new_bottlenecks", [])[:5]
    labels = qt4.get("labels", {})
    parameter_labels = labels.get("parameters", {})
    variable_labels = labels.get("variables", {})
    constraint_labels = labels.get("constraints", {})

    shift_families: dict[str, dict[str, list[str]]] = {}
    for shift in shifts:
        family = shift.get("family", "?")
        shift_families.setdefault(family, {"up": [], "down": []})
        direction = "up" if (shift.get("value_b") or 0) > (shift.get("value_a") or 0) else "down"
        shift_families[family][direction].append(str(shift.get("index", "?")))
    if shifts:
        lines.append("  Input changes that line up with the plan shift:")
        for family, grouped in shift_families.items():
            payload = {
                "family_label": parameter_labels.get(family, family),
                "changed": (
                    [{"index": idx, "value_a": 0, "value_b": 1} for idx in grouped["up"]]
                    + [{"index": idx, "value_a": 1, "value_b": 0} for idx in grouped["down"]]
                ),
                "added": {},
                "removed": {},
            }
            lines.extend(_render_parameter_family_summary(family, payload, parameter_labels, profile))
    if flips or adjustments:
        lines.append("  Decision movements that line up with those changes:")
        seen_families: set[str] = set()
        for change in flips:
            family = change.get("family", "") or change.get("variable", "?").split("[")[0]
            if family in seen_families:
                continue
            seen_families.add(family)
            family_label = _decision_family_label(variable_labels.get(family, family))
            direction = "appeared" if change.get("direction") == "0->1" else "dropped out"
            lines.append(f"  {family_label}: some choices in this area {direction} in the later run.")
        for change in adjustments[:_profile_limit(profile, detailed=2, summary=1)]:
            family = change.get("family", "") or change.get("variable", "?").split("[")[0]
            if family in seen_families:
                continue
            seen_families.add(family)
            family_label = _decision_family_label(variable_labels.get(family, family))
            lines.append(f"  {family_label}: recommendation levels changed materially in this area.")
    if bottlenecks:
        lines.append("  Business rules that matter more in the later run:")
        for entry in bottlenecks:
            constraint = entry.get("constraint", "?")
            family = constraint.split("[")[0]
            lines.append(f"    - {_clean_label(constraint_labels.get(family), _con_label(constraint))}")
    return "\n".join(lines) if lines else "  (No major attribution signals computed)"


def _format_epoch_meta_block(state: dict) -> str:
    meta_a = state.get(RH_EPOCH_A_META, {})
    meta_b = state.get(RH_EPOCH_B_META, {})
    if not meta_a and not meta_b:
        return "  (Session not yet initialized)"

    lines = []
    for title, meta in (("Earlier plan", meta_a), ("Later plan", meta_b)):
        if not meta:
            continue
        label = meta.get("label", meta.get("epoch_id", title))
        status = meta.get("sol_status", "unknown")
        objective = meta.get("objective_value")
        lines.append(f"  {title}: {label} | status={status} | outcome={_safe(objective)}")
    return "\n".join(lines)


def _format_epoch_desc_block(state: dict, profile: str = "detailed") -> str:
    limit = 1600 if profile == "detailed" else 950
    desc_a = _truncate_text(state.get(RH_DESCRIPTION_A, ""), limit=limit)
    desc_b = _truncate_text(state.get(RH_DESCRIPTION_B, ""), limit=limit)
    if desc_a:
        return desc_a
    if desc_b:
        return desc_b
    return "(Model explanation not available yet)"


def _format_root_epoch_info_block(state: dict) -> str:
    meta_a = state.get(RH_EPOCH_A_META, {})
    meta_b = state.get(RH_EPOCH_B_META, {})
    if not meta_a and not meta_b:
        return "No planning runs are loaded yet."
    lines = ["Loaded planning runs:"]
    if meta_a:
        lines.append(f"  Earlier plan: {meta_a.get('label', meta_a.get('epoch_id', 'A'))}")
    if meta_b:
        lines.append(f"  Later plan: {meta_b.get('label', meta_b.get('epoch_id', 'B'))}")
    return "\n".join(lines)


def build_root_agent_prompt(state: dict) -> str:
    initialized = state.get(RH_SESSION_INITIALIZED, False)
    status = "ready — planning runs loaded" if initialized else "awaiting configuration"
    return ROOT_AGENT_PROMPT_TEMPLATE.format(
        RH_SESSION_STATUS=status,
        EPOCH_INFO_BLOCK=_format_root_epoch_info_block(state),
    )


def _qt1_visible_family_count(qt1: dict | None) -> int:
    if not qt1:
        return 0
    labels = qt1.get("labels", {})
    set_labels = labels.get("sets", {})
    variable_labels = labels.get("variables", {})
    parameter_labels = labels.get("parameters", {})
    constraint_labels = labels.get("constraints", {})
    count = 0
    for set_name, payload in qt1.get("index_set_changes", {}).get("modified_sets", {}).items():
        label = _clean_label(payload.get("label") or set_labels.get(set_name), set_name)
        if not _is_internal_label(set_name, label):
            count += 1
    count += sum(
        1 for name in qt1.get("index_set_changes", {}).get("added_sets", [])
        if not _is_internal_label(name, _clean_label(set_labels.get(name), name))
    )
    count += sum(
        1 for name in qt1.get("index_set_changes", {}).get("removed_sets", [])
        if not _is_internal_label(name, _clean_label(set_labels.get(name), name))
    )
    count += sum(
        1 for family in qt1.get("variable_family_changes", {}).get("added_families", [])
        if not _is_internal_label(family, _clean_label(variable_labels.get(family), family))
    )
    count += sum(
        1 for family in qt1.get("variable_family_changes", {}).get("removed_families", [])
        if not _is_internal_label(family, _clean_label(variable_labels.get(family), family))
    )
    count += sum(
        1 for family in qt1.get("parameter_changes", {}).get("by_family", {}).keys()
        if not _is_internal_label(family, _clean_label(parameter_labels.get(family), family))
    )
    count += sum(
        1 for family in qt1.get("constraint_structure_changes", {}).get("added_families", [])
        if not _is_internal_label(family, _constraint_label(family, constraint_labels))
    )
    count += sum(
        1 for family in qt1.get("constraint_structure_changes", {}).get("removed_families", [])
        if not _is_internal_label(family, _constraint_label(family, constraint_labels))
    )
    return count


def _qt2_visible_family_count(qt2: dict | None) -> int:
    if not qt2:
        return 0
    labels = qt2.get("labels", {})
    variable_labels = labels.get("variables", {})
    count = 0
    for family, family_summary in qt2.get("churn_summary", {}).items():
        label = _decision_family_label(family_summary.get("family_label") or variable_labels.get(family, family))
        if not _is_internal_label(family, label):
            count += 1
    return count


def _qt3_visible_family_count(qt3: dict | None) -> int:
    if not qt3:
        return 0
    if qt3.get("feasible"):
        return 1
    return max(1, len(qt3.get("minimal_cover_details", []) or qt3.get("minimal_cover", [])))


def _qt4_visible_family_count(qt4: dict | None) -> int:
    if not qt4:
        return 0
    families = set()
    labels = qt4.get("labels", {})
    parameter_labels = labels.get("parameters", {})
    variable_labels = labels.get("variables", {})
    constraint_labels = labels.get("constraints", {})
    for shift in qt4.get("parameter_shifts", []):
        family = shift.get("family", "?")
        if not _is_internal_label(family, _clean_label(parameter_labels.get(family), family)):
            families.add(f"p:{family}")
    for flip in qt4.get("binary_flips", []):
        family = flip.get("family") or str(flip.get("variable", "?")).split("[")[0]
        if not _is_internal_label(family, _clean_label(variable_labels.get(family), family)):
            families.add(f"v:{family}")
    for entry in qt4.get("bottleneck_evolution", {}).get("new_bottlenecks", []):
        family = str(entry.get("constraint", "?")).split("[")[0]
        if not _is_internal_label(family, _constraint_label(family, constraint_labels)):
            families.add(f"c:{family}")
    return len(families)


def _informational_line_count(block: str) -> int:
    return sum(1 for line in block.splitlines() if line.strip() and not line.strip().endswith(":"))


def _query_profile_metrics(query_type: str, state: dict, subfocus: str = "") -> tuple[int, str]:
    qt1 = state.get(RH_QT1_RESULT)
    qt2 = state.get(RH_QT2_RESULT)
    qt3 = state.get(RH_QT3_RESULT)
    qt4 = state.get(RH_QT4_RESULT)

    if query_type == QueryType.STRUCTURAL_CHANGE:
        block = _format_qt1(qt1, state, profile="detailed", subfocus=subfocus or "setup")
        return _qt1_visible_family_count(qt1), block
    if query_type == QueryType.SOLUTION_DIFF:
        block = _format_qt2(qt2, state, profile="detailed")
        return _qt2_visible_family_count(qt2), block
    if query_type == QueryType.BACKWARD_COMPAT:
        block = _format_qt3(qt3, state, profile="detailed")
        return _qt3_visible_family_count(qt3), block
    if query_type == QueryType.ATTRIBUTION:
        block = _format_qt4(qt4, state, profile="detailed")
        return _qt4_visible_family_count(qt4), block
    if query_type == QueryType.MODEL_DESCRIPTION:
        desc = _format_epoch_desc_block(state, profile="detailed")
        return 4, desc
    if query_type == QueryType.RETRIEVAL:
        return 1, ""
    combined = _format_qt1(qt1, state, profile="detailed", subfocus="setup") + "\n" + _format_qt2(qt2, state, profile="detailed")
    return _qt1_visible_family_count(qt1) + _qt2_visible_family_count(qt2), combined


def _choose_answer_profile(query_type: str, state: dict, user_question: str) -> str:
    if query_type == QueryType.RETRIEVAL:
        return "detailed"
    subfocus = _query_subfocus(query_type, user_question)
    family_count, rendered = _query_profile_metrics(query_type, state, subfocus=subfocus)
    detail_rows = _informational_line_count(rendered)
    if (
        family_count <= _MAX_DETAILED_CHANGED_FAMILIES
        and detail_rows <= _MAX_DETAILED_ROWS
        and len(rendered) <= _MAX_DETAILED_CONTEXT_CHARS
    ):
        return "detailed"
    return "summary"


def _guidance_block(query_type: str, focus: str, profile: str, subfocus: str = "") -> str:
    strategy_map = {
        QueryType.GENERAL: STRATEGY_GENERAL,
        QueryType.RETRIEVAL: STRATEGY_RETRIEVAL,
        QueryType.MODEL_DESCRIPTION: STRATEGY_MODEL_DESCRIPTION,
        QueryType.STRUCTURAL_CHANGE: STRATEGY_STRUCTURAL_CHANGE,
        QueryType.SOLUTION_DIFF: STRATEGY_SOLUTION_DIFF,
        QueryType.BACKWARD_COMPAT: STRATEGY_BACKWARD_COMPAT,
        QueryType.ATTRIBUTION: STRATEGY_ATTRIBUTION,
    }
    strategy = strategy_map.get(query_type, STRATEGY_GENERAL)
    if focus == "broad":
        focus_rule = (
            "For this question, give a report-style answer: start with a short answer, then explain the important changes, "
            "their meaning, and the practical takeaway."
        )
    else:
        focus_rule = (
            "For this question, answer directly in the first sentence. Add only the context needed to make the answer clear."
        )
    profile_rule = (
        "Use all materially relevant changed families that fit cleanly in the answer."
        if profile == "detailed"
        else "Cover every important changed family once at a high level, use only a few representative examples, and stop before the answer becomes crowded."
    )
    subfocus_rule = ""
    if query_type == QueryType.STRUCTURAL_CHANGE and subfocus == "formulation":
        subfocus_rule = (
            "Treat this as a formulation question. Only call something structural if there is a new or removed set family, decision family, or business rule family. "
            "Do not describe changed data values, added members within an existing set, or tighter parameter values as structural changes."
        )
    elif query_type == QueryType.STRUCTURAL_CHANGE and subfocus == "input_data":
        subfocus_rule = (
            "Treat this as an input-data question. Explain changed scope items and changed inputs in plain English. Do not frame these as structural formulation changes."
        )
    elif query_type == QueryType.SOLUTION_DIFF and subfocus != "drivers":
        subfocus_rule = (
            "Focus on how the allocation or recommendation changed. Do not spend the answer explaining why the plan changed unless the user asked for the reasons."
        )
    elif query_type == QueryType.SOLUTION_DIFF and subfocus == "drivers":
        subfocus_rule = (
            "The user asked for drivers, so explain both what changed in the plan and the main reasons behind it."
        )
    elif query_type == QueryType.BACKWARD_COMPAT:
        subfocus_rule = (
            "When the older plan does not fit, say exactly what would need to be increased, decreased, or changed, and mention amounts when available."
        )
    example_family = _QUERY_EXAMPLES.get(query_type, _QUERY_EXAMPLES[QueryType.GENERAL])
    example = example_family["broad" if focus == "broad" else "focused"]
    parts = [
        strategy,
        focus_rule,
        profile_rule,
    ]
    if subfocus_rule:
        parts.append(subfocus_rule)
    parts.extend([
        "Style example (adapt the language, do not copy the placeholders literally):",
        example,
    ])
    return "\n".join(parts)


def build_comparison_agent_prompt(state: dict, query_type: str, user_question: str = "") -> str:
    focus = _question_focus(query_type, user_question)
    subfocus = _query_subfocus(query_type, user_question)
    profile = _choose_answer_profile(query_type, state, user_question)
    base = COMPARISON_BASE_PROMPT.format(
        USER_QUESTION=user_question.strip() or "(not available)",
        QUESTION_FOCUS=focus,
        ANSWER_PROFILE=profile,
        EPOCH_META_BLOCK=_format_epoch_meta_block(state),
        EPOCH_DESC_BLOCK=_format_epoch_desc_block(state, profile=profile),
    )

    qt1 = state.get(RH_QT1_RESULT)
    qt2 = state.get(RH_QT2_RESULT)
    qt3 = state.get(RH_QT3_RESULT)
    qt4 = state.get(RH_QT4_RESULT)

    sections = [base]

    if query_type == QueryType.GENERAL:
        sections.extend([
            "\nCOMPARISON SNAPSHOT\n" + _format_qt1(qt1, state, profile=profile, subfocus="setup"),
            "\nPLAN SNAPSHOT\n" + _format_qt2(qt2, state, profile=profile),
            "\nGUIDANCE\n" + _guidance_block(QueryType.GENERAL, focus, profile, subfocus),
        ])
    elif query_type == QueryType.RETRIEVAL:
        sections.extend([
            "\nGUIDANCE\n" + _guidance_block(QueryType.RETRIEVAL, focus, profile, subfocus),
        ])
    elif query_type == QueryType.MODEL_DESCRIPTION:
        sections.extend([
            "\nGUIDANCE\n" + _guidance_block(QueryType.MODEL_DESCRIPTION, focus, profile, subfocus),
        ])
    elif query_type == QueryType.STRUCTURAL_CHANGE:
        sections.extend([
            "\nSETUP CHANGES\n" + _format_qt1(qt1, state, profile=profile, subfocus=subfocus or "setup"),
            "\nGUIDANCE\n" + _guidance_block(QueryType.STRUCTURAL_CHANGE, focus, profile, subfocus),
        ])
    elif query_type == QueryType.SOLUTION_DIFF:
        sections.extend([
            "\nPLAN CHANGES\n" + _format_qt2(qt2, state, profile=profile),
            "\nGUIDANCE\n" + _guidance_block(QueryType.SOLUTION_DIFF, focus, profile, subfocus),
        ])
    elif query_type == QueryType.BACKWARD_COMPAT:
        sections.extend([
            "\nREUSE CHECK\n" + _format_qt3(qt3, state, profile=profile),
            "\nGUIDANCE\n" + _guidance_block(QueryType.BACKWARD_COMPAT, focus, profile, subfocus),
        ])
    elif query_type == QueryType.ATTRIBUTION:
        sections.extend([
            "\nDRIVER SNAPSHOT\n" + _format_qt4(qt4, state, profile=profile),
            "\nGUIDANCE\n" + _guidance_block(QueryType.ATTRIBUTION, focus, profile, subfocus),
        ])
    else:
        sections.extend([
            "\nCOMPARISON SNAPSHOT\n" + _format_qt1(qt1, state, profile=profile, subfocus="setup"),
            "\nPLAN SNAPSHOT\n" + _format_qt2(qt2, state, profile=profile),
            "\nGUIDANCE\n" + _guidance_block(QueryType.GENERAL, focus, profile, subfocus),
        ])

    sections.append(
        "\nTOOLS\n"
        "- get_epoch_data(epoch_id, component_type, family_filter) for values, counts, memberships, and family drill-downs\n"
        "- get_comparison_json(analysis_type, family_filter) for QT summaries and detailed comparison data\n"
    )

    return "\n".join(sections)
