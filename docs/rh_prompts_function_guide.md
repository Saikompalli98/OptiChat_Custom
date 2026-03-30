# `rh_prompts.py` Function Guide

Source file: `/Users/skompall/OptiChat_Custom/rh_comparison/agents/rh_prompts.py`

This file has four kinds of functions:

1. small text-cleanup helpers
2. question interpretation helpers
3. QT-summary renderers
4. final prompt builders

The explanations below are grouped the same way.

## 1. Small formatting and cleanup helpers

### `_safe(val, fmt=".4g", default="N/A")`
- Purpose: convert a raw value into a safe display string.
- `val`: any raw value from JSON/state.
- `fmt`: Python numeric format string used if `val` can be cast to float.
- `default`: text to return if `val is None`.

### `_con_label(name)`
- Purpose: strip the family prefix from a constraint name like `c1[a] -> a`.
- `name`: full constraint name string.

### `_var_label(name)`
- Purpose: strip the family prefix from a variable name like `x[a] -> a`.
- `name`: full variable name string.

### `_epoch_label(state, which)`
- Purpose: get the human label for the earlier or later run.
- `state`: session state dict.
- `which`: `"a"` for earlier plan or `"b"` for later plan.

### `_truncate_text(text, limit=1400)`
- Purpose: shorten long text blocks before prompt injection.
- `text`: raw explanation text.
- `limit`: max characters to keep.

### `_clean_label(text, fallback)`
- Purpose: normalize a label by trimming spaces and trailing periods.
- `text`: candidate label text.
- `fallback`: string to use if `text` is empty.

### `_natural_join(items)`
- Purpose: join a list into readable English like `a, b, and c`.
- `items`: list of strings.

### `_natural_sort_key(value)`
- Purpose: create a sorting key that keeps numeric-like indices in numeric order.
- `value`: any index-like item.

### `_ordered_unique(items)`
- Purpose: deduplicate and sort a list in a stable, natural way.
- `items`: list of string-like values.

### `_parse_index(index)`
- Purpose: try to parse an index string like `"(A, 3)"` into a Python tuple.
- `index`: raw index value or string.

### `_normalize_question(text)`
- Purpose: lowercase and normalize whitespace for question analysis.
- `text`: raw user question.

## 2. Question-shape helpers

### `_question_focus(query_type, user_question)`
- Purpose: decide whether the question is `broad` or `focused`.
- `query_type`: already-classified RH query type.
- `user_question`: raw user text.

### `_query_subfocus(query_type, user_question)`
- Purpose: refine a query type into a smaller subcase.
- `query_type`: RH query type.
- `user_question`: raw user text.
- Returns:
  - structural: `formulation`, `input_data`, or `setup`
  - solution diff: `drivers` or `plan_changes`
  - backward compat: `change_amounts` or `reuse_check`

### `_profile_limit(profile, *, detailed, summary)`
- Purpose: choose one limit value depending on answer profile.
- `profile`: `"detailed"` or `"summary"`.
- `detailed`: value to use in detailed mode.
- `summary`: value to use in summary mode.

### `_sample_items(items, profile, *, detailed=..., summary=...)`
- Purpose: pick only the first few cleaned items for prompt injection.
- `items`: list of candidate items.
- `profile`: `"detailed"` or `"summary"`.
- `detailed`: max count for detailed mode.
- `summary`: max count for summary mode.

### `_compact_numeric_series(items, profile)`
- Purpose: compress a list like `1,2,3,5,6` into a readable series like `1 through 3 and 5, 6`.
- `items`: list of numeric-like strings.
- `profile`: controls how many ranges are shown.

### `_sample_phrase(items, profile)`
- Purpose: turn a list into a short human-readable phrase.
- `items`: list of strings.
- `profile`: `"detailed"` or `"summary"`.

## 3. Label filtering and label rewriting helpers

### `_is_internal_label(name, label)`
- Purpose: hide internal approximation machinery like `D`, `Dpos`, `lnB`, `exp_r` from default business summaries.
- `name`: raw family/set/component name.
- `label`: cleaned label text.

### `_set_subject(label)`
- Purpose: rewrite a set label into a lowercase subject phrase.
- `label`: set label text.

### `_summary_label(label)`
- Purpose: shorten a long doc string into a cleaner summary label.
- `label`: raw label/doc text.

### `_index_context(label, indices)`
- Purpose: build a phrase like `for 4, 5, and 6`.
- `label`: family label, mainly for readability.
- `indices`: list of changed indices.

### `_dimension_label(set_name, set_labels)`
- Purpose: convert an index-set name into a readable dimension name like `time period`.
- `set_name`: raw set key.
- `set_labels`: map of set names to human labels.

### `_decision_family_label(label)`
- Purpose: strip generic prefixes like `binary variable indicating whether ...` from variable docs.
- `label`: raw variable family label/doc.

### `_constraint_label(name, labels)`
- Purpose: get the cleaned human label for a constraint family.
- `name`: raw constraint family name.
- `labels`: map of constraint family names to docs/labels.

### `_objective_phrase(label)`
- Purpose: strip `maximize` or `minimize` from objective text so the result reads naturally.
- `label`: objective label/doc.

### `_family_activity_term(label, vtype)`
- Purpose: choose a generic business phrase for a variable family.
- `label`: variable family label.
- `vtype`: variable type such as `binary`.

### `_family_change_strength(changed, added, removed)`
- Purpose: convert counts into a phrase like `changed materially` or `changed noticeably`.
- `changed`: count of overlapping items that changed.
- `added`: count of later-only items.
- `removed`: count of earlier-only items.

## 4. Structural and parameter-summary helpers

### `_describe_parameter_family_change(label, increased, decreased)`
- Purpose: create sentence fragments for one-dimensional parameter increases/decreases.
- `label`: family label.
- `increased`: indices where values went up.
- `decreased`: indices where values went down.

### `_collect_dimension_members(indices, position)`
- Purpose: extract one dimension from tuple-style indices.
- `indices`: list of tuple-like index strings.
- `position`: tuple position to read.

### `_describe_multidimensional_parameter_change(family, label, added, removed, changed)`
- Purpose: summarize a multidimensional parameter family.
- `family`: raw parameter family name.
- `label`: cleaned family label.
- `added`: later-only entries.
- `removed`: earlier-only entries.
- `changed`: list of changed-entry dicts.

### `_summarize_parameter_family(family, family_payload, fallback_labels)`
- Purpose: build lines summarizing one parameter family.
- `family`: parameter family key.
- `family_payload`: QT1 payload for that family.
- `fallback_labels`: fallback label map from state/QT labels.

### `_describe_set_change(name, payload, label)`
- Purpose: summarize a changed set in plain language.
- `name`: set name.
- `payload`: QT1 set-change payload.
- `label`: cleaned set label.

### `_summarize_variable_family_changes(variable_changes, variable_labels)`
- Purpose: summarize added or removed variable families.
- `variable_changes`: QT1 variable family change block.
- `variable_labels`: variable family label map.

### `_group_parameter_shift_lines(by_family, fallback_labels)`
- Purpose: loop through all parameter families and combine their summary lines.
- `by_family`: QT1 parameter-change dict keyed by family.
- `fallback_labels`: parameter label map.

### `_index_phrase(index)`
- Purpose: turn an index value into readable text.
- `index`: raw index value.

### `_format_value_shift(label, index, value_a, value_b)`
- Purpose: format a single `old -> new` value change line.
- `label`: what changed.
- `index`: where it changed.
- `value_a`: earlier value.
- `value_b`: later value.

## 5. Variable-name and location helpers

### `_parse_name_index(name)`
- Purpose: parse the bracketed part of a variable name like `x[Front,5]`.
- `name`: full variable name.

### `_format_domain_location(var_name, indexed_over, set_labels)`
- Purpose: turn a raw variable key into a phrase like `Front vehicle at time period 5`.
- `var_name`: full variable name.
- `indexed_over`: list of set names used by that variable family.
- `set_labels`: set label map.

## 6. Generic sentence builders for plan and reuse explanations

### `_describe_direction(label, direction, items, profile)`
- Purpose: describe where a parameter family increased or decreased.
- `label`: family label.
- `direction`: `"up"` or `"down"`.
- `items`: changed members.
- `profile`: `"detailed"` or `"summary"`.

### `_describe_added_removed_entries(label, direction, items, profile)`
- Purpose: describe newly covered or removed entries in a family.
- `label`: family label.
- `direction`: `"added"` or `"removed"`.
- `items`: affected members.
- `profile`: `"detailed"` or `"summary"`.

### `_result_delta_phrase(value_a, value_b)`
- Purpose: convert two numeric results into a phrase like `rose from 10 to 12`.
- `value_a`: earlier result.
- `value_b`: later result.

### `_required_change_phrase(description, amount)`
- Purpose: describe what rule change would be needed to allow the old plan.
- `description`: business rule description.
- `amount`: required change amount from QT3.

### `_revised_current_plan_phrase(amount)`
- Purpose: describe what the new run would need if rules stay unchanged.
- `amount`: required reduction/change amount.

### `_blocking_overage_phrase(description, amount)`
- Purpose: describe the first blocking overage in QT3.
- `description`: rule description.
- `amount`: required slack / overage size.

## 7. Renderers that convert QT JSON into prompt-ready text

### `_render_set_change_sentence(set_name, payload, label, profile)`
- Purpose: convert one changed set into a single business sentence.
- `set_name`: raw set name.
- `payload`: QT1 set-change payload.
- `label`: set label/doc.
- `profile`: `"detailed"` or `"summary"`.

### `_render_parameter_family_summary(family, family_payload, fallback_labels, profile)`
- Purpose: convert one parameter family into one or more business summary lines.
- `family`: parameter family name.
- `family_payload`: QT1 payload for that family.
- `fallback_labels`: parameter label map.
- `profile`: `"detailed"` or `"summary"`.

### `_solution_family_summary(family, family_summary, allocation_summary, variable_labels)`
- Purpose: summarize how one variable family changed between the two plans.
- `family`: variable family name.
- `family_summary`: QT2 family-level change stats.
- `allocation_summary`: QT2 rollup for that family, or `None`.
- `variable_labels`: variable family label map.

### `_summarize_solution_family_examples(top_changes, variable_metadata, set_labels, profile)`
- Purpose: create a few natural-language examples from QT2 top variable changes.
- `top_changes`: QT2 top individual change rows.
- `variable_metadata`: metadata for each variable family.
- `set_labels`: set label map.
- `profile`: `"detailed"` or `"summary"`.

### `_dimension_shift_sentence(dimension_summary, set_labels, profile)`
- Purpose: describe plan shifts by one dimension, such as by time period or by vehicle.
- `dimension_summary`: QT2 dimension-level rollup block.
- `set_labels`: set label map.
- `profile`: `"detailed"` or `"summary"`.

### `_format_qt1(qt1, state=None, profile="detailed", subfocus="setup")`
- Purpose: render the structural/input-difference summary block injected into the prompt.
- `qt1`: QT1 result dict.
- `state`: session state, used for labels and epoch names.
- `profile`: `"detailed"` or `"summary"`.
- `subfocus`: `setup`, `formulation`, or `input_data`.

### `_format_qt2(qt2, state=None, profile="detailed")`
- Purpose: render the solution-difference summary block injected into the prompt.
- `qt2`: QT2 result dict.
- `state`: session state for labels and epoch names.
- `profile`: `"detailed"` or `"summary"`.

### `_format_qt3(qt3, state=None, profile="detailed")`
- Purpose: render the backward-compatibility summary block injected into the prompt.
- `qt3`: QT3 result dict.
- `state`: session state.
- `profile`: `"detailed"` or `"summary"`.

### `_format_qt4(qt4, state=None, profile="detailed")`
- Purpose: render the attribution summary block injected into the prompt.
- `qt4`: QT4 result dict.
- `state`: session state.
- `profile`: `"detailed"` or `"summary"`.

## 8. Metadata and description block builders

### `_format_epoch_meta_block(state)`
- Purpose: build the small earlier/later plan metadata block for the prompt.
- `state`: session state containing epoch labels, solve status, and objective values.

### `_format_epoch_desc_block(state, profile="detailed")`
- Purpose: inject the manager-friendly model explanation built during initialization.
- `state`: session state containing `RH_DESCRIPTION_A` and `RH_DESCRIPTION_B`.
- `profile`: controls truncation length.

### `_format_root_epoch_info_block(state)`
- Purpose: build the short `Loaded planning runs:` block for the root prompt.
- `state`: session state.

## 9. Root prompt builder

### `build_root_agent_prompt(state)`
- Purpose: fill `ROOT_AGENT_PROMPT_TEMPLATE` with current session status and loaded epoch labels.
- `state`: session state.

## 10. Visibility and profile-selection helpers

### `_qt1_visible_family_count(qt1)`
- Purpose: count how many visible QT1 families would appear in a business answer.
- `qt1`: QT1 result dict.

### `_qt2_visible_family_count(qt2)`
- Purpose: count visible QT2 variable families.
- `qt2`: QT2 result dict.

### `_qt3_visible_family_count(qt3)`
- Purpose: estimate QT3 answer complexity from feasibility or blocker count.
- `qt3`: QT3 result dict.

### `_qt4_visible_family_count(qt4)`
- Purpose: estimate QT4 answer complexity from visible parameter, variable, and rule families.
- `qt4`: QT4 result dict.

### `_informational_line_count(block)`
- Purpose: count non-header lines in a rendered block.
- `block`: already-rendered prompt text.

### `_query_profile_metrics(query_type, state, subfocus="")`
- Purpose: render the relevant block in detailed mode and compute complexity metrics.
- `query_type`: RH query type.
- `state`: session state.
- `subfocus`: structural / solution / compatibility subcase.
- Returns: `(family_count, rendered_block)`.

### `_choose_answer_profile(query_type, state, user_question)`
- Purpose: choose `"detailed"` or `"summary"` for the answer.
- `query_type`: RH query type.
- `state`: session state.
- `user_question`: raw user text.

## 11. Guidance and final comparison prompt builders

### `_guidance_block(query_type, focus, profile, subfocus="")`
- Purpose: build the instruction block that tells the comparison agent how to answer this specific question.
- `query_type`: RH query type.
- `focus`: `"broad"` or `"focused"`.
- `profile`: `"detailed"` or `"summary"`.
- `subfocus`: more specific interpretation of the question.
- Combines:
  - strategy text
  - focus rule
  - profile rule
  - subfocus rule
  - one example answer style

### `build_comparison_agent_prompt(state, query_type, user_question="")`
- Purpose: assemble the full final system prompt for `rh_comparison_agent`.
- `state`: session state containing epoch metadata, descriptions, and any cached QT results.
- `query_type`: already-classified RH query type.
- `user_question`: raw user text.
- Internally it:
  - decides `focus`
  - decides `subfocus`
  - decides `profile`
  - builds the base prompt
  - injects the relevant QT block(s)
  - injects the guidance block
  - appends the available tool descriptions

## 12. Most important functions to understand first

If you only want the key control points, focus on these:

1. `build_root_agent_prompt(...)`
2. `_question_focus(...)`
3. `_query_subfocus(...)`
4. `_choose_answer_profile(...)`
5. `_format_qt1(...)`
6. `_format_qt2(...)`
7. `_format_qt3(...)`
8. `_format_qt4(...)`
9. `_guidance_block(...)`
10. `build_comparison_agent_prompt(...)`

Those ten functions are the backbone of how RH turns stored comparison data into the final prompt seen by the comparison agent.
