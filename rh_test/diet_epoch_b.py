#!/usr/bin/env python3
"""
Diet Planning Model — Epoch B (Month 2, Summer).

Extended minimum-cost diet MILP.  This epoch introduces four categories of
changes relative to Epoch A, making it a rich rolling-horizon test instance:

SET CHANGES
-----------
  Foods   : cornmeal removed (supply disruption); eggs and yogurt added
            (newly available from summer suppliers).
  Nutrients: fiber added as a seventh tracked nutrient (updated health policy).

PARAMETER CHANGES
-----------------
  b['iron']      : requirement raised 12 → 15 (revised dietary guidelines)
  b['vitamin-c'] : requirement raised 75 → 90 (revised dietary guidelines)
  b['fiber']     : new requirement = 25       (new nutrient, new policy)
  a['wheat', *]  : all yields reduced by ~15% (seasonal wheat price spike)
  a['spinach', *]: all yields increased by ~20% (summer harvest, price drop)
  max_serving[f] : new per-food daily spending cap (supply / portion limit)
  calorie_max    : new calorie upper bound = 4.5 (avoid excess intake)
  min_foods      : minimum distinct foods in diet = 4 (dietary variety policy)

VARIABLE CHANGES
----------------
  select[f] ∈ {0, 1}  added — binary indicator: 1 if food f is included in
                               the daily diet, 0 otherwise.
  (x[f] and cost carry over from Epoch A unchanged.)

CONSTRAINT CHANGES
------------------
  nutrient_min[n]  : unchanged in form; now also enforces fiber requirement.
  cost_balance     : unchanged.
  calorie_upper    : NEW — sum_f a[f,'calorie']*x[f] <= calorie_max
                     (prevents excessive calorie intake)
  serving_link[f]  : NEW — x[f] <= max_serving[f] * select[f]
                     (spending on food f is zero unless food is selected;
                      also enforces per-food supply / portion cap)
  min_variety      : NEW — sum_f select[f] >= min_foods
                     (dietary variety requirement)

Sets
----
  f  : {wheat, cannedmilk, cheese, peanut-b, liver,
         potatoes, spinach, cabbage, navybeans, eggs, yogurt}
  n  : {calorie, protein, calcium, iron, vitamin-a, vitamin-c, fiber}

Parameters
----------
  b[n]             : minimum daily requirement for nutrient n
  a[f, n]          : nutritive yield of food f per dollar spent
  max_serving[f]   : maximum dollars per day that can be spent on food f
  calorie_max      : maximum total daily calorie intake (scalar)
  min_foods        : minimum number of distinct foods in the diet (scalar)

Variables
---------
  x[f]      >= 0   : dollars spent daily on food f         [dollars/day]
  cost      >= 0   : total daily food expenditure          [dollars/day]
  select[f] in {0,1}: 1 if food f is included in the diet

Constraints
-----------
  nutrient_min[n]  : sum_f a[f,n]*x[f]           >= b[n]
  cost_balance     : cost == sum_f x[f]
  calorie_upper    : sum_f a[f,'calorie']*x[f]    <= calorie_max
  serving_link[f]  : x[f]                         <= max_serving[f]*select[f]
  min_variety      : sum_f select[f]              >= min_foods

Objective
---------
  Minimize  cost
"""

from pyomo.environ import (
    ConcreteModel, Set, Param, Var,
    Constraint, Objective,
    NonNegativeReals, Binary, minimize,
)

# ---------------------------------------------------------------------------
# Data — injected by the RH framework via globals()
# ---------------------------------------------------------------------------
data = globals().get("data", {})

nutrient_requirements = data["nutrient_requirements"]
nutritive_values      = data["nutritive_values"]
max_serving_data      = data["max_serving"]
calorie_max_val       = data["calorie_max"]
min_foods_val         = data["min_foods"]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = ConcreteModel(name="Diet_EpochB")

# --- Sets ---
model.f = Set(
    initialize=list(nutritive_values.keys()),
    doc="Foods available in Month 2 (Summer): cornmeal removed; eggs and yogurt added",
)
model.n = Set(
    initialize=list(nutrient_requirements.keys()),
    doc="Nutrients tracked in Month 2: fiber added",
)

# --- Parameters ---
model.b = Param(
    model.n,
    initialize=nutrient_requirements,
    mutable=True,
    doc="Minimum daily requirement for nutrient n  [per-nutrient units]",
)

model.a = Param(
    model.f, model.n,
    initialize={
        (food, nutrient): value
        for food, nutrients in nutritive_values.items()
        for nutrient, value in nutrients.items()
    },
    mutable=True,
    doc="Nutritive yield of food f per dollar spent  [units / dollar]",
)

model.max_serving = Param(
    model.f,
    initialize=max_serving_data,
    mutable=True,
    doc="Maximum dollars per day that can be spent on food f  [dollars/day]",
)

model.calorie_max = Param(
    initialize=calorie_max_val,
    mutable=True,
    doc="Maximum total daily calorie intake  [calorie units/day]",
)

model.min_foods = Param(
    initialize=min_foods_val,
    mutable=True,
    doc="Minimum number of distinct foods that must appear in the diet",
)

# --- Variables ---
model.x = Var(
    model.f,
    within=NonNegativeReals,
    initialize=0,
    doc="Dollars spent daily on food f  [dollars/day]",
)
model.cost = Var(
    within=NonNegativeReals,
    initialize=0,
    doc="Total daily food expenditure  [dollars/day]",
)
model.select = Var(
    model.f,
    within=Binary,
    initialize=0,
    doc="1 if food f is included in the diet, 0 otherwise",
)

# --- Constraints ---
def nutrient_min_rule(mdl, n):
    """Each nutrient (including the new fiber requirement) must be met."""
    return sum(mdl.a[f, n] * mdl.x[f] for f in mdl.f) >= mdl.b[n]

model.nutrient_min = Constraint(
    model.n,
    rule=nutrient_min_rule,
    doc="Daily nutrient requirement (lower bound)  [units]",
)


def cost_balance_rule(mdl):
    """Cost variable equals the sum of food expenditures."""
    return mdl.cost == sum(mdl.x[f] for f in mdl.f)

model.cost_balance = Constraint(
    rule=cost_balance_rule,
    doc="Total cost definition  [dollars]",
)


def calorie_upper_rule(mdl):
    """Total calorie intake must not exceed the daily maximum."""
    return sum(mdl.a[f, "calorie"] * mdl.x[f] for f in mdl.f) <= mdl.calorie_max

model.calorie_upper = Constraint(
    rule=calorie_upper_rule,
    doc="Calorie intake upper bound (new in Epoch B)  [calorie units/day]",
)


def serving_link_rule(mdl, f):
    """Spending on food f is bounded by its supply cap only if it is selected;
    if select[f] = 0 then x[f] must be 0 (big-M linking constraint)."""
    return mdl.x[f] <= mdl.max_serving[f] * mdl.select[f]

model.serving_link = Constraint(
    model.f,
    rule=serving_link_rule,
    doc="Per-food supply/portion cap linked to selection binary  [dollars/day]",
)


def min_variety_rule(mdl):
    """At least min_foods distinct foods must appear in the diet."""
    return sum(mdl.select[f] for f in mdl.f) >= mdl.min_foods

model.min_variety = Constraint(
    rule=min_variety_rule,
    doc="Dietary variety requirement: minimum number of foods selected",
)

# --- Objective ---
model.obj = Objective(
    expr=model.cost,
    sense=minimize,
    doc="Minimize total daily food cost  [dollars/day]",
)
