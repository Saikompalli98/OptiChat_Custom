#!/usr/bin/env python3
"""
Diet Planning Model — Epoch A (Month 1, Spring).

Classic minimum-cost diet LP.  The planner must purchase a combination of
10 available foods to meet six daily nutrient requirements at minimum cost.

Sets
----
  f  : foods available this planning period
         {wheat, cornmeal, cannedmilk, cheese, peanut-b,
          liver, potatoes, spinach, cabbage, navybeans}
  n  : nutrients tracked
         {calorie, protein, calcium, iron, vitamin-a, vitamin-c}

Parameters
----------
  b[n]       : minimum daily requirement for nutrient n  (per-nutrient units)
  a[f, n]    : nutritive yield of food f per dollar spent (units / dollar)

Variables
---------
  x[f] >= 0  : dollars spent daily on food f              (dollars/day)
  cost >= 0  : total daily food expenditure               (dollars/day)

Constraints
-----------
  nutrient_min[n]  : sum_f a[f,n] * x[f]  >= b[n]        (nutrient balance)
  cost_balance     : cost == sum_f x[f]                   (cost definition)

Objective
---------
  Minimize  cost
"""

from pyomo.environ import (
    ConcreteModel, Set, Param, Var,
    Constraint, Objective,
    NonNegativeReals, minimize,
)

# ---------------------------------------------------------------------------
# Data — injected by the RH framework via globals(); falls back to {} so the
# file is also importable standalone (Pyomo will raise on missing data then).
# ---------------------------------------------------------------------------
data = globals().get("data", {})

nutrient_requirements = data["nutrient_requirements"]   # {nutrient: min_value}
nutritive_values      = data["nutritive_values"]        # {food: {nutrient: yield}}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = ConcreteModel(name="Diet_EpochA")

# --- Sets ---
model.f = Set(
    initialize=list(nutritive_values.keys()),
    doc="Foods available in Month 1 (Spring)",
)
model.n = Set(
    initialize=list(nutrient_requirements.keys()),
    doc="Nutrients tracked in Month 1",
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

# --- Constraints ---
def nutrient_min_rule(mdl, n):
    """Each nutrient must meet its minimum daily requirement."""
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

# --- Objective ---
model.obj = Objective(
    expr=model.cost,
    sense=minimize,
    doc="Minimize total daily food cost  [dollars/day]",
)
