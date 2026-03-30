import math
import pyomo.environ as pyo

def create_model(data):
    model = pyo.ConcreteModel(
        doc=(
            "This model represents a vehicle allocation and boosting optimization problem across multiple "
            "time periods. The goal is to decide which vehicles should be assigned to which time periods "
            "so as to maximize a weighted profit measure. Each assignment of a vehicle to a time period "
            "contributes to a multiplicative boost effect, and the model approximates this nonlinear boost "
            "using a geometric grid. The optimization balances vehicle usage limits and time-period slot "
            "capacities while selecting assignments that produce the highest weighted benefit across all periods."
        )
    )

    # --- Data Extraction ---
    T = data['periods']
    V = data['vehicles']
    L = {int(k): v for k, v in data['slot_capacity'].items()}
    C = data['usage_budgets']
    alpha = {int(k): v for k, v in data['profit_weights'].items()}
    boosts_vt = data['boosts']
    epsilon = data.get('epsilon', 0.05)

    # --- Grid Generation Logic ---
    lnB = {}
    B_values_above_1 = []

    for v in V:
        for t in T:
            B = max(1.0, float(boosts_vt[v][str(t)]))
            lnB[(v, t)] = 0.0 if B == 1.0 else math.log(B)

            if B > 1.0:
                B_values_above_1.append(B)

    if not B_values_above_1:
        raise ValueError("No boosts > 1 found; optimization is trivial.")

    B_plus_min = min(B_values_above_1)
    B_max = max(max(v_dict.values()) for v_dict in boosts_vt.values())
    Delta = max(L.values())

    # --- Build geometric grid ---
    r_min = math.log(B_plus_min)
    r_max = Delta * math.log(B_max)

    M = (Delta / float(epsilon)) * math.log(B_max)
    ratio = (1.0 + 1.0 / M) if M > 0 else 2.0

    D = [0.0]
    r = r_min

    while r <= r_max * (1.0 + 1e-12):
        D.append(float(r))
        r *= ratio

    D = sorted(set(D))
    Dpos = [val for val in D if val > 0.0]

    exp_r = {val: math.exp(val) for val in D}

    # --- Pyomo Sets ---
    model.T = pyo.Set(
        initialize=T,
        doc="Set of time periods in which vehicles may be assigned."
    )

    model.V = pyo.Set(
        initialize=V,
        doc="Set of vehicles available for assignment."
    )

    model.D = pyo.Set(
        initialize=D,
        doc="Internal performance levels used to represent the combined uplift created by the assigned vehicles."
    )

    model.Dpos = pyo.Set(
        initialize=Dpos,
        doc="The positive internal performance levels used in the consistency rules for the uplift calculation."
    )

    # --- Parameters ---
    model.L = pyo.Param(
        model.T,
        initialize=L,
        mutable = True,
        doc="Slot capacity for each time period (maximum number of vehicles that can be assigned)."
    )

    model.C = pyo.Param(
        model.V,
        initialize=C,
        mutable = True,
        doc="Usage budget for each vehicle representing the maximum number of periods it may be assigned."
    )

    model.alpha = pyo.Param(
        model.T,
        initialize=alpha,
        mutable = True,
        doc="Profit weight associated with each time period."
    )

    model.lnB = pyo.Param(
        model.V,
        model.T,
        initialize=lnB,
        mutable = True,
        doc="Expected uplift contribution from assigning vehicle v in period t, expressed in the model's internal scale."
    )

    model.exp_r = pyo.Param(
        model.D,
        initialize=exp_r,
        mutable = True,
        doc="Business uplift value associated with each internal performance level."
    )

    # --- Variables ---
    model.x = pyo.Var(
        model.V,
        model.T,
        within=pyo.Binary,
        doc="Binary variable indicating whether vehicle v is assigned to time period t."
    )

    model.y = pyo.Var(
        model.T,
        model.D,
        within=pyo.Binary,
        doc="Binary decision selecting the internal uplift level used to represent each period's combined vehicle effect."
    )

    # --- Constraints ---
    model.C1_budget = pyo.Constraint(
        model.V,
        rule=lambda m, v: sum(m.x[v, t] for t in m.T) <= m.C[v],
        doc="Vehicle usage budget constraint ensuring each vehicle is assigned at most C[v] times."
    )

    model.C2_slots = pyo.Constraint(
        model.T,
        rule=lambda m, t: sum(m.x[v, t] for v in m.V) <= m.L[t],
        doc="Slot capacity constraint limiting the number of vehicles assigned in each period."
    )

    model.C3_pick_one = pyo.Constraint(
        model.T,
        rule=lambda m, t: sum(m.y[t, r] for r in m.D) == 1,
        doc="Exactly one grid value must be selected for each time period."
    )

    def link_rule(m, t, r):
        return m.y[t, r] <= (1.0 / float(r)) * sum(m.lnB[v, t] * m.x[v, t] for v in m.V)

    model.C4_link = pyo.Constraint(
        model.T,
        model.Dpos,
        rule=link_rule,
        doc="Consistency rule ensuring the selected uplift level is supported by the vehicles assigned in that period."
    )

    # --- Objective ---
    def obj_rule(m):
        return sum(
            m.alpha[t] * sum(m.exp_r[r] * m.y[t, r] for r in m.D)
            for t in m.T
        )

    model.obj = pyo.Objective(
        rule=obj_rule,
        sense=pyo.maximize,
        doc="Maximize the total weighted business uplift captured across all periods."
    )

    return model


data = globals().get("data", {})
if data:
    model = create_model(data)
