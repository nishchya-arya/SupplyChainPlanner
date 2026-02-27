"""
MILP Supply Chain Optimizer using PuLP.

Given a product category, target country, and volume, finds the optimal
factory→hub allocation that minimizes composite cost while respecting
factory capacity, hub throughput, geopolitical restrictions, and lead times.

The solver uses Mixed-Integer Linear Programming (MILP):
  - "Linear" because the objective function and all constraints are linear
    in the decision variables (no x², no x·y, no log(x)).
  - "Mixed-Integer" because we have both continuous variables (x = units per
    flow) and binary/integer variables (y = is this flow active?).
  - This guarantees a provably optimal solution, unlike heuristics or
    metaheuristics which only find approximate solutions.
"""

from dataclasses import dataclass, field
import pandas as pd
import pulp

from solver.data_loader import SupplyChainData


@dataclass
class SolverResult:
    """Container for MILP solver output."""
    status: str                         # "Optimal", "Infeasible", etc.
    chosen_flows: list = field(default_factory=list)    # Tier 1: allocated flows
    total_cost: float = 0.0             # Total cost of optimal allocation (real dollars)
    total_units: int = 0                # Total units allocated (should == volume)
    feasible_flows: pd.DataFrame = field(default_factory=pd.DataFrame)  # All feasible flows with scores


def solve(
    data: SupplyChainData,
    category_id: str,
    country_code: str,
    volume: int,
    cost_weight: int = 8,
    time_weight: int = 5,
    regional_weight: int = 3,
    min_batch: int = 500,
) -> SolverResult:
    """
    Solve the supply chain allocation problem via MILP.

    Args:
        data: Loaded supply chain data.
        category_id: Product category (e.g., "CAT01").
        country_code: Destination country (e.g., "US").
        volume: Number of units to allocate.
        cost_weight: 1-10, importance of minimizing cost.
        time_weight: 1-10, importance of faster delivery.
        regional_weight: 1-10, importance of same-region manufacturing/storage.
        min_batch: Minimum units per active flow (makes it MILP via binary vars).

    Returns:
        SolverResult with optimal allocation and all feasible flows.
    """
    # ── 1. Get feasible flows ────────────────────────────────────────────
    # Feasible = not geopolitically restricted AND within lead-time limits.
    # Pre-filtered by data_loader; typically ~27 flows from ~22K total.
    flows = data.get_feasible_flows(category_id, country_code)

    if flows.empty:
        return SolverResult(status="No feasible flows found")

    dest_region = data.get_region_for_country(country_code)
    factory_cap = data.get_factory_capacity(category_id)   # {factory_id: max_units}
    hub_throughput = data.get_hub_throughput()               # {hub_id: max_units}

    # ── 2. Compute effective cost (normalized weighted score) ────────────
    # We normalize each criterion to [0, 1] using min-max scaling, then
    # combine them with user-provided weights. This "effective_cost" is what
    # the MILP minimizes — it's NOT a dollar amount, it's a unitless score.
    # The actual dollar cost is computed separately for reporting.

    # Cost normalization: cheapest flow → 0.0, most expensive → 1.0
    cost_vals = flows["total_landed_cost"]
    cost_range = cost_vals.max() - cost_vals.min()
    cost_norm = (cost_vals - cost_vals.min()) / cost_range if cost_range > 0 else 0.0

    # Time normalization: fastest flow → 0.0, slowest → 1.0
    time_vals = flows["transit_days"].astype(float)
    time_range = time_vals.max() - time_vals.min()
    time_norm = (time_vals - time_vals.min()) / time_range if time_range > 0 else 0.0

    # Regional penalty: measures geographic proximity of factory/hub to destination
    #   0.0 = factory is in the same region as destination (best case)
    #   0.5 = factory is elsewhere, but hub is in destination's region
    #   1.0 = neither factory nor hub is in destination's region (worst case)
    regional_penalty = flows.apply(
        lambda r: 0.0 if data.get_region_for_factory(r["factory_id"]) == dest_region
        else 0.5 if data.get_region_for_hub(r["hub_id"]) == dest_region
        else 1.0,
        axis=1,
    )

    # Weighted composite: each weight contributes proportionally.
    # Example: cost_weight=8, time_weight=5, regional_weight=3 → w_total=16
    #   cost gets 50% influence, time 31%, regional 19%.
    w_total = cost_weight + time_weight + regional_weight
    flows["effective_cost"] = (
        (cost_weight / w_total) * cost_norm
        + (time_weight / w_total) * time_norm
        + (regional_weight / w_total) * regional_penalty
    )

    # ── 3. Check total capacity ──────────────────────────────────────────
    # Quick feasibility check before building the MILP — if all factories
    # combined can't produce enough, no solution exists.
    available_factories = flows["factory_id"].unique()
    total_capacity = sum(factory_cap.get(f, 0) for f in available_factories)

    if total_capacity < volume:
        return SolverResult(
            status=f"Infeasible: total factory capacity ({total_capacity:,}) < volume ({volume:,})",
            feasible_flows=flows,
        )

    # ── 4. Build MILP ────────────────────────────────────────────────────
    prob = pulp.LpProblem("SupplyChainOptimizer", pulp.LpMinimize)

    flow_ids = list(flows.index)

    # Decision variables:
    #   x[i] = continuous, how many units to allocate to flow i (≥ 0)
    #   y[i] = binary, 1 if flow i is active (receives any allocation), 0 otherwise
    x = pulp.LpVariable.dicts("x", flow_ids, lowBound=0, cat="Continuous")
    y = pulp.LpVariable.dicts("y", flow_ids, cat="Binary")

    # Objective: minimize total weighted score across all flows
    # Note: this is NOT dollars — it's the composite score. Dollar cost is
    # computed after solving from the actual total_landed_cost values.
    prob += pulp.lpSum(x[i] * flows.loc[i, "effective_cost"] for i in flow_ids)

    # Constraint 1 — Meet demand: total allocated units must equal requested volume
    prob += pulp.lpSum(x[i] for i in flow_ids) == volume, "MeetDemand"

    # Constraint 2 — Factory capacity: each factory can only produce up to
    # its monthly capacity for this product category
    for f_id in available_factories:
        flow_indices = flows[flows["factory_id"] == f_id].index.tolist()
        cap = factory_cap.get(f_id, 0)
        prob += (
            pulp.lpSum(x[i] for i in flow_indices) <= cap,
            f"FactoryCap_{f_id}",
        )

    # Constraint 3 — Hub throughput: each hub can only handle up to its
    # monthly throughput capacity across all flows routed through it
    available_hubs = flows["hub_id"].unique()
    for h_id in available_hubs:
        flow_indices = flows[flows["hub_id"] == h_id].index.tolist()
        cap = hub_throughput.get(h_id, 0)
        prob += (
            pulp.lpSum(x[i] for i in flow_indices) <= cap,
            f"HubCap_{h_id}",
        )

    # Constraint 4 — Flow activation: links x[i] to y[i]. If y[i]=0
    # (flow inactive), then x[i] must be 0. If y[i]=1, x[i] can be up
    # to the full volume. This is a "big-M" constraint with M=volume.
    for i in flow_ids:
        prob += x[i] <= volume * y[i], f"Activate_{i}"

    # Constraint 5 — Minimum batch: if a flow is active (y[i]=1), it must
    # receive at least min_batch units. This prevents unrealistically tiny
    # allocations like 3 units to a factory. This constraint, combined with
    # the binary y[i], is what makes this a MILP (not just LP).
    for i in flow_ids:
        prob += x[i] >= min_batch * y[i], f"MinBatch_{i}"

    # ── 5. Solve ─────────────────────────────────────────────────────────
    # CBC = Coin-or Branch and Cut, an open-source MILP solver.
    # msg=0 suppresses solver output. Solves in milliseconds for ~27 flows.
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        return SolverResult(status=status, feasible_flows=flows)

    # ── 6. Extract results ───────────────────────────────────────────────
    # Read the solution: for each flow with a non-trivial allocation,
    # compute the real dollar cost (not the normalized score) for reporting.
    chosen = []
    total_cost = 0.0
    total_units = 0

    for i in flow_ids:
        alloc = x[i].varValue
        # Threshold > 0.5 filters out numerical noise from the solver
        # (CBC sometimes returns tiny non-zero values like 1e-8)
        if alloc is not None and alloc > 0.5:
            row = flows.loc[i]
            units = round(alloc)
            cost = units * row["total_landed_cost"]  # actual dollar cost
            chosen.append({
                "factory_id": row["factory_id"],
                "hub_id": row["hub_id"],
                "units_allocated": units,
                "cost_per_unit": row["total_landed_cost"],
                "effective_cost_per_unit": row["effective_cost"],
                "total_cost": round(cost, 2),
                "manufacturing_cost": row["manufacturing_cost"],
                "transport_cost": row["transport_cost"],
                "hub_handling_cost": row["hub_handling_cost"],
                "last_mile_cost": row["last_mile_cost"],
                "tariff_pct": row["tariff_pct"],
                "tariff_amount": row["tariff_amount"],
                "transit_days": int(row["transit_days"]),
            })
            total_cost += cost
            total_units += units

    # Sort chosen flows by allocation (largest first) for display
    chosen.sort(key=lambda f: f["units_allocated"], reverse=True)

    return SolverResult(
        status="Optimal",
        chosen_flows=chosen,
        total_cost=round(total_cost, 2),
        total_units=total_units,
        feasible_flows=flows,
    )
