"""
Score feasible flows and format results into 3 output tiers:
  1. Chosen Flow(s) — from MILP solution
  2. Other Available Flows — next 3 best by composite score
  3. Alternative Manufacturing — remaining unique factories

The ranker uses the same normalized scoring formula as the optimizer
so that composite scores are consistent between the MILP solution and
the ranked alternatives shown to the user.
"""

import pandas as pd

from solver.data_loader import SupplyChainData
from solver.optimizer import SolverResult


def _compute_composite_scores(
    flows: pd.DataFrame,
    data: SupplyChainData,
    country_code: str,
    cost_weight: int = 8,
    time_weight: int = 5,
    regional_weight: int = 3,
) -> pd.Series:
    """Compute normalized composite score for each flow. Lower = better.

    Uses the same min-max normalization and weighted formula as
    optimizer.py's effective_cost calculation to ensure consistency."""
    dest_region = data.get_region_for_country(country_code)

    cost = flows["total_landed_cost"]
    days = flows["transit_days"].astype(float)

    # Min-max normalize to [0, 1]; if all values are equal, norm = 0 (no differentiation)
    cost_range = cost.max() - cost.min()
    days_range = days.max() - days.min()

    cost_norm = (cost - cost.min()) / cost_range if cost_range > 0 else 0.0
    days_norm = (days - days.min()) / days_range if days_range > 0 else 0.0

    # Regional penalty: how far the factory/hub is from the destination region
    #   0.0 = factory in same region (ideal for regional sourcing)
    #   0.5 = factory elsewhere, but hub in same region (partial benefit)
    #   1.0 = neither factory nor hub in same region (no regional advantage)
    regional_penalty = flows.apply(
        lambda r: 0.0 if data.get_region_for_factory(r["factory_id"]) == dest_region
        else 0.5 if data.get_region_for_hub(r["hub_id"]) == dest_region
        else 1.0,
        axis=1,
    )

    # Weighted sum: each weight is normalized by the total so they sum to 1.0
    w_total = cost_weight + time_weight + regional_weight
    return (
        (cost_weight / w_total) * cost_norm
        + (time_weight / w_total) * days_norm
        + (regional_weight / w_total) * regional_penalty
    )


def _get_restriction_reason(data: SupplyChainData, factory_id: str, country_code: str) -> str:
    """Look up the geopolitical restriction reason for a factory→country pair.

    Checks if the factory's country is restricted for the destination via MADE_IN rules."""
    factory_country = data.factory_country(factory_id)
    matches = data.geopolitical[
        (data.geopolitical["destination_country_code"] == country_code)
        & (data.geopolitical["restricted_country_code"] == factory_country)
        & (data.geopolitical["restriction_type"] == "MADE_IN")
    ]
    if not matches.empty:
        return matches.iloc[0]["reason"]
    return None


def rank_flows(
    result: SolverResult,
    data: SupplyChainData,
    country_code: str,
    cost_weight: int = 8,
    time_weight: int = 5,
    regional_weight: int = 3,
    category_id: str = None,
) -> dict:
    """
    Organize solver results into 3 tiers.

    Returns dict with keys:
        chosen_flows: list[dict]            — Tier 1 (from MILP)
        other_available: list[dict]         — Tier 2 (ranks 2-4)
        alternative_factories: list[dict]   — Tier 3 (unique factories not in T1/T2)

    When category_id is provided, Tier 3 includes ALL factories (even those
    blocked by geopolitical restrictions or lead-time violations), tagged with
    a status field explaining why they're unavailable.
    """
    flows = result.feasible_flows.copy()

    if flows.empty:
        return {"chosen_flows": [], "other_available": [], "alternative_factories": []}

    # Score all feasible flows using the same weights the solver used
    flows["composite_score"] = _compute_composite_scores(
        flows, data, country_code, cost_weight, time_weight, regional_weight
    )

    # ── Tier 1: Chosen flows (from MILP) ─────────────────────────────────
    # These are the flows the optimizer actually allocated units to.
    # We enrich them with human-readable factory/hub names for the UI.
    chosen_keys = set()
    for cf in result.chosen_flows:
        chosen_keys.add((cf["factory_id"], cf["hub_id"]))

    tier1 = []
    for cf in result.chosen_flows:
        tier1.append({
            **cf,
            "factory_name": data.factory_name(cf["factory_id"]),
            "factory_city": data.factory_city(cf["factory_id"]),
            "factory_country": data.factory_country(cf["factory_id"]),
            "hub_name": data.hub_name(cf["hub_id"]),
            "hub_city": data.hub_city(cf["hub_id"]),
            "hub_country": data.hub_country(cf["hub_id"]),
        })

    # ── Tier 2: Other available flows (next 3 by composite score) ────────
    # Exclude flows already in Tier 1, sort by score, take the top 3.
    # These serve as backup options if the primary supply chain is disrupted.
    remaining = flows[
        ~flows.apply(lambda r: (r["factory_id"], r["hub_id"]) in chosen_keys, axis=1)
    ].sort_values("composite_score")

    tier2_factories_used = set()  # track which factories appear in Tier 2
    tier2 = []
    for _, row in remaining.head(3).iterrows():
        tier2_factories_used.add(row["factory_id"])
        tier2.append({
            "rank": len(tier2) + 2,  # ranks 2, 3, 4 (Tier 1 is rank 1)
            "factory_id": row["factory_id"],
            "factory_name": data.factory_name(row["factory_id"]),
            "factory_city": data.factory_city(row["factory_id"]),
            "factory_country": data.factory_country(row["factory_id"]),
            "hub_id": row["hub_id"],
            "hub_name": data.hub_name(row["hub_id"]),
            "hub_city": data.hub_city(row["hub_id"]),
            "cost_per_unit": row["total_landed_cost"],
            "effective_cost_per_unit": row["effective_cost"],
            "transit_days": int(row["transit_days"]),
            "composite_score": round(row["composite_score"], 4),
        })

    # ── Tier 3: Alternative manufacturing (unique factories not in T1/T2)
    # Shows one entry per factory (the best hub route for each), excluding
    # any factory already represented in Tiers 1 or 2. This answers:
    # "What other manufacturing locations are available?"
    used_factories = set(cf["factory_id"] for cf in result.chosen_flows) | tier2_factories_used

    tier3_candidates = remaining[~remaining["factory_id"].isin(used_factories)]

    # For each unused factory, keep only its best-scoring flow (lowest composite_score)
    tier3 = []
    tier3_factory_ids = set()
    if not tier3_candidates.empty:
        best_per_factory = tier3_candidates.sort_values("composite_score").drop_duplicates(
            subset=["factory_id"], keep="first"  # keep="first" after sort = best score
        )
        for _, row in best_per_factory.iterrows():
            tier3_factory_ids.add(row["factory_id"])
            tier3.append({
                "factory_id": row["factory_id"],
                "factory_name": data.factory_name(row["factory_id"]),
                "factory_city": data.factory_city(row["factory_id"]),
                "factory_country": data.factory_country(row["factory_id"]),
                "best_hub_id": row["hub_id"],
                "best_hub_name": data.hub_name(row["hub_id"]),
                "cost_per_unit": row["total_landed_cost"],
                "transit_days": int(row["transit_days"]),
                "composite_score": round(row["composite_score"], 4),
                "status": "Available",
            })

    # ── Blocked factories (only when category_id is provided) ─────────
    # Include factories with zero feasible routes so the user sees ALL
    # 13 manufacturing locations regardless of restrictions.
    if category_id is not None:
        all_factory_ids = set(data.factories["factory_id"])
        missing = all_factory_ids - used_factories - tier3_factory_ids

        if missing:
            # Query ALL flows for this category+country (including restricted ones)
            all_cat_country = data.all_flows[
                (data.all_flows["category_id"] == category_id)
                & (data.all_flows["country_code"] == country_code)
            ]

            for fid in sorted(missing):
                factory_flows = all_cat_country[all_cat_country["factory_id"] == fid]

                # Determine block reason
                geo_reason = _get_restriction_reason(data, fid, country_code)
                if geo_reason:
                    status = f"Restricted: {geo_reason}"
                else:
                    status = "Exceeds lead time"

                # Find best route even among blocked flows (for display)
                if not factory_flows.empty:
                    best = factory_flows.sort_values("total_landed_cost").iloc[0]
                    tier3.append({
                        "factory_id": fid,
                        "factory_name": data.factory_name(fid),
                        "factory_city": data.factory_city(fid),
                        "factory_country": data.factory_country(fid),
                        "best_hub_id": best["hub_id"],
                        "best_hub_name": data.hub_name(best["hub_id"]),
                        "cost_per_unit": best["total_landed_cost"],
                        "transit_days": int(best["transit_days"]),
                        "composite_score": None,
                        "status": status,
                    })
                else:
                    # No flows at all (shouldn't happen with current data)
                    tier3.append({
                        "factory_id": fid,
                        "factory_name": data.factory_name(fid),
                        "factory_city": data.factory_city(fid),
                        "factory_country": data.factory_country(fid),
                        "best_hub_id": None,
                        "best_hub_name": "N/A",
                        "cost_per_unit": None,
                        "transit_days": None,
                        "composite_score": None,
                        "status": status,
                    })

    return {
        "chosen_flows": tier1,
        "other_available": tier2,
        "alternative_factories": tier3,
    }
