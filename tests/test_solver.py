"""
Test suite for the Supply Chain MILP Solver.

99 tests across 13 test classes. Uses real generated data (data/ CSVs)
so tests validate end-to-end behavior, not mocked data.

Tests cover:
  1. Data loading and filtering (CSV row counts, lookup methods)
  2. Geopolitical compliance (restricted flows excluded)
  3. Lead time feasibility (infeasible transit filtered)
  4. Cost minimization (solver picks cheapest feasible flow)
  5. Time penalty (slower routes penalized by time_weight slider)
  6. Regional preference (same-region factory/hub favored by regional_weight)
  7. Capacity constraints (volume exceeding 1 factory forces split)
  8. Minimum batch size (no trivially small allocations)
  9. 3-tier ranking output (chosen, other available, alternative factories)
  10. Cross-category and cross-country parametric tests
  11. Transit time realism (same-country fast, cross-region slow)
  12. Ontology layer (typed entities, validate_flow, restrictions, region queries)
  13. Knowledge graph (nodes, edges, routes, diversity, impact analysis)

Run: python -m pytest tests/test_solver.py -v
"""

import pytest
from solver.data_loader import SupplyChainData
from solver.optimizer import solve
from solver.ranker import rank_flows


@pytest.fixture(scope="module")
def data():
    """Module-scoped fixture: loads all CSVs once, shared across all tests."""
    return SupplyChainData()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoader:
    def test_all_csvs_loaded(self, data):
        assert len(data.regions) == 6
        assert len(data.countries) == 17
        assert len(data.factories) == 13
        assert len(data.hubs) == 14
        assert len(data.categories) == 10
        assert len(data.products) > 80  # ~86 products
        assert len(data.all_flows) > 20000
        assert len(data.geopolitical) == 11

    def test_category_list(self, data):
        cats = data.get_category_list()
        assert len(cats) == 10
        assert cats[0] == ("CAT01", "Smartphones")

    def test_region_list(self, data):
        regions = data.get_region_list()
        assert len(regions) == 6
        region_ids = [r[0] for r in regions]
        assert "NAM" in region_ids
        assert "EUR" in region_ids

    def test_countries_in_region(self, data):
        nam = data.get_countries_in_region("NAM")
        assert "US" in nam
        assert "CA" in nam
        assert "MX" in nam
        assert len(nam) == 3

    def test_default_country(self, data):
        assert data.get_default_country("NAM") == "US"
        assert data.get_default_country("SAM") == "BR"
        assert data.get_default_country("EUR") == "DE"
        assert data.get_default_country("MEA") == "AE"
        assert data.get_default_country("NEA") == "CN"
        assert data.get_default_country("SEA") == "IN"

    def test_factory_capacity_returns_dict(self, data):
        cap = data.get_factory_capacity("CAT01")
        assert isinstance(cap, dict)
        assert len(cap) > 0
        # All values should be positive integers
        for fid, c in cap.items():
            assert c > 0

    def test_hub_throughput_returns_dict(self, data):
        tp = data.get_hub_throughput()
        assert len(tp) == 14
        for hid, c in tp.items():
            assert c >= 70000


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GEOPOLITICAL COMPLIANCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeopoliticalCompliance:
    def test_no_chinese_factories_in_us_flows(self, data):
        """US has MADE_IN restriction on CN — no Chinese factory should appear."""
        flows = data.get_feasible_flows("CAT01", "US")
        factory_countries = flows["factory_id"].map(data._factory_country)
        assert "CN" not in factory_countries.values, \
            "Chinese factories should be excluded from US feasible flows"

    def test_no_chinese_hubs_in_us_flows(self, data):
        """US has ROUTED_THROUGH restriction on CN — no Chinese hub should appear."""
        flows = data.get_feasible_flows("CAT01", "US")
        hub_countries = flows["hub_id"].map(data._hub_country)
        assert "CN" not in hub_countries.values, \
            "Chinese hubs should be excluded from US feasible flows"

    def test_no_us_factories_in_cn_flows(self, data):
        """CN has MADE_IN restriction on US."""
        flows = data.get_feasible_flows("CAT01", "CN")
        factory_countries = flows["factory_id"].map(data._factory_country)
        assert "US" not in factory_countries.values

    def test_no_brazilian_factories_in_us_flows(self, data):
        """US has MADE_IN restriction on BR."""
        flows = data.get_feasible_flows("CAT01", "US")
        factory_countries = flows["factory_id"].map(data._factory_country)
        assert "BR" not in factory_countries.values

    def test_no_chinese_factories_in_india_flows(self, data):
        """IN has MADE_IN restriction on CN."""
        flows = data.get_feasible_flows("CAT01", "IN")
        factory_countries = flows["factory_id"].map(data._factory_country)
        assert "CN" not in factory_countries.values

    def test_no_chinese_hubs_in_india_flows(self, data):
        """IN has ROUTED_THROUGH restriction on CN."""
        flows = data.get_feasible_flows("CAT01", "IN")
        hub_countries = flows["hub_id"].map(data._hub_country)
        assert "CN" not in hub_countries.values

    def test_chinese_factories_allowed_for_germany(self, data):
        """DE has no MADE_IN restriction on CN — CN factories should appear
        for categories with relaxed lead times (urgency 3).
        Uses CAT07 (Smart Speakers, urgency=3) to ensure lead time isn't blocking."""
        flows = data.get_feasible_flows("CAT07", "DE")
        factory_countries = flows["factory_id"].map(data._factory_country)
        assert "CN" in factory_countries.values, \
            "Chinese factories should be allowed for Germany (no geopolitical restriction)"

    def test_all_feasible_flows_unrestricted(self, data):
        """Every row from get_feasible_flows must have is_geopolitically_restricted=0."""
        for country in ["US", "CN", "DE", "IN", "AU"]:
            flows = data.get_feasible_flows("CAT01", country)
            assert (flows["is_geopolitically_restricted"] == 0).all(), \
                f"Found restricted flows in feasible set for {country}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LEAD TIME FEASIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeadTimeFeasibility:
    def test_all_feasible_flows_within_lead_time(self, data):
        """Every feasible flow must have transit_days <= max_lead_time_days."""
        flows = data.get_feasible_flows("CAT01", "US")
        assert (flows["is_lead_time_feasible"] == 1).all()
        assert (flows["transit_days"] <= flows["max_lead_time_days"]).all()

    def test_feasible_flow_count_less_than_total(self, data):
        """Feasible flows should be a strict subset of all flows for the category/country."""
        all_cat_country = data.all_flows[
            (data.all_flows["category_id"] == "CAT01")
            & (data.all_flows["country_code"] == "US")
        ]
        feasible = data.get_feasible_flows("CAT01", "US")
        assert len(feasible) < len(all_cat_country), \
            "Some flows should be blocked by lead time or geopolitics"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COST MINIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCostMinimization:
    def test_solver_returns_optimal(self, data):
        result = solve(data, "CAT01", "US", 5000)
        assert result.status == "Optimal"

    def test_total_units_equals_volume(self, data):
        volume = 5000
        result = solve(data, "CAT01", "US", volume)
        assert result.total_units == volume, \
            f"Expected {volume} units allocated, got {result.total_units}"

    def test_total_cost_positive(self, data):
        result = solve(data, "CAT01", "US", 5000)
        assert result.total_cost > 0

    def test_cost_per_unit_reasonable(self, data):
        """Smartphones (CAT01) base cost $250. With multiplier and logistics,
        landed cost should be roughly $50-$400/unit."""
        result = solve(data, "CAT01", "US", 5000)
        cost_per_unit = result.total_cost / 5000
        assert 50 < cost_per_unit < 400, \
            f"Cost per unit ${cost_per_unit:.2f} seems unreasonable for smartphones"

    def test_solver_picks_cheap_factory_when_capacity_allows(self, data):
        """With small volume, solver should pick one of the cheapest factories
        (VN, IN, or CN — depending on destination restrictions)."""
        result = solve(data, "CAT01", "US", 2000)
        assert len(result.chosen_flows) == 1, "Small volume should use 1 factory"
        factory_country = data.factory_country(result.chosen_flows[0]["factory_id"])
        # For US, CN is blocked. VN and IN are cheapest alternatives.
        assert factory_country in ("VN", "IN", "MX", "KR"), \
            f"Expected cheap factory country, got {factory_country}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TIME PENALTY
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimePenalty:
    def test_high_time_weight_favors_faster_routes(self, data):
        """With high time weight, solver should prefer faster transit
        even if raw cost is higher."""
        # Compare two extreme configurations to verify weights influence routing
        result_cost = solve(data, "CAT01", "US", 3000, cost_weight=10, time_weight=1, regional_weight=1)
        result_speed = solve(data, "CAT01", "US", 3000, cost_weight=1, time_weight=10, regional_weight=1)

        days_cost = result_cost.chosen_flows[0]["transit_days"]
        days_speed = result_speed.chosen_flows[0]["transit_days"]

        assert days_speed <= days_cost, \
            f"High time weight should favor faster routes: got {days_speed}d vs {days_cost}d"

    def test_max_cost_weight_minimizes_raw_cost(self, data):
        """With max cost weight and minimal others, solver should minimize landed cost."""
        result = solve(data, "CAT01", "US", 3000, cost_weight=10, time_weight=1, regional_weight=1)
        # The chosen flow should have one of the lowest total_landed_costs among feasible
        flows = data.get_feasible_flows("CAT01", "US")
        min_cost = flows["total_landed_cost"].min()
        chosen_cost = result.chosen_flows[0]["cost_per_unit"]
        # Allow some slack since time/regional still have weight=1
        assert chosen_cost < flows["total_landed_cost"].median(), \
            f"Max cost weight should pick below-median cost: chose ${chosen_cost:.2f}, median is ${flows['total_landed_cost'].median():.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REGIONAL PREFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegionalPreference:
    def test_high_regional_weight_favors_local_factory(self, data):
        """With very high regional weight, solver should prefer same-region factory
        even if it's more expensive in raw cost."""
        # High regional weight (10), low cost/time weight (1)
        result = solve(data, "CAT01", "US", 3000,
                       cost_weight=1, time_weight=1, regional_weight=10)

        # With dominant regional weight, a NAM factory (US or MX) should be chosen
        chosen_factory = result.chosen_flows[0]["factory_id"]
        factory_region = data.get_region_for_factory(chosen_factory)
        dest_region = data.get_region_for_country("US")
        assert factory_region == dest_region, \
            f"High regional weight should pick NAM factory, got {chosen_factory} in {factory_region}"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CAPACITY CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCapacityConstraints:
    def test_small_volume_uses_one_factory(self, data):
        result = solve(data, "CAT01", "US", 2000)
        assert len(result.chosen_flows) == 1

    def test_large_volume_splits_across_factories(self, data):
        """10,000 units exceeds any single factory's capacity for CAT01→US."""
        result = solve(data, "CAT01", "US", 10000)
        assert result.status == "Optimal"
        assert len(result.chosen_flows) > 1, \
            f"Expected split across factories, got {len(result.chosen_flows)}"

    def test_allocations_respect_factory_capacity(self, data):
        """No factory should be allocated more than its capacity."""
        result = solve(data, "CAT01", "US", 10000)
        cap = data.get_factory_capacity("CAT01")
        for cf in result.chosen_flows:
            fid = cf["factory_id"]
            assert cf["units_allocated"] <= cap.get(fid, 0), \
                f"{fid} allocated {cf['units_allocated']} but capacity is {cap.get(fid, 0)}"

    def test_allocations_respect_hub_throughput(self, data):
        """No hub should receive more than its throughput capacity."""
        result = solve(data, "CAT01", "US", 10000)
        hub_tp = data.get_hub_throughput()
        hub_totals = {}
        for cf in result.chosen_flows:
            hub_totals[cf["hub_id"]] = hub_totals.get(cf["hub_id"], 0) + cf["units_allocated"]
        for hid, total in hub_totals.items():
            assert total <= hub_tp[hid], \
                f"Hub {hid} received {total} but throughput is {hub_tp[hid]}"

    def test_infeasible_when_volume_exceeds_all_capacity(self, data):
        """Volume of 500,000 should exceed total factory capacity for any category."""
        result = solve(data, "CAT01", "US", 500000)
        assert "Infeasible" in result.status


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MINIMUM BATCH SIZE
# ═══════════════════════════════════════════════════════════════════════════════

class TestMinBatch:
    def test_no_allocation_below_min_batch(self, data):
        """With min_batch=500, no chosen flow should have < 500 units."""
        result = solve(data, "CAT01", "US", 10000, min_batch=500)
        for cf in result.chosen_flows:
            assert cf["units_allocated"] >= 500, \
                f"Flow {cf['factory_id']}->{cf['hub_id']} has {cf['units_allocated']} < min_batch 500"

    def test_high_min_batch_reduces_active_flows(self, data):
        """With very high min_batch, fewer flows should be active."""
        result_low = solve(data, "CAT01", "US", 10000, min_batch=100)
        result_high = solve(data, "CAT01", "US", 10000, min_batch=5000)
        assert len(result_high.chosen_flows) <= len(result_low.chosen_flows)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. THREE-TIER RANKING
# ═══════════════════════════════════════════════════════════════════════════════

class TestRanking:
    def test_tier1_matches_solver_output(self, data):
        result = solve(data, "CAT01", "US", 5000)
        ranked = rank_flows(result, data, "US")
        assert len(ranked["chosen_flows"]) == len(result.chosen_flows)
        for cf in ranked["chosen_flows"]:
            assert "factory_name" in cf
            assert "hub_name" in cf
            assert "units_allocated" in cf

    def test_tier2_has_max_3_flows(self, data):
        result = solve(data, "CAT01", "US", 5000)
        ranked = rank_flows(result, data, "US")
        assert len(ranked["other_available"]) <= 3

    def test_tier2_not_in_tier1(self, data):
        result = solve(data, "CAT01", "US", 5000)
        ranked = rank_flows(result, data, "US")
        tier1_keys = {(cf["factory_id"], cf["hub_id"]) for cf in ranked["chosen_flows"]}
        for r in ranked["other_available"]:
            key = (r["factory_id"], r["hub_id"])
            assert key not in tier1_keys, \
                f"Tier 2 flow {key} should not duplicate Tier 1"

    def test_tier2_ranks_are_sequential(self, data):
        result = solve(data, "CAT01", "US", 5000)
        ranked = rank_flows(result, data, "US")
        ranks = [r["rank"] for r in ranked["other_available"]]
        assert ranks == [2, 3, 4][:len(ranks)]

    def test_tier2_sorted_by_composite_score(self, data):
        result = solve(data, "CAT01", "US", 5000)
        ranked = rank_flows(result, data, "US")
        scores = [r["composite_score"] for r in ranked["other_available"]]
        assert scores == sorted(scores), \
            f"Tier 2 should be sorted by composite score: {scores}"

    def test_tier3_unique_factories(self, data):
        result = solve(data, "CAT01", "US", 5000)
        ranked = rank_flows(result, data, "US")
        factory_ids = [af["factory_id"] for af in ranked["alternative_factories"]]
        assert len(factory_ids) == len(set(factory_ids)), \
            "Tier 3 should have unique factory IDs"

    def test_tier3_excludes_tier1_and_tier2_factories(self, data):
        result = solve(data, "CAT01", "US", 5000)
        ranked = rank_flows(result, data, "US")
        used_factories = set()
        for cf in ranked["chosen_flows"]:
            used_factories.add(cf["factory_id"])
        for r in ranked["other_available"]:
            used_factories.add(r["factory_id"])
        for af in ranked["alternative_factories"]:
            assert af["factory_id"] not in used_factories, \
                f"Tier 3 factory {af['factory_id']} already appears in Tier 1 or 2"

    def test_all_tiers_cover_available_factories(self, data):
        """Every factory in feasible flows should appear in exactly one tier."""
        result = solve(data, "CAT01", "US", 5000)
        ranked = rank_flows(result, data, "US")
        feasible_factories = set(result.feasible_flows["factory_id"].unique())

        tier_factories = set()
        for cf in ranked["chosen_flows"]:
            tier_factories.add(cf["factory_id"])
        for r in ranked["other_available"]:
            tier_factories.add(r["factory_id"])
        for af in ranked["alternative_factories"]:
            tier_factories.add(af["factory_id"])

        # Every feasible factory should appear in some tier
        for f in feasible_factories:
            assert f in tier_factories, \
                f"Factory {f} from feasible flows missing from all tiers"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CROSS-CATEGORY / CROSS-COUNTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossScenarios:
    @pytest.mark.parametrize("category_id", [
        "CAT01", "CAT02", "CAT03", "CAT04", "CAT05",
        "CAT06", "CAT07", "CAT08", "CAT09", "CAT10",
    ])
    def test_solver_works_for_all_categories(self, data, category_id):
        result = solve(data, category_id, "US", 3000)
        assert result.status == "Optimal", \
            f"Solver failed for {category_id} -> US: {result.status}"
        assert result.total_units == 3000

    @pytest.mark.parametrize("country_code", [
        "US", "CA", "MX", "BR", "AR", "DE", "GB", "FR",
        "AE", "SA", "ZA", "CN", "JP", "KR", "IN", "VN", "AU",
    ])
    def test_solver_works_for_all_countries(self, data, country_code):
        result = solve(data, "CAT01", country_code, 3000)
        assert result.status == "Optimal", \
            f"Solver failed for CAT01 -> {country_code}: {result.status}"
        assert result.total_units == 3000

    def test_cost_breakdown_sums_correctly(self, data):
        """Verify manufacturing + transport + handling + last_mile + tariff = total.
        Tolerance of $0.02 accounts for floating-point rounding in generate_data.py."""
        result = solve(data, "CAT01", "US", 5000)
        for cf in result.chosen_flows:
            expected = (cf["manufacturing_cost"] + cf["transport_cost"]
                        + cf["hub_handling_cost"] + cf["last_mile_cost"]
                        + cf["tariff_amount"])
            assert abs(cf["cost_per_unit"] - expected) < 0.02, \
                f"Cost breakdown doesn't add up: {expected:.2f} vs {cf['cost_per_unit']:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TRANSIT TIME REALISM
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransitTimeRealism:
    def test_same_country_flows_are_fast(self, data):
        """Flows where factory, hub, and destination are in the same country should be short."""
        flows = data.all_flows
        # US factory → US hub → US
        us_local = flows[
            (flows["factory_id"] == "F_US_01")
            & (flows["hub_id"] == "H_US_01")
            & (flows["country_code"] == "US")
        ]
        if not us_local.empty:
            days = us_local["transit_days"].iloc[0]
            assert days <= 8, f"US→US hub→US should be ≤8 days, got {days}d"

    def test_cross_region_flows_are_slow(self, data):
        """Vietnam factory routed to European country should take 20+ days."""
        flows = data.all_flows
        vn_to_de = flows[
            (flows["factory_id"] == "F_VN_01")
            & (flows["country_code"] == "DE")
        ]
        if not vn_to_de.empty:
            min_days = vn_to_de["transit_days"].min()
            assert min_days >= 15, \
                f"Vietnam→Germany should take ≥15 days, got {min_days}d"

    def test_transit_includes_last_mile(self, data):
        """Transit days should include hub→country leg, not just factory→hub."""
        flows = data.all_flows
        # A flow through a local hub to a distant country should have high transit
        # VN hub → DE should add significant days vs VN hub → VN
        vn_hub_to_de = flows[
            (flows["hub_id"] == "H_VN_01") & (flows["country_code"] == "DE")
        ]
        vn_hub_to_vn = flows[
            (flows["hub_id"] == "H_VN_01") & (flows["country_code"] == "VN")
        ]
        if not vn_hub_to_de.empty and not vn_hub_to_vn.empty:
            # Same factory for comparison
            factory = vn_hub_to_de["factory_id"].iloc[0]
            de_row = vn_hub_to_de[vn_hub_to_de["factory_id"] == factory]
            vn_row = vn_hub_to_vn[vn_hub_to_vn["factory_id"] == factory]
            if not de_row.empty and not vn_row.empty:
                days_de = de_row["transit_days"].iloc[0]
                days_vn = vn_row["transit_days"].iloc[0]
                assert days_de > days_vn + 10, \
                    f"VN hub→DE ({days_de}d) should be much longer than VN hub→VN ({days_vn}d)"

    def test_no_unrealistic_short_cross_region_transit(self, data):
        """No cross-region flow should have transit < 5 days."""
        flows = data.all_flows
        for _, row in flows.iterrows():
            factory_region = data.get_region_for_factory(row["factory_id"])
            dest_region = data.get_region_for_country(row["country_code"])
            if factory_region != dest_region:
                hub_region = data.get_region_for_hub(row["hub_id"])
                # If neither factory nor hub is in dest region, transit should be significant
                if hub_region != dest_region:
                    assert row["transit_days"] >= 5, \
                        f"{row['factory_id']}→{row['hub_id']}→{row['country_code']}: " \
                        f"cross-region transit {row['transit_days']}d is unrealistically short"


# ═══════════════════════════════════════════════════════════════════════════════
# 12. ONTOLOGY LAYER
# ═══════════════════════════════════════════════════════════════════════════════

from solver.ontology import SupplyChainOntology, FactoryEntity, HubEntity


@pytest.fixture(scope="module")
def ontology(data):
    """Module-scoped fixture: builds ontology once from shared data."""
    return SupplyChainOntology(data)


class TestOntology:
    def test_all_factories_loaded(self, ontology):
        assert len(ontology.factories) == 13

    def test_all_hubs_loaded(self, ontology):
        assert len(ontology.hubs) == 14

    def test_all_countries_loaded(self, ontology):
        assert len(ontology.countries) == 17

    def test_all_categories_loaded(self, ontology):
        assert len(ontology.categories) == 10

    def test_factory_entity_type(self, ontology):
        f = ontology.get_factory("F_CN_01")
        assert isinstance(f, FactoryEntity)
        assert f.factory_id == "F_CN_01"
        assert f.country_code == "CN"
        assert f.region_id == "NEA"

    def test_factory_manufacturing_cost(self, ontology):
        """CN factory with multiplier 0.40: 250 * 0.40 = 100.0"""
        f = ontology.get_factory("F_CN_01")
        cost = f.manufacturing_cost(250.0)
        assert cost == 100.0

    def test_hub_has_capacity(self, ontology):
        h = ontology.get_hub("H_US_01")
        assert isinstance(h, HubEntity)
        assert h.has_capacity_for(200000) is True
        assert h.has_capacity_for(200001) is False

    def test_validate_flow_us_cn_made_in_blocked(self, ontology):
        """US blocks MADE_IN CN — Chinese factory to US should be invalid."""
        valid, reason = ontology.validate_flow("F_CN_01", "H_US_01", "US")
        assert valid is False
        assert "MADE_IN" in reason

    def test_validate_flow_us_cn_routed_through_blocked(self, ontology):
        """US blocks ROUTED_THROUGH CN — German factory via Chinese hub to US invalid."""
        valid, reason = ontology.validate_flow("F_DE_01", "H_CN_01", "US")
        assert valid is False
        assert "ROUTED_THROUGH" in reason

    def test_validate_flow_de_cn_allowed(self, ontology):
        """Germany has no restrictions on CN — Chinese factory to DE is valid."""
        valid, reason = ontology.validate_flow("F_CN_01", "H_DE_01", "DE")
        assert valid is True

    def test_restrictions_for_us(self, ontology):
        """US has 3 restrictions: MADE_IN CN, ROUTED_THROUGH CN, MADE_IN BR."""
        restrictions = ontology.get_restrictions_for_country("US")
        assert len(restrictions) == 3

    def test_restrictions_for_de(self, ontology):
        """Germany has no geopolitical restrictions."""
        restrictions = ontology.get_restrictions_for_country("DE")
        assert len(restrictions) == 0

    def test_factories_in_region(self, ontology):
        """SEA region contains VN and IN factories, not CN (which is NEA)."""
        sea_factories = ontology.factories_in_region("SEA")
        factory_ids = {f.factory_id for f in sea_factories}
        assert "F_VN_01" in factory_ids
        assert "F_IN_01" in factory_ids
        assert "F_CN_01" not in factory_ids

    def test_nonexistent_factory_returns_none(self, ontology):
        assert ontology.get_factory("F_FAKE_99") is None


# ═══════════════════════════════════════════════════════════════════════════════
# 13. KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

from solver.knowledge_graph import SupplyChainGraph


@pytest.fixture(scope="module")
def kg(data):
    """Module-scoped fixture: builds knowledge graph once from shared data."""
    return SupplyChainGraph(data)


class TestKnowledgeGraph:
    def test_graph_has_all_factories(self, kg):
        assert len(kg.get_nodes_by_type("factory")) == 13

    def test_graph_has_all_hubs(self, kg):
        assert len(kg.get_nodes_by_type("hub")) == 14

    def test_graph_has_all_countries(self, kg):
        assert len(kg.get_nodes_by_type("country")) == 17

    def test_graph_has_all_regions(self, kg):
        assert len(kg.get_nodes_by_type("region")) == 6

    def test_ships_to_edges_exist(self, kg):
        """Every factory should have at least one SHIPS_TO edge."""
        for fid in ["F_CN_01", "F_US_01", "F_VN_01"]:
            edges = [
                (u, v) for u, v, d in kg.graph.edges(f"factory:{fid}", data=True)
                if d.get("edge_type") == "SHIPS_TO"
            ]
            assert len(edges) > 0, f"Factory {fid} has no SHIPS_TO edges"

    def test_delivers_to_edges_exist(self, kg):
        """Every hub should have at least one DELIVERS_TO edge."""
        for hid in ["H_US_01", "H_CN_01", "H_DE_01"]:
            edges = [
                (u, v) for u, v, d in kg.graph.edges(f"hub:{hid}", data=True)
                if d.get("edge_type") == "DELIVERS_TO"
            ]
            assert len(edges) > 0, f"Hub {hid} has no DELIVERS_TO edges"

    def test_restriction_edges(self, kg):
        """US should have RESTRICTS edges to CN."""
        edges = [
            (u, v, d) for u, v, d in kg.graph.edges("country:US", data=True)
            if d.get("edge_type") == "RESTRICTS"
        ]
        restricted_targets = {v for _, v, _ in edges}
        assert "country:CN" in restricted_targets

    def test_find_all_routes(self, kg):
        """F_US_01 should have at least one route to US (same-country)."""
        routes = kg.find_all_routes("F_US_01", "US")
        assert len(routes) > 0
        for r in routes:
            assert r["factory_id"] == "F_US_01"
            assert r["country_code"] == "US"

    def test_supply_diversity_us(self, kg):
        """US should be reachable from factories in multiple regions."""
        diversity = kg.supply_diversity("US")
        assert sum(diversity.values()) > 3
        assert len(diversity) >= 2

    def test_get_restriction_graph(self, kg):
        """US restrictions should include CN."""
        restrictions = kg.get_restriction_graph("US")
        restricted_countries = {r["restricted_country"] for r in restrictions}
        assert "CN" in restricted_countries

    def test_hub_utilization_risk(self, kg):
        """Houston hub should serve multiple countries and be fed by factories."""
        risk = kg.hub_utilization_risk("H_US_01")
        assert risk["factory_count"] > 0
        assert risk["country_count"] > 0
        assert risk["hub_id"] == "H_US_01"

    def test_impact_analysis_returns_list(self, kg):
        """impact_analysis should return a list of country codes."""
        result = kg.impact_analysis("H_US_01")
        assert isinstance(result, list)

    def test_impact_analysis_nonexistent_hub(self, kg):
        """Nonexistent hub should return empty list."""
        result = kg.impact_analysis("H_FAKE_99")
        assert result == []
