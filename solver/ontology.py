"""
Ontology Layer — typed domain entities wrapping SupplyChainData.

Provides frozen dataclasses for each entity type (Factory, Hub, Country,
Category) with business logic methods, plus a SupplyChainOntology class
that builds typed lookups and offers semantic query methods like
validate_flow() and factories_in_region().

This layer does NOT replace data_loader.py — the MILP solver still uses
DataFrames directly. The ontology adds type safety, domain methods, and
validation on top of the raw data.
"""

from dataclasses import dataclass
from typing import Optional

from solver.data_loader import SupplyChainData


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FactoryEntity:
    """A manufacturing facility with location and cost characteristics."""
    factory_id: str
    factory_name: str
    city: str
    country_code: str
    region_id: str
    cost_multiplier: float

    def manufacturing_cost(self, base_cost: float) -> float:
        """Compute unit manufacturing cost: base_cost * cost_multiplier."""
        return round(base_cost * self.cost_multiplier, 2)

    def is_in_region(self, region_id: str) -> bool:
        return self.region_id == region_id


@dataclass(frozen=True)
class HubEntity:
    """A distribution hub with throughput capacity."""
    hub_id: str
    hub_name: str
    city: str
    country_code: str
    region_id: str
    monthly_throughput_capacity: int

    def has_capacity_for(self, units: int) -> bool:
        """Check if this hub can handle the given volume."""
        return units <= self.monthly_throughput_capacity

    def is_in_region(self, region_id: str) -> bool:
        return self.region_id == region_id


@dataclass(frozen=True)
class CountryEntity:
    """A destination country within a region."""
    country_code: str
    country_name: str
    region_id: str


@dataclass(frozen=True)
class CategoryEntity:
    """A product category with base cost and weight."""
    category_id: str
    category_name: str
    base_manufacturing_cost_usd: float
    representative_weight_kg: float


@dataclass(frozen=True)
class GeopoliticalRestriction:
    """A trade restriction rule: destination blocks a restricted country."""
    destination_country_code: str
    restricted_country_code: str
    restriction_type: str   # "MADE_IN" or "ROUTED_THROUGH"
    reason: str


# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class SupplyChainOntology:
    """Typed domain layer wrapping SupplyChainData's raw DataFrames.

    Provides entity lookups with proper types, business logic validation,
    and semantic query methods. Does not modify or replace the underlying
    data_loader — the solver continues to use DataFrames directly.
    """

    def __init__(self, data: SupplyChainData):
        self._data = data
        self._build_entities()

    def _build_entities(self):
        """Convert DataFrames into typed entity dictionaries."""

        # ── Factories ──
        self.factories: dict[str, FactoryEntity] = {}
        for _, row in self._data.factories.iterrows():
            self.factories[row["factory_id"]] = FactoryEntity(
                factory_id=row["factory_id"],
                factory_name=row["factory_name"],
                city=row["city"],
                country_code=row["country_code"],
                region_id=row["region_id"],
                cost_multiplier=row["cost_multiplier"],
            )

        # ── Hubs ──
        self.hubs: dict[str, HubEntity] = {}
        for _, row in self._data.hubs.iterrows():
            self.hubs[row["hub_id"]] = HubEntity(
                hub_id=row["hub_id"],
                hub_name=row["hub_name"],
                city=row["city"],
                country_code=row["country_code"],
                region_id=row["region_id"],
                monthly_throughput_capacity=int(row["monthly_throughput_capacity"]),
            )

        # ── Countries ──
        self.countries: dict[str, CountryEntity] = {}
        for _, row in self._data.countries.iterrows():
            self.countries[row["country_code"]] = CountryEntity(
                country_code=row["country_code"],
                country_name=row["country_name"],
                region_id=row["region_id"],
            )

        # ── Categories ──
        self.categories: dict[str, CategoryEntity] = {}
        for _, row in self._data.categories.iterrows():
            self.categories[row["category_id"]] = CategoryEntity(
                category_id=row["category_id"],
                category_name=row["category_name"],
                base_manufacturing_cost_usd=row["base_manufacturing_cost_usd"],
                representative_weight_kg=row["representative_weight_kg"],
            )

        # ── Geopolitical Restrictions ──
        self.restrictions: list[GeopoliticalRestriction] = []
        for _, row in self._data.geopolitical.iterrows():
            self.restrictions.append(GeopoliticalRestriction(
                destination_country_code=row["destination_country_code"],
                restricted_country_code=row["restricted_country_code"],
                restriction_type=row["restriction_type"],
                reason=row["reason"],
            ))

    # ── Entity Lookups ─────────────────────────────────────────────────────────

    def get_factory(self, factory_id: str) -> Optional[FactoryEntity]:
        """Return a FactoryEntity by ID, or None if not found."""
        return self.factories.get(factory_id)

    def get_hub(self, hub_id: str) -> Optional[HubEntity]:
        """Return a HubEntity by ID, or None if not found."""
        return self.hubs.get(hub_id)

    def get_country(self, country_code: str) -> Optional[CountryEntity]:
        """Return a CountryEntity by code, or None if not found."""
        return self.countries.get(country_code)

    def get_category(self, category_id: str) -> Optional[CategoryEntity]:
        """Return a CategoryEntity by ID, or None if not found."""
        return self.categories.get(category_id)

    # ── Semantic Queries ───────────────────────────────────────────────────────

    def get_restrictions_for_country(self, country_code: str) -> list[GeopoliticalRestriction]:
        """Return all geopolitical restrictions targeting this destination."""
        return [r for r in self.restrictions
                if r.destination_country_code == country_code]

    def validate_flow(
        self, factory_id: str, hub_id: str, country_code: str
    ) -> tuple[bool, str]:
        """Validate a factory->hub->country route against geopolitical rules.

        Returns (is_valid, reason). Checks both MADE_IN restrictions
        (factory country blocked) and ROUTED_THROUGH restrictions
        (hub country blocked) for the destination.
        """
        factory = self.factories.get(factory_id)
        hub = self.hubs.get(hub_id)
        if not factory or not hub:
            return (False, f"Unknown factory {factory_id} or hub {hub_id}")

        for r in self.get_restrictions_for_country(country_code):
            if (r.restriction_type == "MADE_IN"
                    and factory.country_code == r.restricted_country_code):
                return (False,
                        f"MADE_IN restriction: {country_code} blocks products "
                        f"made in {r.restricted_country_code} ({r.reason})")
            if (r.restriction_type == "ROUTED_THROUGH"
                    and hub.country_code == r.restricted_country_code):
                return (False,
                        f"ROUTED_THROUGH restriction: {country_code} blocks "
                        f"routing via {r.restricted_country_code} ({r.reason})")

        return (True, "Flow is valid")

    def factories_in_region(self, region_id: str) -> list[FactoryEntity]:
        """Return all factories located in the given region."""
        return [f for f in self.factories.values()
                if f.region_id == region_id]

    def hubs_in_region(self, region_id: str) -> list[HubEntity]:
        """Return all hubs located in the given region."""
        return [h for h in self.hubs.values()
                if h.region_id == region_id]
