"""
Load supply chain CSV data and provide lookup methods for the optimizer.

Loads 10 CSV files from data/ into pandas DataFrames, builds dictionary
lookups for fast O(1) access to factory/hub/country metadata, and provides
query methods for feasibility filtering and capacity retrieval.
"""

import os
import pandas as pd


class SupplyChainData:
    """Loads all CSVs from data/ and provides query methods."""

    def __init__(self, data_dir=None):
        if data_dir is None:
            # Default: look for data/ relative to this file's parent directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self._dir = data_dir
        self._load()

    def _load(self):
        # ── Load CSV files ────────────────────────────────────────────────
        self.regions = pd.read_csv(os.path.join(self._dir, "regions.csv"))
        self.countries = pd.read_csv(os.path.join(self._dir, "countries.csv"))
        self.factories = pd.read_csv(os.path.join(self._dir, "factories.csv"))
        self.hubs = pd.read_csv(os.path.join(self._dir, "hubs.csv"))
        self.categories = pd.read_csv(os.path.join(self._dir, "product_categories.csv"))
        self.products = pd.read_csv(os.path.join(self._dir, "products.csv"))
        self.product_availability = pd.read_csv(os.path.join(self._dir, "product_availability.csv"))
        self.factory_capacity = pd.read_csv(os.path.join(self._dir, "factory_category_capacity.csv"))
        self.all_flows = pd.read_csv(os.path.join(self._dir, "all_flows.csv"))
        self.geopolitical = pd.read_csv(os.path.join(self._dir, "geopolitical_restrictions.csv"))

        # ── Build lookup dictionaries ─────────────────────────────────────
        # These provide O(1) access to metadata without DataFrame queries.
        # Used heavily by the optimizer and ranker during scoring.

        # Geographic lookups: map entities to their region for proximity scoring
        self._country_region = dict(zip(self.countries["country_code"], self.countries["region_id"]))
        self._factory_region = dict(zip(self.factories["factory_id"], self.factories["region_id"]))
        self._factory_country = dict(zip(self.factories["factory_id"], self.factories["country_code"]))
        self._hub_region = dict(zip(self.hubs["hub_id"], self.hubs["region_id"]))
        self._hub_country = dict(zip(self.hubs["hub_id"], self.hubs["country_code"]))

        # Display lookups: human-readable names for UI rendering
        self._factory_names = dict(zip(self.factories["factory_id"], self.factories["factory_name"]))
        self._factory_cities = dict(zip(self.factories["factory_id"], self.factories["city"]))
        self._hub_names = dict(zip(self.hubs["hub_id"], self.hubs["hub_name"]))
        self._hub_cities = dict(zip(self.hubs["hub_id"], self.hubs["city"]))

    # ── Query methods ────────────────────────────────────────────────────────

    def get_feasible_flows(self, category_id, country_code):
        """Return DataFrame of flows that pass both filters:
        1. Not geopolitically restricted (no MADE_IN or ROUTED_THROUGH violations)
        2. Within lead-time limits for this country + category urgency level
        Typically returns ~27 flows from ~22K total in all_flows.csv."""
        mask = (
            (self.all_flows["category_id"] == category_id)
            & (self.all_flows["country_code"] == country_code)
            & (self.all_flows["is_geopolitically_restricted"] == 0)
            & (self.all_flows["is_lead_time_feasible"] == 1)
        )
        return self.all_flows[mask].copy().reset_index(drop=True)

    def get_factory_capacity(self, category_id):
        """Return dict: factory_id -> monthly_capacity_units for this category.
        Capacity varies by category because different products require
        different manufacturing processes and line configurations."""
        subset = self.factory_capacity[self.factory_capacity["category_id"] == category_id]
        return dict(zip(subset["factory_id"], subset["monthly_capacity_units"]))

    def get_hub_throughput(self):
        """Return dict: hub_id -> monthly_throughput_capacity.
        Hub throughput is category-independent (it's warehouse space, not mfg)."""
        return dict(zip(self.hubs["hub_id"], self.hubs["monthly_throughput_capacity"]))

    def get_region_for_country(self, country_code):
        return self._country_region.get(country_code)

    def get_region_for_factory(self, factory_id):
        return self._factory_region.get(factory_id)

    def get_region_for_hub(self, hub_id):
        return self._hub_region.get(hub_id)

    def get_countries_in_region(self, region_id):
        """Return list of country_code strings in this region."""
        return self.countries[self.countries["region_id"] == region_id]["country_code"].tolist()

    def get_default_country(self, region_id):
        """Return the primary/largest-demand country per region.
        Hardcoded because demand_scale is only in generate_data.py, not in CSVs."""
        default_map = {
            "NAM": "US", "SAM": "BR", "EUR": "DE",
            "MEA": "AE", "NEA": "CN", "SEA": "IN",
        }
        region_countries = self.get_countries_in_region(region_id)
        return default_map.get(region_id, region_countries[0] if region_countries else None)

    def get_category_list(self):
        """Return list of (category_id, category_name) tuples."""
        return list(zip(self.categories["category_id"], self.categories["category_name"]))

    def get_region_list(self):
        """Return list of (region_id, region_name) tuples."""
        return list(zip(self.regions["region_id"], self.regions["region_name"]))

    # ── Display name lookups (used by ranker for UI-friendly output) ──────

    def factory_name(self, factory_id):
        return self._factory_names.get(factory_id, factory_id)

    def factory_city(self, factory_id):
        return self._factory_cities.get(factory_id, "")

    def factory_country(self, factory_id):
        return self._factory_country.get(factory_id, "")

    def hub_name(self, hub_id):
        return self._hub_names.get(hub_id, hub_id)

    def hub_city(self, hub_id):
        return self._hub_cities.get(hub_id, "")

    def hub_country(self, hub_id):
        return self._hub_country.get(hub_id, "")
