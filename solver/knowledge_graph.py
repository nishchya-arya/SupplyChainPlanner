"""
Knowledge Graph — NetworkX DiGraph of the supply chain network.

Represents factories, hubs, countries, and regions as typed nodes with
edges for shipping routes (SHIPS_TO), delivery routes (DELIVERS_TO),
geographic membership (IN_REGION), and trade restrictions (RESTRICTS).

Enables graph-based queries that are awkward with flat DataFrames:
  - Impact analysis: "Which countries lose supply if hub X fails?"
  - Route enumeration: "All paths from factory X to country Y"
  - Supply diversity: "How many factories per region serve country Z?"
  - Restriction mapping: "What origins are blocked for this destination?"

Node IDs use type prefixes (factory:, hub:, country:, region:) to avoid
collisions between entity types that share similar codes.
"""

import networkx as nx

from solver.data_loader import SupplyChainData


class SupplyChainGraph:
    """NetworkX DiGraph representing the supply chain network topology."""

    def __init__(self, data: SupplyChainData):
        self._data = data
        self.graph = nx.DiGraph()
        self._build()

    def _build(self):
        """Construct the graph from SupplyChainData DataFrames."""
        g = self.graph

        # ── Region nodes ──────────────────────────────────────────────────
        for _, row in self._data.regions.iterrows():
            g.add_node(
                f"region:{row['region_id']}",
                node_type="region",
                name=row["region_name"],
                region_id=row["region_id"],
            )

        # ── Country nodes + IN_REGION edges ───────────────────────────────
        for _, row in self._data.countries.iterrows():
            cc = row["country_code"]
            g.add_node(
                f"country:{cc}",
                node_type="country",
                name=row["country_name"],
                country_code=cc,
                region_id=row["region_id"],
            )
            g.add_edge(
                f"country:{cc}", f"region:{row['region_id']}",
                edge_type="IN_REGION",
            )

        # ── Factory nodes + IN_REGION edges ───────────────────────────────
        for _, row in self._data.factories.iterrows():
            fid = row["factory_id"]
            g.add_node(
                f"factory:{fid}",
                node_type="factory",
                factory_id=fid,
                name=row["factory_name"],
                city=row["city"],
                country_code=row["country_code"],
                region_id=row["region_id"],
                cost_multiplier=row["cost_multiplier"],
            )
            g.add_edge(
                f"factory:{fid}", f"region:{row['region_id']}",
                edge_type="IN_REGION",
            )

        # ── Hub nodes + IN_REGION edges ───────────────────────────────────
        for _, row in self._data.hubs.iterrows():
            hid = row["hub_id"]
            g.add_node(
                f"hub:{hid}",
                node_type="hub",
                hub_id=hid,
                name=row["hub_name"],
                city=row["city"],
                country_code=row["country_code"],
                region_id=row["region_id"],
                monthly_throughput_capacity=int(row["monthly_throughput_capacity"]),
            )
            g.add_edge(
                f"hub:{hid}", f"region:{row['region_id']}",
                edge_type="IN_REGION",
            )

        # ── SHIPS_TO edges (factory → hub) ────────────────────────────────
        # Aggregate from all_flows: one edge per unique (factory, hub) pair
        # with the minimum transport cost and transit days across categories
        ships_to = self._data.all_flows.groupby(
            ["factory_id", "hub_id"]
        ).agg(
            min_transport_cost=("transport_cost", "min"),
            min_transit_days=("transit_days", "min"),
        ).reset_index()

        for _, row in ships_to.iterrows():
            g.add_edge(
                f"factory:{row['factory_id']}", f"hub:{row['hub_id']}",
                edge_type="SHIPS_TO",
                min_transport_cost=row["min_transport_cost"],
                min_transit_days=int(row["min_transit_days"]),
            )

        # ── DELIVERS_TO edges (hub → country) ────────────────────────────
        # One edge per unique (hub, country) pair with minimum costs
        delivers_to = self._data.all_flows.groupby(
            ["hub_id", "country_code"]
        ).agg(
            min_last_mile_cost=("last_mile_cost", "min"),
            min_transit_days=("transit_days", "min"),
        ).reset_index()

        for _, row in delivers_to.iterrows():
            g.add_edge(
                f"hub:{row['hub_id']}", f"country:{row['country_code']}",
                edge_type="DELIVERS_TO",
                min_last_mile_cost=row["min_last_mile_cost"],
                min_transit_days=int(row["min_transit_days"]),
            )

        # ── RESTRICTS edges (country → country) ──────────────────────────
        for _, row in self._data.geopolitical.iterrows():
            g.add_edge(
                f"country:{row['destination_country_code']}",
                f"country:{row['restricted_country_code']}",
                edge_type="RESTRICTS",
                restriction_type=row["restriction_type"],
                reason=row["reason"],
            )

    # ═══════════════════════════════════════════════════════════════════════
    # QUERY METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def get_nodes_by_type(self, node_type: str) -> list[str]:
        """Return node IDs of the given type (factory, hub, country, region)."""
        return [n for n, d in self.graph.nodes(data=True)
                if d.get("node_type") == node_type]

    def impact_analysis(self, hub_id: str) -> list[str]:
        """Which countries lose ALL supply routes if this hub is disabled?

        Returns country codes that are served ONLY by this hub (no
        alternative hub delivers to them). In practice, most countries
        have multiple hubs, so this list is usually short or empty.
        """
        hub_node = f"hub:{hub_id}"
        if hub_node not in self.graph:
            return []

        # Countries served by this hub
        served = set()
        for _, target, attrs in self.graph.edges(hub_node, data=True):
            if attrs.get("edge_type") == "DELIVERS_TO" and target.startswith("country:"):
                served.add(target)

        # For each served country, check if other hubs also deliver to it
        solely_dependent = []
        for country_node in served:
            serving_hubs = [
                source for source, _, attrs
                in self.graph.in_edges(country_node, data=True)
                if attrs.get("edge_type") == "DELIVERS_TO"
            ]
            if len(serving_hubs) == 1 and serving_hubs[0] == hub_node:
                solely_dependent.append(country_node.replace("country:", ""))

        return solely_dependent

    def find_all_routes(self, factory_id: str, country_code: str) -> list[dict]:
        """Find all factory→hub→country paths for a given origin and destination.

        Returns list of dicts: {factory_id, hub_id, country_code,
        transport_cost, last_mile_cost}.
        """
        factory_node = f"factory:{factory_id}"
        country_node = f"country:{country_code}"
        routes = []

        if factory_node not in self.graph:
            return routes

        # 2-hop paths: factory → hub → country
        for _, hub_node, ship_attrs in self.graph.edges(factory_node, data=True):
            if ship_attrs.get("edge_type") != "SHIPS_TO":
                continue
            for _, dest_node, deliver_attrs in self.graph.edges(hub_node, data=True):
                if deliver_attrs.get("edge_type") != "DELIVERS_TO":
                    continue
                if dest_node == country_node:
                    routes.append({
                        "factory_id": factory_id,
                        "hub_id": hub_node.replace("hub:", ""),
                        "country_code": country_code,
                        "transport_cost": ship_attrs.get("min_transport_cost"),
                        "last_mile_cost": deliver_attrs.get("min_last_mile_cost"),
                    })

        return routes

    def supply_diversity(self, country_code: str) -> dict[str, int]:
        """Count factories per region that can ship to this country.

        Returns {region_id: factory_count}. Higher diversity across
        regions means more geographic resilience.
        """
        country_node = f"country:{country_code}"
        factories_by_region: dict[str, set] = {}

        # Walk backwards: country ← hub ← factory
        for hub_node, _, attrs in self.graph.in_edges(country_node, data=True):
            if attrs.get("edge_type") != "DELIVERS_TO":
                continue
            for factory_node, _, ship_attrs in self.graph.in_edges(hub_node, data=True):
                if ship_attrs.get("edge_type") != "SHIPS_TO":
                    continue
                region = self.graph.nodes[factory_node].get("region_id", "unknown")
                if region not in factories_by_region:
                    factories_by_region[region] = set()
                factories_by_region[region].add(factory_node)

        return {region: len(factories)
                for region, factories in factories_by_region.items()}

    def get_restriction_graph(self, country_code: str) -> list[dict]:
        """Get all geopolitical restrictions affecting a destination country.

        Returns list of dicts: {restricted_country, restriction_type, reason}.
        """
        country_node = f"country:{country_code}"
        restrictions = []
        for _, target, attrs in self.graph.edges(country_node, data=True):
            if attrs.get("edge_type") == "RESTRICTS":
                restrictions.append({
                    "restricted_country": target.replace("country:", ""),
                    "restriction_type": attrs["restriction_type"],
                    "reason": attrs["reason"],
                })
        return restrictions

    def hub_utilization_risk(self, hub_id: str) -> dict:
        """Analyze how many factories feed and countries depend on this hub.

        Returns dict: {hub_id, feeding_factories, served_countries,
        factory_count, country_count}.
        """
        hub_node = f"hub:{hub_id}"
        feeding_factories = [
            source.replace("factory:", "")
            for source, _, attrs in self.graph.in_edges(hub_node, data=True)
            if attrs.get("edge_type") == "SHIPS_TO"
        ]
        served_countries = [
            target.replace("country:", "")
            for _, target, attrs in self.graph.edges(hub_node, data=True)
            if attrs.get("edge_type") == "DELIVERS_TO"
        ]
        return {
            "hub_id": hub_id,
            "feeding_factories": feeding_factories,
            "served_countries": served_countries,
            "factory_count": len(feeding_factories),
            "country_count": len(served_countries),
        }
