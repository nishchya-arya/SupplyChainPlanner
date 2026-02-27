"""
Knowledge Graph page — interactive network visualization and disruption analysis.

Built on a NetworkX DiGraph (solver/knowledge_graph.py) that models the supply chain
as typed nodes (factory, hub, country, region) connected by edges (SHIPS_TO,
DELIVERS_TO, IN_REGION, RESTRICTS). This page visualizes that graph on Plotly
geographic maps and lets users simulate disruptions.

Tabs:
  1. Network Map — Full Plotly scatter_geo world map with factories (crimson
     triangles, sized by capacity), hubs (blue squares, sized by throughput), and
     countries (green circles). Sidebar country AND category filters control which
     nodes and flow lines are shown. When a category is selected, only factories
     and hubs with feasible flows for that category are displayed.

  2. Impact Analysis — Disruption simulator. Two multi-selects let you disable
     factories and/or hubs (filtered to relevant nodes when a category is active).
     The map updates in real-time:
       - Active nodes keep their normal colors
       - Disabled nodes turn gray
       - Countries losing ALL supply routes turn red
       - Remaining routes drawn for the sidebar-selected country
     Below the map: 4-column impact metrics, status messages, and per-hub details.

Sidebar controls:
  - Country selector — filters flow lines and route visualization
  - Category selector — filters which factories/hubs are shown (based on feasible
    flows after geopolitical + lead-time filtering)

Shared helpers:
  - _factory_size_dynamic() — dynamic marker sizing by capacity (total or per-category)
  - _hub_size() — dynamic marker sizing by throughput
  - _map_layout() — standard Plotly layout reused by both tabs

Data flow: SupplyChainData → SupplyChainGraph → graph queries → Plotly visualization
          SupplyChainData.all_flows → category filtering → relevant factory/hub sets
"""

import streamlit as st
import plotly.graph_objects as go

from solver.data_loader import SupplyChainData     # CSV data loader with lookup methods
from solver.knowledge_graph import SupplyChainGraph  # NetworkX graph with query methods
from solver.coords import FACTORY_COORDS, HUB_COORDS, COUNTRY_COORDS  # shared coordinates


@st.cache_resource
def load_data():
    return SupplyChainData()


@st.cache_resource
def build_graph(_data):
    return SupplyChainGraph(_data)


st.set_page_config(
    page_title="Knowledge Graph — Supply Chain Planner", layout="wide"
)

data = load_data()
kg = build_graph(data)

st.title("Supply Chain Knowledge Graph")
st.caption(
    "Interactive network visualization and graph-based analysis of the "
    "supply chain topology"
)

# ── Shared lookups (computed once, used by both tabs) ────────────────────────
# Country code → name mapping for display
country_names = dict(
    zip(data.countries["country_code"], data.countries["country_name"])
)

# Factory capacity totals (sum across all 10 categories) for dynamic marker sizing.
# factory_category_capacity.csv has per-category rows; we sum to get total per factory.
_fcc = data.factory_capacity.groupby("factory_id")["monthly_capacity_units"].sum()
_fcc_max, _fcc_min = _fcc.max(), _fcc.min()

# Hub throughput dict (hub_id → monthly_throughput_capacity) for dynamic sizing.
_hub_tp = dict(zip(data.hubs["hub_id"], data.hubs["monthly_throughput_capacity"]))
_hub_tp_max, _hub_tp_min = max(_hub_tp.values()), min(_hub_tp.values())

# Factory cost multipliers (factory_id → cost_multiplier) for enriched hover tooltips.
# Lower multiplier = cheaper manufacturing (e.g., Vietnam 0.38x vs Germany 1.05x).
_factory_cost_mult = dict(
    zip(data.factories["factory_id"], data.factories["cost_multiplier"])
)


def _factory_size_dynamic(fid, cap_series, cap_min, cap_max):
    """Scale factory marker size [8, 18] by capacity (total or category-specific)."""
    cap = cap_series.get(fid, cap_min)
    if cap_max == cap_min:
        return 12
    return 8 + 10 * (cap - cap_min) / (cap_max - cap_min)


def _hub_size(hid):
    """Scale hub marker size [7, 15] by throughput."""
    tp = _hub_tp.get(hid, _hub_tp_min)
    if _hub_tp_max == _hub_tp_min:
        return 10
    return 7 + 8 * (tp - _hub_tp_min) / (_hub_tp_max - _hub_tp_min)


def _map_layout():
    """Standard map layout used by both tabs."""
    return dict(
        geo=dict(
            projection_type="natural earth",
            showland=True, landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
            showocean=True, oceancolor="rgb(230, 240, 250)",
            showcountries=True,
        ),
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Explore Routes")
    country_options = ["(All)"] + list(data.countries["country_code"])
    selected_country = st.selectbox(
        "Highlight routes to country",
        country_options,
        format_func=lambda x: (
            f"{x} — {country_names.get(x, x)}" if x != "(All)" else "All (no filter)"
        ),
    )

    st.divider()

    # Category filter — controls which factories/hubs are visible on both tabs.
    # When a category is selected, only nodes participating in feasible flows
    # (after geopolitical + lead-time filtering) for that category are shown.
    st.header("Filter by Category")
    cat_options = data.get_category_list()  # [(cat_id, cat_name), ...]
    cat_labels = ["(All)"] + [f"{cid} — {cname}" for cid, cname in cat_options]
    selected_cat_idx = st.selectbox(
        "Product category",
        range(len(cat_labels)),
        format_func=lambda i: cat_labels[i],
    )
    selected_category = None if selected_cat_idx == 0 else cat_options[selected_cat_idx - 1][0]

# ── Compute category-filtered factory/hub sets ───────────────────────────────
# When a category is selected, determine which factories and hubs participate in
# feasible flows for that category. "Feasible" means not geopolitically restricted
# AND lead-time feasible. This filtering is done on data.all_flows directly because
# the knowledge graph's find_all_routes() is not category-aware.
if selected_category is not None:
    cat_mask = (
        (data.all_flows["category_id"] == selected_category)
        & (data.all_flows["is_geopolitically_restricted"] == 0)
        & (data.all_flows["is_lead_time_feasible"] == 1)
    )
    if selected_country != "(All)":
        cat_mask = cat_mask & (data.all_flows["country_code"] == selected_country)

    cat_flows = data.all_flows[cat_mask]
    relevant_factories = set(cat_flows["factory_id"].unique())
    relevant_hubs = set(cat_flows["hub_id"].unique())

    # Category-specific factory capacity for marker sizing
    _cat_cap = data.factory_capacity[
        data.factory_capacity["category_id"] == selected_category
    ].set_index("factory_id")["monthly_capacity_units"]
    _cat_cap_max = _cat_cap.max() if not _cat_cap.empty else 1
    _cat_cap_min = _cat_cap.min() if not _cat_cap.empty else 0

    # Category-specific manufacturing cost for tooltips
    _cat_mfg_cost = data.factory_capacity[
        data.factory_capacity["category_id"] == selected_category
    ].set_index("factory_id")["unit_manufacturing_cost_usd"]

    cat_name_display = dict(cat_options).get(selected_category, selected_category)
    st.sidebar.caption(
        f"Showing {len(relevant_factories)} factories and "
        f"{len(relevant_hubs)} hubs with feasible flows for {cat_name_display}"
    )
else:
    relevant_factories = set(FACTORY_COORDS.keys())
    relevant_hubs = set(HUB_COORDS.keys())
    _cat_cap = _fcc
    _cat_cap_max = _fcc_max
    _cat_cap_min = _fcc_min
    _cat_mfg_cost = None

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Network Map", "Impact Analysis"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: GEOGRAPHIC NETWORK MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    fig = go.Figure()

    # ── Factory markers (red triangles, sized by capacity) ──
    # Only show factories in relevant_factories (all when no category filter).
    visible_fids = [fid for fid in FACTORY_COORDS if fid in relevant_factories]

    if visible_fids:
        f_sizes = [
            _factory_size_dynamic(fid, _cat_cap, _cat_cap_min, _cat_cap_max)
            for fid in visible_fids
        ]
        f_texts = []
        for fid in visible_fids:
            name = data.factory_name(fid)
            city = data.factory_city(fid)
            cc = data.factory_country(fid)
            if selected_category is not None:
                # Show category-specific capacity and manufacturing cost
                cap = _cat_cap.get(fid, 0)
                mfg = _cat_mfg_cost.get(fid, 0) if _cat_mfg_cost is not None else 0
                f_texts.append(
                    f"<b>{name}</b><br>{city}, {cc}<br>"
                    f"{cat_name_display} capacity: {cap:,.0f} units/mo<br>"
                    f"Mfg cost: ${mfg:,.2f}/unit"
                )
            else:
                # Show total capacity and cost multiplier (original behavior)
                total_cap = _fcc.get(fid, 0)
                cost_mult = _factory_cost_mult.get(fid, 1.0)
                f_texts.append(
                    f"<b>{name}</b><br>{city}, {cc}<br>"
                    f"Capacity: {total_cap:,.0f} units/mo<br>"
                    f"Cost: {cost_mult:.2f}x"
                )

        fig.add_trace(go.Scattergeo(
            lat=[FACTORY_COORDS[fid][0] for fid in visible_fids],
            lon=[FACTORY_COORDS[fid][1] for fid in visible_fids],
            mode="markers",
            marker=dict(size=f_sizes, color="crimson", symbol="triangle-up"),
            text=f_texts,
            hoverinfo="text",
            name="Factories",
        ))

    # ── Hub markers (blue squares, sized by throughput) ──
    # Only show hubs in relevant_hubs.
    visible_hids = [hid for hid in HUB_COORDS if hid in relevant_hubs]

    if visible_hids:
        h_sizes = [_hub_size(hid) for hid in visible_hids]
        h_texts = []
        for hid in visible_hids:
            name = data.hub_name(hid)
            city = data.hub_city(hid)
            tp = _hub_tp.get(hid, 0)
            h_texts.append(
                f"<b>{name}</b><br>{city}<br>"
                f"Throughput: {tp:,} units/mo"
            )

        fig.add_trace(go.Scattergeo(
            lat=[HUB_COORDS[hid][0] for hid in visible_hids],
            lon=[HUB_COORDS[hid][1] for hid in visible_hids],
            mode="markers",
            marker=dict(size=h_sizes, color="royalblue", symbol="square"),
            text=h_texts,
            hoverinfo="text",
            name="Hubs",
        ))

    # ── Country markers (green circles — always show all 17) ──
    c_texts = [
        f"<b>{country_names.get(cc, cc)}</b> ({cc})" for cc in COUNTRY_COORDS
    ]

    fig.add_trace(go.Scattergeo(
        lat=[COUNTRY_COORDS[cc][0] for cc in COUNTRY_COORDS],
        lon=[COUNTRY_COORDS[cc][1] for cc in COUNTRY_COORDS],
        mode="markers",
        marker=dict(size=7, color="seagreen", symbol="circle"),
        text=c_texts,
        hoverinfo="text",
        name="Countries",
    ))

    # ── Flow lines when a country is selected ──
    if selected_country != "(All)":
        if selected_category is not None:
            # Category-aware: use pre-filtered cat_flows for accurate routes
            country_cat_flows = cat_flows[cat_flows["country_code"] == selected_country]
            drawn = set()
            for _, row in country_cat_flows.iterrows():
                fid, hid = row["factory_id"], row["hub_id"]
                key = (fid, hid)
                if key in drawn:
                    continue
                drawn.add(key)
                if fid in FACTORY_COORDS and hid in HUB_COORDS:
                    f_lat, f_lon = FACTORY_COORDS[fid]
                    h_lat, h_lon = HUB_COORDS[hid]
                    c_lat, c_lon = COUNTRY_COORDS[selected_country]

                    fig.add_trace(go.Scattergeo(
                        lat=[f_lat, h_lat], lon=[f_lon, h_lon],
                        mode="lines",
                        line=dict(width=1.5, color="orange"),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig.add_trace(go.Scattergeo(
                        lat=[h_lat, c_lat], lon=[h_lon, c_lon],
                        mode="lines",
                        line=dict(width=1.5, color="dodgerblue"),
                        hoverinfo="skip", showlegend=False,
                    ))
        else:
            # No category filter: use knowledge graph for all routes (original behavior)
            for fid in FACTORY_COORDS:
                routes = kg.find_all_routes(fid, selected_country)
                for route in routes:
                    hid = route["hub_id"]
                    if fid in FACTORY_COORDS and hid in HUB_COORDS:
                        f_lat, f_lon = FACTORY_COORDS[fid]
                        h_lat, h_lon = HUB_COORDS[hid]
                        c_lat, c_lon = COUNTRY_COORDS[selected_country]

                        fig.add_trace(go.Scattergeo(
                            lat=[f_lat, h_lat], lon=[f_lon, h_lon],
                            mode="lines",
                            line=dict(width=1.5, color="orange"),
                            hoverinfo="skip", showlegend=False,
                        ))
                        fig.add_trace(go.Scattergeo(
                            lat=[h_lat, c_lat], lon=[h_lon, c_lon],
                            mode="lines",
                            line=dict(width=1.5, color="dodgerblue"),
                            hoverinfo="skip", showlegend=False,
                        ))

        restrictions = kg.get_restriction_graph(selected_country)
        if restrictions:
            restricted_names = [
                f"{r['restricted_country']} ({r['restriction_type']})"
                for r in restrictions
            ]
            st.info(
                f"**Geopolitical restrictions for "
                f"{country_names.get(selected_country, selected_country)}:** "
                + ", ".join(restricted_names)
            )

    # Handle edge case: category + country with no feasible flows
    if selected_category is not None and not relevant_factories:
        st.warning(
            f"No feasible flows exist for {cat_name_display} "
            f"to {country_names.get(selected_country, selected_country)}. "
            f"Try selecting a different category or country."
        )

    fig.update_layout(**_map_layout())
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Graph: {kg.graph.number_of_nodes()} nodes, "
        f"{kg.graph.number_of_edges()} edges"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DISRUPTION IMPACT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Disruption Impact Analysis")
    st.caption(
        "Simulate disabling factories and hubs. The map updates to show "
        "remaining routes and highlights countries that lose all supply."
    )

    # ── Disable controls (side by side) ──
    # Only list factories/hubs that are relevant for the selected category.
    dis_col1, dis_col2 = st.columns(2)

    # Build list of factory/hub IDs available for disabling (filtered by category)
    available_factory_ids = [
        fid for fid in data.factories["factory_id"] if fid in relevant_factories
    ]
    available_hub_ids = [
        hid for hid in data.hubs["hub_id"] if hid in relevant_hubs
    ]

    with dis_col1:
        disabled_factories = st.multiselect(
            "Disable factories",
            available_factory_ids,
            format_func=lambda fid: (
                f"{fid} — {data.factory_name(fid)} ({data.factory_city(fid)})"
            ),
        )

    with dis_col2:
        disabled_hubs = st.multiselect(
            "Disable hubs",
            available_hub_ids,
            format_func=lambda hid: (
                f"{hid} — {data.hub_name(hid)} ({data.hub_city(hid)})"
            ),
        )

    # ── Compute remaining routes and affected countries ──
    # For every (factory, country) pair where the factory is still active and relevant,
    # find all 2-hop routes (factory→hub→country) where the hub is also active.
    # When a category is selected, use cat_flows for accurate category-specific routing.
    active_factories = relevant_factories - set(disabled_factories)
    active_hubs = relevant_hubs - set(disabled_hubs)

    countries_with_supply = set()
    remaining_routes = []  # (factory_id, hub_id, country_code) triples for route lines

    if selected_category is not None:
        # Category-aware: filter cat_flows by active factories and hubs
        # Re-query without country filter to get all-country feasible flows for this category
        all_cat_mask = (
            (data.all_flows["category_id"] == selected_category)
            & (data.all_flows["is_geopolitically_restricted"] == 0)
            & (data.all_flows["is_lead_time_feasible"] == 1)
        )
        all_cat_flows = data.all_flows[all_cat_mask]
        for _, row in all_cat_flows.iterrows():
            fid, hid, cc = row["factory_id"], row["hub_id"], row["country_code"]
            if fid in active_factories and hid in active_hubs:
                countries_with_supply.add(cc)
                remaining_routes.append((fid, hid, cc))
    else:
        # No category filter: use knowledge graph (original behavior)
        for cc in COUNTRY_COORDS:
            for fid in active_factories:
                routes = kg.find_all_routes(fid, cc)
                for route in routes:
                    hid = route["hub_id"]
                    if hid in active_hubs:
                        countries_with_supply.add(cc)
                        remaining_routes.append((fid, hid, cc))

    affected_countries = set(COUNTRY_COORDS.keys()) - countries_with_supply

    # ── Build impact map ──
    impact_fig = go.Figure()

    # Active factory markers (crimson triangles)
    active_f_ids = [fid for fid in FACTORY_COORDS if fid in active_factories]
    if active_f_ids:
        impact_fig.add_trace(go.Scattergeo(
            lat=[FACTORY_COORDS[fid][0] for fid in active_f_ids],
            lon=[FACTORY_COORDS[fid][1] for fid in active_f_ids],
            mode="markers",
            marker=dict(
                size=[
                    _factory_size_dynamic(fid, _cat_cap, _cat_cap_min, _cat_cap_max)
                    for fid in active_f_ids
                ],
                color="crimson", symbol="triangle-up",
            ),
            text=[
                f"<b>{data.factory_name(fid)}</b><br>"
                f"{data.factory_city(fid)}, {data.factory_country(fid)}"
                for fid in active_f_ids
            ],
            hoverinfo="text",
            name="Active Factories",
        ))

    # Disabled factory markers (gray triangles)
    dis_f_ids = [fid for fid in disabled_factories if fid in FACTORY_COORDS]
    if dis_f_ids:
        impact_fig.add_trace(go.Scattergeo(
            lat=[FACTORY_COORDS[fid][0] for fid in dis_f_ids],
            lon=[FACTORY_COORDS[fid][1] for fid in dis_f_ids],
            mode="markers",
            marker=dict(size=10, color="lightgray", symbol="triangle-up",
                        line=dict(width=1, color="gray")),
            text=[
                f"<b>{data.factory_name(fid)}</b> (DISABLED)<br>"
                f"{data.factory_city(fid)}, {data.factory_country(fid)}"
                for fid in dis_f_ids
            ],
            hoverinfo="text",
            name="Disabled Factories",
        ))

    # Active hub markers (blue squares)
    active_h_ids = [hid for hid in HUB_COORDS if hid in active_hubs]
    if active_h_ids:
        impact_fig.add_trace(go.Scattergeo(
            lat=[HUB_COORDS[hid][0] for hid in active_h_ids],
            lon=[HUB_COORDS[hid][1] for hid in active_h_ids],
            mode="markers",
            marker=dict(
                size=[_hub_size(hid) for hid in active_h_ids],
                color="royalblue", symbol="square",
            ),
            text=[
                f"<b>{data.hub_name(hid)}</b><br>"
                f"{data.hub_city(hid)}<br>"
                f"Throughput: {_hub_tp.get(hid, 0):,} units/mo"
                for hid in active_h_ids
            ],
            hoverinfo="text",
            name="Active Hubs",
        ))

    # Disabled hub markers (gray squares)
    dis_h_ids = [hid for hid in disabled_hubs if hid in HUB_COORDS]
    if dis_h_ids:
        impact_fig.add_trace(go.Scattergeo(
            lat=[HUB_COORDS[hid][0] for hid in dis_h_ids],
            lon=[HUB_COORDS[hid][1] for hid in dis_h_ids],
            mode="markers",
            marker=dict(size=9, color="lightgray", symbol="square",
                        line=dict(width=1, color="gray")),
            text=[
                f"<b>{data.hub_name(hid)}</b> (DISABLED)<br>"
                f"{data.hub_city(hid)}"
                for hid in dis_h_ids
            ],
            hoverinfo="text",
            name="Disabled Hubs",
        ))

    # Country markers: green if served, red if affected
    served_ccs = [cc for cc in COUNTRY_COORDS if cc not in affected_countries]
    lost_ccs = [cc for cc in COUNTRY_COORDS if cc in affected_countries]

    if served_ccs:
        impact_fig.add_trace(go.Scattergeo(
            lat=[COUNTRY_COORDS[cc][0] for cc in served_ccs],
            lon=[COUNTRY_COORDS[cc][1] for cc in served_ccs],
            mode="markers",
            marker=dict(size=7, color="seagreen", symbol="circle"),
            text=[
                f"<b>{country_names.get(cc, cc)}</b> ({cc})"
                for cc in served_ccs
            ],
            hoverinfo="text",
            name="Countries (served)",
        ))

    if lost_ccs:
        impact_fig.add_trace(go.Scattergeo(
            lat=[COUNTRY_COORDS[cc][0] for cc in lost_ccs],
            lon=[COUNTRY_COORDS[cc][1] for cc in lost_ccs],
            mode="markers",
            marker=dict(size=12, color="red", symbol="circle",
                        line=dict(width=1, color="darkred")),
            text=[
                f"<b>{country_names.get(cc, cc)}</b> ({cc})<br>NO SUPPLY"
                for cc in lost_ccs
            ],
            hoverinfo="text",
            name="Countries (no supply)",
        ))

    # Route lines for sidebar-selected country (through active nodes only)
    if selected_country != "(All)":
        drawn = set()
        for fid, hid, cc in remaining_routes:
            if cc != selected_country:
                continue
            key = (fid, hid, cc)
            if key in drawn:
                continue
            drawn.add(key)

            if fid in FACTORY_COORDS and hid in HUB_COORDS:
                f_lat, f_lon = FACTORY_COORDS[fid]
                h_lat, h_lon = HUB_COORDS[hid]
                c_lat, c_lon = COUNTRY_COORDS[cc]

                impact_fig.add_trace(go.Scattergeo(
                    lat=[f_lat, h_lat], lon=[f_lon, h_lon],
                    mode="lines",
                    line=dict(width=1.5, color="orange"),
                    hoverinfo="skip", showlegend=False,
                ))
                impact_fig.add_trace(go.Scattergeo(
                    lat=[h_lat, c_lat], lon=[h_lon, c_lon],
                    mode="lines",
                    line=dict(width=1.5, color="dodgerblue"),
                    hoverinfo="skip", showlegend=False,
                ))

    impact_fig.update_layout(**_map_layout())
    st.plotly_chart(impact_fig, use_container_width=True)

    # ── Impact metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Factories", len(active_factories))
    m2.metric("Active Hubs", len(active_hubs))
    m3.metric("Countries Served", len(countries_with_supply))
    m4.metric(
        "Countries Without Supply",
        len(affected_countries),
        delta=f"-{len(affected_countries)}" if affected_countries else None,
        delta_color="inverse",
    )

    # ── Status messages ──
    if affected_countries:
        affected_names = sorted(
            f"{country_names.get(cc, cc)} ({cc})" for cc in affected_countries
        )
        st.error(
            "**Countries with no remaining supply route:** "
            + ", ".join(affected_names)
        )
    elif disabled_factories or disabled_hubs:
        st.success("All countries still have at least one supply route.")
    else:
        st.info("Select factories or hubs above to simulate disruptions.")

    # ── Per-hub details ──
    if disabled_hubs:
        with st.expander("Hub disruption details"):
            for hid in disabled_hubs:
                st.markdown("---")
                util = kg.hub_utilization_risk(hid)
                st.markdown(
                    f"**{data.hub_name(hid)}** ({data.hub_city(hid)}, "
                    f"{data.hub_country(hid)})"
                )
                dcol1, dcol2 = st.columns(2)
                dcol1.metric("Factories feeding this hub", util["factory_count"])
                dcol2.metric("Countries served", util["country_count"])

                solely_dependent = kg.impact_analysis(hid)
                if solely_dependent:
                    st.warning(
                        "Countries solely dependent on this hub: "
                        + ", ".join(
                            f"{country_names.get(cc, cc)} ({cc})"
                            for cc in solely_dependent
                        )
                    )
                else:
                    st.success("All served countries have alternative hub routes")
