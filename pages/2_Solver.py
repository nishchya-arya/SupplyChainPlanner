"""
Solver page — configure inputs and run the MILP optimizer.

This is the main functional page of the app. The user configures inputs in the
sidebar (category, country, volume, weight sliders) and clicks Solve. The MILP
optimizer returns results organized into 3 tiers, followed by a geographic
route map visualizing the flows on a world map.

Layout:
  - Sidebar: product category, region/country selectors (cascading), volume,
    solver parameter expander (cost/time/regional weights, min batch), solve button
  - Main area (before solve): instructional empty state with quick-start guide
  - Main area (after solve):
    1. Summary metrics row (total cost, cost/unit, flows used, avg transit)
    2. Tier 1: Chosen Flow(s) — MILP-optimal allocation + cost breakdown donut chart
    3. Tier 2: Other Available Flows — next 3 best alternatives
    4. Tier 3: Alternative Manufacturing — ALL factories with status + best routes
    5. Route Map — Plotly world map with Tier 1 (green), Tier 2 (orange),
       Tier 3 available (gray with routes), blocked (red), destination (gold star)
    6. Tier explanation expander

Data flow: data_loader → optimizer.solve() → ranker.rank_flows() → UI display
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from solver.data_loader import SupplyChainData   # CSV data loader
from solver.optimizer import solve                # MILP solver
from solver.ranker import rank_flows              # 3-tier result organizer
from solver.coords import FACTORY_COORDS, HUB_COORDS, COUNTRY_COORDS  # map coordinates


@st.cache_resource
def load_data():
    return SupplyChainData()


st.set_page_config(page_title="Solver — Supply Chain Planner", layout="wide")

data = load_data()

# ── Sidebar: Inputs ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    solve_btn = st.button("Solve", type="primary", use_container_width=True)

    # Solver parameters (just below Solve button)
    with st.expander("Solver Parameters"):
        cost_weight = st.slider(
            "Cost importance", 1, 10, 8,
            help="How much to prioritize lower cost",
        )
        time_weight = st.slider(
            "Speed importance", 1, 10, 5,
            help="How much to prioritize faster delivery",
        )
        regional_weight = st.slider(
            "Regional preference", 1, 10, 3,
            help="How much to prioritize same-region manufacturing/storage",
        )
        min_batch = st.slider(
            "Minimum batch size", 100, 2000, 500, 100,
            help="Minimum units per active flow",
        )

    st.divider()

    # Category
    cat_options = data.get_category_list()
    cat_labels = [f"{cid} — {cname}" for cid, cname in cat_options]
    cat_idx = st.selectbox(
        "Product Category", range(len(cat_labels)),
        format_func=lambda i: cat_labels[i],
    )
    category_id = cat_options[cat_idx][0]
    category_name = cat_options[cat_idx][1]

    # Region
    region_options = data.get_region_list()
    region_labels = [f"{rid} — {rname}" for rid, rname in region_options]
    region_idx = st.selectbox(
        "Target Region", range(len(region_labels)),
        format_func=lambda i: region_labels[i],
    )
    region_id = region_options[region_idx][0]

    # Country (filtered by region)
    region_countries = data.get_countries_in_region(region_id)
    default_country = data.get_default_country(region_id)
    default_idx = (
        region_countries.index(default_country)
        if default_country in region_countries else 0
    )

    country_names = dict(
        zip(data.countries["country_code"], data.countries["country_name"])
    )
    country_labels = [
        f"{cc} — {country_names.get(cc, cc)}" for cc in region_countries
    ]
    country_idx = st.selectbox(
        "Target Country",
        range(len(country_labels)),
        index=default_idx,
        format_func=lambda i: country_labels[i],
    )
    country_code = region_countries[country_idx]

    # Volume
    volume = st.number_input(
        "Volume (units)", min_value=500, max_value=200000, value=10000, step=500,
    )


# ── Main area ─────────────────────────────────────────────────────────────
st.title("Supply Chain Solver")
st.caption(
    "Find optimal factory-to-hub routing for your product launch"
)

if not solve_btn:
    # Empty state — show instructions
    st.info(
        "Configure your product category, target country, and volume in the "
        "sidebar, then click **Solve** to find the optimal supply chain allocation."
    )

    with st.container():
        st.markdown("#### Quick Start")
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(
            "**1. Select inputs**\n\n"
            "Choose a product category, target region and country, and order volume."
        )
        col_b.markdown(
            "**2. Tune weights**\n\n"
            "Expand *Solver Parameters* to adjust the importance of cost, "
            "speed, and regional preference (1-10)."
        )
        col_c.markdown(
            "**3. Click Solve**\n\n"
            "The MILP optimizer runs in milliseconds and returns the optimal "
            "allocation across three result tiers."
        )
    st.stop()

# ── Run solver ────────────────────────────────────────────────────────────
with st.spinner("Running MILP solver..."):
    result = solve(
        data, category_id, country_code, volume,
        cost_weight=cost_weight,
        time_weight=time_weight,
        regional_weight=regional_weight,
        min_batch=min_batch,
    )

if result.status != "Optimal":
    st.error(f"Solver status: {result.status}")
    st.stop()

st.toast("Solver found optimal solution!", icon="\u2705")

ranked = rank_flows(
    result, data, country_code,
    cost_weight=cost_weight,
    time_weight=time_weight,
    regional_weight=regional_weight,
    category_id=category_id,
)

# ── Summary ───────────────────────────────────────────────────────────────
st.header(
    f"{category_name}  /  "
    f"{country_names.get(country_code, country_code)}  /  "
    f"{volume:,} units"
)

weighted_days = (
    sum(cf["transit_days"] * cf["units_allocated"] for cf in result.chosen_flows)
    / volume
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cost", f"${result.total_cost:,.0f}")
col2.metric("Cost / Unit", f"${result.total_cost / volume:,.2f}")
col3.metric("Flows Used", str(len(result.chosen_flows)))
col4.metric("Avg Transit", f"{weighted_days:.0f} days")

st.divider()

# ── Tier 1: Chosen Flows ─────────────────────────────────────────────────
st.subheader("Chosen Flow(s)")
st.caption(
    "Optimal allocation from the MILP solver — may split across factories "
    "if capacity requires it."
)

tier1_rows = []
for cf in ranked["chosen_flows"]:
    tier1_rows.append({
        "Factory": f"{cf['factory_name']} ({cf['factory_city']}, {cf['factory_country']})",
        "Hub": f"{cf['hub_name']} ({cf['hub_city']}, {cf['hub_country']})",
        "Units": f"{cf['units_allocated']:,}",
        "Cost/Unit": f"${cf['cost_per_unit']:.2f}",
        "Total Cost": f"${cf['total_cost']:,.0f}",
        "Transit": f"{cf['transit_days']}d",
    })

st.dataframe(pd.DataFrame(tier1_rows), use_container_width=True, hide_index=True)

# ── Cost breakdown donut chart ──
# Donut chart replaces the old stacked bar which was dominated by manufacturing
# cost (~90%+). The donut shows proportional breakdown clearly for all components.
st.markdown("**Cost Breakdown (per unit)**")

cost_components = ["Manufacturing", "Transport", "Hub Handling", "Last Mile", "Tariff"]
cost_colors = {
    "Manufacturing": "#2196F3",
    "Transport": "#FF9800",
    "Hub Handling": "#4CAF50",
    "Last Mile": "#9C27B0",
    "Tariff": "#F44336",
}

breakdown_rows = []
for cf in ranked["chosen_flows"]:
    label = f"{cf['factory_name']} → {cf['hub_name']}"
    row = {
        "Flow": label,
        "Manufacturing": cf["manufacturing_cost"],
        "Transport": cf["transport_cost"],
        "Hub Handling": cf["hub_handling_cost"],
        "Last Mile": cf["last_mile_cost"],
        "Tariff": cf["tariff_amount"],
    }
    row["Total"] = sum(row[c] for c in cost_components)
    breakdown_rows.append(row)

if len(breakdown_rows) == 1:
    row = breakdown_rows[0]
    values = [row[c] for c in cost_components]
    fig_pie = go.Figure(data=[go.Pie(
        labels=cost_components,
        values=values,
        marker_colors=[cost_colors[c] for c in cost_components],
        textinfo="none",
        textposition="inside",
        hovertemplate="%{label}: $%{value:.2f} (%{percent})<extra></extra>",
        hole=0.3,
    )])
    fig_pie.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05,
                    xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    # Side-by-side donuts for multi-flow (volume split across factories)
    n_flows = len(breakdown_rows)
    fig_pie = make_subplots(
        rows=1, cols=n_flows,
        specs=[[{"type": "domain"}] * n_flows],
        subplot_titles=[row["Flow"] for row in breakdown_rows],
    )
    for i, row in enumerate(breakdown_rows):
        values = [row[c] for c in cost_components]
        fig_pie.add_trace(
            go.Pie(
                labels=cost_components,
                values=values,
                marker_colors=[cost_colors[c] for c in cost_components],
                textinfo="none",
                hovertemplate="%{label}: $%{value:.2f} (%{percent})<extra></extra>",
                hole=0.3,
                showlegend=(i == 0),
            ),
            row=1, col=i + 1,
        )
    fig_pie.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1,
                    xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Absolute values in an expander for users who need exact dollar amounts
with st.expander("Detailed cost breakdown (absolute values)"):
    detail_rows = []
    for row in breakdown_rows:
        detail_rows.append({
            "Flow": row["Flow"],
            **{c: f"${row[c]:.2f}" for c in cost_components},
            "Total": f"${row['Total']:.2f}",
        })
    st.dataframe(
        pd.DataFrame(detail_rows), use_container_width=True, hide_index=True,
    )

st.divider()

# ── Tier 2: Other Available Flows ─────────────────────────────────────────
st.subheader("Other Available Flows")
st.caption(
    "Next 3 best alternatives by composite score (cost + time + regional preference)"
)

if ranked["other_available"]:
    tier2_rows = []
    for r in ranked["other_available"]:
        tier2_rows.append({
            "Rank": r["rank"],
            "Factory": f"{r['factory_name']} ({r['factory_city']}, {r['factory_country']})",
            "Hub": f"{r['hub_name']} ({r['hub_city']})",
            "Cost/Unit": f"${r['cost_per_unit']:.2f}",
            "Transit": f"{r['transit_days']}d",
            "Score": f"{r['composite_score']:.3f}",
        })
    st.dataframe(
        pd.DataFrame(tier2_rows), use_container_width=True, hide_index=True,
    )
else:
    st.info("No additional flows available.")

st.divider()

# ── Tier 3: Alternative Manufacturing ─────────────────────────────────────
st.subheader("Alternative Manufacturing Options")
st.caption(
    "Other factory locations not in the flows above — showing best route per factory"
)

if ranked["alternative_factories"]:
    tier3_rows = []
    for af in ranked["alternative_factories"]:
        cost_str = f"${af['cost_per_unit']:.2f}" if af["cost_per_unit"] is not None else "N/A"
        transit_str = f"{af['transit_days']}d" if af["transit_days"] is not None else "N/A"
        score_str = f"{af['composite_score']:.3f}" if af["composite_score"] is not None else "—"
        tier3_rows.append({
            "Factory": f"{af['factory_name']} ({af['factory_city']}, {af['factory_country']})",
            "Best Hub": af["best_hub_name"],
            "Cost/Unit": cost_str,
            "Transit": transit_str,
            "Score": score_str,
            "Status": af.get("status", "Available"),
        })
    st.dataframe(
        pd.DataFrame(tier3_rows), use_container_width=True, hide_index=True,
    )
else:
    st.info("No additional manufacturing options.")

st.divider()

# ── Route Map ─────────────────────────────────────────────────────────────
# Geographic visualization of solver results: Tier 1 (green), Tier 2 (orange),
# Tier 3 factories (gray triangles), destination (gold star).
st.subheader("Route Map")
st.caption(
    "Geographic view of solver results. Green = chosen (Tier 1), "
    "orange = alternatives (Tier 2), gray = other available (Tier 3), "
    "red = blocked by restrictions."
)

map_fig = go.Figure()

# Collect factory/hub IDs per tier for color coding
tier1_fids = {cf["factory_id"] for cf in ranked["chosen_flows"]}
tier1_hids = {cf["hub_id"] for cf in ranked["chosen_flows"]}
tier2_fids = {r["factory_id"] for r in ranked["other_available"]}
tier2_hids = {r["hub_id"] for r in ranked["other_available"]}
# Split Tier 3 into available (feasible) and blocked (restricted/lead-time)
tier3_available = [af for af in ranked["alternative_factories"] if af.get("status") == "Available"]
tier3_blocked = [af for af in ranked["alternative_factories"] if af.get("status", "Available") != "Available"]
tier3_fids = {af["factory_id"] for af in tier3_available}
tier3_blocked_fids = {af["factory_id"] for af in tier3_blocked}

# ── Tier 1 flow lines (green): factory → hub → country ──
for cf in ranked["chosen_flows"]:
    fid, hid = cf["factory_id"], cf["hub_id"]
    if fid in FACTORY_COORDS and hid in HUB_COORDS and country_code in COUNTRY_COORDS:
        f_lat, f_lon = FACTORY_COORDS[fid]
        h_lat, h_lon = HUB_COORDS[hid]
        c_lat, c_lon = COUNTRY_COORDS[country_code]
        map_fig.add_trace(go.Scattergeo(
            lat=[f_lat, h_lat], lon=[f_lon, h_lon],
            mode="lines", line=dict(width=2.5, color="green"),
            hoverinfo="skip", showlegend=False,
        ))
        map_fig.add_trace(go.Scattergeo(
            lat=[h_lat, c_lat], lon=[h_lon, c_lon],
            mode="lines", line=dict(width=2.5, color="green"),
            hoverinfo="skip", showlegend=False,
        ))

# ── Tier 2 flow lines (orange): factory → hub → country ──
for r in ranked["other_available"]:
    fid, hid = r["factory_id"], r["hub_id"]
    if fid in FACTORY_COORDS and hid in HUB_COORDS and country_code in COUNTRY_COORDS:
        f_lat, f_lon = FACTORY_COORDS[fid]
        h_lat, h_lon = HUB_COORDS[hid]
        c_lat, c_lon = COUNTRY_COORDS[country_code]
        map_fig.add_trace(go.Scattergeo(
            lat=[f_lat, h_lat], lon=[f_lon, h_lon],
            mode="lines", line=dict(width=1.5, color="orange"),
            hoverinfo="skip", showlegend=False,
        ))
        map_fig.add_trace(go.Scattergeo(
            lat=[h_lat, c_lat], lon=[h_lon, c_lon],
            mode="lines", line=dict(width=1.5, color="orange", dash="dash"),
            hoverinfo="skip", showlegend=False,
        ))

# ── Tier 1 factory markers (green triangles) ──
for fid in tier1_fids:
    if fid in FACTORY_COORDS:
        lat, lon = FACTORY_COORDS[fid]
        map_fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers",
            marker=dict(size=12, color="green", symbol="triangle-up"),
            text=f"<b>{data.factory_name(fid)}</b><br>{data.factory_city(fid)} (Tier 1)",
            hoverinfo="text", showlegend=False,
        ))

# ── Tier 2 factory markers (orange triangles, skip if already Tier 1) ──
for fid in tier2_fids - tier1_fids:
    if fid in FACTORY_COORDS:
        lat, lon = FACTORY_COORDS[fid]
        map_fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers",
            marker=dict(size=10, color="orange", symbol="triangle-up"),
            text=f"<b>{data.factory_name(fid)}</b><br>{data.factory_city(fid)} (Tier 2)",
            hoverinfo="text", showlegend=False,
        ))

# ── Tier 3 available: route lines (gray dashed) + factory markers ──
for af in tier3_available:
    fid, hid = af["factory_id"], af["best_hub_id"]
    if fid not in tier1_fids | tier2_fids and fid in FACTORY_COORDS:
        # Route lines: factory → hub → destination
        if hid in HUB_COORDS and country_code in COUNTRY_COORDS:
            f_lat, f_lon = FACTORY_COORDS[fid]
            h_lat, h_lon = HUB_COORDS[hid]
            c_lat, c_lon = COUNTRY_COORDS[country_code]
            map_fig.add_trace(go.Scattergeo(
                lat=[f_lat, h_lat], lon=[f_lon, h_lon],
                mode="lines", line=dict(width=1, color="gray", dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))
            map_fig.add_trace(go.Scattergeo(
                lat=[h_lat, c_lat], lon=[h_lon, c_lon],
                mode="lines", line=dict(width=1, color="gray", dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))
        # Factory marker
        lat, lon = FACTORY_COORDS[fid]
        map_fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers",
            marker=dict(size=8, color="gray", symbol="triangle-up"),
            text=f"<b>{data.factory_name(fid)}</b><br>{data.factory_city(fid)} (Tier 3)",
            hoverinfo="text", showlegend=False,
        ))

# ── Tier 3 blocked: red triangle markers (no route lines) ──
for af in tier3_blocked:
    fid = af["factory_id"]
    if fid in FACTORY_COORDS:
        lat, lon = FACTORY_COORDS[fid]
        map_fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers",
            marker=dict(size=8, color="red", symbol="triangle-up"),
            text=f"<b>{data.factory_name(fid)}</b><br>{data.factory_city(fid)}<br>{af['status']}",
            hoverinfo="text", showlegend=False,
        ))

# ── Hub markers (blue for Tier 1/2, gray for Tier 3) ──
tier3_hids = {af["best_hub_id"] for af in tier3_available if af["best_hub_id"]}
for hid in tier1_hids | tier2_hids:
    if hid in HUB_COORDS:
        lat, lon = HUB_COORDS[hid]
        map_fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers",
            marker=dict(size=9, color="royalblue", symbol="square"),
            text=f"<b>{data.hub_name(hid)}</b><br>{data.hub_city(hid)}",
            hoverinfo="text", showlegend=False,
        ))
# Tier 3 hubs (only those not already shown as Tier 1/2 hubs)
for hid in tier3_hids - tier1_hids - tier2_hids:
    if hid in HUB_COORDS:
        lat, lon = HUB_COORDS[hid]
        map_fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers",
            marker=dict(size=7, color="gray", symbol="square"),
            text=f"<b>{data.hub_name(hid)}</b><br>{data.hub_city(hid)} (Tier 3)",
            hoverinfo="text", showlegend=False,
        ))

# ── Destination country marker (gold star) ──
if country_code in COUNTRY_COORDS:
    lat, lon = COUNTRY_COORDS[country_code]
    map_fig.add_trace(go.Scattergeo(
        lat=[lat], lon=[lon], mode="markers",
        marker=dict(size=14, color="gold", symbol="star",
                    line=dict(width=1, color="black")),
        text=f"<b>{country_names.get(country_code, country_code)}</b><br>Destination",
        hoverinfo="text", showlegend=False,
    ))

# ── Legend entries (invisible data, visible legend) ──
for label, color, symbol in [
    ("Tier 1 (Chosen)", "green", "triangle-up"),
    ("Tier 2 (Alternatives)", "orange", "triangle-up"),
    ("Tier 3 (Available)", "gray", "triangle-up"),
    ("Blocked (Restricted)", "red", "triangle-up"),
    ("Destination", "gold", "star"),
]:
    map_fig.add_trace(go.Scattergeo(
        lat=[None], lon=[None], mode="markers",
        marker=dict(size=10, color=color, symbol=symbol),
        name=label,
    ))

map_fig.update_layout(
    geo=dict(
        projection_type="natural earth",
        showland=True, landcolor="rgb(243, 243, 243)",
        countrycolor="rgb(204, 204, 204)",
        showocean=True, oceancolor="rgb(230, 240, 250)",
        showcountries=True,
    ),
    height=450,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.01,
        bgcolor="rgba(255,255,255,0.8)",
    ),
)

st.plotly_chart(map_fig, use_container_width=True)

# ── Explanation ───────────────────────────────────────────────────────────
with st.expander("What do these tiers mean?"):
    st.markdown(
        """
        **Tier 1 — Chosen Flow(s):** The mathematically optimal allocation from the
        MILP solver. If capacity allows, a single factory-hub pair handles all units.
        Otherwise, volume is split across multiple flows.

        **Tier 2 — Other Available Flows:** The next 3 best routes by composite score.
        These are backup options if the primary supply chain is disrupted.

        **Tier 3 — Alternative Manufacturing:** Factories not represented in Tiers 1
        or 2, each showing their best hub route. Useful for diversification or
        geopolitical risk mitigation.

        **Composite Score:** A 0-1 value combining cost, speed, and regional proximity
        using your weight settings. Lower = better.
        """
    )
