"""
Supply Chain Planner — Home Page (Streamlit entry point).

This is the landing page users see first. It provides:
  1. Three navigation cards linking to the main pages
  2. A key-stats row showing network dimensions at a glance
  3. A mini Plotly world map previewing the global supply chain topology

Run: streamlit run app.py

Multipage app (sidebar order determined by numeric filename prefix):
  - pages/1_Knowledge_Graph.py → Interactive network map + disruption simulator
  - pages/2_Solver.py          → MILP optimizer with 3-tier results + route map
  - pages/3_About.py           → Documentation, data model, and technical details
"""

import streamlit as st
import plotly.graph_objects as go

from solver.coords import FACTORY_COORDS, HUB_COORDS, COUNTRY_COORDS

st.set_page_config(
    page_title="Supply Chain Planner",
    layout="wide",
)

st.title("Supply Chain Planner")
st.markdown(
    "MILP-based supply chain optimizer for consumer electronics. "
    "Find the optimal factory-to-hub-to-country routing that minimizes cost "
    "while respecting capacity, geopolitical, and lead-time constraints."
)

st.divider()

# ── Navigation Cards ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Knowledge Graph")
    st.markdown(
        "Explore the supply chain network on an interactive world map. "
        "Simulate factory and hub disruptions to assess resilience."
    )
    st.page_link(
        "pages/1_Knowledge_Graph.py",
        label="Explore Network",
        icon="🌐",
    )

with col2:
    st.subheader("Solver")
    st.markdown(
        "Configure a product category, target country, and order volume, "
        "then run the optimizer to find the best allocation."
    )
    st.page_link("pages/2_Solver.py", label="Open Solver", icon="⚙️")

with col3:
    st.subheader("About")
    st.markdown(
        "Learn how the optimizer works, what data it uses, "
        "and how to interpret the results."
    )
    st.page_link("pages/3_About.py", label="Read About", icon="ℹ️")

st.divider()

# ── Key Stats ────────────────────────────────────────────────────────────────
# Quick-glance metrics matching the data model dimensions.
# Hardcoded to match the About page (same pattern as pages/3_About.py).
c1, c2, c3, c4 = st.columns(4)
c1.metric("Factories", "13")
c2.metric("Distribution Hubs", "14")
c3.metric("Destination Countries", "17")
c4.metric("Geopolitical Rules", "11")

# ── Mini Network Map ─────────────────────────────────────────────────────────
# Static preview map (no interactivity, no flow lines). Shows all factories,
# hubs, and countries as markers on a world map. Serves as a visual hook
# that encourages users to visit the Knowledge Graph page for full interaction.
st.markdown("#### Global Supply Chain Network")

fig = go.Figure()

# Factories: crimson triangles (same visual language as Knowledge Graph page)
fig.add_trace(go.Scattergeo(
    lat=[c[0] for c in FACTORY_COORDS.values()],
    lon=[c[1] for c in FACTORY_COORDS.values()],
    mode="markers",
    marker=dict(size=8, color="crimson", symbol="triangle-up"),
    name="Factories",
    hoverinfo="skip",
))

# Hubs: blue squares
fig.add_trace(go.Scattergeo(
    lat=[c[0] for c in HUB_COORDS.values()],
    lon=[c[1] for c in HUB_COORDS.values()],
    mode="markers",
    marker=dict(size=7, color="royalblue", symbol="square"),
    name="Hubs",
    hoverinfo="skip",
))

# Countries: green circles
fig.add_trace(go.Scattergeo(
    lat=[c[0] for c in COUNTRY_COORDS.values()],
    lon=[c[1] for c in COUNTRY_COORDS.values()],
    mode="markers",
    marker=dict(size=5, color="seagreen", symbol="circle"),
    name="Countries",
    hoverinfo="skip",
))

fig.update_layout(
    geo=dict(
        projection_type="natural earth",
        showland=True, landcolor="rgb(243, 243, 243)",
        countrycolor="rgb(204, 204, 204)",
        showocean=True, oceancolor="rgb(230, 240, 250)",
        showcountries=True,
    ),
    height=350,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(
        yanchor="top", y=0.99,
        xanchor="left", x=0.01,
        bgcolor="rgba(255,255,255,0.8)",
    ),
)

st.plotly_chart(fig, use_container_width=True)
