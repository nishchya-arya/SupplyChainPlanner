"""
About page — explains what the Supply Chain Planner does and how to use it.

Sections:
  1. Hero: One-line description of the MILP optimizer
  2. How It Works: 3-column layout (Configure → Optimize → Compare)
  3. Knowledge Graph: Network Map + Impact Analysis feature descriptions
  4. The Data: Key metrics and data dimension explanations
  5. The Solver: Weight sliders, minimum batch size, MILP details
  6. Understanding the Output: Tier 1/2/3 and composite score explanations
  7. Technical Details: Full MILP formulation (expandable)
"""

import streamlit as st

st.set_page_config(page_title="About — Supply Chain Planner", layout="wide")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.title("About Supply Chain Planner")
st.markdown(
    """
    A **Mixed-Integer Linear Programming (MILP)** optimizer for consumer electronics
    supply chains. Given a product category, target country, and order volume, it finds
    the best factory-to-hub-to-country routing that minimizes cost while respecting
    real-world constraints like factory capacity, hub throughput, geopolitical trade
    restrictions, and delivery lead times.
    """
)

st.divider()

# ── How It Works ──────────────────────────────────────────────────────────────
st.header("How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Configure")
    st.markdown(
        """
        Select a **product category** (e.g. Smartphones, Laptops),
        a **target country** for delivery, and the **order volume**
        in units. Optionally adjust the importance weights for cost,
        speed, and regional preference.
        """
    )

with col2:
    st.subheader("2. Optimize")
    st.markdown(
        """
        The MILP solver evaluates all feasible factory-to-hub routes,
        filters out geopolitically restricted and lead-time-infeasible
        paths, and finds the **optimal allocation** that minimizes
        a weighted composite of cost, transit time, and regional
        proximity.
        """
    )

with col3:
    st.subheader("3. Compare")
    st.markdown(
        """
        Results are organized into **3 tiers**: the chosen optimal
        flow(s), the next-best alternatives, and other available
        factory locations. This gives a complete picture for
        decision-making.
        """
    )

st.divider()

# ── Knowledge Graph ──────────────────────────────────────────────────────────
st.header("Knowledge Graph")

st.markdown(
    """
    The **Knowledge Graph** page provides an interactive geographic visualization
    of the entire supply chain network, built on a NetworkX directed graph with
    factories, hubs, countries, and their routing connections.
    """
)

kg_col1, kg_col2 = st.columns(2)

with kg_col1:
    st.subheader("Network Map")
    st.markdown(
        """
        A Plotly world map showing all 13 factories, 14 hubs, and 17
        destination countries. Markers are sized by capacity and throughput.
        Select a country in the sidebar to see all feasible routes drawn
        as flow lines, along with any geopolitical restrictions that apply.
        """
    )

with kg_col2:
    st.subheader("Impact Analysis")
    st.markdown(
        """
        Simulate disruptions by disabling factories and hubs. The map
        dynamically updates to show remaining routes, and countries that
        lose all supply paths are highlighted in red. Useful for assessing
        geographic resilience and identifying single points of failure.
        """
    )

st.divider()

# ── The Data ──────────────────────────────────────────────────────────────────
st.header("The Data")

st.markdown(
    """
    The optimizer works with a synthetic but realistic dataset modeled after
    global consumer electronics supply chains.
    """
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Factories", "13", help="Manufacturing facilities across 9 countries")
c2.metric("Distribution Hubs", "14", help="Warehousing and logistics hubs across 6 regions")
c3.metric("Destination Countries", "17", help="Target markets spanning 6 global regions")
c4.metric("Pre-computed Flows", "22,600+", help="Factory-to-hub-to-country route combinations per category")

st.markdown("")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Product Categories", "10", help="Smartphones, Laptops, Tablets, and 7 more")
c6.metric("Products", "86", help="7-12 products per category with regional availability")
c7.metric("Geopolitical Rules", "11", help="Trade restrictions like US-China, India-China, etc.")
c8.metric("Regions", "6", help="NAM, SAM, EUR, MEA, NEA, SEA")

with st.expander("Data dimensions explained"):
    st.markdown(
        """
        | Dimension | What it represents |
        |---|---|
        | **Factories** | Manufacturing plants with per-category capacity limits and cost multipliers. Located in US, Mexico, Brazil, Germany, UK, UAE, China, South Korea, Vietnam, and India. |
        | **Hubs** | Distribution centers with monthly throughput capacity. Each hub has a handling cost per unit. |
        | **Transport** | Factory-to-hub shipping costs and transit days, computed from geographic distance. |
        | **Last Mile** | Hub-to-country delivery costs and transit days. Same-country deliveries are fast (1 day); cross-region can take 20-30+ days. |
        | **Tariffs** | Import duty rates based on origin and destination country pairs. |
        | **Geopolitical Restrictions** | MADE_IN restrictions (cannot source from a country) and ROUTED_THROUGH restrictions (cannot use hubs in a country). Examples: US restricts Chinese-made goods; India blocks Chinese routing. |
        | **Lead Times** | Maximum acceptable delivery time per country and product urgency level. High-urgency products (smartphones) have tighter limits than low-urgency ones (keyboards). |
        """
    )

st.divider()

# ── The Solver ────────────────────────────────────────────────────────────────
st.header("The Solver")

st.markdown(
    """
    The core optimizer uses **Mixed-Integer Linear Programming (MILP)**, a mathematical
    optimization technique that finds the provably best solution given a set of
    constraints. Unlike heuristic approaches, MILP guarantees optimality.
    """
)

st.subheader("Weight Sliders")
st.markdown(
    """
    Three importance sliders (each 1-10) control what the solver prioritizes:

    - **Cost importance** — How much to prioritize lower total landed cost
      (manufacturing + transport + hub handling + last mile + tariffs).
    - **Speed importance** — How much to prioritize faster end-to-end transit
      (factory-to-hub + hub-to-country).
    - **Regional preference** — How much to favor factories and hubs in the same
      region as the destination country, reducing geopolitical risk and supporting
      local sourcing.

    All three criteria are normalized to a 0-1 scale and combined using the weights
    you set. The solver then minimizes this composite score across all allocated flows.
    """
)

st.subheader("Minimum Batch Size")
st.markdown(
    """
    The minimum batch size (default: 500 units) prevents the solver from making
    unrealistically small allocations. If a factory is used at all, it must produce
    at least this many units. This is what makes the problem a *mixed-integer* program
    rather than a simple linear program — each flow has a binary on/off variable
    linked to the continuous allocation variable.
    """
)

st.divider()

# ── Understanding the Output ─────────────────────────────────────────────────
st.header("Understanding the Output")

st.markdown(
    """
    Results are organized into three tiers to support decision-making at different levels:
    """
)

st.subheader("Tier 1: Chosen Flow(s)")
st.markdown(
    """
    The **optimal allocation** found by the MILP solver. If a single factory can handle
    the full volume, you'll see one flow. If capacity constraints require splitting,
    you'll see multiple flows with unit allocations that sum to your requested volume.

    This tier shows the full cost breakdown: manufacturing, transport, hub handling,
    last mile, and tariff amounts.
    """
)

st.subheader("Tier 2: Other Available Flows")
st.markdown(
    """
    The **next 3 best alternatives** by composite score, using the same weighting you
    configured. These are feasible routes that the solver did not choose but could serve
    as backups if the primary supply chain is disrupted.
    """
)

st.subheader("Tier 3: Alternative Manufacturing")
st.markdown(
    """
    **Other factory locations** not represented in Tiers 1 or 2. For each factory,
    the best hub route is shown. This tier answers the question: *"What if we need
    to source from a completely different region?"*
    """
)

st.subheader("Composite Score")
st.markdown(
    """
    A normalized 0-1 score combining cost, transit time, and regional proximity using
    your weight settings. **Lower is better.** A score of 0.0 means the flow is the
    best possible on all weighted criteria; 1.0 means it is the worst.
    """
)

st.divider()

# ── Technical Details ─────────────────────────────────────────────────────────
with st.expander("Technical Details"):
    st.markdown(
        """
        ### MILP Formulation

        **Decision variables:**
        - `x[i]` — Continuous: units allocated to flow *i*
        - `y[i]` — Binary: 1 if flow *i* is active, 0 otherwise

        **Objective:** Minimize the total weighted composite score across all flows:

        `minimize  sum( x[i] * effective_score[i]  for all i )`

        where `effective_score` is the normalized weighted combination of cost,
        transit time, and regional penalty.

        **Constraints:**
        1. **Demand satisfaction:** Total units allocated = requested volume
        2. **Factory capacity:** Each factory's total allocation <= its monthly capacity for the category
        3. **Hub throughput:** Each hub's total allocation <= its monthly throughput limit
        4. **Flow activation:** `x[i] <= volume * y[i]` (if flow is off, allocation is zero)
        5. **Minimum batch:** `x[i] >= min_batch * y[i]` (if flow is on, at least min_batch units)

        **Solver:** PuLP CBC (open-source, solves typical instances in milliseconds)

        **Pre-filtering:** Before the MILP runs, flows are filtered to exclude:
        - Geopolitically restricted routes (MADE_IN or ROUTED_THROUGH violations)
        - Routes exceeding the destination country's lead time requirement for the product category
        """
    )

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    ---
    Built with [Streamlit](https://streamlit.io), [PuLP](https://coin-or.github.io/pulp/),
    [NetworkX](https://networkx.org), [Plotly](https://plotly.com/python/),
    and [pandas](https://pandas.pydata.org/).
    """,
    unsafe_allow_html=False,
)
