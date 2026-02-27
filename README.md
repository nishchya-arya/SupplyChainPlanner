# Supply Chain Planner

A MILP-based supply chain optimizer for consumer electronics. Given a product category, target country, and order volume, it finds the optimal factory → hub → country routing that minimizes a composite score of cost, transit time, and regional proximity — while respecting factory capacity, hub throughput, geopolitical restrictions, and lead-time constraints.

**Live demo:** *(add your Streamlit Cloud URL here after deploying)*

---

## Features

- **MILP Optimizer** — PuLP/CBC solver guarantees provably optimal allocation in milliseconds across ~27 feasible flows per query
- **3-Tier Results** — Chosen flow(s), alternative routes, and all 13 manufacturing locations with restriction status
- **Knowledge Graph** — NetworkX-powered network with disruption impact analysis and category filtering
- **Geopolitical Awareness** — 11 trade restriction rules enforced at solve time (US-China, India-China, AU-China, etc.)
- **Geographic Route Map** — Plotly world map with factory → hub → destination flow lines, tiered by optimality
- **Cost Breakdown** — Donut chart decomposing manufacturing, transport, hub handling, last-mile, and tariff costs per unit

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Optimizer | PuLP (CBC solver) |
| Data | pandas + 16 CSV files |
| Network analysis | NetworkX |
| Visualization | Plotly |
| Language | Python 3.12 |

---

## Data Model

- **13 factories** across US, MX, BR, DE, GB, AE, CN (×2), KR, VN (×2), IN (×2)
- **14 distribution hubs** worldwide
- **17 destination countries** across 6 regions
- **10 product categories** (Smartphones, Laptops, Tablets, Smartwatches, Earbuds, Monitors, Speakers, Power Banks, Keyboards, Webcams)
- **~22,600 pre-computed flows** with landed cost, transit days, geopolitical flags, and lead-time feasibility

---

## Project Structure

```
Supply/
├── app.py                         Home page — nav cards, stats, mini network map
├── pages/
│   ├── 1_Knowledge_Graph.py       Network map + disruption impact analysis
│   ├── 2_Solver.py                MILP optimizer UI + 3-tier results + route map
│   └── 3_About.py                 Documentation
├── solver/
│   ├── data_loader.py             Load CSVs, build lookups, query feasible flows
│   ├── optimizer.py               MILP formulation (PuLP CBC)
│   ├── ranker.py                  3-tier ranking of solver results
│   ├── ontology.py                Typed entity layer (factory, hub, country, category)
│   ├── knowledge_graph.py         NetworkX DiGraph for network analysis
│   └── coords.py                  Geographic coordinates for map rendering
├── scripts/
│   └── generate_data.py           Synthetic data generator (produces all 16 CSVs)
├── tests/
│   └── test_solver.py             99 tests across 13 test classes
├── data/                          16 CSV files
└── requirements.txt
```

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/supply-chain-planner.git
cd supply-chain-planner
pip install -r requirements.txt
```

### 2. (Optional) Regenerate data

The `data/` directory is already populated. Only run this if you want to regenerate it from scratch.

```bash
python scripts/generate_data.py
```

### 3. Run tests

```bash
python -m pytest tests/test_solver.py -v
```

All 99 tests should pass.

### 4. Launch the app

```bash
streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo, branch `main`, main file `app.py`
4. Click **Deploy** — Streamlit auto-installs from `requirements.txt`

---

## MILP Formulation

The optimizer minimizes a weighted composite score across all feasible (factory, hub, country, category) flows:

```
minimize  SUM( x[i] * effective_cost[i] )

where effective_cost[i] = w_cost * cost_norm[i]
                        + w_time * time_norm[i]
                        + w_region * regional_penalty[i]
```

Subject to:
1. **Demand**: total allocated units = requested volume
2. **Factory capacity**: units per factory ≤ monthly capacity
3. **Hub throughput**: units per hub ≤ monthly throughput
4. **Flow activation** (big-M): units on flow i = 0 if flow i is inactive
5. **Minimum batch**: if a flow is active, at least `min_batch` units must be allocated

Binary variables (`y[i]`) determine which flows are active; continuous variables (`x[i]`) determine unit allocation. Solved by CBC in milliseconds for typical problem sizes (~27 feasible flows).

---

## License

MIT
