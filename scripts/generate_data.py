"""
Supply Chain Data Generator
===========================
Generates realistic synthetic data for a Consumer Electronics supply chain
optimization project. Costs and flows operate at the CATEGORY level (not
individual products). Products are a catalog with regional availability.

The core output is all_flows.csv (~22K rows), which pre-computes the total
landed cost for every (factory, hub, country, category) combination:
  total_landed_cost = manufacturing + transport + hub_handling + last_mile + tariff

Each flow also has transit_days (factory→hub + hub→country), a lead-time
feasibility flag, and a geopolitical restriction flag. The MILP solver
consumes this table directly.

Produces 16 CSV files in the data/ directory.

Usage: python scripts/generate_data.py
"""

import os
import math
import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════════════

ALL_REGIONS = ["NAM", "SAM", "EUR", "MEA", "NEA", "SEA"]

# ── Regions ──────────────────────────────────────────────────────────────────
REGIONS = [
    {"region_id": "NAM", "region_name": "North America"},
    {"region_id": "SAM", "region_name": "South America"},
    {"region_id": "EUR", "region_name": "Europe"},
    {"region_id": "MEA", "region_name": "Middle East and Africa"},
    {"region_id": "NEA", "region_name": "North East Asia"},
    {"region_id": "SEA", "region_name": "South East Asia and Oceania"},
]

# ── Countries ────────────────────────────────────────────────────────────────
COUNTRIES = [
    {"country_code": "US", "country_name": "United States",  "region_id": "NAM", "lat": 37.09, "lon": -95.71,  "demand_scale": 3.0,  "developed": True},
    {"country_code": "CA", "country_name": "Canada",         "region_id": "NAM", "lat": 56.13, "lon": -106.35, "demand_scale": 1.2,  "developed": True},
    {"country_code": "MX", "country_name": "Mexico",         "region_id": "NAM", "lat": 23.63, "lon": -102.55, "demand_scale": 1.0,  "developed": False},
    {"country_code": "BR", "country_name": "Brazil",         "region_id": "SAM", "lat": -14.24, "lon": -51.93, "demand_scale": 1.5,  "developed": False},
    {"country_code": "AR", "country_name": "Argentina",      "region_id": "SAM", "lat": -38.42, "lon": -63.62, "demand_scale": 0.6,  "developed": False},
    {"country_code": "DE", "country_name": "Germany",        "region_id": "EUR", "lat": 51.17, "lon": 10.45,   "demand_scale": 2.0,  "developed": True},
    {"country_code": "GB", "country_name": "United Kingdom", "region_id": "EUR", "lat": 55.38, "lon": -3.44,   "demand_scale": 1.8,  "developed": True},
    {"country_code": "FR", "country_name": "France",         "region_id": "EUR", "lat": 46.23, "lon": 2.21,    "demand_scale": 1.6,  "developed": True},
    {"country_code": "AE", "country_name": "United Arab Emirates", "region_id": "MEA", "lat": 23.42, "lon": 53.85, "demand_scale": 0.8, "developed": False},
    {"country_code": "SA", "country_name": "Saudi Arabia",   "region_id": "MEA", "lat": 23.89, "lon": 45.08,   "demand_scale": 0.7,  "developed": False},
    {"country_code": "ZA", "country_name": "South Africa",   "region_id": "MEA", "lat": -30.56, "lon": 22.94,  "demand_scale": 0.5,  "developed": False},
    {"country_code": "CN", "country_name": "China",          "region_id": "NEA", "lat": 35.86, "lon": 104.20,  "demand_scale": 2.8,  "developed": False},
    {"country_code": "JP", "country_name": "Japan",          "region_id": "NEA", "lat": 36.20, "lon": 138.25,  "demand_scale": 2.2,  "developed": True},
    {"country_code": "KR", "country_name": "South Korea",    "region_id": "NEA", "lat": 35.91, "lon": 127.77,  "demand_scale": 1.4,  "developed": True},
    {"country_code": "IN", "country_name": "India",          "region_id": "SEA", "lat": 20.59, "lon": 78.96,   "demand_scale": 2.0,  "developed": False},
    {"country_code": "VN", "country_name": "Vietnam",        "region_id": "SEA", "lat": 14.06, "lon": 108.28,  "demand_scale": 0.7,  "developed": False},
    {"country_code": "AU", "country_name": "Australia",      "region_id": "SEA", "lat": -25.27, "lon": 133.78, "demand_scale": 1.3,  "developed": True},
]

# ── Factories ────────────────────────────────────────────────────────────────
FACTORIES = [
    {"factory_id": "F_US_01", "factory_name": "Detroit Electronics",      "city": "Detroit",          "country_code": "US", "region_id": "NAM", "lat": 42.33, "lon": -83.05, "cost_multiplier": 1.00},
    {"factory_id": "F_MX_01", "factory_name": "Monterrey Assembly",       "city": "Monterrey",        "country_code": "MX", "region_id": "NAM", "lat": 25.67, "lon": -100.31, "cost_multiplier": 0.65},
    {"factory_id": "F_BR_01", "factory_name": "Sao Paulo Manufacturing",  "city": "Sao Paulo",        "country_code": "BR", "region_id": "SAM", "lat": -23.55, "lon": -46.63, "cost_multiplier": 0.70},
    {"factory_id": "F_DE_01", "factory_name": "Munich Precision Works",   "city": "Munich",           "country_code": "DE", "region_id": "EUR", "lat": 48.14, "lon": 11.58, "cost_multiplier": 1.05},
    {"factory_id": "F_GB_01", "factory_name": "Birmingham Tech Plant",    "city": "Birmingham",       "country_code": "GB", "region_id": "EUR", "lat": 52.49, "lon": -1.90, "cost_multiplier": 0.95},
    {"factory_id": "F_AE_01", "factory_name": "Dubai Industrial City",    "city": "Dubai",            "country_code": "AE", "region_id": "MEA", "lat": 25.20, "lon": 55.27, "cost_multiplier": 0.75},
    {"factory_id": "F_CN_01", "factory_name": "Shenzhen Electronics Hub", "city": "Shenzhen",         "country_code": "CN", "region_id": "NEA", "lat": 22.54, "lon": 114.06, "cost_multiplier": 0.40},
    {"factory_id": "F_CN_02", "factory_name": "Guangzhou Tech Factory",   "city": "Guangzhou",        "country_code": "CN", "region_id": "NEA", "lat": 23.13, "lon": 113.26, "cost_multiplier": 0.42},
    {"factory_id": "F_KR_01", "factory_name": "Suwon Innovation Plant",   "city": "Suwon",            "country_code": "KR", "region_id": "NEA", "lat": 37.26, "lon": 127.03, "cost_multiplier": 0.72},
    {"factory_id": "F_VN_01", "factory_name": "Hanoi Assembly Plant",     "city": "Hanoi",            "country_code": "VN", "region_id": "SEA", "lat": 21.03, "lon": 105.85, "cost_multiplier": 0.38},
    {"factory_id": "F_VN_02", "factory_name": "HCMC Electronics",         "city": "Ho Chi Minh City", "country_code": "VN", "region_id": "SEA", "lat": 10.82, "lon": 106.63, "cost_multiplier": 0.40},
    {"factory_id": "F_IN_01", "factory_name": "Chennai Manufacturing",    "city": "Chennai",          "country_code": "IN", "region_id": "SEA", "lat": 13.08, "lon": 80.27, "cost_multiplier": 0.45},
    {"factory_id": "F_IN_02", "factory_name": "Bangalore Tech Works",     "city": "Bangalore",        "country_code": "IN", "region_id": "SEA", "lat": 12.97, "lon": 77.59, "cost_multiplier": 0.50},
]

# ── Hubs ─────────────────────────────────────────────────────────────────────
HUBS = [
    {"hub_id": "H_US_01", "hub_name": "Houston Distribution Center",      "city": "Houston",          "country_code": "US", "region_id": "NAM", "lat": 29.76, "lon": -95.37, "monthly_throughput_capacity": 200000},
    {"hub_id": "H_CA_01", "hub_name": "Toronto Logistics Hub",            "city": "Toronto",          "country_code": "CA", "region_id": "NAM", "lat": 43.65, "lon": -79.38, "monthly_throughput_capacity": 120000},
    {"hub_id": "H_MX_01", "hub_name": "Mexico City Warehouse",            "city": "Mexico City",      "country_code": "MX", "region_id": "NAM", "lat": 19.43, "lon": -99.13, "monthly_throughput_capacity": 100000},
    {"hub_id": "H_BR_01", "hub_name": "Sao Paulo Distribution",           "city": "Sao Paulo",        "country_code": "BR", "region_id": "SAM", "lat": -23.55, "lon": -46.63, "monthly_throughput_capacity": 100000},
    {"hub_id": "H_DE_01", "hub_name": "Frankfurt Logistics Center",       "city": "Frankfurt",        "country_code": "DE", "region_id": "EUR", "lat": 50.11, "lon": 8.68, "monthly_throughput_capacity": 180000},
    {"hub_id": "H_GB_01", "hub_name": "London Distribution Hub",          "city": "London",           "country_code": "GB", "region_id": "EUR", "lat": 51.51, "lon": -0.13, "monthly_throughput_capacity": 150000},
    {"hub_id": "H_AE_01", "hub_name": "Dubai Logistics Hub",              "city": "Dubai",            "country_code": "AE", "region_id": "MEA", "lat": 25.20, "lon": 55.27, "monthly_throughput_capacity": 130000},
    {"hub_id": "H_ZA_01", "hub_name": "Johannesburg Distribution Center", "city": "Johannesburg",     "country_code": "ZA", "region_id": "MEA", "lat": -26.20, "lon": 28.05, "monthly_throughput_capacity": 70000},
    {"hub_id": "H_CN_01", "hub_name": "Shanghai Mega Hub",                "city": "Shanghai",         "country_code": "CN", "region_id": "NEA", "lat": 31.23, "lon": 121.47, "monthly_throughput_capacity": 200000},
    {"hub_id": "H_JP_01", "hub_name": "Tokyo Distribution Center",        "city": "Tokyo",            "country_code": "JP", "region_id": "NEA", "lat": 35.68, "lon": 139.69, "monthly_throughput_capacity": 140000},
    {"hub_id": "H_KR_01", "hub_name": "Busan Port Logistics",             "city": "Busan",            "country_code": "KR", "region_id": "NEA", "lat": 35.18, "lon": 129.08, "monthly_throughput_capacity": 120000},
    {"hub_id": "H_IN_01", "hub_name": "Mumbai Logistics Center",          "city": "Mumbai",           "country_code": "IN", "region_id": "SEA", "lat": 19.08, "lon": 72.88, "monthly_throughput_capacity": 130000},
    {"hub_id": "H_VN_01", "hub_name": "HCMC Distribution Hub",            "city": "Ho Chi Minh City", "country_code": "VN", "region_id": "SEA", "lat": 10.82, "lon": 106.63, "monthly_throughput_capacity": 80000},
    {"hub_id": "H_AU_01", "hub_name": "Sydney Logistics Center",          "city": "Sydney",           "country_code": "AU", "region_id": "SEA", "lat": -33.87, "lon": 151.21, "monthly_throughput_capacity": 90000},
]

# ── Product Categories ───────────────────────────────────────────────────────
# Costs and weights live at category level. All products within a category
# have the same manufacturing cost at a given factory.
# urgency: 1=high (smartphones: tight lead times), 2=medium, 3=low (keyboards: relaxed)
CATEGORIES = [
    {"category_id": "CAT01", "category_name": "Smartphones",      "urgency": 1, "base_manufacturing_cost_usd": 250, "representative_weight_kg": 0.22},
    {"category_id": "CAT02", "category_name": "Laptops",           "urgency": 1, "base_manufacturing_cost_usd": 580, "representative_weight_kg": 1.80},
    {"category_id": "CAT03", "category_name": "Tablets",           "urgency": 1, "base_manufacturing_cost_usd": 270, "representative_weight_kg": 0.55},
    {"category_id": "CAT04", "category_name": "Smartwatches",      "urgency": 2, "base_manufacturing_cost_usd": 125, "representative_weight_kg": 0.07},
    {"category_id": "CAT05", "category_name": "Wireless Earbuds",  "urgency": 2, "base_manufacturing_cost_usd": 55,  "representative_weight_kg": 0.07},
    {"category_id": "CAT06", "category_name": "Monitors",          "urgency": 2, "base_manufacturing_cost_usd": 300, "representative_weight_kg": 5.50},
    {"category_id": "CAT07", "category_name": "Smart Speakers",    "urgency": 3, "base_manufacturing_cost_usd": 50,  "representative_weight_kg": 0.75},
    {"category_id": "CAT08", "category_name": "Power Banks",       "urgency": 3, "base_manufacturing_cost_usd": 28,  "representative_weight_kg": 0.30},
    {"category_id": "CAT09", "category_name": "Keyboards",         "urgency": 2, "base_manufacturing_cost_usd": 55,  "representative_weight_kg": 0.85},
    {"category_id": "CAT10", "category_name": "Webcams",           "urgency": 2, "base_manufacturing_cost_usd": 100, "representative_weight_kg": 0.40},
]

# ── Products (catalog with regional availability) ───────────────────────────
# Products within a category share manufacturing cost/weight.
# They differ by: price tier, demand weight, regional availability.
G = ALL_REGIONS  # shorthand for global

PRODUCTS = [
    # ── CAT01 Smartphones (10) ──
    {"product_id": "P001", "product_name": "Alpha Pro Max",     "category_id": "CAT01", "retail_price_tier": "premium", "relative_demand_weight": 1.5, "regions": G},
    {"product_id": "P002", "product_name": "Galaxy Ultra",      "category_id": "CAT01", "retail_price_tier": "premium", "relative_demand_weight": 1.4, "regions": G},
    {"product_id": "P003", "product_name": "Nova Standard",     "category_id": "CAT01", "retail_price_tier": "mid",     "relative_demand_weight": 1.2, "regions": G},
    {"product_id": "P004", "product_name": "Pixel Lite",        "category_id": "CAT01", "retail_price_tier": "mid",     "relative_demand_weight": 1.0, "regions": ["NAM", "EUR"]},
    {"product_id": "P005", "product_name": "Redmi Value",       "category_id": "CAT01", "retail_price_tier": "budget",  "relative_demand_weight": 1.8, "regions": ["NEA", "SEA"]},
    {"product_id": "P006", "product_name": "Samba Phone",       "category_id": "CAT01", "retail_price_tier": "budget",  "relative_demand_weight": 0.8, "regions": ["SAM"]},
    {"product_id": "P007", "product_name": "Desert Connect",    "category_id": "CAT01", "retail_price_tier": "mid",     "relative_demand_weight": 0.6, "regions": ["MEA"]},
    {"product_id": "P008", "product_name": "Euro Slim",         "category_id": "CAT01", "retail_price_tier": "mid",     "relative_demand_weight": 0.9, "regions": ["EUR"]},
    {"product_id": "P009", "product_name": "Pacific Phone",     "category_id": "CAT01", "retail_price_tier": "mid",     "relative_demand_weight": 0.7, "regions": ["SEA"]},
    {"product_id": "P010", "product_name": "Liberty Mobile",    "category_id": "CAT01", "retail_price_tier": "budget",  "relative_demand_weight": 1.0, "regions": ["NAM"]},

    # ── CAT02 Laptops (9) ──
    {"product_id": "P011", "product_name": "Ultrabook Pro",     "category_id": "CAT02", "retail_price_tier": "premium", "relative_demand_weight": 1.3, "regions": G},
    {"product_id": "P012", "product_name": "Gaming Beast",      "category_id": "CAT02", "retail_price_tier": "premium", "relative_demand_weight": 1.4, "regions": G},
    {"product_id": "P013", "product_name": "Business Elite",    "category_id": "CAT02", "retail_price_tier": "mid",     "relative_demand_weight": 1.1, "regions": G},
    {"product_id": "P014", "product_name": "Student Laptop",    "category_id": "CAT02", "retail_price_tier": "budget",  "relative_demand_weight": 1.6, "regions": ["NAM", "EUR", "SEA"]},
    {"product_id": "P015", "product_name": "Creator Studio",    "category_id": "CAT02", "retail_price_tier": "premium", "relative_demand_weight": 0.8, "regions": ["NAM", "EUR", "NEA"]},
    {"product_id": "P016", "product_name": "Asia Slim Book",    "category_id": "CAT02", "retail_price_tier": "mid",     "relative_demand_weight": 0.9, "regions": ["NEA", "SEA"]},
    {"product_id": "P017", "product_name": "Sahara Notebook",   "category_id": "CAT02", "retail_price_tier": "budget",  "relative_demand_weight": 0.5, "regions": ["MEA"]},
    {"product_id": "P018", "product_name": "Pampas Laptop",     "category_id": "CAT02", "retail_price_tier": "budget",  "relative_demand_weight": 0.6, "regions": ["SAM"]},
    {"product_id": "P019", "product_name": "Euro WorkStation",  "category_id": "CAT02", "retail_price_tier": "mid",     "relative_demand_weight": 0.7, "regions": ["EUR"]},

    # ── CAT03 Tablets (8) ──
    {"product_id": "P020", "product_name": "Tab Pro 12",        "category_id": "CAT03", "retail_price_tier": "premium", "relative_demand_weight": 1.3, "regions": G},
    {"product_id": "P021", "product_name": "Tab Standard",      "category_id": "CAT03", "retail_price_tier": "mid",     "relative_demand_weight": 1.4, "regions": G},
    {"product_id": "P022", "product_name": "Tab Lite",          "category_id": "CAT03", "retail_price_tier": "budget",  "relative_demand_weight": 1.5, "regions": G},
    {"product_id": "P023", "product_name": "Tab Education",     "category_id": "CAT03", "retail_price_tier": "budget",  "relative_demand_weight": 1.2, "regions": ["NAM", "EUR", "SEA"]},
    {"product_id": "P024", "product_name": "Tab Kids",          "category_id": "CAT03", "retail_price_tier": "budget",  "relative_demand_weight": 0.9, "regions": ["NAM", "EUR"]},
    {"product_id": "P025", "product_name": "Silk Pad",          "category_id": "CAT03", "retail_price_tier": "mid",     "relative_demand_weight": 0.8, "regions": ["NEA", "SEA"]},
    {"product_id": "P026", "product_name": "Arena Tablet",      "category_id": "CAT03", "retail_price_tier": "mid",     "relative_demand_weight": 0.5, "regions": ["SAM", "MEA"]},
    {"product_id": "P027", "product_name": "Nordic Slate",      "category_id": "CAT03", "retail_price_tier": "premium", "relative_demand_weight": 0.6, "regions": ["EUR"]},

    # ── CAT04 Smartwatches (9) ──
    {"product_id": "P028", "product_name": "Chrono Elite",      "category_id": "CAT04", "retail_price_tier": "premium", "relative_demand_weight": 1.3, "regions": G},
    {"product_id": "P029", "product_name": "Fitness Band Pro",  "category_id": "CAT04", "retail_price_tier": "mid",     "relative_demand_weight": 1.5, "regions": G},
    {"product_id": "P030", "product_name": "Sport Watch X",     "category_id": "CAT04", "retail_price_tier": "mid",     "relative_demand_weight": 1.2, "regions": G},
    {"product_id": "P031", "product_name": "Health Tracker",    "category_id": "CAT04", "retail_price_tier": "budget",  "relative_demand_weight": 1.6, "regions": ["NAM", "EUR", "SEA"]},
    {"product_id": "P032", "product_name": "Zen Watch",         "category_id": "CAT04", "retail_price_tier": "premium", "relative_demand_weight": 0.7, "regions": ["NEA"]},
    {"product_id": "P033", "product_name": "Outback Band",      "category_id": "CAT04", "retail_price_tier": "budget",  "relative_demand_weight": 0.6, "regions": ["SEA"]},
    {"product_id": "P034", "product_name": "Gulf Timer",        "category_id": "CAT04", "retail_price_tier": "premium", "relative_demand_weight": 0.5, "regions": ["MEA"]},
    {"product_id": "P035", "product_name": "Rio Pulse",         "category_id": "CAT04", "retail_price_tier": "budget",  "relative_demand_weight": 0.6, "regions": ["SAM"]},
    {"product_id": "P036", "product_name": "Maple Fit",         "category_id": "CAT04", "retail_price_tier": "mid",     "relative_demand_weight": 0.8, "regions": ["NAM"]},

    # ── CAT05 Wireless Earbuds (8) ──
    {"product_id": "P037", "product_name": "SoundPods Pro",     "category_id": "CAT05", "retail_price_tier": "premium", "relative_demand_weight": 1.4, "regions": G},
    {"product_id": "P038", "product_name": "BassBuds",          "category_id": "CAT05", "retail_price_tier": "mid",     "relative_demand_weight": 1.3, "regions": G},
    {"product_id": "P039", "product_name": "EcoBuds",           "category_id": "CAT05", "retail_price_tier": "budget",  "relative_demand_weight": 1.7, "regions": G},
    {"product_id": "P040", "product_name": "ActivePods",        "category_id": "CAT05", "retail_price_tier": "mid",     "relative_demand_weight": 1.0, "regions": ["NAM", "EUR"]},
    {"product_id": "P041", "product_name": "Harmony Buds",      "category_id": "CAT05", "retail_price_tier": "mid",     "relative_demand_weight": 0.9, "regions": ["NEA", "SEA"]},
    {"product_id": "P042", "product_name": "Carnival Pods",     "category_id": "CAT05", "retail_price_tier": "budget",  "relative_demand_weight": 0.7, "regions": ["SAM"]},
    {"product_id": "P043", "product_name": "Dune Audio",        "category_id": "CAT05", "retail_price_tier": "mid",     "relative_demand_weight": 0.5, "regions": ["MEA"]},
    {"product_id": "P044", "product_name": "Studio Silence",    "category_id": "CAT05", "retail_price_tier": "premium", "relative_demand_weight": 0.8, "regions": ["NAM", "EUR", "NEA"]},

    # ── CAT06 Monitors (10) ──
    {"product_id": "P045", "product_name": "Office Display 24", "category_id": "CAT06", "retail_price_tier": "budget",  "relative_demand_weight": 1.5, "regions": G},
    {"product_id": "P046", "product_name": "Gaming Display 27", "category_id": "CAT06", "retail_price_tier": "mid",     "relative_demand_weight": 1.3, "regions": G},
    {"product_id": "P047", "product_name": "Pro Display 32",    "category_id": "CAT06", "retail_price_tier": "premium", "relative_demand_weight": 1.1, "regions": G},
    {"product_id": "P048", "product_name": "Ultrawide 34",      "category_id": "CAT06", "retail_price_tier": "premium", "relative_demand_weight": 0.9, "regions": ["NAM", "EUR", "NEA"]},
    {"product_id": "P049", "product_name": "Curved Gaming 32",  "category_id": "CAT06", "retail_price_tier": "mid",     "relative_demand_weight": 1.0, "regions": ["NAM", "EUR"]},
    {"product_id": "P050", "product_name": "Budget Panel 22",   "category_id": "CAT06", "retail_price_tier": "budget",  "relative_demand_weight": 1.4, "regions": ["SEA", "SAM", "MEA"]},
    {"product_id": "P051", "product_name": "K-Display",         "category_id": "CAT06", "retail_price_tier": "mid",     "relative_demand_weight": 0.7, "regions": ["NEA"]},
    {"product_id": "P052", "product_name": "Studio Color",      "category_id": "CAT06", "retail_price_tier": "premium", "relative_demand_weight": 0.6, "regions": ["EUR"]},
    {"product_id": "P053", "product_name": "Portable Monitor",  "category_id": "CAT06", "retail_price_tier": "mid",     "relative_demand_weight": 0.8, "regions": ["NAM", "NEA", "SEA"]},
    {"product_id": "P054", "product_name": "Boardroom Screen",  "category_id": "CAT06", "retail_price_tier": "premium", "relative_demand_weight": 0.5, "regions": ["NAM", "EUR", "MEA"]},

    # ── CAT07 Smart Speakers (7) ──
    {"product_id": "P055", "product_name": "Echo Mini",         "category_id": "CAT07", "retail_price_tier": "budget",  "relative_demand_weight": 1.6, "regions": G},
    {"product_id": "P056", "product_name": "Home Hub",          "category_id": "CAT07", "retail_price_tier": "mid",     "relative_demand_weight": 1.3, "regions": G},
    {"product_id": "P057", "product_name": "Sound Tower",       "category_id": "CAT07", "retail_price_tier": "premium", "relative_demand_weight": 0.9, "regions": ["NAM", "EUR"]},
    {"product_id": "P058", "product_name": "Smart Orb",         "category_id": "CAT07", "retail_price_tier": "mid",     "relative_demand_weight": 0.8, "regions": ["NEA", "SEA"]},
    {"product_id": "P059", "product_name": "Voice Cube",        "category_id": "CAT07", "retail_price_tier": "budget",  "relative_demand_weight": 1.0, "regions": ["SAM", "MEA"]},
    {"product_id": "P060", "product_name": "Bamboo Speaker",    "category_id": "CAT07", "retail_price_tier": "mid",     "relative_demand_weight": 0.5, "regions": ["SEA"]},
    {"product_id": "P061", "product_name": "Alto Speaker",      "category_id": "CAT07", "retail_price_tier": "premium", "relative_demand_weight": 0.6, "regions": ["EUR"]},

    # ── CAT08 Power Banks (8) ──
    {"product_id": "P062", "product_name": "Pocket Charge",     "category_id": "CAT08", "retail_price_tier": "budget",  "relative_demand_weight": 1.7, "regions": G},
    {"product_id": "P063", "product_name": "Mega Power",        "category_id": "CAT08", "retail_price_tier": "mid",     "relative_demand_weight": 1.3, "regions": G},
    {"product_id": "P064", "product_name": "Solar Charge",      "category_id": "CAT08", "retail_price_tier": "premium", "relative_demand_weight": 0.9, "regions": G},
    {"product_id": "P065", "product_name": "Rugged Power",      "category_id": "CAT08", "retail_price_tier": "mid",     "relative_demand_weight": 0.8, "regions": ["NAM", "EUR", "MEA"]},
    {"product_id": "P066", "product_name": "Slim Charge",       "category_id": "CAT08", "retail_price_tier": "budget",  "relative_demand_weight": 1.2, "regions": ["NEA", "SEA"]},
    {"product_id": "P067", "product_name": "Tropic Power",      "category_id": "CAT08", "retail_price_tier": "budget",  "relative_demand_weight": 0.7, "regions": ["SAM", "SEA"]},
    {"product_id": "P068", "product_name": "Desert Reserve",    "category_id": "CAT08", "retail_price_tier": "mid",     "relative_demand_weight": 0.5, "regions": ["MEA"]},
    {"product_id": "P069", "product_name": "Fast Charge Pro",   "category_id": "CAT08", "retail_price_tier": "premium", "relative_demand_weight": 0.7, "regions": ["NAM", "EUR"]},

    # ── CAT09 Keyboards (9) ──
    {"product_id": "P070", "product_name": "Mech Warrior",      "category_id": "CAT09", "retail_price_tier": "premium", "relative_demand_weight": 1.2, "regions": G},
    {"product_id": "P071", "product_name": "Wireless Comfort",  "category_id": "CAT09", "retail_price_tier": "mid",     "relative_demand_weight": 1.4, "regions": G},
    {"product_id": "P072", "product_name": "Gaming Clicker",    "category_id": "CAT09", "retail_price_tier": "premium", "relative_demand_weight": 1.1, "regions": G},
    {"product_id": "P073", "product_name": "Ergo Board",        "category_id": "CAT09", "retail_price_tier": "mid",     "relative_demand_weight": 0.9, "regions": ["NAM", "EUR"]},
    {"product_id": "P074", "product_name": "Compact Keys",      "category_id": "CAT09", "retail_price_tier": "budget",  "relative_demand_weight": 1.3, "regions": ["NEA", "SEA"]},
    {"product_id": "P075", "product_name": "Office Basic",      "category_id": "CAT09", "retail_price_tier": "budget",  "relative_demand_weight": 1.5, "regions": ["SAM", "MEA", "SEA"]},
    {"product_id": "P076", "product_name": "Silent Type",       "category_id": "CAT09", "retail_price_tier": "mid",     "relative_demand_weight": 0.7, "regions": ["EUR"]},
    {"product_id": "P077", "product_name": "Sakura Board",      "category_id": "CAT09", "retail_price_tier": "mid",     "relative_demand_weight": 0.6, "regions": ["NEA"]},
    {"product_id": "P078", "product_name": "Flex Keyboard",     "category_id": "CAT09", "retail_price_tier": "budget",  "relative_demand_weight": 0.8, "regions": ["NAM", "SAM"]},

    # ── CAT10 Webcams (8) ──
    {"product_id": "P079", "product_name": "ClearView HD",      "category_id": "CAT10", "retail_price_tier": "budget",  "relative_demand_weight": 1.5, "regions": G},
    {"product_id": "P080", "product_name": "ProCam 4K",         "category_id": "CAT10", "retail_price_tier": "premium", "relative_demand_weight": 1.2, "regions": G},
    {"product_id": "P081", "product_name": "StreamCam",         "category_id": "CAT10", "retail_price_tier": "mid",     "relative_demand_weight": 1.3, "regions": G},
    {"product_id": "P082", "product_name": "Conference Eye",    "category_id": "CAT10", "retail_price_tier": "premium", "relative_demand_weight": 0.9, "regions": ["NAM", "EUR", "NEA"]},
    {"product_id": "P083", "product_name": "Budget Cam",        "category_id": "CAT10", "retail_price_tier": "budget",  "relative_demand_weight": 1.4, "regions": ["SEA", "SAM", "MEA"]},
    {"product_id": "P084", "product_name": "Night Vision Cam",  "category_id": "CAT10", "retail_price_tier": "mid",     "relative_demand_weight": 0.7, "regions": ["NAM", "EUR"]},
    {"product_id": "P085", "product_name": "Mini Cam",          "category_id": "CAT10", "retail_price_tier": "budget",  "relative_demand_weight": 0.8, "regions": ["NEA", "SEA"]},
    {"product_id": "P086", "product_name": "Boardroom Cam",     "category_id": "CAT10", "retail_price_tier": "premium", "relative_demand_weight": 0.5, "regions": ["NAM", "EUR", "MEA"]},
]

# ── Factory-Category Assignments ────────────────────────────────────────────
# Which categories each factory can manufacture.
# Every category has at least 3 factories, at least 1 in unrestricted country.
ALL_CAT_IDS = [c["category_id"] for c in CATEGORIES]

FACTORY_CATEGORY_MAP = {
    # China: makes everything (cheapest)
    "F_CN_01": ALL_CAT_IDS[:],
    "F_CN_02": ALL_CAT_IDS[:],
    # Vietnam: most categories (second cheapest)
    "F_VN_01": ["CAT01", "CAT03", "CAT04", "CAT05", "CAT07", "CAT08", "CAT09", "CAT10"],
    "F_VN_02": ["CAT01", "CAT02", "CAT05", "CAT06", "CAT07", "CAT08", "CAT09", "CAT10"],
    # India: growing hub
    "F_IN_01": ["CAT01", "CAT03", "CAT04", "CAT05", "CAT07", "CAT08", "CAT09"],
    "F_IN_02": ["CAT01", "CAT02", "CAT04", "CAT06", "CAT08", "CAT09", "CAT10"],
    # South Korea: quality
    "F_KR_01": ["CAT01", "CAT02", "CAT03", "CAT04", "CAT05", "CAT06", "CAT09", "CAT10"],
    # US: high-end focus
    "F_US_01": ["CAT01", "CAT02", "CAT03", "CAT06", "CAT07", "CAT09", "CAT10"],
    # Germany: precision/premium
    "F_DE_01": ["CAT02", "CAT03", "CAT04", "CAT06", "CAT09", "CAT10"],
    # UK: moderate range
    "F_GB_01": ["CAT01", "CAT02", "CAT03", "CAT05", "CAT07", "CAT09"],
    # Mexico: USMCA-friendly
    "F_MX_01": ["CAT01", "CAT03", "CAT05", "CAT07", "CAT08", "CAT09", "CAT10"],
    # Brazil: regional
    "F_BR_01": ["CAT01", "CAT03", "CAT05", "CAT07", "CAT08", "CAT09"],
    # UAE: niche
    "F_AE_01": ["CAT01", "CAT04", "CAT05", "CAT07", "CAT08"],
}

# ── Geopolitical Restrictions ────────────────────────────────────────────────
GEOPOLITICAL_RESTRICTIONS = [
    {"destination_country_code": "US", "restricted_country_code": "CN", "restriction_type": "MADE_IN",         "reason": "US-China trade restrictions"},
    {"destination_country_code": "US", "restricted_country_code": "CN", "restriction_type": "ROUTED_THROUGH",  "reason": "US security compliance"},
    {"destination_country_code": "US", "restricted_country_code": "BR", "restriction_type": "MADE_IN",         "reason": "US-Brazil trade policy"},
    {"destination_country_code": "CA", "restricted_country_code": "CN", "restriction_type": "MADE_IN",         "reason": "Aligned with US-China policy"},
    {"destination_country_code": "CA", "restricted_country_code": "CN", "restriction_type": "ROUTED_THROUGH",  "reason": "Aligned with US security policy"},
    {"destination_country_code": "IN", "restricted_country_code": "CN", "restriction_type": "MADE_IN",         "reason": "India-China border tensions"},
    {"destination_country_code": "IN", "restricted_country_code": "CN", "restriction_type": "ROUTED_THROUGH",  "reason": "India security concerns"},
    {"destination_country_code": "CN", "restricted_country_code": "US", "restriction_type": "MADE_IN",         "reason": "China reciprocal restrictions"},
    {"destination_country_code": "CN", "restricted_country_code": "US", "restriction_type": "ROUTED_THROUGH",  "reason": "China reciprocal restrictions"},
    {"destination_country_code": "AU", "restricted_country_code": "CN", "restriction_type": "MADE_IN",         "reason": "Australia-China trade dispute"},
    {"destination_country_code": "JP", "restricted_country_code": "CN", "restriction_type": "ROUTED_THROUGH",  "reason": "Japan regional security policy"},
]

# ── Tariff Rates ─────────────────────────────────────────────────────────────
TARIFF_RULES = {
    # Within-region / FTA rates (low)
    ("US", "CA"): 0.02, ("US", "MX"): 0.01, ("CA", "US"): 0.02, ("CA", "MX"): 0.03,
    ("MX", "US"): 0.01, ("MX", "CA"): 0.03,
    ("BR", "AR"): 0.04, ("AR", "BR"): 0.04,
    ("DE", "GB"): 0.03, ("DE", "FR"): 0.01, ("GB", "DE"): 0.04, ("GB", "FR"): 0.04,
    ("CN", "KR"): 0.05, ("CN", "JP"): 0.06, ("KR", "CN"): 0.05, ("KR", "JP"): 0.04,
    ("JP", "KR"): 0.04,
    ("VN", "IN"): 0.06, ("IN", "VN"): 0.06, ("VN", "AU"): 0.05, ("IN", "AU"): 0.06,
    ("AE", "SA"): 0.02, ("SA", "AE"): 0.02, ("AE", "ZA"): 0.08, ("ZA", "AE"): 0.08,
    # China to developed markets (HIGH tariffs)
    ("CN", "US"): 0.30, ("CN", "CA"): 0.28, ("CN", "DE"): 0.28, ("CN", "GB"): 0.25,
    ("CN", "FR"): 0.28, ("CN", "AU"): 0.32, ("CN", "JP"): 0.15,
    # China to emerging markets (lower)
    ("CN", "MX"): 0.10, ("CN", "BR"): 0.12, ("CN", "AR"): 0.12,
    ("CN", "AE"): 0.05, ("CN", "SA"): 0.05, ("CN", "ZA"): 0.08,
    ("CN", "IN"): 0.18, ("CN", "VN"): 0.08,
    # Vietnam to US/EU (low, trade agreements)
    ("VN", "US"): 0.05, ("VN", "CA"): 0.06, ("VN", "DE"): 0.08, ("VN", "GB"): 0.07,
    ("VN", "FR"): 0.08, ("VN", "JP"): 0.05, ("VN", "KR"): 0.05,
    # India to developed (moderate)
    ("IN", "US"): 0.08, ("IN", "CA"): 0.09, ("IN", "DE"): 0.10, ("IN", "GB"): 0.06,
    ("IN", "FR"): 0.10, ("IN", "JP"): 0.08, ("IN", "KR"): 0.08,
    # South Korea (KORUS FTA)
    ("KR", "US"): 0.03, ("KR", "CA"): 0.05, ("KR", "DE"): 0.06, ("KR", "GB"): 0.05,
    ("KR", "FR"): 0.06, ("KR", "AU"): 0.05,
    # US to others
    ("US", "DE"): 0.06, ("US", "GB"): 0.05, ("US", "FR"): 0.06, ("US", "JP"): 0.04,
    ("US", "KR"): 0.05, ("US", "AU"): 0.04, ("US", "CN"): 0.25,
    ("US", "IN"): 0.08, ("US", "BR"): 0.10, ("US", "AR"): 0.12,
    ("US", "AE"): 0.05, ("US", "SA"): 0.05, ("US", "ZA"): 0.08,
    # Germany to non-EU
    ("DE", "US"): 0.05, ("DE", "CA"): 0.06, ("DE", "JP"): 0.05, ("DE", "KR"): 0.06,
    ("DE", "AU"): 0.06, ("DE", "CN"): 0.08, ("DE", "IN"): 0.08,
    ("DE", "BR"): 0.12, ("DE", "AR"): 0.14, ("DE", "AE"): 0.04, ("DE", "SA"): 0.04,
    ("DE", "ZA"): 0.08, ("DE", "MX"): 0.08, ("DE", "VN"): 0.07,
    # UK to non-EU
    ("GB", "US"): 0.05, ("GB", "CA"): 0.06, ("GB", "JP"): 0.06, ("GB", "KR"): 0.07,
    ("GB", "AU"): 0.04, ("GB", "CN"): 0.10, ("GB", "IN"): 0.07,
    ("GB", "BR"): 0.12, ("GB", "AR"): 0.14, ("GB", "AE"): 0.05, ("GB", "SA"): 0.05,
    ("GB", "ZA"): 0.09, ("GB", "MX"): 0.09, ("GB", "VN"): 0.08,
    # Brazil to others
    ("BR", "US"): 0.10, ("BR", "CA"): 0.12, ("BR", "DE"): 0.14, ("BR", "GB"): 0.14,
    ("BR", "FR"): 0.14, ("BR", "JP"): 0.12, ("BR", "KR"): 0.12, ("BR", "AU"): 0.14,
    ("BR", "CN"): 0.10, ("BR", "IN"): 0.10, ("BR", "AE"): 0.08, ("BR", "SA"): 0.08,
    ("BR", "ZA"): 0.10, ("BR", "MX"): 0.08, ("BR", "VN"): 0.12,
    # Mexico
    ("MX", "DE"): 0.08, ("MX", "GB"): 0.08, ("MX", "FR"): 0.08, ("MX", "JP"): 0.10,
    ("MX", "KR"): 0.10, ("MX", "AU"): 0.12, ("MX", "CN"): 0.10, ("MX", "IN"): 0.10,
    ("MX", "BR"): 0.08, ("MX", "AR"): 0.10, ("MX", "AE"): 0.08, ("MX", "SA"): 0.08,
    ("MX", "ZA"): 0.12, ("MX", "VN"): 0.10,
    # UAE
    ("AE", "US"): 0.06, ("AE", "CA"): 0.07, ("AE", "DE"): 0.05, ("AE", "GB"): 0.05,
    ("AE", "FR"): 0.05, ("AE", "JP"): 0.06, ("AE", "KR"): 0.06, ("AE", "AU"): 0.08,
    ("AE", "CN"): 0.05, ("AE", "IN"): 0.04, ("AE", "BR"): 0.10, ("AE", "AR"): 0.12,
    ("AE", "MX"): 0.10, ("AE", "VN"): 0.06,
    # India extras
    ("IN", "MX"): 0.10, ("IN", "BR"): 0.10, ("IN", "AR"): 0.12,
    ("IN", "AE"): 0.04, ("IN", "SA"): 0.05, ("IN", "ZA"): 0.08,
    ("IN", "CN"): 0.18,
    # Vietnam extras
    ("VN", "MX"): 0.10, ("VN", "BR"): 0.12, ("VN", "AR"): 0.14,
    ("VN", "AE"): 0.06, ("VN", "SA"): 0.06, ("VN", "ZA"): 0.10,
    ("VN", "CN"): 0.08,
}
DEFAULT_TARIFF = 0.10
SAME_COUNTRY_TARIFF = 0.00


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points on Earth in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def add_noise(value, pct=0.10):
    """Add random noise of +/-pct to a value."""
    return round(value * (1 + rng.uniform(-pct, pct)), 2)


def seasonality_factor(month):
    """Monthly seasonality multiplier for consumer electronics demand.
    Q1 is low (post-holiday), Q4 peaks (Nov=1.45x, Dec=1.55x for holiday season)."""
    factors = {
        1: 0.80, 2: 0.85, 3: 0.90, 4: 0.92, 5: 0.95, 6: 1.00,
        7: 1.00, 8: 1.05, 9: 1.10, 10: 1.15, 11: 1.45, 12: 1.55,
    }
    return factors[month]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_lookups():
    """Build lookup dicts for quick access."""
    factory_lookup = {f["factory_id"]: f for f in FACTORIES}
    hub_lookup = {h["hub_id"]: h for h in HUBS}
    country_lookup = {c["country_code"]: c for c in COUNTRIES}
    category_lookup = {c["category_id"]: c for c in CATEGORIES}
    return factory_lookup, hub_lookup, country_lookup, category_lookup


def generate_product_availability():
    """Generate product_availability rows from PRODUCTS regions field."""
    rows = []
    for p in PRODUCTS:
        for region_id in p["regions"]:
            rows.append({
                "product_id": p["product_id"],
                "region_id": region_id,
            })
    return rows


def generate_factory_category_capacity(factory_lookup, category_lookup):
    """Generate manufacturing costs and capacity for each factory-category pair.
    Cost formula: base_cost * factory_cost_multiplier * (1 ± 8% noise).
    Capacity: random within country-specific ranges (China highest, US/EU lowest)."""
    # Monthly capacity ranges by factory country (units)
    capacity_ranges = {
        "CN": (8000, 15000),
        "VN": (5000, 10000),
        "IN": (5000, 10000),
        "KR": (6000, 12000),
        "US": (3000, 7000),
        "DE": (3000, 7000),
        "GB": (3000, 7000),
        "MX": (3000, 6000),
        "BR": (3000, 6000),
        "AE": (3000, 6000),
    }

    rows = []
    for factory_id, cat_ids in FACTORY_CATEGORY_MAP.items():
        factory = factory_lookup[factory_id]
        cc = factory["country_code"]
        cap_lo, cap_hi = capacity_ranges[cc]

        for cat_id in cat_ids:
            cat = category_lookup[cat_id]
            base_cost = cat["base_manufacturing_cost_usd"]
            mfg_cost = round(base_cost * factory["cost_multiplier"] * (1 + rng.uniform(-0.08, 0.08)), 2)
            capacity = int(rng.integers(cap_lo, cap_hi + 1))
            rows.append({
                "factory_id": factory_id,
                "category_id": cat_id,
                "unit_manufacturing_cost_usd": mfg_cost,
                "monthly_capacity_units": capacity,
            })
    return rows


def generate_transport_costs(factory_lookup, hub_lookup):
    """Generate transport cost and transit days for every factory-hub pair.
    Distance uses haversine * 1.3 (effective distance accounts for routing,
    customs, not-straight-line shipping). Transit days = effective_dist / KM_PER_DAY.
    Cost is per kg, later scaled by category weight in generate_all_flows()."""
    BASE_RATE_PER_KM_KG = 0.003   # $/km/kg base shipping rate
    MIN_COST_PER_UNIT = 1.50       # minimum cost floor (local shipments)
    AVG_WEIGHT_KG = 1.0            # base rate for 1kg; scaled by actual weight later
    KM_PER_DAY = 400               # assumed avg shipping speed

    rows = []
    for f in FACTORIES:
        for h in HUBS:
            dist = haversine_km(f["lat"], f["lon"], h["lat"], h["lon"])
            effective_dist = dist * 1.3
            cost = max(MIN_COST_PER_UNIT, round(effective_dist * BASE_RATE_PER_KM_KG * AVG_WEIGHT_KG, 2))
            cost = add_noise(cost, 0.12)
            transit = max(2, int(round(effective_dist / KM_PER_DAY)))
            if transit > 10:
                transit += int(rng.integers(-2, 3))
            transit = max(2, transit)
            rows.append({
                "factory_id": f["factory_id"],
                "hub_id": h["hub_id"],
                "cost_per_unit_usd": cost,
                "transit_days": transit,
            })
    return rows


def generate_hub_handling_costs(hub_lookup):
    """Generate per-unit handling cost for each hub."""
    rows = []
    for h in HUBS:
        country = next(c for c in COUNTRIES if c["country_code"] == h["country_code"])
        if country["developed"]:
            base_handling = rng.uniform(3.0, 5.0)
        else:
            base_handling = rng.uniform(1.0, 3.0)
        rows.append({
            "hub_id": h["hub_id"],
            "handling_cost_per_unit_usd": round(base_handling, 2),
        })
    return rows


def generate_last_mile_costs(hub_lookup, country_lookup):
    """Generate last-mile cost and transit days from each hub to each country.
    Three tiers:
      - Same country: $0.50-2.00, 1 day (domestic ground shipping)
      - Same region: $2.00-8.00, 2+ days (regional freight)
      - Cross-region: $5.00-15.00, 3-30+ days (international ocean/air)"""
    KM_PER_DAY = 400
    rows = []
    for h in HUBS:
        for c in COUNTRIES:
            dist = haversine_km(h["lat"], h["lon"], c["lat"], c["lon"])
            effective_dist = dist * 1.3
            if h["country_code"] == c["country_code"]:
                cost = rng.uniform(0.50, 2.00)
                transit = 1
            elif h["region_id"] == c["region_id"]:
                cost = rng.uniform(2.00, 8.00)
                transit = max(2, int(round(effective_dist / KM_PER_DAY)))
            else:
                cost = min(15.0, max(5.0, dist * 0.0008))
                cost = add_noise(cost, 0.15)
                transit = max(3, int(round(effective_dist / KM_PER_DAY)))
            rows.append({
                "hub_id": h["hub_id"],
                "country_code": c["country_code"],
                "cost_per_unit_usd": round(cost, 2),
                "transit_days": transit,
            })
    return rows


def generate_tariffs():
    """Generate tariff rates for all factory-country to destination-country pairs."""
    factory_countries = list(set(f["country_code"] for f in FACTORIES))
    rows = []
    for origin in factory_countries:
        for dest_c in COUNTRIES:
            dest = dest_c["country_code"]
            if origin == dest:
                tariff = SAME_COUNTRY_TARIFF
            else:
                tariff = TARIFF_RULES.get((origin, dest), DEFAULT_TARIFF)
            rows.append({
                "origin_country_code": origin,
                "destination_country_code": dest,
                "tariff_pct": tariff,
            })
    return rows


def generate_lead_time_requirements(country_lookup, category_lookup):
    """Generate max lead time (days) for each (country, category) pair."""
    # Transit now includes factory→hub + hub→country (total 3-60 days),
    # so lead time limits must accommodate realistic cross-region shipping.
    urgency_base = {1: 30, 2: 45, 3: 60}
    developed_offset = -7
    emerging_offset = 7

    rows = []
    for c in COUNTRIES:
        for cat in CATEGORIES:
            base = urgency_base[cat["urgency"]]
            if c["developed"]:
                lead_time = base + developed_offset
            else:
                lead_time = base + emerging_offset
            lead_time += int(rng.integers(-2, 3))
            lead_time = max(10, lead_time)
            rows.append({
                "country_code": c["country_code"],
                "category_id": cat["category_id"],
                "max_lead_time_days": lead_time,
            })
    return rows


def generate_demand(country_lookup, category_lookup):
    """Generate 12-month demand for each (country, category) pair."""
    # Tuned so peak-month demand is ~75-85% of total factory capacity.
    # With ~110 factory-category pairs averaging ~8K capacity each = ~880K total.
    # 17 countries, 10 categories = 170 combos. Peak demand ~750K total.
    BASE_DEMAND = 1900

    rows = []
    for c in COUNTRIES:
        for cat in CATEGORIES:
            for month in range(1, 13):
                base = BASE_DEMAND * c["demand_scale"]
                price_factor = max(0.3, 1.0 - (cat["base_manufacturing_cost_usd"] / 800))
                seasonal = seasonality_factor(month)
                demand = int(round(base * price_factor * seasonal * (1 + rng.uniform(-0.15, 0.15))))
                demand = max(50, demand)
                rows.append({
                    "country_code": c["country_code"],
                    "category_id": cat["category_id"],
                    "month": month,
                    "demand_units": demand,
                })
    return rows


def is_flow_restricted(factory_country, hub_country, dest_country):
    """Check if a flow is geopolitically restricted."""
    for r in GEOPOLITICAL_RESTRICTIONS:
        if r["destination_country_code"] != dest_country:
            continue
        if r["restriction_type"] == "MADE_IN" and r["restricted_country_code"] == factory_country:
            return True
        if r["restriction_type"] == "ROUTED_THROUGH" and r["restricted_country_code"] == hub_country:
            return True
    return False


def generate_all_flows(factory_category_capacity, transport_costs, hub_handling_costs,
                       last_mile_costs, tariffs, lead_time_reqs,
                       factory_lookup, hub_lookup, category_lookup):
    """Generate the master all_flows table (~22K rows) with pre-computed costs.

    Iterates: for each factory → for each category it makes → for each hub →
    for each destination country, compute:
      total_landed_cost = mfg + transport + handling + last_mile + tariff
      transit_days = factory→hub days + hub→country days
      is_lead_time_feasible = 1 if transit_days <= max allowed for (country, category)
      is_geopolitically_restricted = 1 if MADE_IN or ROUTED_THROUGH rule applies
    """
    # Build lookup dicts for O(1) access during the nested loop
    fcc_dict = {}
    for row in factory_category_capacity:
        fcc_dict[(row["factory_id"], row["category_id"])] = row["unit_manufacturing_cost_usd"]

    tc_dict = {}
    for row in transport_costs:
        tc_dict[(row["factory_id"], row["hub_id"])] = {
            "cost": row["cost_per_unit_usd"],
            "days": row["transit_days"],
        }

    hh_dict = {}
    for row in hub_handling_costs:
        hh_dict[row["hub_id"]] = row["handling_cost_per_unit_usd"]

    lm_dict = {}
    for row in last_mile_costs:
        lm_dict[(row["hub_id"], row["country_code"])] = {
            "cost": row["cost_per_unit_usd"],
            "days": row["transit_days"],
        }

    tariff_dict = {}
    for row in tariffs:
        tariff_dict[(row["origin_country_code"], row["destination_country_code"])] = row["tariff_pct"]

    lt_dict = {}
    for row in lead_time_reqs:
        lt_dict[(row["country_code"], row["category_id"])] = row["max_lead_time_days"]

    flows = []
    for factory_id, cat_ids in FACTORY_CATEGORY_MAP.items():
        factory = factory_lookup[factory_id]
        factory_cc = factory["country_code"]

        for cat_id in cat_ids:
            cat = category_lookup[cat_id]
            mfg_cost = fcc_dict[(factory_id, cat_id)]
            weight = cat["representative_weight_kg"]

            for hub in HUBS:
                hub_id = hub["hub_id"]
                hub_cc = hub["country_code"]
                transport = tc_dict[(factory_id, hub_id)]
                # Scale transport cost by category weight (base transport is for 1kg)
                transport_cost = round(transport["cost"] * weight, 2)
                factory_to_hub_days = transport["days"]
                handling_cost = hh_dict[hub_id]

                for country in COUNTRIES:
                    cc = country["country_code"]
                    last_mile_info = lm_dict[(hub_id, cc)]
                    last_mile = last_mile_info["cost"]
                    last_mile_days = last_mile_info["days"]
                    transit_days = factory_to_hub_days + last_mile_days
                    tariff_pct = tariff_dict.get((factory_cc, cc), DEFAULT_TARIFF)
                    tariff_amount = round(mfg_cost * tariff_pct, 2)
                    total = round(mfg_cost + transport_cost + handling_cost + last_mile + tariff_amount, 2)

                    max_lt = lt_dict.get((cc, cat_id), 999)
                    is_lt_feasible = 1 if transit_days <= max_lt else 0
                    is_restricted = 1 if is_flow_restricted(factory_cc, hub_cc, cc) else 0

                    flows.append({
                        "factory_id": factory_id,
                        "hub_id": hub_id,
                        "country_code": cc,
                        "category_id": cat_id,
                        "manufacturing_cost": mfg_cost,
                        "transport_cost": transport_cost,
                        "hub_handling_cost": handling_cost,
                        "last_mile_cost": last_mile,
                        "tariff_pct": tariff_pct,
                        "tariff_amount": tariff_amount,
                        "total_landed_cost": total,
                        "transit_days": transit_days,
                        "max_lead_time_days": max_lt,
                        "is_lead_time_feasible": is_lt_feasible,
                        "is_geopolitically_restricted": is_restricted,
                    })
    return flows


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_data(all_flows, demand, factory_category_capacity):
    """Run validation checks and print summary."""
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    df = pd.DataFrame(all_flows)
    feasible = df[(df["is_geopolitically_restricted"] == 0) & (df["is_lead_time_feasible"] == 1)]

    # 1. Feasibility: every (country, category) has at least 1 feasible flow
    print("\n-- Feasibility Check --")
    all_ok = True
    for c in COUNTRIES:
        for cat in CATEGORIES:
            mask = (feasible["country_code"] == c["country_code"]) & (feasible["category_id"] == cat["category_id"])
            count = mask.sum()
            if count == 0:
                print(f"  FAIL: {c['country_code']}/{cat['category_id']} has NO feasible flow!")
                all_ok = False
    if all_ok:
        print(f"  PASS: All {len(COUNTRIES)} countries x {len(CATEGORIES)} categories have at least 1 feasible flow")

    # 2. Cost range
    print("\n-- Cost Summary --")
    print(f"  Total flows:       {len(df):,}")
    print(f"  Feasible flows:    {len(feasible):,} ({len(feasible)/len(df)*100:.1f}%)")
    print(f"  Restricted flows:  {(df['is_geopolitically_restricted'] == 1).sum():,}")
    print(f"  Lead-time blocked: {(df['is_lead_time_feasible'] == 0).sum():,}")
    print(f"  Min total cost:    ${feasible['total_landed_cost'].min():.2f}")
    print(f"  Max total cost:    ${feasible['total_landed_cost'].max():.2f}")
    print(f"  Mean total cost:   ${feasible['total_landed_cost'].mean():.2f}")

    # 3. Demand vs capacity
    print("\n-- Demand vs Capacity --")
    demand_df = pd.DataFrame(demand)
    monthly_demand = demand_df.groupby("month")["demand_units"].sum()

    fcc_df = pd.DataFrame(factory_category_capacity)
    total_monthly_capacity = fcc_df["monthly_capacity_units"].sum()

    print(f"  Total monthly factory capacity: {total_monthly_capacity:,} units")
    for month in range(1, 13):
        md = monthly_demand[month]
        pct = md / total_monthly_capacity * 100
        print(f"  Month {month:2d}: demand={md:>10,}  ({pct:.1f}% of capacity)")

    # 4. Product catalog stats
    print("\n-- Product Catalog --")
    print(f"  Total products: {len(PRODUCTS)}")
    for cat in CATEGORIES:
        cat_products = [p for p in PRODUCTS if p["category_id"] == cat["category_id"]]
        global_count = sum(1 for p in cat_products if len(p["regions"]) == 6)
        multi_count = sum(1 for p in cat_products if 2 <= len(p["regions"]) < 6)
        single_count = sum(1 for p in cat_products if len(p["regions"]) == 1)
        print(f"  {cat['category_id']} {cat['category_name']:<20s}: {len(cat_products)} products "
              f"({global_count} global, {multi_count} multi-region, {single_count} single-region)")

    # 5. Sample flows
    print("\n-- Top 5 Cheapest Feasible Flows --")
    cheapest = feasible.nsmallest(5, "total_landed_cost")
    for _, row in cheapest.iterrows():
        print(f"  {row['factory_id']} -> {row['hub_id']} -> {row['country_code']} "
              f"[{row['category_id']}] = ${row['total_landed_cost']:.2f} ({row['transit_days']}d)")

    print("\n-- Top 5 Most Expensive Feasible Flows --")
    expensive = feasible.nlargest(5, "total_landed_cost")
    for _, row in expensive.iterrows():
        print(f"  {row['factory_id']} -> {row['hub_id']} -> {row['country_code']} "
              f"[{row['category_id']}] = ${row['total_landed_cost']:.2f} ({row['transit_days']}d)")

    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(data, filename, columns=None):
    """Write list of dicts to CSV."""
    df = pd.DataFrame(data)
    if columns:
        df = df[columns]
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  {filename:<45s} {len(df):>8,} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Supply Chain Data Generator (Category-Level Model)")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seed: {SEED}")

    factory_lookup, hub_lookup, country_lookup, category_lookup = build_lookups()

    # Generate all data
    print("\nGenerating data...")
    product_availability = generate_product_availability()
    fcc = generate_factory_category_capacity(factory_lookup, category_lookup)
    tc = generate_transport_costs(factory_lookup, hub_lookup)
    hhc = generate_hub_handling_costs(hub_lookup)
    lmc = generate_last_mile_costs(hub_lookup, country_lookup)
    tariffs = generate_tariffs()
    lt_reqs = generate_lead_time_requirements(country_lookup, category_lookup)
    demand = generate_demand(country_lookup, category_lookup)
    all_flows = generate_all_flows(fcc, tc, hhc, lmc, tariffs, lt_reqs,
                                   factory_lookup, hub_lookup, category_lookup)

    # Write CSVs
    print("\nWriting CSV files...")
    write_csv(REGIONS, "regions.csv", ["region_id", "region_name"])
    write_csv(COUNTRIES, "countries.csv", ["country_code", "country_name", "region_id"])
    write_csv(FACTORIES, "factories.csv",
              ["factory_id", "factory_name", "city", "country_code", "region_id", "cost_multiplier"])
    write_csv(HUBS, "hubs.csv",
              ["hub_id", "hub_name", "city", "country_code", "region_id", "monthly_throughput_capacity"])
    write_csv(CATEGORIES, "product_categories.csv",
              ["category_id", "category_name", "base_manufacturing_cost_usd", "representative_weight_kg"])
    write_csv(PRODUCTS, "products.csv",
              ["product_id", "product_name", "category_id", "retail_price_tier", "relative_demand_weight"])
    write_csv(product_availability, "product_availability.csv",
              ["product_id", "region_id"])
    write_csv(fcc, "factory_category_capacity.csv",
              ["factory_id", "category_id", "unit_manufacturing_cost_usd", "monthly_capacity_units"])
    write_csv(tc, "transport_costs.csv",
              ["factory_id", "hub_id", "cost_per_unit_usd", "transit_days"])
    write_csv(hhc, "hub_handling_costs.csv",
              ["hub_id", "handling_cost_per_unit_usd"])
    write_csv(lmc, "last_mile_costs.csv",
              ["hub_id", "country_code", "cost_per_unit_usd", "transit_days"])
    write_csv(tariffs, "tariffs.csv",
              ["origin_country_code", "destination_country_code", "tariff_pct"])
    write_csv(GEOPOLITICAL_RESTRICTIONS, "geopolitical_restrictions.csv",
              ["destination_country_code", "restricted_country_code", "restriction_type", "reason"])
    write_csv(lt_reqs, "lead_time_requirements.csv",
              ["country_code", "category_id", "max_lead_time_days"])
    write_csv(demand, "demand.csv",
              ["country_code", "category_id", "month", "demand_units"])
    write_csv(all_flows, "all_flows.csv",
              ["factory_id", "hub_id", "country_code", "category_id",
               "manufacturing_cost", "transport_cost", "hub_handling_cost",
               "last_mile_cost", "tariff_pct", "tariff_amount", "total_landed_cost",
               "transit_days", "max_lead_time_days", "is_lead_time_feasible",
               "is_geopolitically_restricted"])

    # Validate
    validate_data(all_flows, demand, fcc)

    print("\nDone!")


if __name__ == "__main__":
    main()
