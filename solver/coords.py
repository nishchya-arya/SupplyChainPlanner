"""
Geographic coordinates for supply chain entities.

Lat/lon pairs for factories, hubs, and destination countries.
Used by the Knowledge Graph page and the Solver results route map.
These coordinates come from generate_data.py but aren't stored in CSVs.
"""

# 13 factories: keyed by factory_id, values are (latitude, longitude).
# Locations correspond to the city in factories.csv (e.g., F_US_01 = Detroit).
FACTORY_COORDS = {
    "F_US_01": (42.33, -83.05), "F_MX_01": (25.67, -100.31),
    "F_BR_01": (-23.55, -46.63), "F_DE_01": (48.14, 11.58),
    "F_GB_01": (52.49, -1.90), "F_AE_01": (25.20, 55.27),
    "F_CN_01": (22.54, 114.06), "F_CN_02": (23.13, 113.26),
    "F_KR_01": (37.26, 127.03), "F_VN_01": (21.03, 105.85),
    "F_VN_02": (10.82, 106.63), "F_IN_01": (13.08, 80.27),
    "F_IN_02": (12.97, 77.59),
}

# 14 distribution hubs: keyed by hub_id, values are (latitude, longitude).
# Locations correspond to the city in hubs.csv (e.g., H_US_01 = Houston).
HUB_COORDS = {
    "H_US_01": (29.76, -95.37), "H_CA_01": (43.65, -79.38),
    "H_MX_01": (19.43, -99.13), "H_BR_01": (-23.55, -46.63),
    "H_DE_01": (50.11, 8.68), "H_GB_01": (51.51, -0.13),
    "H_AE_01": (25.20, 55.27), "H_ZA_01": (-26.20, 28.05),
    "H_CN_01": (31.23, 121.47), "H_JP_01": (35.68, 139.69),
    "H_KR_01": (35.18, 129.08), "H_IN_01": (19.08, 72.88),
    "H_VN_01": (10.82, 106.63), "H_AU_01": (-33.87, 151.21),
}

# 17 destination countries: keyed by 2-letter country code, values are (latitude, longitude).
# These are geographic centroids used for map marker placement and route line endpoints.
COUNTRY_COORDS = {
    "US": (37.09, -95.71), "CA": (56.13, -106.35), "MX": (23.63, -102.55),
    "BR": (-14.24, -51.93), "AR": (-38.42, -63.62), "DE": (51.17, 10.45),
    "GB": (55.38, -3.44), "FR": (46.23, 2.21), "AE": (23.42, 53.85),
    "SA": (23.89, 45.08), "ZA": (-30.56, 22.94), "CN": (35.86, 104.20),
    "JP": (36.20, 138.25), "KR": (35.91, 127.77), "IN": (20.59, 78.96),
    "VN": (14.06, 108.28), "AU": (-25.27, 133.77),
}
