# step10_a3_rolling_entropy.py

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ===============================
# CONFIG
# ===============================

MASTER_PATH = Path("out/messages_master.csv")
DOMAIN_MAP_PATH = Path("out/node_to_macro_domain.csv")

WINDOW_SIZE = 250
STRIDE = 250

OUT_PATH = Path("out/rolling_entropy_250.csv")
OUT_CONFIG = Path("out/run_config_step10_a3.json")

# ===============================
# LOAD DATA
# ===============================

print("Loading master...")
df = pd.read_csv(MASTER_PATH)

print("Loading domain map...")
domain_map = pd.read_csv(DOMAIN_MAP_PATH)

# Merge on node_id
df = df.merge(domain_map, on="node_id", how="inner")

# Parse datetime
df["dt"] = pd.to_datetime(df["message_create_time_iso"], errors="coerce")
df = df.sort_values("dt").reset_index(drop=True)

print("Total merged messages:", len(df))

# ===============================
# ENTROPY FUNCTION
# ===============================

def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    probs = counts.values
    return -np.sum(probs * np.log2(probs))

# ===============================
# ROLLING WINDOWS
# ===============================

results = []

for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
    end = start + WINDOW_SIZE
    window = df.iloc[start:end]

    # Combined entropy
    combined_entropy = compute_entropy(window["macro_domain"])

    # User-only entropy
    user_window = window[window["role"] == "user"]
    if len(user_window) > 0:
        user_entropy = compute_entropy(user_window["macro_domain"])
    else:
        user_entropy = np.nan

    results.append({
        "window_start_index": start,
        "window_end_index": end,
        "combined_entropy": combined_entropy,
        "user_entropy": user_entropy,
        "mid_timestamp": window["dt"].iloc[len(window)//2]
    })

rolling_df = pd.DataFrame(results)

rolling_df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)

# ===============================
# SAVE CONFIG
# ===============================

config = {
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "metrics": ["combined_entropy", "user_entropy"],
    "entropy_base": "log2"
}

with open(OUT_CONFIG, "w") as f:
    json.dump(config, f, indent=2)

print("Saved:", OUT_CONFIG)

print("\nDONE.")
print("Next: plot entropy curves and compute rolling slopes.")