# step10_a2_state_segmentation_refined.py

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ===============================
# CONFIG
# ===============================

MONTHLY_METRICS_PATH = Path("out/macro_monthly_metrics.csv")

OUT_STATES = Path("out/monthly_states_refined.csv")
OUT_TRANSITIONS = Path("out/state_transition_summary_refined.csv")
OUT_CONFIG = Path("out/run_config_step10_a2_refined.json")

# ===============================
# LOAD DATA
# ===============================

print("Loading monthly metrics...")
df = pd.read_csv(MONTHLY_METRICS_PATH)

required_cols = ["month", "macro_entropy_user", "macro_js_divergence"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")

df = df.sort_values("month").reset_index(drop=True)

# ===============================
# COMPUTE MONTH-TO-MONTH CHANGE
# ===============================

df["delta_entropy"] = df["macro_entropy_user"].diff()
df["delta_divergence"] = df["macro_js_divergence"].diff()

# ===============================
# STATE ASSIGNMENT (RATE-BASED)
# ===============================

states = []

for i, row in df.iterrows():
    if i == 0:
        states.append("Initial")
        continue

    dE = row["delta_entropy"]
    dD = row["delta_divergence"]

    if dE > 0 and dD > 0:
        state = "Exploration"

    elif dE < 0 and dD < 0:
        state = "Consolidation"

    else:
        state = "Transitional"

    states.append(state)

df["state"] = states

# ===============================
# SAVE STATES
# ===============================

df[[
    "month",
    "macro_entropy_user",
    "macro_js_divergence",
    "delta_entropy",
    "delta_divergence",
    "state"
]].to_csv(OUT_STATES, index=False)

print("Saved:", OUT_STATES)

# ===============================
# TRANSITION SUMMARY
# ===============================

transitions = []

for i in range(1, len(df)):
    transitions.append((df.loc[i-1, "state"], df.loc[i, "state"]))

transition_df = pd.DataFrame(transitions, columns=["from_state", "to_state"])

transition_summary = (
    transition_df
    .groupby(["from_state", "to_state"])
    .size()
    .reset_index(name="count")
)

transition_summary.to_csv(OUT_TRANSITIONS, index=False)
print("Saved:", OUT_TRANSITIONS)

# ===============================
# SAVE CONFIG
# ===============================

config = {
    "method": "rate-based",
    "columns_used": required_cols,
    "rule": "Exploration if delta_entropy>0 and delta_divergence>0; Consolidation if both negative; else Transitional"
}

with open(OUT_CONFIG, "w") as f:
    json.dump(config, f, indent=2)

print("Saved:", OUT_CONFIG)

print("\nDONE.")