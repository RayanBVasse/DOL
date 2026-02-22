# step10_a2_state_segmentation.py

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ===============================
# CONFIG
# ===============================

MONTHLY_METRICS_PATH = Path("out/macro_monthly_metrics.csv")

OUT_STATES = Path("out/monthly_states.csv")
OUT_TRANSITIONS = Path("out/state_transition_summary.csv")
OUT_CONFIG = Path("out/run_config_step10_a2.json")

# ===============================
# LOAD DATA
# ===============================

print("Loading monthly metrics...")
df = pd.read_csv(MONTHLY_METRICS_PATH)

required_cols = ["month", "macro_entropy_user", "macro_js_divergence"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in macro_monthly_metrics.csv")

df = df.sort_values("month").reset_index(drop=True)

# ===============================
# COMPUTE RELATIVE THRESHOLDS
# ===============================

entropy_high = df["macro_entropy_user"].quantile(0.66)
entropy_low  = df["macro_entropy_user"].quantile(0.33)

div_high = df["macro_js_divergence"].quantile(0.66)
div_low  = df["macro_js_divergence"].quantile(0.33)

# ===============================
# ASSIGN STATES
# ===============================

states = []

for i, row in df.iterrows():
    entropy = row["macro_entropy_user"]
    div = row["macro_js_divergence"]

    if entropy >= entropy_high and div >= div_high:
        state = "Exploration"

    elif entropy <= entropy_low and div <= div_low:
        state = "Consolidation"

    else:
        state = "Transitional"

    states.append(state)

df["state"] = states

# ===============================
# SAVE STATES
# ===============================

df[["month", "macro_entropy_user", "macro_js_divergence", "state"]].to_csv(OUT_STATES, index=False)
print("Saved:", OUT_STATES)

# ===============================
# TRANSITION SUMMARY
# ===============================

transitions = []

for i in range(1, len(df)):
    prev_state = df.loc[i-1, "state"]
    curr_state = df.loc[i, "state"]

    transitions.append((prev_state, curr_state))

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
    "entropy_high_quantile": 0.66,
    "entropy_low_quantile": 0.33,
    "divergence_high_quantile": 0.66,
    "divergence_low_quantile": 0.33,
    "columns_used": required_cols
}

with open(OUT_CONFIG, "w") as f:
    json.dump(config, f, indent=2)

print("Saved:", OUT_CONFIG)

print("\nDONE.")
print("Interpretation:")
print("- Exploration = high entropy + high divergence.")
print("- Consolidation = low entropy + low divergence.")
print("- Transitional = intermediate or switching regime.")