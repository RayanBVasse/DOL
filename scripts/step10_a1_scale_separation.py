# step10_a1_scale_separation.py

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ===============================
# CONFIG
# ===============================

WEEKLY_SUMMARY_PATH = Path("out/weekly_directional_summary.csv")
MONTHLY_METRICS_PATH = Path("out/macro_monthly_metrics.csv")

OUT_REPORT = Path("out/scale_separation_report.csv")
OUT_CONFIG = Path("out/run_config_step10_a1.json")

# ===============================
# LOAD DATA
# ===============================

print("Loading weekly directional summary...")
weekly = pd.read_csv(WEEKLY_SUMMARY_PATH)

print("Loading monthly macro metrics...")
monthly = pd.read_csv(MONTHLY_METRICS_PATH)

# ===============================
# WEEKLY SCALE METRICS
# ===============================

# absolute directional difference at weekly level
weekly_diff_abs = np.abs(weekly["difference"]).mean()

# permutation p-value
weekly_perm_p = weekly["perm_p_abs"].iloc[0]

# number of weeks used
weeks_used = weekly["weeks_used"].iloc[0]

# ===============================
# MONTHLY SCALE METRICS
# ===============================

# assume macro_monthly_metrics contains:
# month, topic_entropy, alignment_score (or similar)

required_cols = ["month", "macro_entropy_user", "macro_js_divergence"]
for col in required_cols:
    if col not in monthly.columns:
        raise ValueError(f"Column '{col}' not found in macro_monthly_metrics.csv")

monthly = monthly.sort_values("month")

# entropy change magnitude
entropy_range = monthly["macro_entropy_user"].max() - monthly["macro_entropy_user"].min()

# alignment change magnitude
alignment_range = monthly["macro_js_divergence"].max() - monthly["macro_js_divergence"].min()

# month-to-month volatility
entropy_volatility = monthly["macro_entropy_user"].diff().abs().mean()
alignment_volatility = monthly["macro_js_divergence"].diff().abs().mean()

# ===============================
# STRUCTURE INDEX
# ===============================

# Simple scale contrast index:
# monthly structural variation divided by weekly directional difference

structure_index_entropy = entropy_range / (weekly_diff_abs + 1e-6)
structure_index_alignment = alignment_range / (weekly_diff_abs + 1e-6)

# ===============================
# BUILD REPORT
# ===============================

report = pd.DataFrame([{
    "weeks_used": weeks_used,
    "weekly_abs_directional_diff": weekly_diff_abs,
    "weekly_perm_p": weekly_perm_p,
    "monthly_entropy_range": entropy_range,
    "monthly_alignment_range": alignment_range,
    "monthly_entropy_volatility": entropy_volatility,
    "monthly_alignment_volatility": alignment_volatility,
    "structure_index_entropy": structure_index_entropy,
    "structure_index_alignment": structure_index_alignment
}])

report.to_csv(OUT_REPORT, index=False)

print("Saved:", OUT_REPORT)

# ===============================
# SAVE CONFIG
# ===============================

config = {
    "weekly_summary_path": str(WEEKLY_SUMMARY_PATH),
    "monthly_metrics_path": str(MONTHLY_METRICS_PATH),
    "computed_metrics": [
        "weekly_abs_directional_diff",
        "weekly_perm_p",
        "monthly_entropy_range",
        "monthly_alignment_range",
        "structure_index_entropy",
        "structure_index_alignment"
    ]
}

with open(OUT_CONFIG, "w") as f:
    json.dump(config, f, indent=2)

print("Saved:", OUT_CONFIG)

print("\nDONE.")
print("Interpretation guide:")
print("- If weekly_abs_directional_diff ~ 0 and perm_p high → micro reciprocity.")
print("- If monthly_entropy_range and alignment_range large → macro structure present.")
print("- High structure_index values → scale separation confirmed.")