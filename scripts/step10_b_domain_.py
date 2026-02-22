import pandas as pd
import numpy as np
from pathlib import Path

# -------- PATHS --------
MASTER_PATH = Path("out/messages_master.csv")
DOMAIN_MAP_PATH = Path("out/node_to_macro_domain.csv")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

# -------- LOAD DATA --------
print("Loading master file...")
df = pd.read_csv(MASTER_PATH)

print("Loading domain map...")
domain_map = pd.read_csv(DOMAIN_MAP_PATH)

# Ensure consistent naming
df["dt"] = pd.to_datetime(df["message_create_time_iso"], errors="coerce")

# Merge domain labels
df = df.merge(domain_map, on="node_id", how="inner")

# Keep only user and assistant
df = df[df["role"].isin(["user", "assistant"])]

# Sort properly
df = df.sort_values(["thread_id", "dt"]).reset_index(drop=True)

# -------- DETECT DOMAIN SHIFTS --------
shift_records = []

for thread_id, group in df.groupby("thread_id"):
    group = group.reset_index(drop=True)
    for i in range(1, len(group)):
        prev_domain = group.loc[i-1, "macro_domain"]
        curr_domain = group.loc[i, "macro_domain"]
        
        if prev_domain != curr_domain:
            shift_records.append({
                "thread_id": thread_id,
                "initiator_role": group.loc[i, "role"],
                "from_domain": prev_domain,
                "to_domain": curr_domain
            })

shift_df = pd.DataFrame(shift_records)

print(f"Total domain shifts detected: {len(shift_df)}")

# -------- BASIC COUNTS --------
counts = shift_df["initiator_role"].value_counts()
total_shifts = counts.sum()

user_prop = counts.get("user", 0) / total_shifts if total_shifts else 0
assistant_prop = counts.get("assistant", 0) / total_shifts if total_shifts else 0

# -------- PERMUTATION TEST --------
def permutation_test(labels, n_perm=5000):
    observed = np.mean(labels == "user")
    perm_vals = []
    
    for _ in range(n_perm):
        shuffled = np.random.permutation(labels)
        perm_vals.append(np.mean(shuffled == "user"))
    
    perm_vals = np.array(perm_vals)
    p_value = np.mean(np.abs(perm_vals - 0.5) >= np.abs(observed - 0.5))
    
    return observed, p_value

observed_user_share, perm_p = permutation_test(shift_df["initiator_role"].values)

# -------- OUTPUT --------
summary = pd.DataFrame([{
    "total_shifts": total_shifts,
    "user_initiated_prop": user_prop,
    "assistant_initiated_prop": assistant_prop,
    "perm_p_user_vs_random": perm_p
}])

summary_path = OUT_DIR / "step10b_shift_initiation_summary.csv"
detail_path = OUT_DIR / "step10b_shift_initiation_detail.csv"

summary.to_csv(summary_path, index=False)
shift_df.to_csv(detail_path, index=False)

print("\nRESULT SUMMARY:")
print(summary)

print("\nSaved:")
print(summary_path)
print(detail_path)