#!/usr/bin/env python3

"""
Step 9.1 – Weekly directional coupling (macro domains)

Uses:
- out/macro_domain_map.csv  (message_id → macro_domain)
- threads_master.csv (or your master message file with dt + role + message_id)

Output:
- out/weekly_directional_summary.csv
- out/weekly_directional_by_domain.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------- PATHS ----------
MASTER_PATH = Path("out/messages_master.csv")  # adjust if needed
DOMAIN_MAP_PATH = Path("out/node_to_macro_domain.csv")

OUT_SUMMARY = Path("out/weekly_directional_summary.csv")
OUT_DOMAIN  = Path("out/weekly_directional_by_domain.csv")

MIN_MSGS_PER_ROLE_PER_WEEK = 40
N_PERM = 2000
SEED = 42

#print("MAP_PATH resolved to:", DOMAIN_MAP_PATH.resolve())
#print("domain_map columns:", DOMAIN_MAP_PATH.columns.tolist())
#print("domain_map head:")
#print(DOMAIN_MAP_PATH.head())
# ---------- HELPERS ----------

def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return np.dot(a, b) / (na * nb)

def perm_p_abs(null_vals, real):
    null_vals = np.array(null_vals)
    return np.mean(np.abs(null_vals) >= abs(real))

# ---------- LOAD DATA ----------

df = pd.read_csv(MASTER_PATH)
domain_map = pd.read_csv(DOMAIN_MAP_PATH)

df["dt"] = pd.to_datetime(df["message_create_time_iso"], errors="coerce")
df["week"] = df["dt"].dt.to_period("W").astype(str)

df = df.merge(domain_map, on="node_id", how="inner")

# Filter only user + assistant
df = df[df["role"].isin(["user", "assistant"])]

# ---------- WEEKLY AGGREGATION ----------

weekly = (
    df.groupby(["week", "role", "macro_domain"])
      .size()
      .reset_index(name="count")
)

# Pivot to shares
pivot = weekly.pivot_table(
    index=["week", "role"],
    columns="macro_domain",
    values="count",
    fill_value=0
)

pivot = pivot.reset_index()

# Keep only weeks with sufficient data
week_counts = (
    df.groupby(["week", "role"])
      .size()
      .unstack()
      .fillna(0)
)

valid_weeks = week_counts[
    (week_counts["user"] >= MIN_MSGS_PER_ROLE_PER_WEEK) &
    (week_counts["assistant"] >= MIN_MSGS_PER_ROLE_PER_WEEK)
].index

pivot = pivot[pivot["week"].isin(valid_weeks)]

weeks = sorted(pivot["week"].unique())
domains = sorted(df["macro_domain"].unique())

# ---------- BUILD MATRICES ----------

U = []
A = []

for w in weeks:
    u_row = pivot[(pivot["week"] == w) & (pivot["role"] == "user")]
    a_row = pivot[(pivot["week"] == w) & (pivot["role"] == "assistant")]

    if len(u_row) == 0 or len(a_row) == 0:
        continue

    u_vec = u_row[domains].values[0]
    a_vec = a_row[domains].values[0]

    u_vec = u_vec / u_vec.sum()
    a_vec = a_vec / a_vec.sum()

    U.append(u_vec)
    A.append(a_vec)

U = np.array(U)
A = np.array(A)

T = len(U)

if T < 6:
    raise ValueError("Too few valid weeks for weekly coupling.")

# ---------- DELTAS ----------

dU = U[1:] - U[:-1]
dA = A[1:] - A[:-1]

# Lag 1 coupling
forward = []
reverse = []

for t in range(T - 2):
    forward.append(cosine(dU[t], dA[t+1]))
    reverse.append(cosine(dA[t], dU[t+1]))

forward_mean = np.nanmean(forward)
reverse_mean = np.nanmean(reverse)
diff = forward_mean - reverse_mean

# ---------- PERMUTATION ----------

rng = np.random.default_rng(SEED)
null_diff = []

for _ in range(N_PERM):
    perm = rng.permutation(len(dA))
    sims = []
    sims_rev = []

    for t in range(T - 2):
        sims.append(cosine(dU[t], dA[perm][t+1]))
        sims_rev.append(cosine(dA[perm][t], dU[t+1]))

    null_diff.append(np.nanmean(sims) - np.nanmean(sims_rev))

p_value = perm_p_abs(null_diff, diff)

# ---- BY DOMAIN (macro_domain) ----
by_rows = []

domains = sorted(df["macro_domain"].dropna().unique().tolist())

for dom in domains:
    dfd = df[df["macro_domain"] == dom].copy()

    # weekly counts per role
    wk_user = dfd[dfd["role"] == "user"].groupby("week").size()
    wk_ai   = dfd[dfd["role"] == "assistant"].groupby("week").size()

    # align on same week index
    weeks = sorted(set(wk_user.index).union(set(wk_ai.index)))
    u = np.array([wk_user.get(w, 0) for w in weeks], dtype=float)
    a = np.array([wk_ai.get(w, 0) for w in weeks], dtype=float)

    # require enough active weeks to be meaningful
    if (u > 0).sum() < 8 or (a > 0).sum() < 8:
        continue

    # normalize to shares within this domain across weeks (optional)
    if u.sum() > 0: u = u / u.sum()
    if a.sum() > 0: a = a / a.sum()

    # lag series
    u_t = u[:-1]
    a_t = a[:-1]
    u_next = u[1:]
    a_next = a[1:]

    fwd = cosine(u_t, a_next)   # user -> ai
    rev = cosine(a_t, u_next)   # ai -> user
    diff = fwd - rev

    # permutation null: shuffle weeks of one partner
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_PERM):
        perm = rng.permutation(len(a_next))
        fwd_p = cosine(u_t, a_next[perm])
        rev_p = cosine(a_t, u_next)  # keep rev fixed, or perm both for stricter null
        diffs.append(fwd_p - rev_p)
    diffs = np.array(diffs)
    p_abs = (np.abs(diffs) >= abs(diff)).mean()

    by_rows.append({
        "macro_domain": int(dom),
        "weeks_total": len(weeks),
        "weeks_user_active": int((u > 0).sum()),
        "weeks_ai_active": int((a > 0).sum()),
        "forward_user_to_ai": float(fwd),
        "reverse_ai_to_user": float(rev),
        "difference": float(diff),
        "perm_p_abs": float(p_abs),
    })

by_df = pd.DataFrame(by_rows).sort_values("perm_p_abs")
by_df.to_csv(OUT_DOMAIN, index=False)
print("Saved:", OUT_DOMAIN)

# ---------- SAVE SUMMARY ----------

summary = pd.DataFrame([{
    "weeks_used": T,
    "forward_mean_user_to_ai": forward_mean,
    "reverse_mean_ai_to_user": reverse_mean,
    "difference": diff,
    "perm_p_abs": p_value
}])

summary.to_csv(OUT_SUMMARY, index=False)

print(summary)
print("Saved weekly summary.")
