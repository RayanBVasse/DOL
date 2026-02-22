import pandas as pd
import numpy as np
from pathlib import Path
from math import comb

# ---------------- PATHS ----------------
MASTER_PATH = Path("out/messages_master.csv")
DOMAIN_MAP_PATH = Path("out/node_to_macro_domain.csv")

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

OUT_EPISODES = OUT_DIR / "step10b2_episode_detail.csv"
OUT_SUMMARY = OUT_DIR / "step10b2_episode_summary.csv"
OUT_THREAD_SUMMARY = OUT_DIR / "step10b2_thread_summary.csv"

# ---------------- PARAMS ----------------
MIN_EPISODE_LEN = 20          # sustained topic episode length (messages)
MIN_THREAD_MSGS = 20          # only analyze threads with enough content
SEED = 42

np.random.seed(SEED)

# ---------------- HELPERS ----------------
def binom_test_two_sided(k, n, p=0.5):
    """
    Exact two-sided binomial test under null p.
    Returns p-value.
    """
    if n == 0:
        return np.nan
    # compute probability of observed
    def pmf(x):
        return comb(n, x) * (p**x) * ((1-p)**(n-x))

    p_obs = pmf(k)
    # two-sided: sum probs <= p_obs
    pval = sum(pmf(x) for x in range(n+1) if pmf(x) <= p_obs + 1e-12)
    return min(1.0, pval)

# ---------------- LOAD ----------------
print("Loading master...")
df = pd.read_csv(MASTER_PATH)

print("Loading domain map...")
domain_map = pd.read_csv(DOMAIN_MAP_PATH)

# Parse time + filter roles
df["dt"] = pd.to_datetime(df["message_create_time_iso"], errors="coerce")
df = df[df["role"].isin(["user", "assistant"])]

# Merge domains
df = df.merge(domain_map, on="node_id", how="inner")

# Sort within threads
df = df.sort_values(["thread_id", "dt"]).reset_index(drop=True)

# Remove empty domains if any
df = df.dropna(subset=["macro_domain"])

print("Total messages after merge/filter:", len(df))

# Keep only sufficiently large threads
thread_sizes = df.groupby("thread_id").size()
keep_threads = thread_sizes[thread_sizes >= MIN_THREAD_MSGS].index
df = df[df["thread_id"].isin(keep_threads)].copy()

print("Threads kept:", len(keep_threads))
print("Messages kept:", len(df))

# ---------------- EPISODE SEGMENTATION ----------------
episode_rows = []
thread_level_rows = []

for thread_id, g in df.groupby("thread_id", sort=False):
    g = g.reset_index(drop=True)

    # Identify contiguous runs of macro_domain
    run_id = (g["macro_domain"] != g["macro_domain"].shift(1)).cumsum()
    g["_run_id"] = run_id

    # Summarize runs
    runs = (
        g.groupby("_run_id")
         .agg(
             macro_domain=("macro_domain", "first"),
             start_idx=("dt", "min"),
             end_idx=("dt", "max"),
             run_len=("macro_domain", "size"),
             initiator_role=("role", "first"),
             initiator_node=("node_id", "first"),
         )
         .reset_index(drop=True)
    )

    # Keep sustained episodes only
    sustained = runs[runs["run_len"] >= MIN_EPISODE_LEN].copy()

    # Record episode-level details
    if len(sustained) > 0:
        sustained["thread_id"] = thread_id
        episode_rows.append(sustained)

        # Thread-level: who initiated the FIRST sustained episode in this thread?
        first_ep = sustained.sort_values("start_idx").iloc[0]
        thread_level_rows.append({
            "thread_id": thread_id,
            "first_sustained_domain": first_ep["macro_domain"],
            "first_sustained_len": int(first_ep["run_len"]),
            "first_sustained_initiator_role": first_ep["initiator_role"],
            "first_sustained_start": first_ep["start_idx"],
        })

# Combine
episodes_df = pd.concat(episode_rows, ignore_index=True) if episode_rows else pd.DataFrame()
thread_df = pd.DataFrame(thread_level_rows)

# Save details
episodes_df.to_csv(OUT_EPISODES, index=False)
thread_df.to_csv(OUT_THREAD_SUMMARY, index=False)

print("Saved:", OUT_EPISODES)
print("Saved:", OUT_THREAD_SUMMARY)

# ---------------- SUMMARY STATS ----------------
def summarize_initiation(table, role_col):
    if table is None or len(table) == 0:
        return {
            "n": 0,
            "user_count": 0,
            "assistant_count": 0,
            "user_share": np.nan,
            "assistant_share": np.nan,
            "binom_p_two_sided_vs_0.5": np.nan
        }
    counts = table[role_col].value_counts()
    n = int(counts.sum())
    k_user = int(counts.get("user", 0))
    k_ai = int(counts.get("assistant", 0))
    return {
        "n": n,
        "user_count": k_user,
        "assistant_count": k_ai,
        "user_share": k_user / n if n else np.nan,
        "assistant_share": k_ai / n if n else np.nan,
        "binom_p_two_sided_vs_0.5": binom_test_two_sided(k_user, n, p=0.5)
    }

episode_summary = summarize_initiation(episodes_df, "initiator_role")
thread_summary = summarize_initiation(thread_df, "first_sustained_initiator_role")

summary_df = pd.DataFrame([
    {"unit": "sustained_episodes", "min_len": MIN_EPISODE_LEN, **episode_summary},
    {"unit": "threads_first_sustained_episode", "min_len": MIN_EPISODE_LEN, **thread_summary},
])

summary_df.to_csv(OUT_SUMMARY, index=False)

print("\nSUMMARY:")
print(summary_df)
print("\nSaved:", OUT_SUMMARY)

print("\nDONE.")
print("Interpretation tip: episode-level is 'who started sustained topic runs' within threads.")