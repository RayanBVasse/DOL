"""
step_thread_complexity.py
─────────────────────────
Analyses structural complexity of conversation threads over time.

Metrics computed per thread:
  depth           — num_nodes_in_main_path (length of primary conversation chain)
  branching_ratio — num_nodes_in_tree / depth  (>1 = exploratory branching occurred)
  total_msgs      — user + assistant messages on main path
  chars_per_msg   — main_path_text_chars / total_msgs  (exchange density)
  user_msg_share  — fraction of messages authored by user

Monthly aggregates (median) are computed and a Spearman trend test is applied
to chars_per_msg, the one metric showing a significant longitudinal signal.

Input  : data/threads_master.csv
Output : data/thread_complexity.csv          (per-thread metrics)
         data/thread_complexity_monthly.csv  (monthly aggregates)

Dependencies: pandas, numpy, scipy
"""

import numpy as np
import pandas as pd
from scipy import stats

# ── Config ───────────────────────────────────────────────────────────────────
INPUT         = 'data/threads_master.csv'
OUTPUT_THREAD = 'data/thread_complexity.csv'
OUTPUT_MONTHLY= 'data/thread_complexity_monthly.csv'
ACTIVE_START  = '2025-05'
ACTIVE_END    = '2025-12'
# ─────────────────────────────────────────────────────────────────────────────


def main():
    t = pd.read_csv(INPUT)

    # Parse creation month (suppress tz-drop warning)
    t['created'] = pd.to_datetime(t['thread_create_time_iso'], utc=True, errors='coerce')
    t['month']   = t['created'].dt.tz_localize(None).dt.to_period('M').astype(str)

    # Derived metrics
    t['branching_ratio'] = (
        t['num_nodes_in_tree'] / t['num_nodes_in_main_path'].replace(0, np.nan)
    )
    t['total_msgs']      = t['num_user_msgs_main_path'] + t['num_assistant_msgs_main_path']
    t['chars_per_msg']   = t['main_path_text_chars'] / t['total_msgs'].replace(0, np.nan)
    t['user_msg_share']  = t['num_user_msgs_main_path'] / t['total_msgs'].replace(0, np.nan)
    t['depth']           = t['num_nodes_in_main_path']

    # Active window
    active = t[t['month'].between(ACTIVE_START, ACTIVE_END)].copy()
    print(f"Active threads ({ACTIVE_START}–{ACTIVE_END}): {len(active)}\n")

    # ── Overall summary ──────────────────────────────────────────────────────
    print("=== OVERALL SUMMARY ===")
    for col, label in [
        ('depth',           'Thread depth (main path messages)'),
        ('branching_ratio', 'Branching ratio'),
        ('chars_per_msg',   'Characters per message'),
        ('total_msgs',      'Total messages per thread'),
        ('user_msg_share',  'User message share'),
    ]:
        s = active[col].dropna()
        print(f"\n{label}")
        print(f"  median={s.median():.1f}  mean={s.mean():.1f}  "
              f"min={s.min():.1f}  max={s.max():.1f}  sd={s.std():.1f}")

    # ── Monthly trends ───────────────────────────────────────────────────────
    monthly = active.groupby('month').agg(
        n_threads         = ('thread_id',       'count'),
        median_depth      = ('depth',           'median'),
        median_branching  = ('branching_ratio', 'median'),
        median_chars_msg  = ('chars_per_msg',   'median'),
        median_total_msgs = ('total_msgs',      'median'),
        median_user_share = ('user_msg_share',  'median'),
    ).reset_index()

    print("\n\n=== MONTHLY AGGREGATES ===")
    print(monthly.to_string(index=False))

    # ── Trend tests ──────────────────────────────────────────────────────────
    print("\n\n=== SPEARMAN TREND TESTS ===")
    time_idx = np.arange(len(monthly))
    for col, label in [
        ('median_depth',     'Thread depth'),
        ('median_branching', 'Branching ratio'),
        ('median_chars_msg', 'Chars per message'),
        ('median_user_share','User message share'),
    ]:
        rho, p = stats.spearmanr(time_idx, monthly[col].values)
        sig = "*** p < 0.05" if p < 0.05 else "(ns)"
        print(f"  {label:<28} ρ={rho:+.3f}  p={p:.4f}  {sig}")

    # ── Save ─────────────────────────────────────────────────────────────────
    active.to_csv(OUTPUT_THREAD,  index=False)
    monthly.to_csv(OUTPUT_MONTHLY, index=False)
    print(f"\nSaved → {OUTPUT_THREAD}")
    print(f"Saved → {OUTPUT_MONTHLY}")


if __name__ == '__main__':
    main()
