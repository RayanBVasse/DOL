"""
step_shuffled_null_cognitive.py
───────────────────────────────
Permutation-based significance test for monotonic trends in cognitive markers.

For each marker (system/structural thinking, uncertainty/epistemic language,
relational/collaborative framing), computes the observed Spearman rank
correlation with time and compares it against a null distribution of 10,000
permutations in which month labels are randomly shuffled.

This tests whether each marker's temporal trajectory is statistically
distinguishable from chance ordering — addressing the concern that any
8-point time series might yield a spurious trend.

Input  : data/trajectory_monthly.csv
Output : data/shuffled_null_results.csv  +  printed report

Dependencies: pandas, numpy, scipy
"""

import numpy as np
import pandas as pd
from scipy import stats

# ── Config ───────────────────────────────────────────────────────────────────
INPUT        = 'data/trajectory_monthly.csv'
OUTPUT       = 'data/shuffled_null_results.csv'
N_PERM       = 10_000
RANDOM_SEED  = 42
MIN_MESSAGES = 100          # drop sparse early months
MARKERS = {
    'system_words_per_1k_msgs':      'System / Structural Thinking',
    'uncertainty_words_per_1k_msgs': 'Uncertainty / Epistemic Humility',
    'relational_words_per_1k_msgs':  'Relational / Collaborative Framing',
}
# ─────────────────────────────────────────────────────────────────────────────


def permutation_trend_test(values: np.ndarray, n_perm: int, seed: int):
    """
    Observed Spearman rho vs. shuffled null.
    Returns (obs_rho, parametric_p, perm_p_1tail, perm_p_2tail, null_rhos).
    """
    rng  = np.random.default_rng(seed)
    time = np.arange(len(values))
    obs_rho, param_p = stats.spearmanr(time, values)
    null_rhos = np.array([
        stats.spearmanr(time, rng.permutation(values))[0]
        for _ in range(n_perm)
    ])
    perm_p_1 = np.mean(null_rhos >= obs_rho)         # one-tailed (positive trend)
    perm_p_2 = np.mean(np.abs(null_rhos) >= abs(obs_rho))  # two-tailed
    return obs_rho, param_p, perm_p_1, perm_p_2, null_rhos


def main():
    df = pd.read_csv(INPUT)
    df = df[df['user_messages'] >= MIN_MESSAGES].reset_index(drop=True)

    print(f"Active months: {list(df['month'])}\n")
    print(f"Permutations : {N_PERM:,}\n")
    print("=" * 60)

    records = []
    for col, label in MARKERS.items():
        vals = df[col].values
        rho, p_param, p1, p2, nulls = permutation_trend_test(vals, N_PERM, RANDOM_SEED)

        sig = "*** p < 0.05" if p1 < 0.05 else "(ns)"
        print(f"\n{label}")
        print(f"  Observed Spearman ρ  : {rho:+.3f}")
        print(f"  Parametric p         : {p_param:.4f}")
        print(f"  Permutation p 1-tail : {p1:.4f}  {sig}")
        print(f"  Permutation p 2-tail : {p2:.4f}")
        print(f"  Null rho range       : [{nulls.min():.3f}, {nulls.max():.3f}]")

        records.append({
            'marker':           label,
            'spearman_rho':     round(rho, 3),
            'parametric_p':     round(p_param, 4),
            'perm_p_1tail':     round(p1, 4),
            'perm_p_2tail':     round(p2, 4),
            'significant_1tail': p1 < 0.05,
        })

    out = pd.DataFrame(records)
    out.to_csv(OUTPUT, index=False)
    print(f"\n\nResults saved → {OUTPUT}")


if __name__ == '__main__':
    main()
