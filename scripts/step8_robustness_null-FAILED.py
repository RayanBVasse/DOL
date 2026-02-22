#!/usr/bin/env python3
"""
Step 8: Robustness + Null-model tests for monthly topic entropy and dyadic JS divergence.

Core pipeline (locked):
- TF-IDF (1-2 grams, English stopwords)
- TruncatedSVD
- KMeans (global topic space)
- Metrics per month:
    * topic_entropy (user only)
    * js_divergence (user vs assistant topic distribution)

Robustness grid:
- k in {40, 60, 80}
- svd_dims in {100, 200}

Null models (computed without re-fitting model):
A) Month-shuffle: shuffle month labels across messages
B) Role-shuffle within month: shuffle role labels within each month
C) Volume-control: downsample each month to equal counts per role (min across months)

Summary statistics (pre-defined calendar windows if present):
- Entropy contrast: mean(Entropy Aug-Oct) - mean(Entropy May-Jul)
- JS spike contrast: mean(JS Sep-Oct) - mean(JS May-Aug)
- JS recovery contrast: mean(JS Nov-Dec) - mean(JS Sep-Oct)

Outputs:
- out/robustness_null_tests.csv  (one row per grid setting with real stats + p-values)
- out/robustness_curves_<k>_<svd>.csv (monthly curves for each setting)
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.stats import entropy as shannon_entropy
from scipy.spatial.distance import jensenshannon

DB_PATH = Path("out/corpus.sqlite")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

N_PERM = 200
SEED = 42

# Robustness grid
K_LIST = [40, 60, 80]
SVD_LIST = [100, 200]

# TF-IDF params (locked)
MAX_FEATURES = 60000
MIN_DF = 5
MAX_DF = 0.6
NGRAM_RANGE = (1, 2)
STOP_WORDS = "english"


def pick_windows(months_sorted):
    """Prefer fixed calendar windows if present, else fall back to terciles."""
    months_set = set(months_sorted)

    early = [m for m in ["2025-05", "2025-06", "2025-07"] if m in months_set]
    peak = [m for m in ["2025-08", "2025-09", "2025-10"] if m in months_set]
    late = [m for m in ["2025-11", "2025-12"] if m in months_set]

    if len(early) >= 2 and len(peak) >= 2 and len(late) >= 1:
        return early, peak, late

    # Fallback: split into early/peak/late terciles
    n = len(months_sorted)
    if n < 6:
        # Minimal fallback
        early = months_sorted[: max(1, n // 3)]
        peak = months_sorted[max(1, n // 3): max(2, 2 * n // 3)]
        late = months_sorted[max(2, 2 * n // 3):]
        return early, peak, late

    early = months_sorted[: n // 3]
    peak = months_sorted[n // 3: 2 * n // 3]
    late = months_sorted[2 * n // 3:]
    return early, peak, late


def monthly_distributions(df, months_sorted, labels, role_col="role", month_col="month"):
    """
    Compute monthly:
      - user entropy
      - dyadic JS divergence between user vs assistant cluster distributions
    labels: np.array cluster label per row of df.
    Returns DataFrame monthly curves.
    """
    df = df.copy()
    df["cluster"] = labels

    rows = []
    for month in months_sorted:
        g = df[df[month_col] == month]
        user_g = g[g[role_col] == "user"]
        asst_g = g[g[role_col] == "assistant"]

        # User entropy (exclude empty)
        if len(user_g) > 0:
            uc = user_g["cluster"].value_counts()
            up = (uc / uc.sum()).values
            H = float(shannon_entropy(up))
        else:
            H = np.nan

        # Dyadic JS divergence
        if len(user_g) >= 10 and len(asst_g) >= 10:
            user_counts = user_g["cluster"].value_counts(normalize=True)
            asst_counts = asst_g["cluster"].value_counts(normalize=True)
            all_clusters = sorted(set(user_counts.index) | set(asst_counts.index))
            u = np.array([user_counts.get(c, 0.0) for c in all_clusters], dtype=float)
            a = np.array([asst_counts.get(c, 0.0) for c in all_clusters], dtype=float)
            js = float(jensenshannon(u, a))
        else:
            js = np.nan

        rows.append({
            "month": month,
            "user_msgs": len(user_g),
            "assistant_msgs": len(asst_g),
            "topic_entropy_user": H,
            "js_divergence": js
        })

    return pd.DataFrame(rows)


def compute_summary_stats(curves, early, peak, late):
    """
    Compute pre-defined contrasts.
    peak (Aug-Oct) vs early (May-Jul) for entropy
    Sep-Oct vs May-Aug for JS spike
    Nov-Dec vs Sep-Oct for JS recovery
    """
    # Helper
    def mean_over(month_list, col):
        sub = curves[curves["month"].isin(month_list)][col].dropna()
        return float(sub.mean()) if len(sub) else np.nan

    # Entropy contrast
    ent_peak = mean_over(peak, "topic_entropy_user")
    ent_early = mean_over(early, "topic_entropy_user")
    entropy_contrast = ent_peak - ent_early

    # JS spike contrast
    js_spike_months = [m for m in ["2025-09", "2025-10"] if m in curves["month"].values]
    js_base_months = [m for m in ["2025-05", "2025-06", "2025-07", "2025-08"] if m in curves["month"].values]

    js_spike = mean_over(js_spike_months, "js_divergence")
    js_base = mean_over(js_base_months, "js_divergence")
    js_spike_contrast = js_spike - js_base

    # JS recovery contrast
    js_late_months = [m for m in ["2025-11", "2025-12"] if m in curves["month"].values]
    js_recovery_contrast = mean_over(js_late_months, "js_divergence") - mean_over(js_spike_months, "js_divergence")

    return {
        "entropy_contrast_peak_minus_early": entropy_contrast,
        "js_spike_contrast_sep_oct_minus_may_aug": js_spike_contrast,
        "js_recovery_contrast_nov_dec_minus_sep_oct": js_recovery_contrast,
    }


def perm_pvalue(null_stats, real_stat, two_sided=False):
    null_stats = np.array([x for x in null_stats if np.isfinite(x)], dtype=float)
    if not np.isfinite(real_stat) or len(null_stats) == 0:
        return np.nan
    if two_sided:
        return float((np.abs(null_stats) >= np.abs(real_stat)).mean())
    else:
        # one-sided in direction of real_stat
        if real_stat >= 0:
            return float((null_stats >= real_stat).mean())
        else:
            return float((null_stats <= real_stat).mean())


def main():
    rng = np.random.default_rng(SEED)

    # Load user + assistant only
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT role, text, message_create_time_iso
        FROM messages
        WHERE role IN ('user','assistant')
          AND text IS NOT NULL AND LENGTH(text) > 0
    """, con)
    con.close()

    df["dt"] = pd.to_datetime(df["message_create_time_iso"], errors="coerce", utc=True)
    df = df.dropna(subset=["dt"])
    df["month"] = df["dt"].dt.to_period("M").astype(str)

    # Stable order
    df = df.reset_index(drop=True)

    months_sorted = sorted(df["month"].unique())
    early, peak, late = pick_windows(months_sorted)

    print("Months:", months_sorted)
    print("Early window:", early)
    print("Peak window:", peak)
    print("Late window:", late)

    results_rows = []

    for k in K_LIST:
        for svd_dims in SVD_LIST:
            print(f"\n=== Running grid setting k={k}, svd={svd_dims} ===")

            # Vectorize & cluster (fit once per setting)
            vec = TfidfVectorizer(
                max_features=MAX_FEATURES,
                min_df=MIN_DF,
                max_df=MAX_DF,
                stop_words=STOP_WORDS,
                ngram_range=NGRAM_RANGE
            )
            X = vec.fit_transform(df["text"].astype(str).tolist())

            svd = TruncatedSVD(n_components=svd_dims, random_state=SEED)
            X_red = svd.fit_transform(X)
            X_red = normalize(X_red)

            km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            labels = km.fit_predict(X_red)

            # Real curves + stats
            curves_real = monthly_distributions(df, months_sorted, labels)
            stats_real = compute_summary_stats(curves_real, early, peak, late)

            curves_path = OUT_DIR / f"robustness_curves_k{k}_svd{svd_dims}.csv"
            curves_real.to_csv(curves_path, index=False)

            # --- Null model A: shuffle month labels globally ---
            nullA = {key: [] for key in stats_real.keys()}
            months_arr = df["month"].to_numpy()

            for _ in range(N_PERM):
                shuffled_months = rng.permutation(months_arr)
                dfA = df.copy()
                dfA["month"] = shuffled_months
                curvesA = monthly_distributions(dfA, months_sorted, labels)
                sA = compute_summary_stats(curvesA, early, peak, late)
                for key in nullA:
                    nullA[key].append(sA[key])

            # --- Null model B: shuffle roles within each month ---
            nullB = {key: [] for key in stats_real.keys()}

            for _ in range(N_PERM):
                dfB = df.copy()
                for m in months_sorted:
                    idx = dfB.index[dfB["month"] == m].to_numpy()
                    if len(idx) > 1:
                        dfB.loc[idx, "role"] = rng.permutation(dfB.loc[idx, "role"].to_numpy())
                curvesB = monthly_distributions(dfB, months_sorted, labels)
                sB = compute_summary_stats(curvesB, early, peak, late)
                for key in nullB:
                    nullB[key].append(sB[key])

            # --- Null model C: downsample to equal volume per month per role ---
            # For each month, sample min_user and min_asst across months
            nullC = {key: [] for key in stats_real.keys()}

            # Determine minimum counts across months (for user and assistant separately)
            counts = df.groupby(["month", "role"]).size().unstack(fill_value=0)
            min_user = int(counts["user"].min()) if "user" in counts else 0
            min_asst = int(counts["assistant"].min()) if "assistant" in counts else 0

            for _ in range(N_PERM):
                sampled_idx = []
                for m in months_sorted:
                    g = df[df["month"] == m]
                    ug = g[g["role"] == "user"]
                    ag = g[g["role"] == "assistant"]
                    if len(ug) >= min_user and len(ag) >= min_asst and min_user > 0 and min_asst > 0:
                        sampled_idx.extend(rng.choice(ug.index.to_numpy(), size=min_user, replace=False).tolist())
                        sampled_idx.extend(rng.choice(ag.index.to_numpy(), size=min_asst, replace=False).tolist())
                dfC = df.loc[sampled_idx].copy()
                curvesC = monthly_distributions(dfC, months_sorted, labels)
                sC = compute_summary_stats(curvesC, early, peak, late)
                for key in nullC:
                    nullC[key].append(sC[key])

            # p-values (one-sided in direction of real)
            pA = {f"p_month_shuffle__{k}": perm_pvalue(nullA[k], stats_real[k]) for k in stats_real}
            pB = {f"p_role_shuffle__{k}": perm_pvalue(nullB[k], stats_real[k]) for k in stats_real}
            pC = {f"p_volume_control__{k}": perm_pvalue(nullC[k], stats_real[k]) for k in stats_real}

            row = {
                "k": k,
                "svd_dims": svd_dims,
                **stats_real,
                **pA,
                **pB,
                **pC,
                "min_user_per_month_for_volume_control": min_user,
                "min_asst_per_month_for_volume_control": min_asst,
                "curves_csv": str(curves_path)
            }
            results_rows.append(row)

    out_df = pd.DataFrame(results_rows)
    out_path = OUT_DIR / "robustness_null_tests.csv"
    out_df.to_csv(out_path, index=False)

    print("\nDone ✅")
    print("Saved:", out_path)
    print("\nPreview:\n", out_df)


if __name__ == "__main__":
    main()
