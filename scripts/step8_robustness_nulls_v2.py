#!/usr/bin/env python3
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

K_LIST = [40, 60, 80]
SVD_LIST = [100, 200]

MAX_FEATURES = 60000
MIN_DF = 5
MAX_DF = 0.6
NGRAM_RANGE = (1, 2)
STOP_WORDS = "english"

# Drop tiny months (helps remove stray 2023-02 etc.)
MIN_PER_MONTH_PER_ROLE = 50


def pick_windows(months_sorted):
    months_set = set(months_sorted)
    early = [m for m in ["2025-05", "2025-06", "2025-07"] if m in months_set]
    peak  = [m for m in ["2025-08", "2025-09", "2025-10"] if m in months_set]
    late  = [m for m in ["2025-11", "2025-12"] if m in months_set]
    if len(early) >= 2 and len(peak) >= 2 and len(late) >= 1:
        return early, peak, late

    n = len(months_sorted)
    early = months_sorted[: max(1, n // 3)]
    peak  = months_sorted[max(1, n // 3): max(2, 2 * n // 3)]
    late  = months_sorted[max(2, 2 * n // 3):]
    return early, peak, late


def monthly_curves(df, months_sorted):
    rows = []
    for month in months_sorted:
        g = df[df["month"] == month]
        user_g = g[g["role"] == "user"]
        asst_g = g[g["role"] == "assistant"]

        # User entropy
        if len(user_g) > 0:
            uc = user_g["cluster"].value_counts()
            up = (uc / uc.sum()).values
            H = float(shannon_entropy(up))
        else:
            H = np.nan

        # Dyadic JS
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
    def mean_over(month_list, col):
        sub = curves[curves["month"].isin(month_list)][col].dropna()
        return float(sub.mean()) if len(sub) else np.nan

    ent_peak = mean_over(peak, "topic_entropy_user")
    ent_early = mean_over(early, "topic_entropy_user")
    entropy_contrast = ent_peak - ent_early

    js_spike_months = [m for m in ["2025-09", "2025-10"] if m in curves["month"].values]
    js_base_months  = [m for m in ["2025-05", "2025-06", "2025-07", "2025-08"] if m in curves["month"].values]
    js_spike = mean_over(js_spike_months, "js_divergence")
    js_base  = mean_over(js_base_months, "js_divergence")
    js_spike_contrast = js_spike - js_base

    js_late_months = [m for m in ["2025-11", "2025-12"] if m in curves["month"].values]
    js_recovery_contrast = mean_over(js_late_months, "js_divergence") - mean_over(js_spike_months, "js_divergence")

    return {
        "entropy_contrast_peak_minus_early": entropy_contrast,
        "js_spike_contrast_sep_oct_minus_may_aug": js_spike_contrast,
        "js_recovery_contrast_nov_dec_minus_sep_oct": js_recovery_contrast,
    }


def perm_pvalue(null_stats, real_stat):
    null_stats = np.array([x for x in null_stats if np.isfinite(x)], dtype=float)
    if not np.isfinite(real_stat) or len(null_stats) == 0:
        return np.nan
    if real_stat >= 0:
        return float((null_stats >= real_stat).mean())
    else:
        return float((null_stats <= real_stat).mean())


def main():
    rng = np.random.default_rng(SEED)

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
    df = df.reset_index(drop=True)

    # Drop tiny months (removes stray months like 2023-02)
    counts = df.groupby(["month", "role"]).size().unstack(fill_value=0)
    keep_months = counts[(counts.get("user", 0) >= MIN_PER_MONTH_PER_ROLE) &
                         (counts.get("assistant", 0) >= MIN_PER_MONTH_PER_ROLE)].index.tolist()
    df = df[df["month"].isin(keep_months)].reset_index(drop=True)

    months_sorted = sorted(df["month"].unique())
    early, peak, late = pick_windows(months_sorted)

    print("Months:", months_sorted)
    print("Early window:", early)
    print("Peak window:", peak)
    print("Late window:", late)

    results_rows = []

    for k in K_LIST:
        for svd_dims in SVD_LIST:
            print(f"\n=== Running k={k}, svd={svd_dims} ===")

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
            df["cluster"] = km.fit_predict(X_red)

            # Real
            curves_real = monthly_curves(df, months_sorted)
            stats_real = compute_summary_stats(curves_real, early, peak, late)
            curves_path = OUT_DIR / f"robustness_curves_k{k}_svd{svd_dims}.csv"
            curves_real.to_csv(curves_path, index=False)

            # Null A: shuffle months globally
            nullA = {key: [] for key in stats_real.keys()}
            months_arr = df["month"].to_numpy()
            for _ in range(N_PERM):
                dfA = df.copy()
                dfA["month"] = rng.permutation(months_arr)
                curvesA = monthly_curves(dfA, months_sorted)
                sA = compute_summary_stats(curvesA, early, peak, late)
                for key in nullA:
                    nullA[key].append(sA[key])

            # Null B: shuffle roles within month
            nullB = {key: [] for key in stats_real.keys()}
            for _ in range(N_PERM):
                dfB = df.copy()
                for m in months_sorted:
                    idx = dfB.index[dfB["month"] == m].to_numpy()
                    if len(idx) > 1:
                        dfB.loc[idx, "role"] = rng.permutation(dfB.loc[idx, "role"].to_numpy())
                curvesB = monthly_curves(dfB, months_sorted)
                sB = compute_summary_stats(curvesB, early, peak, late)
                for key in nullB:
                    nullB[key].append(sB[key])

            # Null C: volume control by downsampling within each month (keep clusters fixed)
            nullC = {key: [] for key in stats_real.keys()}
            counts = df.groupby(["month", "role"]).size().unstack(fill_value=0)
            min_user = int(counts["user"].min())
            min_asst = int(counts["assistant"].min())

            for _ in range(N_PERM):
                sampled_idx = []
                for m in months_sorted:
                    g = df[df["month"] == m]
                    ug = g[g["role"] == "user"]
                    ag = g[g["role"] == "assistant"]
                    sampled_idx.extend(rng.choice(ug.index.to_numpy(), size=min_user, replace=False).tolist())
                    sampled_idx.extend(rng.choice(ag.index.to_numpy(), size=min_asst, replace=False).tolist())

                dfC = df.loc[sampled_idx].copy()
                curvesC = monthly_curves(dfC, months_sorted)
                sC = compute_summary_stats(curvesC, early, peak, late)
                for key in nullC:
                    nullC[key].append(sC[key])

            row = {
                "k": k,
                "svd_dims": svd_dims,
                **stats_real,
                "p_month_shuffle_entropy": perm_pvalue(nullA["entropy_contrast_peak_minus_early"], stats_real["entropy_contrast_peak_minus_early"]),
                "p_month_shuffle_js_spike": perm_pvalue(nullA["js_spike_contrast_sep_oct_minus_may_aug"], stats_real["js_spike_contrast_sep_oct_minus_may_aug"]),
                "p_month_shuffle_js_recovery": perm_pvalue(nullA["js_recovery_contrast_nov_dec_minus_sep_oct"], stats_real["js_recovery_contrast_nov_dec_minus_sep_oct"]),
                "p_role_shuffle_entropy": perm_pvalue(nullB["entropy_contrast_peak_minus_early"], stats_real["entropy_contrast_peak_minus_early"]),
                "p_role_shuffle_js_spike": perm_pvalue(nullB["js_spike_contrast_sep_oct_minus_may_aug"], stats_real["js_spike_contrast_sep_oct_minus_may_aug"]),
                "p_role_shuffle_js_recovery": perm_pvalue(nullB["js_recovery_contrast_nov_dec_minus_sep_oct"], stats_real["js_recovery_contrast_nov_dec_minus_sep_oct"]),
                "p_volume_control_entropy": perm_pvalue(nullC["entropy_contrast_peak_minus_early"], stats_real["entropy_contrast_peak_minus_early"]),
                "p_volume_control_js_spike": perm_pvalue(nullC["js_spike_contrast_sep_oct_minus_may_aug"], stats_real["js_spike_contrast_sep_oct_minus_may_aug"]),
                "p_volume_control_js_recovery": perm_pvalue(nullC["js_recovery_contrast_nov_dec_minus_sep_oct"], stats_real["js_recovery_contrast_nov_dec_minus_sep_oct"]),
                "min_user_per_month": min_user,
                "min_assistant_per_month": min_asst,
                "curves_csv": str(curves_path),
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
