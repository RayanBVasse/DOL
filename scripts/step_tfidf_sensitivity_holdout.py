"""
step_tfidf_sensitivity_holdout.py
──────────────────────────────────
Temporal holdout sensitivity check for TF-IDF/SVD/KMeans pipeline.

WHAT THIS TESTS
───────────────
Steps 5-8 of the main pipeline fit TF-IDF and SVD on the full corpus, then
compute monthly metrics on the same data. This introduces potential temporal
leakage: later months influence the vocabulary representation of earlier ones.

This script tests whether the key finding — that late months (Sep-Dec 2025)
show a structurally distinct topic composition from early months (May-Aug 2025)
— survives when the model is trained exclusively on the early period.

METHOD
──────
1. Split corpus into TRAIN (May–Aug 2025) and HOLDOUT (Sep–Dec 2025).
2. Fit TF-IDF → TruncatedSVD → KMeans on TRAIN user messages only.
3. Transform ALL 8 months using the fitted model (no re-fitting on holdout).
4. Compute each month's cluster distribution (60-dim probability vector).
5. Measure Jensen-Shannon divergence of each month from the TRAIN mean.
   - If late months show higher divergence than early months in both the
     global model and the holdout model, the shift is robust to leakage.
6. Compare divergence profiles: global model vs holdout model.

KEY QUESTION
────────────
Does the December compositional shift persist when the model has never
seen September–December data during training?

Input  : data/messages_master.csv
         data/macro_monthly_domain_shares.csv  (global model, for comparison)
Output : data/sensitivity_holdout_divergences.csv
         data/sensitivity_holdout_cluster_shares.csv

Dependencies: pandas, numpy, scipy, scikit-learn
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_MESSAGES = 'out/messages_master.csv'
INPUT_GLOBAL   = 'out/macro_monthly_domain_shares.csv'   # existing output
OUTPUT_DIV     = 'out/sensitivity_holdout_divergences.csv'
OUTPUT_SHARES  = 'out/sensitivity_holdout_cluster_shares.csv'

TRAIN_MONTHS   = ['2025-05', '2025-06', '2025-07', '2025-08']
HOLDOUT_MONTHS = ['2025-09', '2025-10', '2025-11', '2025-12']
ALL_MONTHS     = TRAIN_MONTHS + HOLDOUT_MONTHS

N_SVD       = 200
N_CLUSTERS  = 60
MAX_FEATURES= 5000
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def month_cluster_distribution(messages, pipeline, n_clusters):
    """Transform messages and return cluster probability vector."""
    if len(messages) == 0:
        return np.zeros(n_clusters)
    tfidf_matrix = pipeline.named_steps['tfidf'].transform(messages)
    svd_matrix   = pipeline.named_steps['svd'].transform(tfidf_matrix)
    labels       = pipeline.named_steps['kmeans'].predict(svd_matrix)
    counts = np.bincount(labels, minlength=n_clusters).astype(float)
    return counts / counts.sum()


def js_from_reference(dist, reference):
    """Jensen-Shannon divergence between dist and a reference distribution."""
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    d = dist + eps;   d /= d.sum()
    r = reference + eps; r /= r.sum()
    return float(jensenshannon(d, r, base=2))


def main():
    # ── Load corpus ──────────────────────────────────────────────────────────
    print("Loading corpus...")
    df = pd.read_csv(INPUT_MESSAGES)
    df['message_create_time_iso'] = pd.to_datetime(
        df['message_create_time_iso'], utc=True, errors='coerce'
    )
    df['month'] = df['message_create_time_iso'].dt.to_period('M').astype(str)
    df = df[df['role'] == 'user'].copy()
    df = df[df['month'].isin(ALL_MONTHS)].copy()
    df['text'] = df['text'].fillna('').astype(str)
    print(f"  User messages in active window: {len(df):,}")

    # ── Build holdout pipeline: fit on TRAIN only ─────────────────────────────
    print("\nFitting TF-IDF / SVD / KMeans on TRAIN months (May–Aug 2025)...")
    train_texts = df[df['month'].isin(TRAIN_MONTHS)]['text'].tolist()
    print(f"  Training messages: {len(train_texts):,}")

    tfidf   = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english',
                               ngram_range=(1, 2))
    svd     = TruncatedSVD(n_components=N_SVD, random_state=RANDOM_SEED)
    kmeans  = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED,
                     n_init=10, max_iter=300)

    # Fit
    tfidf_matrix  = tfidf.fit_transform(train_texts)
    svd_matrix    = svd.fit_transform(tfidf_matrix)
    kmeans.fit(svd_matrix)

    # Wrap in a namespace for convenience
    class FittedPipeline:
        pass
    pipe = FittedPipeline()
    pipe.named_steps = {'tfidf': tfidf, 'svd': svd, 'kmeans': kmeans}

    # ── Compute monthly cluster distributions (holdout model) ─────────────────
    print("\nComputing monthly cluster distributions (holdout model)...")
    monthly_dists = {}
    for month in ALL_MONTHS:
        texts = df[df['month'] == month]['text'].tolist()
        monthly_dists[month] = month_cluster_distribution(texts, pipe, N_CLUSTERS)
        print(f"  {month}: {len(texts):,} messages")

    # ── Reference = mean of TRAIN months ─────────────────────────────────────
    train_mean = np.mean(
        [monthly_dists[m] for m in TRAIN_MONTHS], axis=0
    )

    # ── Compute JS divergence from train mean: holdout model ──────────────────
    print("\n=== HOLDOUT MODEL: JS divergence from train-period mean ===\n")
    holdout_divs = {}
    for month in ALL_MONTHS:
        jsd = js_from_reference(monthly_dists[month], train_mean)
        holdout_divs[month] = jsd
        period = "TRAIN   " if month in TRAIN_MONTHS else "HOLDOUT "
        print(f"  {period} {month}: JSD = {jsd:.4f}")

    train_jsd_mean    = np.mean([holdout_divs[m] for m in TRAIN_MONTHS])
    holdout_jsd_mean  = np.mean([holdout_divs[m] for m in HOLDOUT_MONTHS])
    print(f"\n  Mean JSD — TRAIN months    : {train_jsd_mean:.4f}")
    print(f"  Mean JSD — HOLDOUT months  : {holdout_jsd_mean:.4f}")

    # Mann-Whitney test: are holdout divergences larger than train divergences?
    train_vals   = [holdout_divs[m] for m in TRAIN_MONTHS]
    holdout_vals = [holdout_divs[m] for m in HOLDOUT_MONTHS]
    _, mw_p = stats.mannwhitneyu(holdout_vals, train_vals, alternative='greater')
    print(f"  Mann-Whitney p (holdout > train): {mw_p:.4f}")

    # ── Load global model results for comparison ──────────────────────────────
    comparison_rows = []
    global_available = False
    try:
        global_df = pd.read_csv(INPUT_GLOBAL)
        print(f"\n  Global domain shares loaded for comparison.")
        global_available = True
    except FileNotFoundError:
        print(f"\n  Note: {INPUT_GLOBAL} not found — skipping global comparison.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    div_rows = []
    for month in ALL_MONTHS:
        div_rows.append({
            'month':        month,
            'period':       'train' if month in TRAIN_MONTHS else 'holdout',
            'jsd_from_train_mean': round(holdout_divs[month], 4),
        })
    div_df = pd.DataFrame(div_rows)
    div_df.to_csv(OUTPUT_DIV, index=False)

    shares_rows = []
    for month in ALL_MONTHS:
        for cluster_id, share in enumerate(monthly_dists[month]):
            shares_rows.append({
                'month': month,
                'period': 'train' if month in TRAIN_MONTHS else 'holdout',
                'cluster': cluster_id,
                'share': round(float(share), 6),
            })
    shares_df = pd.DataFrame(shares_rows)
    shares_df.to_csv(OUTPUT_SHARES, index=False)

    print(f"\nSaved → {OUTPUT_DIV}")
    print(f"Saved → {OUTPUT_SHARES}")

    # ── Interpretation guide ──────────────────────────────────────────────────
    print("""
=== HOW TO READ THIS ===

If holdout months (Sep-Dec) show higher JSD from the train mean than
the train months themselves, the compositional shift is NOT an artefact
of global fitting — it persists even when the model has never seen late data.

If the pattern collapses (holdout months look like train months), the
shift may be partly driven by vocabulary leakage from later months
influencing the global model's cluster assignments.
    """)


if __name__ == '__main__':
    main()
