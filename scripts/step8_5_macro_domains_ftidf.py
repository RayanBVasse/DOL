#!/usr/bin/env python3
"""
Step 8.5: Build macro-domains by clustering the fine-grained topic clusters.

Pipeline:
- Load messages (user + assistant) from out/corpus.sqlite
- TF-IDF -> SVD -> KMeans (fine clusters)
- Compute centroids of each fine cluster in reduced space
- Meta-cluster those centroids into M macro-domains (KMeans)
- Auto-label macro-domains via top TF-IDF terms aggregated across their fine clusters
- Compute monthly macro-domain distributions and metrics:
    * user macro-entropy
    * dyadic macro JS divergence (user vs assistant)

Outputs:
- out/macro_cluster_map.csv             (fine cluster -> macro domain)
- out/macro_domain_summary.csv          (macro domain sizes + auto-label terms)
- out/macro_monthly_metrics.csv         (monthly macro entropy + dyadic macro JS)
- out/macro_monthly_domain_shares.csv   (monthly shares by macro domain, user & assistant)
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

# Fine-grained topic model (keep consistent with earlier)
K_FINE = 60
SVD_DIMS = 200
RANDOM_SEED = 42

# Macro-domains (you can change to 6, 8, 10 later)
M_MACRO = 8

# TF-IDF params (locked-ish)
MAX_FEATURES = 60000
MIN_DF = 5
MAX_DF = 0.6
STOP_WORDS = "english"
NGRAM_RANGE = (1, 2)

TOP_TERMS_FINE = 20
TOP_TERMS_MACRO = 25


def main():
    print("Loading user + assistant messages...")
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

    months_sorted = sorted(df["month"].unique())

    print(f"Messages: {len(df):,} | Months: {months_sorted}")

    # 1) TF-IDF
    print("TF-IDF vectorizing...")
    vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words=STOP_WORDS,
        ngram_range=NGRAM_RANGE
    )
    X = vec.fit_transform(df["text"].astype(str).tolist())
    terms = np.array(vec.get_feature_names_out())

    # 2) SVD
    print("SVD reducing...")
    svd = TruncatedSVD(n_components=SVD_DIMS, random_state=RANDOM_SEED)
    X_red = svd.fit_transform(X)
    X_red = normalize(X_red)

    # 3) Fine KMeans clustering
    print(f"Fine clustering: KMeans(k={K_FINE})...")
    km_fine = KMeans(n_clusters=K_FINE, random_state=RANDOM_SEED, n_init=10)
    fine_labels = km_fine.fit_predict(X_red)
    df["fine_cluster"] = fine_labels

    # 4) Fine cluster top terms (interpretable)
    # Use KMeans centers in reduced space to find representative docs -> but simplest:
    # use mean TF-IDF vector per cluster (costly but manageable) OR use centroids in reduced space
    # We'll compute fine top terms via aggregate TF-IDF weights in each cluster using sparse mean
    print("Computing fine-cluster top terms...")
    fine_top_terms = {}
    for c in range(K_FINE):
        idx = np.where(fine_labels == c)[0]
        if len(idx) == 0:
            fine_top_terms[c] = []
            continue
        # Mean TF-IDF vector for the cluster
        mean_vec = X[idx].mean(axis=0)
        mean_vec = np.asarray(mean_vec).ravel()
        top_idx = mean_vec.argsort()[-TOP_TERMS_FINE:][::-1]
        fine_top_terms[c] = terms[top_idx].tolist()

    # 5) Meta-cluster fine cluster centroids in reduced space
    # centroid = mean of X_red rows belonging to fine cluster
    print(f"Meta-clustering fine clusters into M={M_MACRO} macro-domains...")
    centroids = np.zeros((K_FINE, X_red.shape[1]), dtype=float)
    for c in range(K_FINE):
        idx = np.where(fine_labels == c)[0]
        centroids[c] = X_red[idx].mean(axis=0) if len(idx) else 0.0

    km_macro = KMeans(n_clusters=M_MACRO, random_state=RANDOM_SEED, n_init=20)
    macro_labels = km_macro.fit_predict(centroids)

    # Map fine->macro
    fine_to_macro = {c: int(macro_labels[c]) for c in range(K_FINE)}
    df["macro_domain"] = df["fine_cluster"].map(fine_to_macro)

    # 6) Auto-label macro-domains by aggregating top terms of fine clusters
    # Score terms by frequency across fine-top-terms lists (weighted by cluster size)
    print("Auto-labeling macro-domains...")
    macro_rows = []
    for m in range(M_MACRO):
        fine_in_m = [c for c in range(K_FINE) if fine_to_macro[c] == m]
        # weight terms by cluster size
        term_scores = {}
        for c in fine_in_m:
            c_size = int((df["fine_cluster"] == c).sum())
            for rank, t in enumerate(fine_top_terms[c]):
                # higher rank -> slightly higher weight
                w = c_size * (TOP_TERMS_FINE - rank)
                term_scores[t] = term_scores.get(t, 0) + w

        top_terms_macro = [t for t, _ in sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_TERMS_MACRO]]
        macro_label = ", ".join(top_terms_macro[:6]) if top_terms_macro else f"macro_{m}"

        macro_size = int((df["macro_domain"] == m).sum())
        macro_rows.append({
            "macro_domain": m,
            "size_msgs_total": macro_size,
            "num_fine_clusters": len(fine_in_m),
            "auto_label": macro_label,
            "top_terms": ", ".join(top_terms_macro),
            "fine_clusters": ",".join(map(str, fine_in_m))
        })

    macro_summary = pd.DataFrame(macro_rows).sort_values("size_msgs_total", ascending=False)

    # 7) Monthly macro metrics: user entropy + dyadic JS at macro level
    print("Computing monthly macro metrics...")
    metrics_rows = []
    shares_rows = []

    for month in months_sorted:
        g = df[df["month"] == month]
        ug = g[g["role"] == "user"]
        ag = g[g["role"] == "assistant"]

        # user macro distribution
        u_counts = ug["macro_domain"].value_counts().sort_index()
        a_counts = ag["macro_domain"].value_counts().sort_index()

        # fill missing domains
        u = np.array([u_counts.get(i, 0) for i in range(M_MACRO)], dtype=float)
        a = np.array([a_counts.get(i, 0) for i in range(M_MACRO)], dtype=float)

        u_share = u / u.sum() if u.sum() > 0 else np.zeros(M_MACRO)
        a_share = a / a.sum() if a.sum() > 0 else np.zeros(M_MACRO)

        # entropy (user)
        H = float(shannon_entropy(u_share)) if u.sum() > 0 else np.nan

        # JS (macro level)
        js = float(jensenshannon(u_share, a_share)) if (u.sum() >= 10 and a.sum() >= 10) else np.nan

        metrics_rows.append({
            "month": month,
            "user_msgs": int(len(ug)),
            "assistant_msgs": int(len(ag)),
            "macro_entropy_user": H,
            "macro_js_divergence": js
        })

        for i in range(M_MACRO):
            shares_rows.append({
                "month": month,
                "macro_domain": i,
                "user_share": float(u_share[i]),
                "assistant_share": float(a_share[i]),
            })

    monthly_metrics = pd.DataFrame(metrics_rows)
    monthly_shares = pd.DataFrame(shares_rows)

    # 8) Save outputs
    map_df = pd.DataFrame([{"fine_cluster": c, "macro_domain": fine_to_macro[c],
                            "fine_top_terms": ", ".join(fine_top_terms[c])}
                           for c in range(K_FINE)]).sort_values(["macro_domain","fine_cluster"])

    (OUT_DIR / "macro_cluster_map.csv").write_text(map_df.to_csv(index=False), encoding="utf-8")
    (OUT_DIR / "macro_domain_summary.csv").write_text(macro_summary.to_csv(index=False), encoding="utf-8")
    (OUT_DIR / "macro_monthly_metrics.csv").write_text(monthly_metrics.to_csv(index=False), encoding="utf-8")
    (OUT_DIR / "macro_monthly_domain_shares.csv").write_text(monthly_shares.to_csv(index=False), encoding="utf-8")

    print("\nDone ✅ Outputs written to out/:")
    print(" - macro_cluster_map.csv")
    print(" - macro_domain_summary.csv")
    print(" - macro_monthly_metrics.csv")
    print(" - macro_monthly_domain_shares.csv")

    print("\nMacro domain summary (top):")
    print(macro_summary.head(10))

    print("\nMonthly macro metrics preview:")
    print(monthly_metrics)


if __name__ == "__main__":
    main()
