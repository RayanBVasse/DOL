#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.spatial.distance import jensenshannon

DB_PATH = Path("out/corpus.sqlite")
OUT_PATH = Path("out/dyadic_alignment_monthly.csv")

MAX_FEATURES = 60000
SVD_DIMS = 200
K_TOPICS = 60
MIN_DF = 5
MAX_DF = 0.6


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

    texts = df["text"].astype(str).tolist()

    print("Vectorizing...")
    vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = vec.fit_transform(texts)

    print("Reducing dimensionality...")
    svd = TruncatedSVD(n_components=SVD_DIMS, random_state=42)
    X_red = svd.fit_transform(X)
    X_red = normalize(X_red)

    print("Clustering global space...")
    km = KMeans(n_clusters=K_TOPICS, random_state=42, n_init=10)
    labels = km.fit_predict(X_red)
    df["cluster"] = labels

    print("Computing monthly alignment...")
    results = []

    for month, g in df.groupby("month"):

        user_g = g[g["role"] == "user"]
        asst_g = g[g["role"] == "assistant"]

        if len(user_g) < 10 or len(asst_g) < 10:
            continue

        user_counts = user_g["cluster"].value_counts(normalize=True)
        asst_counts = asst_g["cluster"].value_counts(normalize=True)

        # align index
        all_clusters = sorted(set(user_counts.index) | set(asst_counts.index))
        user_dist = np.array([user_counts.get(c, 0) for c in all_clusters])
        asst_dist = np.array([asst_counts.get(c, 0) for c in all_clusters])

        js = jensenshannon(user_dist, asst_dist)

        results.append({
            "month": month,
            "user_msgs": len(user_g),
            "assistant_msgs": len(asst_g),
            "js_divergence": float(js)
        })

    out = pd.DataFrame(results).sort_values("month")
    out.to_csv(OUT_PATH, index=False)

    print("Done.")
    print("Saved:", OUT_PATH)
    print("\nPreview:\n", out)


if __name__ == "__main__":
    main()
