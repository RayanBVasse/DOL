#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

DB_PATH = Path("out/corpus.sqlite")
OUT_PATH = Path("out/cluster_summary_tfidf.csv")

MAX_FEATURES = 60000
SVD_DIMS = 200
K_TOPICS = 60
MIN_DF = 5
MAX_DF = 0.6
TOP_TERMS = 15


def main():
    print("Loading user messages...")
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT text, message_create_time_iso
        FROM messages
        WHERE role='user' AND text IS NOT NULL AND LENGTH(text) > 0
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

    print("Clustering...")
    km = KMeans(n_clusters=K_TOPICS, random_state=42, n_init=10)
    labels = km.fit_predict(X_red)
    df["cluster"] = labels

    print("Extracting top terms per cluster...")
    terms = np.array(vec.get_feature_names_out())

    cluster_centers = km.cluster_centers_

    rows = []

    for i in range(K_TOPICS):
        center = cluster_centers[i]
        top_indices = center.argsort()[-TOP_TERMS:][::-1]
        top_words = terms[top_indices]

        cluster_size = (df["cluster"] == i).sum()

        # Month distribution
        month_counts = df[df["cluster"] == i]["month"].value_counts().to_dict()

        rows.append({
            "cluster_id": i,
            "size": cluster_size,
            "auto_label": ", ".join(top_words[:5]),
            "top_terms": ", ".join(top_words),
            "month_distribution": str(month_counts)
        })

    summary_df = pd.DataFrame(rows).sort_values("size", ascending=False)
    summary_df.to_csv(OUT_PATH, index=False)

    print("Done.")
    print("Saved:", OUT_PATH)
    print("\nTop 10 clusters:\n")
    print(summary_df.head(10))


if __name__ == "__main__":
    main()
