#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.stats import entropy

DB_PATH = Path("out/corpus.sqlite")
OUT_PATH = Path("out/monthly_topic_entropy_tfidf.csv")

# You can tweak these later
MAX_FEATURES = 60000
SVD_DIMS = 200
K_TOPICS = 60   # global topic clusters for first pass
MIN_DF = 5
MAX_DF = 0.6


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
    print(f"User messages: {len(texts):,}")

    print("TF-IDF vectorizing...")
    vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = vec.fit_transform(texts)

    print("SVD reducing...")
    svd = TruncatedSVD(n_components=SVD_DIMS, random_state=42)
    X_red = svd.fit_transform(X)
    X_red = normalize(X_red)

    print(f"KMeans clustering (k={K_TOPICS})...")
    km = KMeans(n_clusters=K_TOPICS, random_state=42, n_init=10)
    labels = km.fit_predict(X_red)
    df["cluster"] = labels

    print("Computing monthly entropy...")
    rows = []
    for month, g in df.groupby("month"):
        counts = g["cluster"].value_counts().sort_index()
        probs = counts / counts.sum()
        H = entropy(probs)  # natural log units
        rows.append({
            "month": month,
            "num_messages": len(g),
            "num_clusters_present": counts.shape[0],
            "topic_entropy": float(H)
        })

    out = pd.DataFrame(rows).sort_values("month")
    out.to_csv(OUT_PATH, index=False)

    print("Done ✅")
    print("Saved:", OUT_PATH)
    print("\nPreview:\n", out)


if __name__ == "__main__":
    main()
