#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import hdbscan
from scipy.stats import entropy

DB_PATH = Path("out/corpus.sqlite")
OUT_PATH = Path("out/monthly_topic_entropy_local.csv")

MODEL_NAME = "all-MiniLM-L6-v2"

def main():

    print("Loading user messages from DB...")
    con = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT text, message_create_time_iso
        FROM messages
        WHERE role = 'user'
    """, con)

    con.close()

    df["dt"] = pd.to_datetime(df["message_create_time_iso"], errors="coerce", utc=True)
    df = df.dropna(subset=["dt"])

    df["month"] = df["dt"].dt.to_period("M").astype(str)

    texts = df["text"].astype(str).tolist()

    print(f"Total user messages: {len(texts)}")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Normalize for cosine similarity clustering
    embeddings = normalize(embeddings)

    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)

    df["cluster"] = labels

    print("Computing monthly entropy...")
    results = []

    for month, group in df.groupby("month"):

        cluster_counts = group["cluster"].value_counts()
        # Exclude noise cluster (-1)
        cluster_counts = cluster_counts[cluster_counts.index != -1]

        if len(cluster_counts) == 0:
            H = 0
        else:
            probs = cluster_counts / cluster_counts.sum()
            H = entropy(probs)

        results.append({
            "month": month,
            "num_messages": len(group),
            "num_clusters_present": len(cluster_counts),
            "topic_entropy": H
        })

    result_df = pd.DataFrame(results).sort_values("month")

    result_df.to_csv(OUT_PATH, index=False)

    print("Done.")
    print("Saved to:", OUT_PATH)
    print("\nPreview:\n")
    print(result_df)


if __name__ == "__main__":
    main()
