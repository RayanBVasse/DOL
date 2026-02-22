#!/usr/bin/env python3
import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

MASTER_PATH = Path("out/messages_master.csv")
MACRO_SUMMARY_PATH = Path("out/macro_domain_summary.csv")  # you uploaded this earlier
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

OUT_FINE = OUT_DIR / "node_to_fine_cluster.csv"
OUT_MACRO = OUT_DIR / "node_to_macro_domain.csv"
OUT_CHECK = OUT_DIR / "diag_macro_repro_check.csv"
OUT_CFG = OUT_DIR / "run_config_step8_6b.json"

# ----------- PARAMETERS (match your earlier run as closely as possible) -----------
SEED = 42
K_FINE = 60          # inferred from fine cluster IDs 0..59 in macro_domain_summary.csv
SVD_DIMS = 100       # used elsewhere in your robustness scripts; good default
MAX_FEATURES = 60000
MIN_DF = 2
MAX_DF = 0.90
NGRAM_RANGE = (1, 2)

# If memory becomes tight, drop MAX_FEATURES to 40000 and/or set NGRAM_RANGE=(1,1)

def parse_fine_clusters(s):
    """
    macro_domain_summary.csv has 'fine_clusters' as a comma-separated string like '0,3,6,...'
    """
    if pd.isna(s):
        return []
    nums = re.findall(r"\d+", str(s))
    return [int(x) for x in nums]

def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing {MASTER_PATH}")

    if not MACRO_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing {MACRO_SUMMARY_PATH} (expected mapping of macro_domain -> fine_clusters)")

    print("Loading message_master.csv...")
    df = pd.read_csv(
        MASTER_PATH,
        usecols=["node_id", "role", "text", "text_len", "message_create_time_iso"],
        low_memory=False
    )

    # Filter to user+assistant, non-empty text
    df = df[df["role"].isin(["user", "assistant"])].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].copy()

    print(f"Messages kept (user+assistant, non-empty): {len(df)}")

    texts = df["text"].tolist()

    print("TF-IDF vectorizing...")
    vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        stop_words="english"
    )
    X = vec.fit_transform(texts)

    print(f"TF-IDF matrix: {X.shape}")

    print(f"SVD reducing to {SVD_DIMS} dims...")
    svd = TruncatedSVD(n_components=SVD_DIMS, random_state=SEED)
    Xr = svd.fit_transform(X)

    print("KMeans clustering...")
    km = KMeans(
        n_clusters=K_FINE,
        random_state=SEED,
        n_init=20
    )
    fine_labels = km.fit_predict(Xr)

    # Save node_id -> fine cluster
    out_fine = pd.DataFrame({
        "node_id": df["node_id"].astype(str).values,
        "fine_cluster": fine_labels.astype(int)
    })
    out_fine.to_csv(OUT_FINE, index=False)
    print("Wrote:", OUT_FINE)

    # Build fine_cluster -> macro_domain mapping from macro_domain_summary.csv
    ms = pd.read_csv(MACRO_SUMMARY_PATH)
    fine_to_macro = {}

    for _, row in ms.iterrows():
        m = int(row["macro_domain"])
        fine_list = parse_fine_clusters(row.get("fine_clusters", ""))
        for c in fine_list:
            fine_to_macro[int(c)] = m

    # Map to macro_domain
    out_fine["macro_domain"] = out_fine["fine_cluster"].map(fine_to_macro)

    missing = out_fine["macro_domain"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} messages have fine_cluster not found in macro mapping.")
        # If this happens, it likely means the clustering params differ from the original run.

    out_fine[["node_id", "macro_domain"]].to_csv(OUT_MACRO, index=False)
    print("Wrote:", OUT_MACRO)

    # Diagnostics: compare macro sizes with macro_domain_summary sizes
    obs = out_fine.dropna(subset=["macro_domain"]).groupby("macro_domain").size().reset_index(name="observed_msgs")
    exp = ms[["macro_domain", "size_msgs_total"]].rename(columns={"size_msgs_total": "expected_msgs"})
    chk = exp.merge(obs, on="macro_domain", how="left").fillna({"observed_msgs": 0})
    chk["diff"] = chk["observed_msgs"] - chk["expected_msgs"]
    chk.to_csv(OUT_CHECK, index=False)
    print("Wrote:", OUT_CHECK)

    # Save config for reproducibility
    cfg = dict(
        SEED=SEED, K_FINE=K_FINE, SVD_DIMS=SVD_DIMS,
        MAX_FEATURES=MAX_FEATURES, MIN_DF=MIN_DF, MAX_DF=MAX_DF,
        NGRAM_RANGE=list(NGRAM_RANGE),
        input_master=str(MASTER_PATH),
        input_macro_summary=str(MACRO_SUMMARY_PATH),
        outputs=dict(node_to_fine=str(OUT_FINE), node_to_macro=str(OUT_MACRO), diag_check=str(OUT_CHECK))
    )
    OUT_CFG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("Wrote:", OUT_CFG)

    print("\nDONE.")
    print("Next: run step9.1 weekly coupling using:")
    print(" - message_master.csv")
    print(" - out/node_to_macro_domain.csv")

if __name__ == "__main__":
    main()