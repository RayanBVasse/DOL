#!/usr/bin/env python3

import sqlite3
import pandas as pd
import re
from pathlib import Path

DB_PATH = Path("out/corpus.sqlite")
OUT_SUMMARY = Path("out/ai_view_profile_v01.csv")

# --- Lexical marker dictionaries ---
SYSTEM_WORDS = [
    "model", "framework", "structure", "system", "pipeline",
    "architecture", "taxonomy", "theory", "design", "mechanism"
]

FUTURE_WORDS = [
    "build", "develop", "implement", "publish", "launch",
    "create", "scale", "design", "construct"
]

META_WORDS = [
    "conceptually", "structurally", "let's step back",
    "zoom out", "in principle", "abstractly"
]

RELATIONAL_WORDS = [
    "you", "we", "us", "together", "partner", "brother",
    "hermano", "coach", "companion"
]

UNCERTAINTY_WORDS = [
    "maybe", "perhaps", "not sure", "i guess", "possibly"
]


def count_matches(text_series, word_list):
    pattern = r"\b(" + "|".join(map(re.escape, word_list)) + r")\b"
    return text_series.str.lower().str.count(pattern).sum()


def main():

    if not DB_PATH.exists():
        raise FileNotFoundError("corpus.sqlite not found.")

    con = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT role, text, text_len
        FROM messages
        WHERE role IN ('user','assistant')
    """, con)

    con.close()

    # --- Interaction Depth ---
    user_df = df[df["role"] == "user"]
    asst_df = df[df["role"] == "assistant"]

    interaction_stats = {
        "total_messages": len(df),
        "user_messages": len(user_df),
        "assistant_messages": len(asst_df),
        "avg_user_len_chars": user_df["text_len"].mean(),
        "avg_assistant_len_chars": asst_df["text_len"].mean(),
        "user_to_assistant_ratio": len(user_df) / max(len(asst_df), 1)
    }

    # --- Cognitive Style (User only) ---
    cognitive_stats = {
        "system_word_count": count_matches(user_df["text"], SYSTEM_WORDS),
        "future_word_count": count_matches(user_df["text"], FUTURE_WORDS),
        "meta_word_count": count_matches(user_df["text"], META_WORDS),
        "uncertainty_word_count": count_matches(user_df["text"], UNCERTAINTY_WORDS),
    }

    # Normalize per 1000 user messages
    for k in list(cognitive_stats.keys()):
        cognitive_stats[k + "_per_1k_msgs"] = (
            cognitive_stats[k] / max(len(user_df), 1) * 1000
        )

    # --- Relational Language (Both sides) ---
    relational_stats = {
        "relational_word_count_total": count_matches(df["text"], RELATIONAL_WORDS),
        "relational_word_count_user": count_matches(user_df["text"], RELATIONAL_WORDS),
        "relational_word_count_assistant": count_matches(asst_df["text"], RELATIONAL_WORDS),
    }

    summary = {**interaction_stats, **cognitive_stats, **relational_stats}

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("Done.")
    print("Summary written to:", OUT_SUMMARY)
    print("\nPreview:\n")
    print(summary_df.T)


if __name__ == "__main__":
    main()
