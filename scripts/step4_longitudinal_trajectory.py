#!/usr/bin/env python3

import sqlite3
import pandas as pd
import re
from pathlib import Path

DB_PATH = Path("out/corpus.sqlite")
OUT_PATH = Path("out/trajectory_monthly.csv")

SYSTEM_WORDS = ["model","framework","structure","system","pipeline",
                "architecture","taxonomy","theory","design","mechanism"]

UNCERTAINTY_WORDS = ["maybe","perhaps","not sure","i guess","possibly"]

RELATIONAL_WORDS = ["you","we","us","together","partner",
                    "brother","hermano","coach","companion"]


def count_matches(text_series, word_list):
    pattern = r"\b(" + "|".join(map(re.escape, word_list)) + r")\b"
    return text_series.str.lower().str.count(pattern)


def main():

    con = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT role, text, message_create_time_iso
        FROM messages
        WHERE role IN ('user','assistant')
    """, con)

    con.close()

    df["dt"] = pd.to_datetime(df["message_create_time_iso"], errors="coerce", utc=True)
    df = df.dropna(subset=["dt"])

    df["month"] = df["dt"].dt.to_period("M").astype(str)

    # Only user language for cognitive markers
    user_df = df[df["role"] == "user"].copy()

    user_df["system_count"] = count_matches(user_df["text"], SYSTEM_WORDS)
    user_df["uncertainty_count"] = count_matches(user_df["text"], UNCERTAINTY_WORDS)
    user_df["relational_count"] = count_matches(user_df["text"], RELATIONAL_WORDS)

    monthly_user = user_df.groupby("month").agg(
        user_messages=("text","count"),
        system_words=("system_count","sum"),
        uncertainty_words=("uncertainty_count","sum"),
        relational_words=("relational_count","sum")
    ).reset_index()

    # Normalize per 1000 user messages
    for col in ["system_words","uncertainty_words","relational_words"]:
        monthly_user[col+"_per_1k_msgs"] = (
            monthly_user[col] / monthly_user["user_messages"] * 1000
        )

    monthly_user.to_csv(OUT_PATH, index=False)

    print("Done.")
    print("Saved:", OUT_PATH)
    print("\nPreview:\n")
    print(monthly_user)


if __name__ == "__main__":
    main()
