#!/usr/bin/env python3
"""
Step 2: Diagnostics + SQLite build

Inputs:
  out/messages_master.csv
  out/threads_master.csv

Outputs:
  out/corpus.sqlite
  out/diag_role_counts.csv
  out/diag_daily_counts.csv
  out/diag_top_threads_by_chars.csv
"""

import sqlite3
import pandas as pd
from pathlib import Path

OUT_DIR = Path("out")
MSG_CSV = OUT_DIR / "messages_master.csv"
THR_CSV = OUT_DIR / "threads_master.csv"

DB_PATH = OUT_DIR / "corpus.sqlite"

ROLE_COUNTS_CSV = OUT_DIR / "diag_role_counts.csv"
DAILY_COUNTS_CSV = OUT_DIR / "diag_daily_counts.csv"
TOP_THREADS_CSV = OUT_DIR / "diag_top_threads_by_chars.csv"


def main():
    if not MSG_CSV.exists():
        raise FileNotFoundError(f"Missing {MSG_CSV}")
    if not THR_CSV.exists():
        raise FileNotFoundError(f"Missing {THR_CSV}")

    print(f"Loading {MSG_CSV} ... (this can take a bit)")
    msgs = pd.read_csv(MSG_CSV, low_memory=False)

    print(f"Loading {THR_CSV} ...")
    thrs = pd.read_csv(THR_CSV, low_memory=False)

    # --- Basic cleanup / types ---
    # Timestamps: message_create_time_iso is ISO string. Parse for time analytics.
    # Some rows may be empty; coerce to NaT.
    msgs["message_dt"] = pd.to_datetime(msgs.get("message_create_time_iso", ""), errors="coerce", utc=True)

    # Normalize role labels
    msgs["role"] = msgs["role"].fillna("").astype(str).str.strip().str.lower()
    msgs["thread_title"] = msgs["thread_title"].fillna("").astype(str)

    # --- Diagnostics ---
    total_rows = len(msgs)
    total_threads = msgs["thread_id"].nunique()
    dt_min = msgs["message_dt"].min()
    dt_max = msgs["message_dt"].max()
    ts_coverage = msgs["message_dt"].notna().mean()

    print("\n=== Corpus diagnostics ===")
    print(f"Messages (rows): {total_rows:,}")
    print(f"Threads (unique): {total_threads:,}")
    print(f"Timestamp coverage: {ts_coverage*100:.2f}%")
    print(f"Date range: {dt_min}  ->  {dt_max}")

    # Role counts
    role_counts = msgs["role"].value_counts(dropna=False).reset_index()
    role_counts.columns = ["role", "count"]
    role_counts["pct"] = role_counts["count"] / role_counts["count"].sum()
    role_counts.to_csv(ROLE_COUNTS_CSV, index=False)
    print(f"\nSaved role counts -> {ROLE_COUNTS_CSV}")

    # Daily counts (for quick timeline sanity)
    msgs["day"] = msgs["message_dt"].dt.date
    daily = (
        msgs.dropna(subset=["message_dt"])
            .groupby(["day", "role"])
            .size()
            .reset_index(name="count")
            .sort_values(["day", "role"])
    )
    daily.to_csv(DAILY_COUNTS_CSV, index=False)
    print(f"Saved daily counts -> {DAILY_COUNTS_CSV}")

    # Thread size ranking (chars)
    # Use thread-level table if it exists; else compute from messages.
    if "main_path_text_chars" in thrs.columns:
        top_threads = thrs.sort_values("main_path_text_chars", ascending=False).head(30)
        top_threads[["thread_id", "thread_title", "main_path_text_chars",
                     "num_user_msgs_main_path", "num_assistant_msgs_main_path"]].to_csv(TOP_THREADS_CSV, index=False)
    else:
        tmp = msgs.groupby(["thread_id", "thread_title"])["text_len"].sum().reset_index()
        tmp = tmp.rename(columns={"text_len": "main_path_text_chars"})
        tmp.sort_values("main_path_text_chars", ascending=False).head(30).to_csv(TOP_THREADS_CSV, index=False)

    print(f"Saved top threads -> {TOP_THREADS_CSV}")

    # --- Build SQLite DB for fast querying later ---
    if DB_PATH.exists():
        DB_PATH.unlink()

    print(f"\nBuilding SQLite DB -> {DB_PATH}")
    con = sqlite3.connect(DB_PATH)

    # Keep only columns we need for v1
    msgs_db = msgs[[
        "thread_id", "thread_title",
        "node_id", "parent_id",
        "role", "author_name",
        "message_create_time", "message_create_time_iso",
        "text", "text_len"
    ]].copy()

    # SQLite writes
    msgs_db.to_sql("messages", con, index=False)
    thrs.to_sql("threads", con, index=False)

    # Indexes (speed)
    con.execute("CREATE INDEX idx_messages_thread_id ON messages(thread_id);")
    con.execute("CREATE INDEX idx_messages_role ON messages(role);")
    con.execute("CREATE INDEX idx_messages_time ON messages(message_create_time);")
    con.commit()
    con.close()

    print("Done ✅")
    print(f"DB ready: {DB_PATH}")
    print("\nNext: we can exclude TOOL/SYSTEM rows (or keep separate) and start embeddings/clustering.")


if __name__ == "__main__":
    main()
