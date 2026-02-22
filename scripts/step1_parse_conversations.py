#!/usr/bin/env python3
"""
Step 1: Parse ChatGPT export conversations.json, reconstruct linear threads,
and write:
  - out/messages_master.csv         (message-level table)
  - out/threads_master.csv          (thread-level table)
  - out/threads_txt/<thread_id>.txt (human-readable thread transcripts)

Designed to be robust to branching trees and mixed content blocks.
"""

import json
import csv
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Helpers
# ----------------------------

def safe_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\s\-\.\(\)]", "", s, flags=re.UNICODE).strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return "untitled"
    return s[:max_len]


def unix_to_iso(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return ""


def extract_text_from_content(content: Any) -> str:
    """
    ChatGPT export content often looks like:
      {"content_type":"text","parts":["...","..."]}
    But can include other shapes (code blocks, images, etc.).
    We pull text-ish parts safely.
    """
    if content is None:
        return ""

    # Typical structure
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            # parts can contain strings or dicts; join strings and stringify lightly otherwise
            out_parts = []
            for p in parts:
                if isinstance(p, str):
                    out_parts.append(p)
                elif isinstance(p, dict):
                    # Some exports store structured parts; try common keys
                    if "text" in p and isinstance(p["text"], str):
                        out_parts.append(p["text"])
                    else:
                        out_parts.append(json.dumps(p, ensure_ascii=False))
                else:
                    out_parts.append(str(p))
            return "\n".join(out_parts).strip()

        # Sometimes content is directly a string in other key
        for k in ("text", "value"):
            if k in content and isinstance(content[k], str):
                return content[k].strip()

        return ""

    # If it's already a string
    if isinstance(content, str):
        return content.strip()

    return ""


def pick_main_path(message_nodes: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Cycle-safe, non-recursive main-path picker.

    Heuristic:
      - Consider roots as nodes with no parent (or missing parent)
      - Compute longest path lengths using an iterative postorder traversal
      - Break ties by earliest create_time
      - Reconstruct best path by following best child pointers

    If cycles exist, they are ignored (we cut traversal when detected).
    """

    # parent->children adjacency
    children_map: Dict[str, List[str]] = {}
    roots: List[str] = []

    def node_time(nid: str) -> float:
        msg = (message_nodes.get(nid, {}).get("message") or {})
        t = msg.get("create_time")
        try:
            return float(t) if t is not None else float("inf")
        except Exception:
            return float("inf")

    # Build children map + roots
    for node_id, node in message_nodes.items():
        parent = node.get("parent")
        if not parent or parent not in message_nodes:
            roots.append(node_id)
        else:
            children_map.setdefault(parent, []).append(node_id)

    if not roots:
        # fallback: pick any node as root
        roots = list(message_nodes.keys())[:1]

    # Sort children lists for deterministic tie-breaking by time
    for p, kids in children_map.items():
        children_map[p] = sorted(kids, key=node_time)

    # We will compute:
    #   best_len[nid] = length of best path starting at nid (inclusive)
    #   best_next[nid] = child id to follow for best path (or None)
    best_len: Dict[str, int] = {}
    best_next: Dict[str, Optional[str]] = {}

    # Iterative postorder traversal for each root
    # state: (nid, stage) where stage 0=enter, 1=exit
    for root in roots:
        stack: List[Tuple[str, int]] = [(root, 0)]
        visiting: set = set()  # current stack nodes to detect cycles

        while stack:
            nid, stage = stack.pop()

            if stage == 0:
                if nid in visiting:
                    # cycle detected; skip expanding further
                    continue
                visiting.add(nid)
                stack.append((nid, 1))  # exit stage
                for kid in children_map.get(nid, []):
                    if kid not in best_len:  # only traverse if not computed yet
                        stack.append((kid, 0))
            else:
                # exit stage: compute best for nid
                visiting.discard(nid)

                kids = children_map.get(nid, [])
                if not kids:
                    best_len[nid] = 1
                    best_next[nid] = None
                else:
                    # choose child with max best_len; tie-break by earliest child time
                    chosen = None
                    chosen_len = 0
                    for kid in kids:
                        # If kid not computed due to cycles, treat as length 1
                        klen = best_len.get(kid, 1)
                        if klen > chosen_len:
                            chosen_len = klen
                            chosen = kid
                        elif klen == chosen_len and chosen is not None:
                            if node_time(kid) < node_time(chosen):
                                chosen = kid

                    best_len[nid] = 1 + (chosen_len if chosen is not None else 0)
                    best_next[nid] = chosen

    # Pick best root by best_len; tie-break by earliest root time
    roots_sorted = sorted(
        roots,
        key=lambda r: (-best_len.get(r, 1), node_time(r))
    )
    best_root = roots_sorted[0]

    # Reconstruct path following best_next pointers (cycle-safe)
    path: List[str] = []
    seen: set = set()
    cur = best_root
    while cur and cur not in seen:
        seen.add(cur)
        path.append(cur)
        cur = best_next.get(cur)

    return path


    # Choose best root = longest path from that root (ties by earliest)
    def best_path_from(start: str) -> List[str]:
        # DFS with memo for longest path
        memo: Dict[str, List[str]] = {}

        def dfs(nid: str) -> List[str]:
            if nid in memo:
                return memo[nid]
            kids = children_map.get(nid, [])
            if not kids:
                memo[nid] = [nid]
                return memo[nid]

            # evaluate each child path
            best: List[str] = []
            # sort kids for deterministic tie-break (earlier time first)
            kids_sorted = sorted(kids, key=node_time)
            for kid in kids_sorted:
                candidate = [nid] + dfs(kid)
                if len(candidate) > len(best):
                    best = candidate
                elif len(candidate) == len(best) and candidate:
                    # tie-break by earliest next-node time
                    # compare create_time of second element if possible
                    if len(candidate) > 1 and len(best) > 1:
                        if node_time(candidate[1]) < node_time(best[1]):
                            best = candidate
            memo[nid] = best if best else [nid]
            return memo[nid]

        return dfs(start)

    if not roots:
        # fallback: pick any node as root
        roots = list(message_nodes.keys())[:1]

    root_paths = [best_path_from(r) for r in roots]
    # choose longest root path; tie-break by earliest root create_time
    root_paths_sorted = sorted(
        root_paths,
        key=lambda p: (-len(p), node_time(p[0]) if p else float("inf"))
    )
    return root_paths_sorted[0] if root_paths_sorted else []


# ----------------------------
# Main
# ----------------------------

def main():
    in_path = Path("conversations.json")
    if not in_path.exists():
        raise FileNotFoundError("conversations.json not found in current directory.")

    out_dir = Path("out")
    threads_txt_dir = out_dir / "threads_txt"
    out_dir.mkdir(exist_ok=True)
    threads_txt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...")
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Export file sometimes is a list of conversations
    if not isinstance(data, list):
        raise ValueError("Expected conversations.json to be a list of conversation objects.")

    messages_csv_path = out_dir / "messages_master.csv"
    threads_csv_path = out_dir / "threads_master.csv"

    messages_fields = [
        "thread_id", "thread_title", "thread_create_time", "thread_update_time",
        "node_id", "parent_id",
        "role", "author_name",
        "message_create_time", "message_create_time_iso",
        "text", "text_len"
    ]
    threads_fields = [
        "thread_id", "thread_title",
        "thread_create_time", "thread_create_time_iso",
        "thread_update_time", "thread_update_time_iso",
        "num_nodes_in_tree", "num_nodes_in_main_path",
        "num_user_msgs_main_path", "num_assistant_msgs_main_path",
        "main_path_text_chars"
    ]

    total_threads = 0
    total_msgs_written = 0

    with messages_csv_path.open("w", newline="", encoding="utf-8") as mf, \
         threads_csv_path.open("w", newline="", encoding="utf-8") as tf:

        m_writer = csv.DictWriter(mf, fieldnames=messages_fields)
        t_writer = csv.DictWriter(tf, fieldnames=threads_fields)
        m_writer.writeheader()
        t_writer.writeheader()

        for conv in data:
            total_threads += 1

            thread_id = str(conv.get("id", ""))
            title = conv.get("title") or "Untitled"
            create_time = conv.get("create_time")
            update_time = conv.get("update_time")

            mapping = conv.get("mapping") or {}
            if not isinstance(mapping, dict) or not mapping:
                # Some conversations can be empty
                continue

            # Build message_nodes in a normalized form: node_id -> {parent, message, children}
            message_nodes: Dict[str, Dict[str, Any]] = {}
            for node_id, node in mapping.items():
                if not isinstance(node, dict):
                    continue
                message_nodes[str(node_id)] = {
                    "parent": node.get("parent"),
                    "children": node.get("children") or [],
                    "message": node.get("message"),
                }

            main_path = pick_main_path(message_nodes)

            # Prepare transcript
            transcript_lines: List[str] = []
            num_user = 0
            num_asst = 0
            transcript_chars = 0

            for node_id in main_path:
                node = message_nodes.get(node_id) or {}
                parent_id = node.get("parent") or ""
                msg = node.get("message") or {}

                author = msg.get("author") or {}
                role = author.get("role") or ""
                author_name = author.get("name") or ""

                msg_ct = msg.get("create_time")
                msg_ct_iso = unix_to_iso(msg_ct)

                content = msg.get("content")
                text = extract_text_from_content(content)

                # Skip empty system nodes, tool nodes, etc. but still record them if they have text
                if role == "user":
                    num_user += 1
                    speaker = "USER"
                elif role == "assistant":
                    num_asst += 1
                    speaker = "ASSISTANT"
                else:
                    speaker = role.upper() if role else "OTHER"

                # Write message row (only for nodes on main path; tree-wide export is possible later)
                if text:
                    m_writer.writerow({
                        "thread_id": thread_id,
                        "thread_title": title,
                        "thread_create_time": create_time if create_time is not None else "",
                        "thread_update_time": update_time if update_time is not None else "",
                        "node_id": node_id,
                        "parent_id": parent_id,
                        "role": role,
                        "author_name": author_name,
                        "message_create_time": msg_ct if msg_ct is not None else "",
                        "message_create_time_iso": msg_ct_iso,
                        "text": text,
                        "text_len": len(text),
                    })
                    total_msgs_written += 1

                    header = f"[{speaker}] {msg_ct_iso}".strip()
                    transcript_lines.append(header)
                    transcript_lines.append(text)
                    transcript_lines.append("")  # spacer
                    transcript_chars += len(text)

            # Save per-thread transcript
            thread_safe_title = safe_filename(title)
            thread_txt_path = threads_txt_dir / f"{thread_id}__{thread_safe_title}.txt"
            with thread_txt_path.open("w", encoding="utf-8") as out_f:
                out_f.write("\n".join(transcript_lines).strip() + "\n")

            # Thread summary row
            t_writer.writerow({
                "thread_id": thread_id,
                "thread_title": title,
                "thread_create_time": create_time if create_time is not None else "",
                "thread_create_time_iso": unix_to_iso(create_time),
                "thread_update_time": update_time if update_time is not None else "",
                "thread_update_time_iso": unix_to_iso(update_time),
                "num_nodes_in_tree": len(message_nodes),
                "num_nodes_in_main_path": len(main_path),
                "num_user_msgs_main_path": num_user,
                "num_assistant_msgs_main_path": num_asst,
                "main_path_text_chars": transcript_chars,
            })

            if total_threads % 50 == 0:
                print(f"Processed {total_threads} threads...")

    print("\nDone ✅")
    print(f"Threads processed: {total_threads}")
    print(f"Messages written (main-path, non-empty): {total_msgs_written}")
    print(f"Outputs:")
    print(f"  - {messages_csv_path}")
    print(f"  - {threads_csv_path}")
    print(f"  - {threads_txt_dir}/")


if __name__ == "__main__":
    main()
