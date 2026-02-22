import os, pandas as pd

candidates = []
for root, _, files in os.walk("."):
    for f in files:
        if f.lower().endswith(".csv"):
            path = os.path.join(root, f)
            try:
                df = pd.read_csv(path, nrows=5)
                cols = set(df.columns)
                if "node_id" in cols and ("cluster" in cols or "fine_cluster" in cols or "topic" in cols or "macro_domain" in cols):
                    candidates.append((path, list(df.columns)))
            except:
                pass

print("Found candidates:")
for p, cols in candidates[:50]:
    print(p, cols)