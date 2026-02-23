"""
step_ttr_lexical_richness.py
────────────────────────────
Computes monthly Type-Token Ratio (TTR) for user messages.

TTR = unique_word_types / total_word_tokens (per month, pooled across messages).
A declining TTR indicates vocabulary specialisation — the user repeats a narrower
set of domain-specific words more frequently. A rising TTR indicates broadening.

Input  : data/messages_master.csv
Output : data/ttr_monthly.csv

Dependencies: pandas, re (stdlib)
"""

import re
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
INPUT         = 'data/messages_master.csv'
OUTPUT        = 'data/ttr_monthly.csv'
ACTIVE_START  = '2025-05'
ACTIVE_END    = '2025-12'
# ─────────────────────────────────────────────────────────────────────────────


def compute_ttr(text_series: pd.Series):
    """Pool all text in the series, tokenise, return (TTR, n_types, n_tokens)."""
    all_text = ' '.join(text_series.dropna().astype(str).str.lower())
    tokens = re.findall(r'\b[a-z]+\b', all_text)
    if not tokens:
        return None, None, None
    n_types  = len(set(tokens))
    n_tokens = len(tokens)
    return round(n_types / n_tokens, 4), n_types, n_tokens


def main():
    df = pd.read_csv(INPUT)

    # User messages only
    df = df[df['role'] == 'user'].copy()

    # Parse month
    df['message_create_time_iso'] = pd.to_datetime(
        df['message_create_time_iso'], utc=True, errors='coerce'
    )
    df['month'] = df['message_create_time_iso'].dt.to_period('M').astype(str)

    # Active window
    df = df[df['month'].between(ACTIVE_START, ACTIVE_END)]

    rows = []
    for month, grp in df.groupby('month'):
        ratio, types, tokens = compute_ttr(grp['text'])
        rows.append({
            'month':        month,
            'ttr':          ratio,
            'unique_words': types,
            'total_words':  tokens,
        })

    out = pd.DataFrame(rows).sort_values('month')
    out.to_csv(OUTPUT, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved → {OUTPUT}")


if __name__ == '__main__':
    main()
