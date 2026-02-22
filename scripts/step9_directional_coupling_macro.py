#!/usr/bin/env python3
"""
Step 9: Directional coupling (lead–lag) on macro-domains.

Inputs (from step 8.5):
- out/macro_monthly_domain_shares.csv   (month, macro_domain, user_share, assistant_share)
- out/macro_monthly_metrics.csv         (month, user_msgs, assistant_msgs, macro_entropy_user, macro_js_divergence)

What it computes:
A) Multivariate coupling:
   mean cosine(ΔU(t), ΔA(t+1))  vs  mean cosine(ΔA(t), ΔU(t+1))
   with permutation p-values (shuffle delta order)

B) Per-domain coupling:
   For each macro_domain d:
   corr(ΔU_d(t), ΔA_d(t+1)) and corr(ΔA_d(t), ΔU_d(t+1))
   with permutation p-values

Notes:
- Uses only months passing a minimum message threshold (to avoid tiny months like 2023-02).
- Uses lag=1 month.
"""

from pathlib import Path
import numpy as np
import pandas as pd

SHARES_PATH = Path("out/macro_monthly_domain_shares.csv")
METRICS_PATH = Path("out/macro_monthly_metrics.csv")

OUT_SUMMARY = Path("out/step9_directional_coupling_summary.csv")
OUT_DOMAIN  = Path("out/step9_directional_coupling_by_domain.csv")
OUT_PAIRS   = Path("out/step9_directional_coupling_pairs.csv")

SEED = 42
N_PERM = 2000
MIN_MSGS_PER_ROLE_PER_MONTH = 200  # adjust if you want stricter/looser


def cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def perm_pvalue_greater(null_vals, real_val):
    null_vals = np.asarray([v for v in null_vals if np.isfinite(v)], dtype=float)
    if not np.isfinite(real_val) or len(null_vals) == 0:
        return np.nan
    return float((null_vals >= real_val).mean())


def perm_pvalue_abs(null_vals, real_val):
    null_vals = np.asarray([v for v in null_vals if np.isfinite(v)], dtype=float)
    if not np.isfinite(real_val) or len(null_vals) == 0:
        return np.nan
    return float((np.abs(null_vals) >= abs(real_val)).mean())


def main():
    rng = np.random.default_rng(SEED)

    if not SHARES_PATH.exists():
        raise FileNotFoundError(f"Missing {SHARES_PATH}. Run step8_5_macro_domains_tfidf.py first.")
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing {METRICS_PATH}. Run step8_5_macro_domains_tfidf.py first.")

    shares = pd.read_csv(SHARES_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    # Filter months by message threshold to avoid tiny months
    keep_months = metrics[
        (metrics["user_msgs"] >= MIN_MSGS_PER_ROLE_PER_MONTH) &
        (metrics["assistant_msgs"] >= MIN_MSGS_PER_ROLE_PER_MONTH)
    ]["month"].tolist()

    shares = shares[shares["month"].isin(keep_months)].copy()

    months = sorted(shares["month"].unique())
    domains = sorted(shares["macro_domain"].unique())
    M = len(domains)

    print("Using months:", months)
    print("Macro domains:", domains)

    # Build U(t), A(t) vectors per month (length M)
    U = []
    A = []
    for m in months:
        g = shares[shares["month"] == m].sort_values("macro_domain")
        # Ensure full domain coverage
        u_vec = np.array([g[g["macro_domain"] == d]["user_share"].values[0] if (g["macro_domain"] == d).any() else 0.0 for d in domains])
        a_vec = np.array([g[g["macro_domain"] == d]["assistant_share"].values[0] if (g["macro_domain"] == d).any() else 0.0 for d in domains])
        U.append(u_vec)
        A.append(a_vec)

    U = np.vstack(U)  # shape T x M
    A = np.vstack(A)  # shape T x M
    T = len(months)

    if T < 4:
        raise ValueError("Need at least 4 usable months for lagged Δ coupling.")

    # Δ vectors: ΔU(t) corresponds to change into month t (from t-1)
    dU = U[1:] - U[:-1]   # shape (T-1) x M, indexed by t=1..T-1
    dA = A[1:] - A[:-1]

    # For lag=1 directional coupling, we compare:
    # dU[t] with dA[t+1]  (t = 0..(T-3))  because dU has length T-1, so t+1 must exist
    # dA[t] with dU[t+1]
    pair_rows = []
    forward_sims = []
    reverse_sims = []
    for t in range(0, T-2):  # up to T-3 inclusive
        sim_U_to_A = cosine(dU[t], dA[t+1])
        sim_A_to_U = cosine(dA[t], dU[t+1])
        forward_sims.append(sim_U_to_A)
        reverse_sims.append(sim_A_to_U)
        pair_rows.append({
            "t_index": t,
            "month_t": months[t+1],          # dU[t] is change into months[t+1]
            "month_tplus1": months[t+2],     # dA[t+1] is change into months[t+2]
            "cos_dU_t__dA_tplus1": sim_U_to_A,
            "cos_dA_t__dU_tplus1": sim_A_to_U,
        })

    forward_mean = float(np.nanmean(forward_sims))
    reverse_mean = float(np.nanmean(reverse_sims))
    lead_diff = forward_mean - reverse_mean  # positive suggests user leads more than assistant

    # Permutation nulls (multivariate):
    # shuffle the order of dA rows (break temporal linkage), keep dU fixed
    null_forward = []
    null_reverse = []
    null_diff = []

    idx = np.arange(dA.shape[0])  # 0..T-2
    for _ in range(N_PERM):
        perm = rng.permutation(idx)

        # compute permuted forward mean: dU[t] with dA_perm[t+1]
        sims_f = []
        sims_r = []
        for t in range(0, T-2):
            sims_f.append(cosine(dU[t], dA[perm][t+1]))
            sims_r.append(cosine(dA[perm][t], dU[t+1]))
        mf = float(np.nanmean(sims_f))
        mr = float(np.nanmean(sims_r))
        null_forward.append(mf)
        null_reverse.append(mr)
        null_diff.append(mf - mr)

    p_forward = perm_pvalue_greater(null_forward, forward_mean)
    p_reverse = perm_pvalue_greater(null_reverse, reverse_mean)
    p_diff_abs = perm_pvalue_abs(null_diff, lead_diff)

    # Per-domain coupling (lag=1):
    # arrays length (T-2) for pairwise aligned series: dU[0..T-3] vs dA[1..T-2]
    X = dU[0:T-2, :]        # (T-2) x M
    Y = dA[1:T-1, :]        # (T-2) x M
    Xr = dA[0:T-2, :]
    Yr = dU[1:T-1, :]

    domain_rows = []
    for j, d in enumerate(domains):
        xu = X[:, j]
        ya = Y[:, j]
        xa = Xr[:, j]
        yu = Yr[:, j]

        # Pearson r (with tiny samples; handle degenerate)
        def pearson(x, y):
            x = np.asarray(x, float); y = np.asarray(y, float)
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                return np.nan
            return float(np.corrcoef(x, y)[0,1])

        r_u_to_a = pearson(xu, ya)
        r_a_to_u = pearson(xa, yu)

        # permutation p-values for per-domain r: shuffle y-series order
        null_r_u_to_a = []
        null_r_a_to_u = []
        for _ in range(N_PERM):
            perm = rng.permutation(len(ya))
            null_r_u_to_a.append(pearson(xu, ya[perm]))
            perm2 = rng.permutation(len(yu))
            null_r_a_to_u.append(pearson(xa, yu[perm2]))

        p_u_to_a = perm_pvalue_abs(null_r_u_to_a, r_u_to_a)
        p_a_to_u = perm_pvalue_abs(null_r_a_to_u, r_a_to_u)

        domain_rows.append({
            "macro_domain": d,
            "r_user_leads_assistant": r_u_to_a,
            "p_perm_abs_user_leads": p_u_to_a,
            "r_assistant_leads_user": r_a_to_u,
            "p_perm_abs_assistant_leads": p_a_to_u,
        })

    # Save outputs
    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(OUT_PAIRS, index=False)

    summary_df = pd.DataFrame([{
        "months_used": ", ".join(months),
        "num_months": T,
        "num_pairs": T-2,
        "forward_mean_cos_user_to_asst": forward_mean,
        "reverse_mean_cos_asst_to_user": reverse_mean,
        "lead_diff_forward_minus_reverse": lead_diff,
        "p_perm_forward_greater": p_forward,
        "p_perm_reverse_greater": p_reverse,
        "p_perm_abs_diff": p_diff_abs,
        "N_perm": N_PERM,
        "min_msgs_per_role_per_month": MIN_MSGS_PER_ROLE_PER_MONTH
    }])
    summary_df.to_csv(OUT_SUMMARY, index=False)

    domain_df = pd.DataFrame(domain_rows).sort_values("p_perm_abs_user_leads")
    domain_df.to_csv(OUT_DOMAIN, index=False)

    print("\nDone ✅")
    print("Saved:")
    print(" -", OUT_SUMMARY)
    print(" -", OUT_DOMAIN)
    print(" -", OUT_PAIRS)

    print("\nSystem-level coupling summary:")
    print(summary_df.to_string(index=False))

    print("\nTop per-domain signals (sorted by user->assistant p):")
    print(domain_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
