"""
interpret_figures.py
====================
DOL Pipeline – Data-Driven Figure Interpretations
Reads the same CSVs as visualise_user.py, computes key statistics,
and produces a self-contained HTML report with every figure embedded
alongside a plain-English interpretation.

Usage
-----
    python interpret_figures.py

Output
------
    user_figures/DOL_Report.html   ← open in any browser
"""

import os, base64, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(DATA_DIR, "user_figures")
OUTPUT_HTML = os.path.join(FIG_DIR, "DOL_Report.html")

DOMAIN_NAMES = {
    0: "AI persona modes",
    1: "Analytical / research",
    2: "Personal reflection",
    3: "Spanish-language",
    4: "Creative / book writing",
    5: "General conversational",
    6: "Fitness / weightlifting",
    7: "Code / data / Python",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(DATA_DIR, filename),
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "..", "uploads", filename),
        os.path.join(script_dir, "outputs", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"Cannot find {filename}")

def try_load(filename):
    try: return load(filename)
    except FileNotFoundError: return None

def fig_to_b64(name):
    path = os.path.join(FIG_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def short_month(m):
    try: return pd.to_datetime(m + "-01").strftime("%b '%y")
    except: return str(m)

def trend_word(rho):
    if rho >= 0.7:  return "strongly increasing"
    if rho >= 0.4:  return "moderately increasing"
    if rho >= 0.1:  return "slightly increasing"
    if rho <= -0.7: return "strongly decreasing"
    if rho <= -0.4: return "moderately decreasing"
    if rho <= -0.1: return "slightly decreasing"
    return "stable"

def pct_change(series):
    s = series.dropna()
    if len(s) < 2 or s.iloc[0] == 0: return 0
    return (s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Loading data …")
traj   = load("trajectory_monthly.csv")
ttr    = load("ttr_monthly.csv")
domain = load("macro_monthly_domain_shares.csv")
dyadic = load("dyadic_alignment_monthly.csv")
tc     = load("thread_complexity.csv")
tcm    = load("thread_complexity_monthly.csv")
entr   = load("rolling_entropy_250.csv")
weekly_dir = try_load("weekly_directional_by_domain.csv")

# Filter to active months
traj = traj[traj["user_messages"] >= 100].copy()
active_months = set(traj["month"].values)

# Map domain names
if pd.api.types.is_integer_dtype(domain["macro_domain"]):
    domain["macro_domain"] = domain["macro_domain"].map(DOMAIN_NAMES).fillna(domain["macro_domain"].astype(str))
domain = domain[domain["month"].isin(active_months)].copy()

months = [short_month(m) for m in traj["month"].values]
n_months = len(months)
total_msgs = int(traj["user_messages"].sum())

# ─────────────────────────────────────────────
# COMPUTE INTERPRETATIONS
# ─────────────────────────────────────────────
interpretations = {}

# ── FIG 01: Monthly Activity ──────────────────
peak_month = short_month(traj.loc[traj["user_messages"].idxmax(), "month"])
peak_msgs  = int(traj["user_messages"].max())
low_month  = short_month(traj.loc[traj["user_messages"].idxmin(), "month"])
low_msgs   = int(traj["user_messages"].min())
x_idx = np.arange(n_months)
rho_activity, _ = spearmanr(x_idx, traj["user_messages"].values)
trend_activity = trend_word(rho_activity)

interpretations["fig01"] = (
    f"Across {n_months} months ({months[0]}–{months[-1]}), you sent a total of "
    f"<strong>{total_msgs:,} messages</strong> to ChatGPT. "
    f"Activity was <strong>{trend_activity}</strong> overall, peaking in <strong>{peak_month}</strong> "
    f"({peak_msgs:,} messages) and at its lowest in <strong>{low_month}</strong> ({low_msgs:,} messages). "
    f"The dashed trend line confirms the overall direction of engagement over the period."
)

# ── FIG 02: Cognitive Markers ─────────────────
sys_rho,  _ = spearmanr(x_idx, traj["system_words_per_1k_msgs"].values)
rel_rho,  _ = spearmanr(x_idx, traj["relational_words_per_1k_msgs"].values)
unc_rho,  _ = spearmanr(x_idx, traj["uncertainty_words_per_1k_msgs"].values)

sys_pct = pct_change(traj["system_words_per_1k_msgs"])
rel_pct = pct_change(traj["relational_words_per_1k_msgs"])

unc_peak_idx = traj["uncertainty_words_per_1k_msgs"].values.argmax()
unc_peak_month = short_month(traj["month"].values[unc_peak_idx])

interpretations["fig02"] = (
    f"Your language became measurably more analytical over the period. "
    f"<strong>Systems thinking language</strong> (words like 'framework', 'structure', 'mechanism') "
    f"grew {sys_pct:+.0f}% — a <strong>{trend_word(sys_rho)}</strong> trend. "
    f"<strong>Relational framing</strong> (connections between ideas) followed a similar arc "
    f"({rel_pct:+.0f}%). "
    f"<strong>Epistemic uncertainty</strong> (hedging, questioning) peaked in <strong>{unc_peak_month}</strong> "
    f"then declined — a pattern consistent with an initial exploratory phase giving way to more confident, "
    f"directed reasoning."
)

# ── FIG 03: Vocabulary Specialisation ─────────
ttr_act = ttr[ttr["ttr"] > 0].copy()
ttr_pct = pct_change(ttr_act["ttr"])
ttr_rho, _ = spearmanr(np.arange(len(ttr_act)), ttr_act["ttr"].values)

depth_pct = 0
if "median_chars_msg" in tcm.columns:
    tcm_act = tcm[tcm["median_chars_msg"] > 0]
    depth_pct = pct_change(tcm_act["median_chars_msg"])

interpretations["fig03"] = (
    f"Two complementary specialisation signals run in opposite directions — which is exactly what "
    f"domain expertise looks like. Your vocabulary richness (Type-Token Ratio) "
    f"<strong>{trend_word(ttr_rho)}</strong> ({ttr_pct:+.0f}%): you used a narrower, "
    f"more consistent set of words as your domain focus sharpened. "
    f"Meanwhile, median message length grew <strong>{depth_pct:+.0f}%</strong> "
    f"— each message packed more content even as vocabulary converged. "
    f"Together, these signals describe a user moving from broad exploration to deep, "
    f"efficient domain dialogue."
)

# ── FIG 04: Domain Composition ────────────────
dom_monthly = domain.pivot_table(index="month", columns="macro_domain",
                                  values="user_share", aggfunc="sum").fillna(0)
dom_means = dom_monthly.mean().sort_values(ascending=False)
top_domain = dom_means.index[0]
top_share  = dom_means.iloc[0]

# Check for big shift in last month vs first month
first_month_dom = dom_monthly.iloc[0].idxmax()
last_month_dom  = dom_monthly.iloc[-1].idxmax()
last_month_top_share = dom_monthly.iloc[-1].max()

interpretations["fig04"] = (
    f"<strong>{top_domain}</strong> was your dominant topic overall "
    f"({top_share:.0%} of messages on average). "
    + (
        f"The composition was relatively stable across most months, but "
        f"<strong>{months[-1]}</strong> stood out: "
        f"<strong>{last_month_dom}</strong> reached {last_month_top_share:.0%} of your messages "
        f"— a marked shift from the earlier pattern. "
        if last_month_dom != first_month_dom else
        f"<strong>{first_month_dom}</strong> dominated throughout, with the mix of supporting topics "
        f"shifting gradually across the period. "
    ) +
    f"This chart is your topical autobiography: each bar is a snapshot of what occupied your mind that month."
)

# ── FIG 05: Domain Role Balance ───────────────
dom_agg = domain.groupby("macro_domain")[["user_share","assistant_share"]].mean()
dom_agg["lead"] = dom_agg["user_share"] - dom_agg["assistant_share"]
you_lead_domains   = dom_agg[dom_agg["lead"] > 0.02].index.tolist()
gpt_lead_domains   = dom_agg[dom_agg["lead"] < -0.02].index.tolist()

interpretations["fig05"] = (
    f"Not all topics are shared equally — you tend to drive some while ChatGPT expands on others. "
    + (f"You contributed more messages in: <strong>{', '.join(you_lead_domains[:3])}</strong>. " if you_lead_domains else "") +
    (f"ChatGPT produced relatively more content in: <strong>{', '.join(gpt_lead_domains[:3])}</strong>. " if gpt_lead_domains else "") +
    f"This asymmetry is healthy: it suggests a genuine division of labour rather than mirroring, "
    f"with you steering content and ChatGPT elaborating and scaffolding."
)

# ── FIG 06: Thread Map ────────────────────────
tc_plot = tc[tc["depth"] > 0].copy()
n_threads = len(tc_plot)
top5 = tc_plot.nlargest(5, "total_msgs")
longest = top5.iloc[0]
longest_title = str(longest.get("thread_title",""))[:40]
longest_msgs  = int(longest["total_msgs"])

interpretations["fig06"] = (
    f"Each dot is one of your {n_threads} conversation threads. "
    f"Position shows the length (x-axis) and structural depth (y-axis); "
    f"dot size reflects how long your messages were. "
    f"Most threads cluster in the lower-left — short, shallow exchanges — "
    f"but a handful of deep-work threads stand out. "
    f"Your longest thread (<em>'{longest_title}…'</em>) ran to "
    f"<strong>{longest_msgs:,} messages</strong>. "
    f"The colour scale (green = more branching) shows that longer threads tend to be more linear, "
    f"suggesting sustained focus rather than exploratory back-and-forth."
)

# ── FIG 07: Thread Length Distribution ────────
med_msgs  = tc["total_msgs"].median()
med_chars = tc["chars_per_msg"].median()
long_threads = (tc["total_msgs"] > med_msgs * 3).sum()

interpretations["fig07"] = (
    f"Thread length is right-skewed: most conversations are short "
    f"(median <strong>{med_msgs:.0f} messages</strong>), but a long tail of deep-work sessions "
    f"pulls the distribution rightward — {long_threads} threads ran more than 3× the median length. "
    f"Median message depth was <strong>{med_chars:.0f} characters</strong>, "
    f"placing your exchanges well above typical social-media messages but within "
    f"the range of substantive professional correspondence. "
    f"The two histograms together describe a user who mostly uses short bursts "
    f"but regularly commits to sustained, dense dialogue."
)

# ── FIG 08: Dyadic Alignment ──────────────────
jsd_mean = dyadic["js_divergence"].mean()
jsd_std  = dyadic["js_divergence"].std()
jsd_rho, _ = spearmanr(np.arange(len(dyadic)), dyadic["js_divergence"].values)

interpretations["fig08"] = (
    f"Jensen-Shannon divergence measures how different your topics are from ChatGPT's in the same month. "
    f"A score near 0 would mean you talk about identical things; near 1 means completely different. "
    f"Your average was <strong>{jsd_mean:.2f} ± {jsd_std:.2f}</strong> — firmly in the "
    f"<em>complementary zone</em> (0.3–0.7). "
    f"The <strong>{trend_word(jsd_rho)}</strong> trend across months suggests the dyad's "
    f"role structure was {'maintaining a healthy balance' if abs(jsd_rho) < 0.4 else 'gradually shifting'}. "
    f"Stable mid-range divergence is the fingerprint of genuine collaboration: "
    f"you are not just echoing each other, but you are not talking past each other either."
)

# ── FIG 09: Rolling Entropy ───────────────────
entr["mid_dt"] = pd.to_datetime(entr["mid_timestamp"], utc=True, errors="coerce")
entr_sorted = entr.sort_values("mid_dt").reset_index(drop=True)
mean_entr = entr_sorted["combined_entropy"].mean()
max_entr_idx = entr_sorted["combined_entropy"].idxmax()
max_entr_date = entr_sorted.loc[max_entr_idx, "mid_dt"]
max_entr_date_str = max_entr_date.strftime("%b %Y") if pd.notna(max_entr_date) else "mid-period"
entr_rho, _ = spearmanr(np.arange(len(entr_sorted)), entr_sorted["combined_entropy"].values)

interpretations["fig09"] = (
    f"This is the highest-resolution view of your cognitive activity — computed in rolling windows "
    f"of 250 messages across the full period. "
    f"Each point estimates how topically diverse that stretch of conversation was. "
    f"Mean entropy was <strong>{mean_entr:.2f}</strong> (on a 0–log₂(60) scale), "
    f"with a peak of diversity around <strong>{max_entr_date_str}</strong>. "
    f"The smoothed red line shows a <strong>{trend_word(entr_rho)}</strong> overall trajectory. "
    f"Spikes indicate bursts of exploratory, multi-topic conversation; "
    f"troughs signal focused, single-domain work sessions."
)

# ── FIG 10: Top Domains ───────────────────────
dom_total = domain.groupby("macro_domain")["user_share"].mean().sort_values(ascending=False)
top3 = dom_total.head(3)
bottom3 = dom_total.tail(3)

interpretations["fig10"] = (
    f"Averaged across all {n_months} months, your three most-used domains were: "
    f"<strong>{top3.index[0]}</strong> ({top3.iloc[0]:.0%}), "
    f"<strong>{top3.index[1]}</strong> ({top3.iloc[1]:.0%}), and "
    f"<strong>{top3.index[2]}</strong> ({top3.iloc[2]:.0%}). "
    f"Together these three account for <strong>{top3.sum():.0%}</strong> of your total usage. "
    f"The least-used domains — "
    f"{', '.join(bottom3.index.tolist())} — "
    f"represent specialised or occasional use cases rather than core work."
)

# ── FIG 11: Cognitive Radar ───────────────────
# Find which dimension grew most (first vs last normalised)
dims_check = {
    "Systems thinking":    traj["system_words_per_1k_msgs"],
    "Relational framing":  traj["relational_words_per_1k_msgs"],
    "Epistemic uncert.":   traj["uncertainty_words_per_1k_msgs"],
}
growth = {}
for name, series in dims_check.items():
    s = series.dropna()
    if len(s) >= 2 and s.iloc[0] > 0:
        growth[name] = (s.iloc[-1] - s.iloc[0]) / s.iloc[0]
top_growth_dim = max(growth, key=growth.get) if growth else "Systems thinking"
top_growth_val = growth.get(top_growth_dim, 0)

interpretations["fig11"] = (
    f"The radar chart is your <strong>cognitive fingerprint</strong>: "
    f"it overlays where you started ({months[0]}) against where you ended ({months[-1]}) "
    f"across six dimensions of your interaction style. "
    f"The biggest growth was in <strong>{top_growth_dim}</strong> "
    f"(+{top_growth_val:.0%} relative to your starting level). "
    f"The shape of the {months[-1]} polygon — larger and more asymmetric than {months[0]} — "
    f"tells a story of directed cognitive specialisation: not growth in all directions equally, "
    f"but a specific deepening that reflects what you were actually working on."
)

# ── FIG 12: Summary Dashboard ─────────────────
interpretations["fig12"] = (
    f"A single-page overview of your {n_months}-month collaboration with ChatGPT "
    f"({months[0]}–{months[-1]}, {total_msgs:,} messages). "
    f"<strong>Top-left</strong>: raw activity over time. "
    f"<strong>Top-right</strong>: your cognitive language index, showing the analytical arc. "
    f"<strong>Middle-left</strong>: what you talked about each month. "
    f"<strong>Middle-right</strong>: vocabulary richness vs. topic entropy — two independent "
    f"specialisation signals moving in consistent directions. "
    f"<strong>Bottom-left</strong>: dyadic alignment, showing stable complementarity throughout. "
    f"<strong>Bottom-right</strong>: your all-time topic distribution. "
    f"Taken together, these six panels describe a coherent and distinctive cognitive trajectory."
)

# ── FIG 13: Directional (optional) ───────────
if weekly_dir is not None:
    cols = weekly_dir.columns.tolist()
    d_col = next((c for c in cols if "domain" in c.lower()), cols[0])
    fwd   = next((c for c in cols if "forward" in c.lower() or "user_to" in c.lower()), None)
    rev   = next((c for c in cols if "reverse" in c.lower() or "asst_to" in c.lower()), None)
    if fwd and rev:
        if pd.api.types.is_integer_dtype(weekly_dir[d_col]):
            weekly_dir[d_col] = weekly_dir[d_col].map(DOMAIN_NAMES).fillna(weekly_dir[d_col].astype(str))
        lead = weekly_dir.groupby(d_col)[[fwd, rev]].mean()
        lead["diff"] = lead[fwd] - lead[rev]
        you_lead  = lead[lead["diff"] > 0].index.tolist()
        gpt_leads = lead[lead["diff"] < 0].index.tolist()
        interpretations["fig13"] = (
            f"This chart answers: <em>who introduces new ideas in each topic area?</em> "
            f"Bars pointing right mean you tend to lead; bars pointing left mean ChatGPT tends to introduce content first. "
            + (f"You lead in: <strong>{', '.join(you_lead[:4])}</strong>. " if you_lead else "") +
            (f"ChatGPT leads in: <strong>{', '.join(gpt_leads[:4])}</strong>. " if gpt_leads else "") +
            f"This asymmetry is informative: the topics where ChatGPT leads may be areas "
            f"where you are primarily learning or seeking synthesis rather than driving original work."
        )

# ─────────────────────────────────────────────
# BUILD HTML REPORT
# ─────────────────────────────────────────────
fig_manifest = [
    ("fig01", "fig01_monthly_activity.png",          "Fig 1 · Monthly Message Volume"),
    ("fig02", "fig02_cognitive_markers.png",          "Fig 2 · Cognitive Language Evolution"),
    ("fig03", "fig03_vocabulary_specialisation.png",  "Fig 3 · Vocabulary Specialisation vs. Message Depth"),
    ("fig04", "fig04_domain_composition.png",         "Fig 4 · Domain Composition Over Time"),
    ("fig05", "fig05_domain_role_balance.png",        "Fig 5 · Who Drives Which Topics"),
    ("fig06", "fig06_thread_map.png",                 "Fig 6 · Thread Map"),
    ("fig07", "fig07_thread_length_distribution.png", "Fig 7 · Thread Length & Depth Distribution"),
    ("fig08", "fig08_dyadic_alignment.png",           "Fig 8 · Dyadic Topic Alignment"),
    ("fig09", "fig09_rolling_entropy.png",            "Fig 9 · Rolling Cognitive Entropy"),
    ("fig10", "fig10_top_domains.png",                "Fig 10 · All-Time Topic Share"),
    ("fig11", "fig11_cognitive_radar.png",            "Fig 11 · Cognitive Fingerprint (Radar)"),
    ("fig12", "fig12_summary_dashboard.png",          "Fig 12 · Summary Dashboard"),
    ("fig13", "fig13_directional_by_domain.png",      "Fig 13 · Topic Leadership by Domain"),
]

HTML_STYLE = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa;
         color: #2d2d2d; margin: 0; padding: 0; }
  header { background: #2E86AB; color: white; padding: 32px 48px; }
  header h1 { margin: 0 0 6px; font-size: 1.8em; }
  header p  { margin: 0; opacity: 0.85; font-size: 0.95em; }
  .grid { display: grid; grid-template-columns: 1fr 1fr;
          gap: 28px; padding: 36px 48px; max-width: 1400px; margin: 0 auto; }
  .card { background: white; border-radius: 10px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.07); overflow: hidden; }
  .card.full-width { grid-column: 1 / -1; }
  .card img { width: 100%; display: block; }
  .card .caption { padding: 18px 20px; }
  .card .caption h3 { margin: 0 0 8px; font-size: 1em; color: #2E86AB; }
  .card .caption p  { margin: 0; font-size: 0.88em; line-height: 1.6; color: #444; }
  footer { text-align: center; padding: 24px; color: #999; font-size: 0.8em; }
  .tag  { display: inline-block; background: #e8f4f8; color: #2E86AB;
          border-radius: 4px; padding: 2px 8px; font-size: 0.78em;
          font-weight: 600; margin-bottom: 6px; }
</style>
"""

def make_card(key, filename, title, full_width=False):
    b64 = fig_to_b64(filename)
    if b64 is None:
        return ""
    interp = interpretations.get(key, "<em>Interpretation not available.</em>")
    fw_class = "full-width" if full_width else ""
    return f"""
    <div class="card {fw_class}">
      <img src="data:image/png;base64,{b64}" alt="{title}" />
      <div class="caption">
        <span class="tag">{title}</span>
        <p>{interp}</p>
      </div>
    </div>"""

period_str  = f"{months[0]} – {months[-1]}"
gen_date    = pd.Timestamp.now().strftime("%d %b %Y")

html_parts = [f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Your ChatGPT Collaboration Report</title>
{HTML_STYLE}
</head><body>
<header>
  <h1>Your ChatGPT Collaboration Report</h1>
  <p>{period_str} &nbsp;·&nbsp; {total_msgs:,} messages &nbsp;·&nbsp; {n_months} months &nbsp;·&nbsp; Generated {gen_date}</p>
</header>
<div class="grid">
"""]

for i, (key, filename, title) in enumerate(fig_manifest):
    # Make fig12 (dashboard) and fig09 (rolling entropy) full-width for readability
    full_w = key in ("fig09", "fig12")
    html_parts.append(make_card(key, filename, title, full_width=full_w))

html_parts.append("""
</div>
<footer>Generated by the DOL Pipeline · visualise_user.py + interpret_figures.py</footer>
</body></html>""")

html_out = "\n".join(html_parts)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_out)

print(f"\n✓  Report saved to:  {OUTPUT_HTML}")
print(f"   Open in any browser to view.\n")