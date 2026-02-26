"""
visualise_user.py
=================
DOL Pipeline – User-Facing Visualisations
Generates 12 interpretable figures from your ChatGPT export analysis.

Usage
-----
    python visualise_user.py

Inputs  (edit DATA_DIR to match your folder)
------
    trajectory_monthly.csv
    ttr_monthly.csv
    macro_monthly_domain_shares.csv
    dyadic_alignment_monthly.csv
    threads_master.csv
    thread_complexity.csv
    thread_complexity_monthly.csv
    rolling_entropy_250.csv

Outputs (saved to OUTPUT_DIR)
-------
    fig01_monthly_activity.png
    fig02_cognitive_markers.png
    fig03_vocabulary_specialisation.png
    fig04_domain_composition.png
    fig05_domain_role_balance.png
    fig06_thread_map.png
    fig07_thread_length_distribution.png
    fig08_dyadic_alignment.png
    fig09_rolling_entropy.png
    fig10_top_domains.png
    fig11_cognitive_radar.png
    fig12_summary_dashboard.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as cm

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION  ← edit DATA_DIR to where your CSVs live
# ─────────────────────────────────────────────
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))   # default: same folder as this script
OUTPUT_DIR = os.path.join(DATA_DIR, "user_figures")       # where PNGs are saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
BLUE   = "#2E86AB"   # user / human
RED    = "#E84855"   # assistant / AI
GOLD   = "#F6AE2D"   # uncertainty / accent
GREEN  = "#6BBA75"   # relational
PURPLE = "#7B5EA7"   # systems thinking
GREY   = "#8D99AE"   # neutral
BG     = "#FAFAFA"

DOMAIN_COLOURS = [
    "#2E86AB","#E84855","#F6AE2D","#6BBA75",
    "#7B5EA7","#F26419","#44BBA4","#E94F37",
]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#E0E0E0",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
})

def savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved → {name}")

def short_month(m):
    """'2025-07' → 'Jul'"""
    try:
        return pd.to_datetime(m + "-01").strftime("%b")
    except Exception:
        return str(m)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load(filename):
    """Search script folder, uploads/, outputs/, and parent folder for filename."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(DATA_DIR, filename),
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "..", "uploads", filename),
        os.path.join(script_dir, "outputs", filename),
        os.path.join(script_dir, "..", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Cannot find {filename}.\n"
        f"Searched: {[os.path.normpath(c) for c in candidates]}\n"
        f"Set DATA_DIR at the top of this script to the folder containing your CSVs."
    )

print("Loading data …")
traj   = load("trajectory_monthly.csv")
ttr    = load("ttr_monthly.csv")
domain = load("macro_monthly_domain_shares.csv")
dyadic = load("dyadic_alignment_monthly.csv")
thread = load("threads_master.csv")
tc     = load("thread_complexity.csv")
tcm    = load("thread_complexity_monthly.csv")
entr   = load("rolling_entropy_250.csv")

# Optional bonus files — script skips gracefully if not present
def try_load(filename):
    try:
        return load(filename)
    except FileNotFoundError:
        return None

weekly_dir   = try_load("weekly_directional_by_domain.csv")
episode_summ = try_load("step10b2_episode_summary.csv")
shift_summ   = try_load("step10b_shift_initiation_summary.csv")
print("  all files loaded.\n")

# Clean month labels
for df in [traj, ttr, dyadic, tcm]:
    if "month" in df.columns:
        df["month_label"] = df["month"].apply(short_month)

# Active months only — keep months with substantial activity (≥100 messages)
# This drops any stray/sparse months outside the main analysis window
traj = traj[traj["user_messages"] >= 100].copy()

# Map integer domain labels 0-7 → readable names
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
# Update DOMAIN_NAMES above if your pipeline produced different domain groupings.
if domain["macro_domain"].dtype in [int, "int64"] or pd.api.types.is_integer_dtype(domain["macro_domain"]):
    domain["macro_domain"] = domain["macro_domain"].map(DOMAIN_NAMES).fillna(domain["macro_domain"].astype(str))

# Restrict domain shares to active months only
active_months = set(traj["month"].values)
domain = domain[domain["month"].isin(active_months)].copy()

# ─────────────────────────────────────────────
# FIG 01 · Monthly Activity
# ─────────────────────────────────────────────
print("Generating fig01 – Monthly Activity …")
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(traj))
bars = ax.bar(x, traj["user_messages"], color=BLUE, alpha=0.85, zorder=3, width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(traj["month_label"])
ax.set_ylabel("User messages")
ax.set_title("Your Monthly Message Volume")

# annotate bars
for bar, val in zip(bars, traj["user_messages"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            str(int(val)), ha="center", va="bottom", fontsize=8, color="#333333")

# trend line
z = np.polyfit(x, traj["user_messages"], 1)
p = np.poly1d(z)
ax.plot(x, p(x), color=RED, linewidth=1.5, linestyle="--", alpha=0.7, label="trend")
ax.legend(frameon=False)
ax.set_xlabel("Month")
fig.tight_layout()
savefig(fig, "fig01_monthly_activity.png")

# ─────────────────────────────────────────────
# FIG 02 · Cognitive Markers Trajectory
# ─────────────────────────────────────────────
print("Generating fig02 – Cognitive Markers …")

# Normalise to May = 100
markers = ["system_words_per_1k_msgs", "relational_words_per_1k_msgs", "uncertainty_words_per_1k_msgs"]
labels  = ["Systems thinking", "Relational framing", "Epistemic uncertainty"]
colours = [PURPLE, GREEN, GOLD]

traj_act = traj.reset_index(drop=True)
base = traj_act.iloc[0]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(traj_act))

for col, label, colour in zip(markers, labels, colours):
    norm = traj_act[col] / base[col] * 100
    ax.plot(x, norm, marker="o", markersize=6, linewidth=2, color=colour, label=label)
    ax.fill_between(x, 100, norm, alpha=0.07, color=colour)

ax.axhline(100, color=GREY, linewidth=0.8, linestyle=":", label="Baseline (May = 100)")
ax.set_xticks(x)
ax.set_xticklabels(traj_act["month_label"])
ax.set_ylabel("Index (May = 100)")
ax.set_title("How Your Cognitive Language Has Evolved")
ax.legend(frameon=False, loc="upper left")
ax.set_xlabel("Month")
fig.tight_layout()
savefig(fig, "fig02_cognitive_markers.png")

# ─────────────────────────────────────────────
# FIG 03 · Vocabulary Specialisation
# ─────────────────────────────────────────────
print("Generating fig03 – Vocabulary Specialisation …")

ttr_act = ttr[ttr["ttr"] > 0].copy()
ttr_act["month_label"] = ttr_act["month"].apply(short_month)

fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()

x = np.arange(len(ttr_act))
ax1.plot(x, ttr_act["ttr"], marker="o", color=RED, linewidth=2, markersize=6, label="Vocabulary richness (TTR)")
ax1.set_ylabel("Type-Token Ratio", color=RED)
ax1.tick_params(axis="y", labelcolor=RED)
ax1.set_ylim(0, ttr_act["ttr"].max() * 1.2)

# chars/msg from thread_complexity_monthly if available
if "median_chars_msg" in tcm.columns:
    tcm_act = tcm[tcm["median_chars_msg"] > 0].copy()
    tcm_act["month_label"] = tcm_act["month"].apply(short_month)
    # align on matching months
    merged = ttr_act.merge(tcm_act[["month","median_chars_msg"]], on="month", how="left")
    if merged["median_chars_msg"].notna().any():
        ax2.plot(np.arange(len(merged)), merged["median_chars_msg"],
                 marker="s", color=BLUE, linewidth=2, markersize=6, linestyle="--",
                 label="Median chars / message")
        ax2.set_ylabel("Median chars per message", color=BLUE)
        ax2.tick_params(axis="y", labelcolor=BLUE)

ax1.set_xticks(x)
ax1.set_xticklabels(ttr_act["month_label"])
ax1.set_xlabel("Month")
ax1.set_title("Vocabulary Specialisation vs. Message Depth")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower left")
ax1.spines["right"].set_visible(False)
fig.tight_layout()
savefig(fig, "fig03_vocabulary_specialisation.png")

# ─────────────────────────────────────────────
# FIG 04 · Domain Composition Over Time
# ─────────────────────────────────────────────
print("Generating fig04 – Domain Composition …")

# Use user_share; pivot to wide
dom_user = domain.pivot_table(index="month", columns="macro_domain",
                               values="user_share", aggfunc="sum").fillna(0)
dom_user = dom_user.loc[traj["month"].values] if all(m in dom_user.index for m in traj["month"].values) else dom_user
dom_labels = [short_month(m) for m in dom_user.index]
domains    = dom_user.columns.tolist()

fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(dom_user))
for i, d in enumerate(domains):
    colour = DOMAIN_COLOURS[i % len(DOMAIN_COLOURS)]
    vals = dom_user[d].values
    ax.bar(np.arange(len(dom_user)), vals, bottom=bottom, color=colour,
           label=d, alpha=0.88, width=0.7)
    bottom += vals

ax.set_xticks(np.arange(len(dom_user)))
ax.set_xticklabels(dom_labels)
ax.set_ylabel("Share of user messages")
ax.set_title("What You Talked About Each Month")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
ax.set_xlabel("Month")
fig.tight_layout()
savefig(fig, "fig04_domain_composition.png")

# ─────────────────────────────────────────────
# FIG 05 · Domain Role Balance
# ─────────────────────────────────────────────
print("Generating fig05 – Domain Role Balance …")

dom_agg = domain.groupby("macro_domain")[["user_share","assistant_share"]].mean().sort_values("user_share", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(dom_agg))
bar_w = 0.38
ax.barh(y - bar_w/2, dom_agg["user_share"], height=bar_w, color=BLUE,  alpha=0.85, label="You")
ax.barh(y + bar_w/2, dom_agg["assistant_share"], height=bar_w, color=RED, alpha=0.85, label="ChatGPT")
ax.set_yticks(y)
ax.set_yticklabels(dom_agg.index, fontsize=9)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("Average share of messages")
ax.set_title("Who Drives Which Topics")
ax.legend(frameon=False)
fig.tight_layout()
savefig(fig, "fig05_domain_role_balance.png")

# ─────────────────────────────────────────────
# FIG 06 · Thread Map
# ─────────────────────────────────────────────
print("Generating fig06 – Thread Map …")

tc_plot = tc[tc["depth"] > 0].copy()
size = np.clip(tc_plot["chars_per_msg"] / tc_plot["chars_per_msg"].max() * 300, 10, 300)

fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(tc_plot["total_msgs"], tc_plot["depth"],
                s=size, c=tc_plot["branching_ratio"],
                cmap="RdYlGn", alpha=0.75, edgecolors="white", linewidths=0.5)
cb = plt.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Branching ratio", fontsize=9)
ax.set_xlabel("Total messages in thread")
ax.set_ylabel("Conversation depth")
ax.set_title("Thread Map  (size = chars/msg)")

# label the 5 longest
top5 = tc_plot.nlargest(5, "total_msgs")
for _, row in top5.iterrows():
    label = str(row.get("thread_title",""))[:22] + "…" if len(str(row.get("thread_title",""))) > 22 else str(row.get("thread_title",""))
    ax.annotate(label, (row["total_msgs"], row["depth"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 4), textcoords="offset points", color="#333")
fig.tight_layout()
savefig(fig, "fig06_thread_map.png")

# ─────────────────────────────────────────────
# FIG 07 · Thread Length Distribution
# ─────────────────────────────────────────────
print("Generating fig07 – Thread Length Distribution …")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(tc["total_msgs"].dropna(), bins=20, color=BLUE, alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Messages per thread")
axes[0].set_ylabel("Number of threads")
axes[0].set_title("Thread Length Distribution")
med_msgs = tc["total_msgs"].median()
axes[0].axvline(med_msgs, color=RED, linestyle="--", linewidth=1.5, label=f"Median = {med_msgs:.0f}")
axes[0].legend(frameon=False)

axes[1].hist(tc["chars_per_msg"].dropna(), bins=20, color=PURPLE, alpha=0.8, edgecolor="white")
axes[1].set_xlabel("Chars per message")
axes[1].set_ylabel("Number of threads")
axes[1].set_title("Message Depth Distribution")
med_chars = tc["chars_per_msg"].median()
axes[1].axvline(med_chars, color=RED, linestyle="--", linewidth=1.5, label=f"Median = {med_chars:.0f}")
axes[1].legend(frameon=False)

fig.tight_layout()
savefig(fig, "fig07_thread_length_distribution.png")

# ─────────────────────────────────────────────
# FIG 08 · Dyadic Alignment
# ─────────────────────────────────────────────
print("Generating fig08 – Dyadic Alignment …")

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(dyadic))
ax.plot(x, dyadic["js_divergence"], marker="o", color=BLUE,
        linewidth=2, markersize=7, label="JS divergence (you vs ChatGPT)")
ax.fill_between(x,
                dyadic["js_divergence"] - 0.02,
                dyadic["js_divergence"] + 0.02,
                alpha=0.12, color=BLUE)

ax.axhline(0.5, color=GREY, linestyle=":", linewidth=1, label="Midpoint (0.5)")
ax.set_ylim(0, 1)
ax.set_xticks(x)
ax.set_xticklabels(dyadic["month_label"])
ax.set_ylabel("Jensen-Shannon divergence\n(0 = identical, 1 = completely different)")
ax.set_title("How Different Are Your Topics from ChatGPT's?")
ax.legend(frameon=False)
ax.set_xlabel("Month")

# Interpretation band
ax.axhspan(0.3, 0.7, alpha=0.04, color=GREEN, label="Complementary zone")
fig.tight_layout()
savefig(fig, "fig08_dyadic_alignment.png")

# ─────────────────────────────────────────────
# FIG 09 · Rolling Entropy
# ─────────────────────────────────────────────
print("Generating fig09 – Rolling Entropy …")

entr["mid_dt"] = pd.to_datetime(entr["mid_timestamp"], utc=True, errors="coerce")
entr_sorted = entr.sort_values("mid_dt").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(entr_sorted["mid_dt"], entr_sorted["combined_entropy"],
        color=PURPLE, linewidth=1.2, alpha=0.85, label="Combined entropy (user + ChatGPT)")
ax.plot(entr_sorted["mid_dt"], entr_sorted["user_entropy"],
        color=BLUE, linewidth=1.2, alpha=0.7, linestyle="--", label="Your entropy only")

# Rolling smoothed line
roll = entr_sorted["combined_entropy"].rolling(5, center=True).mean()
ax.plot(entr_sorted["mid_dt"], roll,
        color=RED, linewidth=2.2, alpha=0.9, label="Smoothed trend")

ax.set_ylabel("Topic entropy (higher = more diverse)")
ax.set_title("Cognitive Texture Over Time  (rolling 250-message windows)")
ax.legend(frameon=False, loc="lower right")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
fig.autofmt_xdate()
fig.tight_layout()
savefig(fig, "fig09_rolling_entropy.png")

# ─────────────────────────────────────────────
# FIG 10 · Top Domains (All Time)
# ─────────────────────────────────────────────
print("Generating fig10 – Top Domains …")

dom_total = domain.groupby("macro_domain")["user_share"].mean().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colours_sorted = [DOMAIN_COLOURS[i % len(DOMAIN_COLOURS)] for i in range(len(dom_total))]
bars = ax.barh(dom_total.index, dom_total.values, color=colours_sorted, alpha=0.85)

for bar, val in zip(bars, dom_total.values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va="center", fontsize=9)

ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("Average share of your messages")
ax.set_title("Your Top Topics  (all time average)")
ax.set_xlim(0, dom_total.max() * 1.2)
fig.tight_layout()
savefig(fig, "fig10_top_domains.png")

# ─────────────────────────────────────────────
# FIG 11 · Cognitive Radar (Fingerprint)
# ─────────────────────────────────────────────
print("Generating fig11 – Cognitive Radar …")

# Normalise each metric 0-1 across the period
def norm01(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn) if mx > mn else series * 0 + 0.5

first = traj_act.iloc[0]
last  = traj_act.iloc[-1]

dims = {
    "Systems\nthinking":   ("system_words_per_1k_msgs",      traj_act),
    "Relational\nframing": ("relational_words_per_1k_msgs",   traj_act),
    "Epistemic\nuncert.":  ("uncertainty_words_per_1k_msgs",  traj_act),
    "Vocab\nrichness":     ("ttr",                            ttr_act),
    "Message\ndepth":      ("median_chars_msg",               tcm),
    "Thread\ndepth":       ("median_depth",                   tcm),
}

angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
angles += angles[:1]

def get_norm_val(df, col, row_idx):
    if col not in df.columns: return 0.5
    series = df[col].dropna()
    if len(series) < 2: return 0.5
    mn, mx = series.min(), series.max()
    if mx == mn: return 0.5
    val = df[col].iloc[row_idx] if row_idx < len(df) else df[col].iloc[-1]
    return float((val - mn) / (mx - mn))

vals_first = [get_norm_val(df, col, 0)  for col, df in dims.values()]
vals_last  = [get_norm_val(df, col, -1) for col, df in dims.values()]
vals_first += vals_first[:1]
vals_last  += vals_last[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_facecolor(BG)

ax.plot(angles, vals_first, color=GREY,  linewidth=1.5, linestyle="--", label=traj_act["month_label"].iloc[0])
ax.fill(angles, vals_first, alpha=0.08, color=GREY)
ax.plot(angles, vals_last,  color=BLUE,  linewidth=2,   label=traj_act["month_label"].iloc[-1])
ax.fill(angles, vals_last,  alpha=0.15, color=BLUE)

ax.set_thetagrids(np.degrees(angles[:-1]), list(dims.keys()), fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75])
ax.set_yticklabels(["low", "mid", "high"], fontsize=7, color=GREY)
ax.set_title("Your Cognitive Fingerprint\n(first month vs last)", pad=20)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1.1))
fig.tight_layout()
savefig(fig, "fig11_cognitive_radar.png")

# ─────────────────────────────────────────────
# FIG 12 · Summary Dashboard
# ─────────────────────────────────────────────
print("Generating fig12 – Summary Dashboard …")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(BG)
gs = GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

# --- Panel A: monthly activity (col 0-1, row 0)
ax_a = fig.add_subplot(gs[0, :2])
x = np.arange(len(traj_act))
ax_a.bar(x, traj_act["user_messages"], color=BLUE, alpha=0.8, width=0.6)
ax_a.set_xticks(x); ax_a.set_xticklabels(traj_act["month_label"], fontsize=8)
ax_a.set_title("Monthly Activity", fontsize=11, fontweight="bold")
ax_a.set_ylabel("Messages", fontsize=9)
ax_a.tick_params(labelsize=8)

# --- Panel B: cognitive markers (col 2-3, row 0)
ax_b = fig.add_subplot(gs[0, 2:])
for col, label, colour in zip(markers, labels, colours):
    norm = traj_act[col] / traj_act[col].iloc[0] * 100
    ax_b.plot(x, norm, marker="o", markersize=4, linewidth=1.8, color=colour, label=label)
ax_b.axhline(100, color=GREY, linewidth=0.7, linestyle=":")
ax_b.set_xticks(x); ax_b.set_xticklabels(traj_act["month_label"], fontsize=8)
ax_b.set_title("Cognitive Language Index", fontsize=11, fontweight="bold")
ax_b.legend(frameon=False, fontsize=7, loc="upper left")
ax_b.tick_params(labelsize=8)

# --- Panel C: domain composition (col 0-1, row 1)
ax_c = fig.add_subplot(gs[1, :2])
bottom = np.zeros(len(dom_user))
for i, d in enumerate(domains):
    ax_c.bar(np.arange(len(dom_user)), dom_user[d].values,
             bottom=bottom, color=DOMAIN_COLOURS[i % len(DOMAIN_COLOURS)],
             alpha=0.88, width=0.7, label=d)
    bottom += dom_user[d].values
ax_c.set_xticks(np.arange(len(dom_user)))
ax_c.set_xticklabels([short_month(m) for m in dom_user.index], fontsize=8)
ax_c.set_title("Domain Mix", fontsize=11, fontweight="bold")
ax_c.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax_c.legend(frameon=False, fontsize=6, loc="upper left", ncol=2)
ax_c.tick_params(labelsize=8)

# --- Panel D: TTR + entropy mini (col 2-3, row 1)
ax_d = fig.add_subplot(gs[1, 2:])
x_ttr = np.arange(len(ttr_act))
ax_d.plot(x_ttr, ttr_act["ttr"], color=RED, marker="o", markersize=4, linewidth=1.8, label="TTR")
ax_d.set_ylabel("TTR", color=RED, fontsize=9)
ax_d.tick_params(axis="y", labelcolor=RED, labelsize=8)
ax_d2 = ax_d.twinx()
roll_small = entr_sorted["combined_entropy"].rolling(8, center=True).mean()
ax_d2.plot(np.linspace(0, len(ttr_act)-1, len(roll_small)),
           roll_small, color=PURPLE, linewidth=1.5, linestyle="--", alpha=0.7, label="Entropy")
ax_d2.set_ylabel("Entropy", color=PURPLE, fontsize=9)
ax_d2.tick_params(axis="y", labelcolor=PURPLE, labelsize=8)
ax_d.set_xticks(x_ttr); ax_d.set_xticklabels(ttr_act["month_label"], fontsize=8)
ax_d.set_title("Vocabulary & Entropy", fontsize=11, fontweight="bold")
ax_d.spines["right"].set_visible(False)

# --- Panel E: dyadic alignment (col 0-1, row 2)
ax_e = fig.add_subplot(gs[2, :2])
ax_e.plot(np.arange(len(dyadic)), dyadic["js_divergence"],
          marker="o", color=BLUE, linewidth=1.8, markersize=5)
ax_e.axhline(0.5, color=GREY, linestyle=":", linewidth=0.8)
ax_e.set_ylim(0, 1)
ax_e.set_xticks(np.arange(len(dyadic)))
ax_e.set_xticklabels(dyadic["month_label"], fontsize=8)
ax_e.set_title("Dyadic Alignment (JS divergence)", fontsize=11, fontweight="bold")
ax_e.set_ylabel("Divergence", fontsize=9)
ax_e.tick_params(labelsize=8)

# --- Panel F: top domains bar (col 2-3, row 2)
ax_f = fig.add_subplot(gs[2, 2:])
ax_f.barh(dom_total.index, dom_total.values,
          color=[DOMAIN_COLOURS[i % len(DOMAIN_COLOURS)] for i in range(len(dom_total))],
          alpha=0.85)
ax_f.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax_f.set_title("All-Time Topic Share", fontsize=11, fontweight="bold")
ax_f.tick_params(labelsize=8)

fig.suptitle("Your ChatGPT Collaboration — Cognitive & Behavioural Overview",
             fontsize=14, fontweight="bold", y=1.01)
savefig(fig, "fig12_summary_dashboard.png")

# ─────────────────────────────────────────────
# FIG 13 · Weekly Directional Coupling by Domain  (optional)
# ─────────────────────────────────────────────
if weekly_dir is not None and not weekly_dir.empty:
    print("Generating fig13 – Weekly Directional Coupling by Domain …")
    try:
        # Detect column names flexibly
        cols = weekly_dir.columns.tolist()
        # Expect columns like: domain, week/period, forward_cos, reverse_cos (or similar)
        domain_col = next((c for c in cols if "domain" in c.lower()), cols[0])
        # Map domain integers if needed
        if pd.api.types.is_integer_dtype(weekly_dir[domain_col]):
            weekly_dir[domain_col] = weekly_dir[domain_col].map(DOMAIN_NAMES).fillna(weekly_dir[domain_col].astype(str))

        fwd_col = next((c for c in cols if "forward" in c.lower() or "user_to" in c.lower()), None)
        rev_col = next((c for c in cols if "reverse" in c.lower() or "asst_to" in c.lower()), None)

        if fwd_col and rev_col:
            dom_lead = weekly_dir.groupby(domain_col)[[fwd_col, rev_col]].mean().copy()
            dom_lead["lead"] = dom_lead[fwd_col] - dom_lead[rev_col]
            dom_lead = dom_lead.sort_values("lead")

            fig, ax = plt.subplots(figsize=(9, 5))
            colours_lead = [BLUE if v >= 0 else RED for v in dom_lead["lead"]]
            ax.barh(dom_lead.index, dom_lead["lead"], color=colours_lead, alpha=0.85)
            ax.axvline(0, color=GREY, linewidth=1)
            ax.set_xlabel("Lead difference (positive = you drive, negative = ChatGPT drives)")
            ax.set_title("Who Leads Each Topic?")
            blue_patch  = mpatches.Patch(color=BLUE, label="You lead")
            red_patch   = mpatches.Patch(color=RED,  label="ChatGPT leads")
            ax.legend(handles=[blue_patch, red_patch], frameon=False)
            fig.tight_layout()
            savefig(fig, "fig13_directional_by_domain.png")
        else:
            print(f"  skipped fig13 – could not identify forward/reverse columns in {cols}")
    except Exception as e:
        print(f"  skipped fig13 – {e}")
else:
    print("  fig13 skipped (weekly_directional_by_domain.csv not found)")

# ─────────────────────────────────────────────
# FIG 14 · Episode Length Distribution  (optional)
# ─────────────────────────────────────────────
if episode_summ is not None and not episode_summ.empty:
    print("Generating fig14 – Episode Summary …")
    try:
        cols = episode_summ.columns.tolist()
        len_col  = next((c for c in cols if "length" in c.lower() or "msgs" in c.lower() or "size" in c.lower()), None)
        dom_col  = next((c for c in cols if "domain" in c.lower() or "macro" in c.lower()), None)

        if len_col:
            fig, axes = plt.subplots(1, 2 if dom_col else 1, figsize=(11 if dom_col else 7, 4))
            if not dom_col:
                axes = [axes]

            axes[0].hist(episode_summ[len_col].dropna(), bins=25, color=PURPLE, alpha=0.8, edgecolor="white")
            axes[0].set_xlabel("Episode length (messages)")
            axes[0].set_ylabel("Count")
            axes[0].set_title("How Long Are Your Topic Episodes?")
            med = episode_summ[len_col].median()
            axes[0].axvline(med, color=RED, linestyle="--", linewidth=1.5, label=f"Median = {med:.0f}")
            axes[0].legend(frameon=False)

            if dom_col:
                if pd.api.types.is_integer_dtype(episode_summ[dom_col]):
                    episode_summ[dom_col] = episode_summ[dom_col].map(DOMAIN_NAMES).fillna(episode_summ[dom_col].astype(str))
                ep_by_dom = episode_summ.groupby(dom_col)[len_col].median().sort_values()
                axes[1].barh(ep_by_dom.index,
                             ep_by_dom.values,
                             color=[DOMAIN_COLOURS[i % len(DOMAIN_COLOURS)] for i in range(len(ep_by_dom))],
                             alpha=0.85)
                axes[1].set_xlabel("Median episode length")
                axes[1].set_title("Episode Length by Topic")

            fig.tight_layout()
            savefig(fig, "fig14_episode_summary.png")
        else:
            print(f"  skipped fig14 – no length column found in {cols}")
    except Exception as e:
        print(f"  skipped fig14 – {e}")
else:
    print("  fig14 skipped (step10b2_episode_summary.csv not found)")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
n_figs = 14
print(f"\n✓  Figures saved to:  {OUTPUT_DIR}")
print(f"   (12 core plots always generated; bonus plots 13-14 require optional CSVs)\n")