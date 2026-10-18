"""
src/create_weighted_vs_convex_selection.py

Generates results/convex_figures/weighted_vs_convex_selection.png
Visualizes candidate population categorization, biological constraint filtering,
CVXOPT selection, and comparison with unconstrained weighted optimization.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Paths
DATA_DIR = Path("data")
OUT_FILE = Path("results/convex_figures/weighted_vs_convex_selection.png")

# Load data
spec = pd.read_csv(DATA_DIR / "gRNA_features_specificity.csv")
opt = pd.read_csv(DATA_DIR / "optimized_guides.csv")
cvx = pd.read_csv(DATA_DIR / "convex_optimized_guides.csv")

# Compute feasibility mask for all 2860 guides
gc_ok = spec["gc_content"].between(0.40, 0.60)
tm_ok = spec["melting_temp_c"].between(55.0, 65.0)
ot_ok = (spec["off_target_count"] <= 20) & (~spec["off_target_capped"].astype(bool))
feasible = gc_ok & tm_ok & ot_ok

n_total = len(spec)
n_rejected = n_total - feasible.sum()
n_feasible = feasible.sum()
n_selected = len(cvx)

# Setup Figure with 2 main subplots
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig = plt.figure(figsize=(14, 9), dpi=200)

# Create grid layout
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

# -------------------------------------------------------------------------
# Subplot 1: Pipeline Flow / Funnel Breakdown (Total -> Filtered -> Selected)
# -------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

categories = ["Total\nCandidates", "Rejected by\nConstraints", "Feasible\nCandidates", "CVXOPT\nSelected"]
values = [n_total, n_rejected, n_feasible, n_selected]
colors = ["#4A90E2", "#E74C3C", "#2ECC71", "#27AE60"]

bars = ax1.bar(categories, values, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

# Log scale for y to show huge contrast between 2860 and 5 clearly
ax1.set_yscale("log")
ax1.set_ylim(1, n_total * 3)

for bar, val in zip(bars, values):
    height = bar.get_height()
    pct = (val / n_total) * 100
    if val >= 100:
        label_text = f"{val:,}\n({pct:.1f}%)"
    else:
        label_text = f"{val}\n({pct:.2f}%)"
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height * 1.35,
        label_text,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#2C3E50",
    )

ax1.set_title("A. Guide Filtering & Selection Funnel (2860 → 5)", fontsize=12, fontweight="bold", pad=12)
ax1.set_ylabel("Number of gRNAs (Log Scale)", fontsize=10, fontweight="bold")
ax1.grid(True, which="both", linestyle="--", alpha=0.4)

# -------------------------------------------------------------------------
# Subplot 2: Categorization Donut / Pie
# -------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])

pie_sizes = [n_rejected, n_feasible]
pie_labels = [f"Rejected ({n_rejected:,})", f"Feasible & Selected ({n_feasible})"]
pie_colors = ["#E74C3C", "#2ECC71"]
explode = (0.05, 0.15)

wedges, texts, autotexts = ax2.pie(
    pie_sizes,
    explode=explode,
    labels=pie_labels,
    colors=pie_colors,
    autopct="%1.2f%%",
    pctdistance=0.75,
    startangle=140,
    textprops=dict(color="#2C3E50", fontweight="bold", fontsize=10),
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
)

for autotext in autotexts:
    autotext.set_color("black")
    autotext.set_fontsize(10)

ax2.set_title("B. Feasibility Breakdown of Candidate Population", fontsize=12, fontweight="bold", pad=12)

# -------------------------------------------------------------------------
# Subplot 3: Unconstrained Top Weighted Guides vs CVXOPT Feasible Selected Guides
# -------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, :])

# Merge top 5 weighted guides with spec features to show their rejection reasons
top_opt = opt.head(5).copy()
top_opt_merged = top_opt.merge(spec[["guide_rna", "gc_content", "melting_temp_c", "off_target_count"]], on="guide_rna", how="left")

# Format top 5 weighted guides info
opt_labels = []
for idx, row in top_opt_merged.iterrows():
    reasons = []
    if not (0.40 <= row["gc_content"] <= 0.60):
        reasons.append(f"GC={row['gc_content']:.2f}")
    if not (55.0 <= row["melting_temp_c"] <= 65.0):
        reasons.append(f"Tm={row['melting_temp_c']:.1f}")
    if row["off_target_count"] > 20:
        reasons.append(f"OT={int(row['off_target_count'])}")
    if row["off_target_capped"]:
        reasons.append("Capped")
    
    reason_str = ", ".join(reasons) if reasons else "Feasible"
    opt_labels.append(f"W-Rank {int(row['rank'])}: {row['guide_rna'][:10]}..\n[REJECTED: {reason_str}]")

# Format 5 CVXOPT guides info
cvx_labels = []
for idx, row in cvx.iterrows():
    cvx_labels.append(f"CVX-{int(row['rank'])}: {row['guide_rna'][:10]}..\n[FEASIBLE: c_i={row['objective_coefficient']:.3f}]")

all_compare_labels = opt_labels + cvx_labels
all_compare_scores = list(top_opt_merged["final_score"]) + list(cvx["objective_coefficient"])
bar_types = ["Unconstrained Weighted (Rejected)"] * 5 + ["CVXOPT Selected (Feasible)"] * 5

x_pos = np.arange(len(all_compare_labels))
bar_colors = ["#E74C3C"] * 5 + ["#2ECC71"] * 5

bars3 = ax3.bar(x_pos, all_compare_scores, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.8)

# Add scores on top of bars
for bar, score in zip(bars3, all_compare_scores):
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        score + 0.015,
        f"{score:.4f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#2C3E50",
    )

ax3.set_xticks(x_pos)
ax3.set_xticklabels(all_compare_labels, fontsize=8, fontweight="bold")
ax3.set_ylim(0.70, 1.02)
ax3.set_ylabel("Weighted Objective Score", fontsize=10, fontweight="bold")
ax3.set_title("C. Objective Score Comparison: Top 5 Unconstrained Weighted Guides vs 5 CVXOPT Feasible Selected Guides", fontsize=12, fontweight="bold", pad=12)

# Add separator line between Weighted and CVXOPT
ax3.axvline(4.5, color="#7F8C8D", linestyle="--", linewidth=1.5)
ax3.text(2, 0.98, "Top 5 Unconstrained Weighted Guides\n(All violate ≥1 biological constraint)", ha="center", va="center", fontsize=10, fontweight="bold", color="#C0392B", bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", edgecolor="#E74C3C", alpha=0.9))
ax3.text(7, 0.98, "CVXOPT Selected Guides\n(100% Feasible under all constraints)", ha="center", va="center", fontsize=10, fontweight="bold", color="#1E8449", bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAFAF1", edgecolor="#2ECC71", alpha=0.9))

# Legend
patch_rej = mpatches.Patch(color="#E74C3C", label="Unconstrained Weighted (Rejected by Constraints)")
patch_sel = mpatches.Patch(color="#2ECC71", label="CVXOPT Selected (Feasible)")
ax3.legend(handles=[patch_rej, patch_sel], loc="upper right", fontsize=9, framealpha=0.9)

fig.suptitle("CRISPR gRNA Optimization: Candidate Categorization & CVXOPT Feasibility Selection", fontsize=14, fontweight="bold", y=0.98)

# Save figure
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FILE, bbox_inches="tight")
plt.close(fig)

print(f"Successfully generated {OUT_FILE}")
