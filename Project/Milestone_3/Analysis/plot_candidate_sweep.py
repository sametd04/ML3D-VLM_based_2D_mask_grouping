"""Plots build_qwen_candidates.py threshold-sweep results from sweep_candidates.py's sweep_log.csv.

Pure pandas + matplotlib, no MaskClustering/Qwen imports needed. Run from
Project/Milestone_3/ (after sweep_candidates.py has produced sweep_log.csv):

    python plot_candidate_sweep.py

Produces two figures:

1. sweep_lines.png -- one line per room, sweeping one threshold at a time while
   holding the other fixed at build_qwen_candidates.py's own CLI default
   (depth_overlap_threshold=0.05, vc_threshold=0.30), for gt_positive_kept_pct,
   gt_negative_kept_pct, and candidate_pairs_kept.
2. sweep_heatmaps.png -- per room, both thresholds' joint influence on each of
   those same three metrics (a 2D parameter space needs a 2D form, not lines).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

LOG_PATH = Path("data/candidates/sweep/sweep_log.csv")
LINES_OUT = Path("data/candidates/sweep/sweep_lines.png")
HEATMAP_OUT = Path("data/candidates/sweep/sweep_heatmaps.png")

# Held fixed when sweeping the other threshold -- matches build_qwen_candidates.py's
# own --depth-overlap-threshold/--vc-threshold CLI defaults, not arbitrary picks.
FIXED_VC_THRESHOLD = 0.30
FIXED_DEPTH_THRESHOLD = 0.05

# Okabe-Ito colorblind-safe palette -- same set as SCENE_PALETTE in
# Training_Data_Exploration.ipynb. Fixed assignment by scene (sorted), never
# re-cycled, so a given room is always the same color across every subplot.
SCENE_PALETTE = ["#009E73", "#CC79A7", "#56B4E9", "#F0E442", "#D55E00", "#000000"]

# (column, y-axis label, value formatter) -- one row per metric in both figures.
METRICS = [
    ("gt_positive_kept_pct", "% positives kept", lambda v: f"{v:.0f}%"),
    ("gt_negative_kept_pct", "% negatives kept", lambda v: f"{v:.0f}%"),
    ("candidate_pairs_kept", "# candidate pairs kept", lambda v: f"{v:,.0f}"),
]
# Sequential, single-hue colormap per metric -- different units (%, raw count) never
# share a color scale, but each is still a magnitude job, so one hue light->dark.
HEATMAP_CMAPS = {
    "gt_positive_kept_pct": "Oranges",
    "gt_negative_kept_pct": "Oranges",
    "candidate_pairs_kept": "Oranges",
}
# Fixed text color for every heatmap cell.
CELL_TEXT_COLOR = "black"

df = pd.read_csv(LOG_PATH)
df = df[df["depth_overlap_threshold"] != 0.0]  # degenerate -- keeps ~everything, skews the trend/color scale
scenes = sorted(df["scene"].unique())
scene_colors = {s: SCENE_PALETTE[i % len(SCENE_PALETTE)] for i, s in enumerate(scenes)}

# ---------------------------------------------------------------------------
# 1. Line plots: one line per room, one threshold swept at a time.
# ---------------------------------------------------------------------------
sweep_axes = [
    ("depth_overlap_threshold", "vc_threshold", FIXED_VC_THRESHOLD),
    ("vc_threshold", "depth_overlap_threshold", FIXED_DEPTH_THRESHOLD),
]

fig, axes = plt.subplots(
    len(METRICS), len(sweep_axes), figsize=(6 * len(sweep_axes), 3.5 * len(METRICS)), squeeze=False,
)

for row, (col_name, ylabel, fmt) in enumerate(METRICS):
    for col, (x_col, fixed_col, fixed_val) in enumerate(sweep_axes):
        ax = axes[row][col]
        held = df[df[fixed_col] == fixed_val]
        for scene in scenes:
            line = held[held.scene == scene].sort_values(x_col)
            ax.plot(
                line[x_col], line[col_name],
                color=scene_colors[scene], marker="o", markersize=6, linewidth=2, label=scene,
            )

        if "_pct" in col_name:
            ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _, fmt=fmt: fmt(y)))
        ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if row == 0:
            ax.set_title(f"sweep {x_col}\n({fixed_col} fixed at {fixed_val})", fontsize=10)
        if col == 0:
            ax.set_ylabel(ylabel)
        if row == len(METRICS) - 1:
            ax.set_xlabel(x_col)

axes[0][0].legend(frameon=False, title="scene", loc="best")
fig.suptitle("Candidate filter sweep: threshold trends by room (one threshold fixed at a time)")
plt.tight_layout()
plt.savefig(LINES_OUT, dpi=150)
print(f"Saved line plot to {LINES_OUT}")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Heatmaps: per room, both thresholds' joint influence on each metric.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    len(METRICS), len(scenes), figsize=(3.4 * len(scenes), 3.4 * len(METRICS)), squeeze=False,
)

for row, (col_name, ylabel, fmt) in enumerate(METRICS):
    for col, scene in enumerate(scenes):
        ax = axes[row][col]
        sub = df[df.scene == scene]
        pivot = (
            sub.pivot(index="vc_threshold", columns="depth_overlap_threshold", values=col_name)
            .sort_index(ascending=False)
        )

        im = ax.imshow(pivot.values, cmap=HEATMAP_CMAPS[col_name], aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7, color=CELL_TEXT_COLOR)

        if row == 0:
            ax.set_title(scene)
        if col == 0:
            ax.set_ylabel(f"{ylabel}\nvc_threshold")
        if row == len(METRICS) - 1:
            ax.set_xlabel("depth_overlap_threshold")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Candidate filter sweep: joint threshold influence per room")
plt.tight_layout()
plt.savefig(HEATMAP_OUT, dpi=150)
print(f"Saved heatmaps to {HEATMAP_OUT}")
