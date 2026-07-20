"""Grouped bar chart of class-agnostic AP results for the poster.

Numbers live in a JSON file so the figure regenerates when results update:
    {"Baseline (view consensus)": {"AP": 0.149, "AP50": 0.305, "AP25": 0.489}, ...}

Usage:
    python ap_chart.py --results results.json --out ap_chart.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TUM_BLUE = "#0065BD"
ACCENT = "#E37222"  # TUM orange
GREY = "#999999"
LIGHT_BLUE = "#64A0C8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--highlight", default=None, help="Variant name to draw in the accent color.")
    args = parser.parse_args()

    results: dict[str, dict[str, float]] = json.loads(args.results.read_text())
    variants = list(results)
    metrics = ["AP", "AP50", "AP25"]
    metric_labels = ["AP", "AP@50", "AP@25"]

    x = np.arange(len(metrics))
    n = len(variants)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    palette = [GREY, LIGHT_BLUE, TUM_BLUE, "#003359", ACCENT]
    for i, name in enumerate(variants):
        color = ACCENT if name == args.highlight else palette[i % len(palette)]
        vals = [results[name].get(m, np.nan) for m in metrics]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.92, label=name, color=color)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(
                    f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=15)
    ax.set_ylabel("class-agnostic AP (room0)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, max(v for r in results.values() for v in r.values()) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", transparent=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
