"""Render predicted 3D instance clusters as poster figures.

Loads a Replica scene mesh and one or more prediction .npz files (from
MaskClustering's post_process / run_qwen_clustering.py) and renders each
prediction's instance clusters over the scene from a fixed viewpoint, so
variants are directly comparable side by side.

Only needs numpy, matplotlib, plyfile — no Open3D/CUDA.

Usage:
    python render_clusters.py \
        --mesh room0_mesh.ply \
        --pred baseline=path/to/baseline/room0.npz \
               zeroshot=path/to/qwen_zeroshot/room0.npz \
        --out-dir renders/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

# Fixed color palette (tab20-ish, bright) cycled per instance id, so the same
# instance index gets the same color across variants of the same scene.
_PALETTE = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors


def load_mesh_vertices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    if all(c in v.data.dtype.names for c in ("red", "green", "blue")):
        colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64) / 255.0
        colors = np.power(colors, 1 / 2.2)  # same tone mapping as MaskClustering's visualizer
    else:
        colors = np.full((len(points), 3), 0.7)
    return points, colors


def instance_colors(pred_masks: np.ndarray, base_colors: np.ndarray, dim: float) -> tuple[np.ndarray, int]:
    """Color each point by its instance; unassigned points keep dimmed RGB."""
    colors = base_colors * dim
    n_instances = pred_masks.shape[1]
    # Larger instances first so small objects stay visible on top.
    order = np.argsort(-pred_masks.sum(axis=0))
    for rank, idx in enumerate(order):
        colors[pred_masks[:, idx]] = _PALETTE[rank % len(_PALETTE)][:3]
    return colors, n_instances


def render(
    points: np.ndarray,
    colors: np.ndarray,
    out_path: Path,
    title: str,
    elev: float,
    azim: float,
    point_size: float,
    max_points: int,
) -> None:
    if len(points) > max_points:
        sel = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[sel], colors[sel]

    fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=point_size, linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    # Equal aspect so the room isn't distorted.
    spans = points.max(axis=0) - points.min(axis=0)
    ax.set_box_aspect(spans)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=16, pad=0)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render instance clustering results for the poster.")
    parser.add_argument("--mesh", type=Path, required=True, help="Scene mesh .ply (e.g. room0_mesh.ply).")
    parser.add_argument(
        "--pred", nargs="+", required=True,
        help="One or more name=path/to/pred.npz entries; name becomes the filename/title.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("renders"))
    parser.add_argument("--elev", type=float, default=55, help="Camera elevation (deg). ~90 = top-down.")
    parser.add_argument("--azim", type=float, default=-60, help="Camera azimuth (deg).")
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=400_000, help="Subsample cap for rendering speed.")
    parser.add_argument("--dim", type=float, default=0.25, help="Brightness of unclustered points (0..1).")
    parser.add_argument("--titles", action="store_true", help="Draw variant name + cluster count above each render.")
    parser.add_argument(
        "--crop-top", type=float, default=None,
        help="Drop points above this height quantile (e.g. 0.9 removes the ceiling for interior views).",
    )
    args = parser.parse_args()

    points, base_colors = load_mesh_vertices(args.mesh)
    points = points - points.mean(axis=0)

    keep = np.ones(len(points), dtype=bool)
    if args.crop_top is not None:
        # Height axis = smallest bounding-box extent (rooms are wider than tall).
        height_axis = int(np.argmin(points.max(axis=0) - points.min(axis=0)))
        keep = points[:, height_axis] <= np.quantile(points[:, height_axis], args.crop_top)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # RGB reference render (leftmost image on the poster).
    render(points[keep], base_colors[keep], args.out_dir / "rgb.png", "", args.elev, args.azim, args.point_size, args.max_points)

    for entry in args.pred:
        name, _, path = entry.partition("=")
        if not path:
            raise ValueError(f"--pred entries must be name=path, got {entry!r}")
        pred = np.load(path)
        masks = pred["pred_masks"]
        if masks.shape[0] != len(points):
            raise ValueError(
                f"{path}: pred_masks has {masks.shape[0]} points but mesh has {len(points)} vertices."
            )
        colors, n_instances = instance_colors(masks, base_colors, args.dim)
        title = f"{name} ({n_instances} instances)" if args.titles else ""
        render(points[keep], colors[keep], args.out_dir / f"{name}.png", title, args.elev, args.azim, args.point_size, args.max_points)


if __name__ == "__main__":
    main()
