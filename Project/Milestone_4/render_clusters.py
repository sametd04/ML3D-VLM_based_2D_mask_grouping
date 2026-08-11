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


def instance_colors(pred_masks: np.ndarray, base_colors: np.ndarray, dim: float,
                    dim_instances: set[int] | None = None) -> tuple[np.ndarray, int]:
    """Color each point by its instance; unassigned points keep dimmed RGB.

    Instances in `dim_instances` are treated as background (kept dimmed), so a
    huge floor/wall megacluster does not flood the whole render with one color.
    """
    dim_instances = dim_instances or set()
    colors = base_colors * dim
    n_instances = pred_masks.shape[1]
    # Larger instances first so small objects stay visible on top.
    order = np.argsort(-pred_masks.sum(axis=0))
    rank = 0
    for idx in order:
        if idx in dim_instances:
            continue  # leave as dimmed background
        colors[pred_masks[:, idx]] = _PALETTE[rank % len(_PALETTE)][:3]
        rank += 1
    return colors, n_instances - len(dim_instances)


def _height_axis(points: np.ndarray) -> int:
    return int(np.argmin(points.max(axis=0) - points.min(axis=0)))


def pick_disagreement_instance(
    ref_masks: np.ndarray, other_masks: np.ndarray, max_frac: float = 0.15
) -> int:
    """Pick the ref instance (excluding huge floor/wall megaclusters) whose points
    are labelled most differently by `other`, i.e. where the two methods disagree.

    Returns the ref instance column index whose points split across the largest
    number of `other` instances (weighted by size), among instances smaller than
    `max_frac` of the scene.
    """
    n_pts = ref_masks.shape[0]
    ref_sizes = ref_masks.sum(axis=0)
    # candidate objects: not tiny noise, not the huge structural megacluster
    cand = np.where((ref_sizes >= 0.005 * n_pts) & (ref_sizes <= max_frac * n_pts))[0]
    if len(cand) == 0:
        return int(np.argmax(ref_sizes))
    best_idx, best_score = int(cand[0]), -1.0
    other_assign = other_masks.argmax(axis=1)  # dominant other-instance per point
    other_any = other_masks.any(axis=1)
    for idx in cand:
        pts = ref_masks[:, idx]
        if pts.sum() == 0:
            continue
        labels = other_assign[pts & other_any]
        if len(labels) == 0:
            continue
        # fragmentation: how many distinct other-instances cover this object,
        # scaled by how evenly (entropy-ish). Higher = more disagreement.
        _, counts = np.unique(labels, return_counts=True)
        frac = counts / counts.sum()
        n_frag = (frac >= 0.1).sum()          # meaningful pieces only
        score = n_frag + (1.0 - frac.max())   # tie-break by how split the object is
        if score > best_score:
            best_idx, best_score = int(idx), float(score)
    return best_idx


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
    parser.add_argument(
        "--zoom-from", type=str, default=None,
        help="name=path.npz: the reference prediction whose instance defines the zoom bbox (shared by all).",
    )
    parser.add_argument(
        "--zoom-vs", type=str, default=None,
        help="Optional path.npz: pick the --zoom-from instance that disagrees most with THIS prediction.",
    )
    parser.add_argument(
        "--zoom-instance", type=int, default=None,
        help="Explicit instance column in --zoom-from to zoom into (overrides --zoom-vs auto-pick).",
    )
    parser.add_argument(
        "--zoom-margin", type=float, default=0.6,
        help="Fractional padding around the zoomed instance's bounding box.",
    )
    parser.add_argument(
        "--highlight", nargs="+", default=None,
        help="Per-pred override name=idx,idx,...: color only these instances (bright, "
             "distinct), dimming everything else. Makes a specific merge/split obvious.",
    )
    parser.add_argument(
        "--dim-largest", nargs="+", default=None,
        help="Pred names whose single largest instance is dimmed as background "
             "(e.g. a floor/wall megacluster), so the remaining objects stay visible.",
    )
    parser.add_argument(
        "--hide-unassigned", action="store_true",
        help="Drop points not assigned to any instance (no dark background).",
    )
    args = parser.parse_args()

    points, base_colors = load_mesh_vertices(args.mesh)
    points = points - points.mean(axis=0)

    keep = np.ones(len(points), dtype=bool)
    if args.crop_top is not None:
        # Height axis = smallest bounding-box extent (rooms are wider than tall).
        height_axis = int(np.argmin(points.max(axis=0) - points.min(axis=0)))
        keep = points[:, height_axis] <= np.quantile(points[:, height_axis], args.crop_top)

    # ----- optional zoom: restrict `keep` to a bbox around one instance --------
    if args.zoom_from is not None:
        zname, _, zpath = args.zoom_from.partition("=")
        ref_masks = np.load(zpath)["pred_masks"]
        if args.zoom_instance is not None:
            zi = args.zoom_instance
        elif args.zoom_vs is not None:
            other_masks = np.load(args.zoom_vs)["pred_masks"]
            zi = pick_disagreement_instance(ref_masks, other_masks)
        else:
            zi = int(np.argmax(ref_masks.sum(axis=0)))
        inst_pts = points[ref_masks[:, zi]]
        lo, hi = inst_pts.min(axis=0), inst_pts.max(axis=0)
        pad = (hi - lo) * args.zoom_margin
        lo, hi = lo - pad, hi + pad
        in_box = np.all((points >= lo) & (points <= hi), axis=1)
        keep = keep & in_box
        print(f"zoom: {zname} instance {zi} ({int(ref_masks[:, zi].sum())} pts) "
              f"-> bbox keeps {int(keep.sum())} / {len(points)} points")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Parse --highlight into {pred_name: [instance indices]}.
    highlight_map: dict[str, list[int]] = {}
    if args.highlight:
        for h in args.highlight:
            hname, _, idxs = h.partition("=")
            highlight_map[hname] = [int(x) for x in idxs.split(",") if x != ""]
    # Bright, well-separated colors for highlighted instances.
    _HL = [(0.90, 0.16, 0.16), (0.13, 0.47, 0.90), (0.13, 0.75, 0.30),
           (0.95, 0.60, 0.10), (0.60, 0.20, 0.80)]

    # RGB reference render (leftmost image on the poster) -- skip when zooming.
    if args.zoom_from is None:
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
        if name in highlight_map:
            # Dim everything, then paint only the chosen instances in bright colors.
            colors = base_colors * args.dim
            for k, idx in enumerate(highlight_map[name]):
                colors[masks[:, idx]] = _HL[k % len(_HL)]
            n_instances = len(highlight_map[name])
        else:
            dim_set: set[int] = set()
            if args.dim_largest and name in args.dim_largest:
                dim_set = {int(np.argmax(masks.sum(axis=0)))}
            colors, n_instances = instance_colors(masks, base_colors, args.dim, dim_set)
        title = f"{name} ({n_instances} instances)" if args.titles else ""
        show = keep.copy()
        if args.hide_unassigned and name not in highlight_map:
            assigned = masks.any(axis=1)
            show = show & assigned
        render(points[show], colors[show], args.out_dir / f"{name}.png", title, args.elev, args.azim, args.point_size, args.max_points)


if __name__ == "__main__":
    main()
