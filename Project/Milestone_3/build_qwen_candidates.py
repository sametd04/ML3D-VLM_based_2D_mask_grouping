"""Build Qwen candidate mask pairs from cheap 3D geometric overlap.

This script is the bridge between the MaskClustering environment and the Qwen
environment. It runs only the geometry-heavy part:

    2D masks + depth + poses -> 3D mask point sets -> candidate CSV

It does not import transformers or Qwen. The output CSV can be consumed later
by a Qwen scoring script in a separate environment.

Example:
    python Project/Milestone_3/build_qwen_candidates.py \
        --scene room0 \
        --config replica \
        --depth-overlap-threshold 0.05 \
        --train \
        --negative-ratio 2 \
        --output Project/Milestone_3/data/candidates/room0_qwen_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm


def find_repo_root(start: Optional[Path] = None) -> Path:
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for path in (start, *start.parents):
        if (path / "MaskClustering").is_dir() and (path / "Project").is_dir():
            return path
    raise RuntimeError("Could not find repository root from current working directory.")


REPO_ROOT = find_repo_root()
MASKCLUSTERING_ROOT = REPO_ROOT / "MaskClustering"
if str(MASKCLUSTERING_ROOT) not in sys.path:
    sys.path.insert(0, str(MASKCLUSTERING_ROOT))

from dataset.replica import ReplicaDataset  # noqa: E402
from graph.construction import build_point_in_mask_matrix, process_masks  # noqa: E402


def load_config(config_name: str) -> dict:
    config_path = MASKCLUSTERING_ROOT / "configs" / f"{config_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text())


def parse_frame_list(value: Optional[str], dataset: ReplicaDataset, step: int) -> list[int]:
    if not value:
        return [int(frame_id) for frame_id in dataset.get_frame_list(step)]
    frames = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        frames.append(int(part))
    if not frames:
        raise ValueError("--frames was provided but no valid frame ids were parsed.")
    return sorted(set(int(frame_id) for frame_id in frames))


def majority_gt_instance(
    point_ids: Iterable[int],
    gt_ids: np.ndarray,
    min_labeled_points: int,
) -> Optional[int]:
    ids = np.fromiter(point_ids, dtype=np.int64)
    if ids.size == 0:
        return None
    labels = gt_ids[ids]
    labels = labels[labels > 0]
    if labels.size < min_labeled_points:
        return None
    unique, counts = np.unique(labels, return_counts=True)
    return int(unique[np.argmax(counts)])


def compute_overlap(point_ids_a: set[int], point_ids_b: set[int]) -> dict[str, float | int]:
    intersection = len(point_ids_a.intersection(point_ids_b))
    union = len(point_ids_a.union(point_ids_b))
    len_a = len(point_ids_a)
    len_b = len(point_ids_b)
    return {
        "points_a": len_a,
        "points_b": len_b,
        "points_intersection": intersection,
        "depth_iou": intersection / union if union else 0.0,
        "overlap_a_to_b": intersection / len_a if len_a else 0.0,
        "overlap_b_to_a": intersection / len_b if len_b else 0.0,
    }


def passes_overlap_filter(
    overlap: dict[str, float | int],
    threshold: float,
    mode: str,
    view_consensus_score: float = 0.0,
    vc_threshold: float = 0.30,
) -> bool:
    if mode == "iou":
        return float(overlap["depth_iou"]) >= threshold
    if mode == "any_asymmetric":
        return (
            float(overlap["depth_iou"]) >= threshold
            or float(overlap["overlap_a_to_b"]) >= threshold
            or float(overlap["overlap_b_to_a"]) >= threshold
        )
    if mode == "or_rule":
        # Recall-oriented filter: geometric overlap OR a relaxed view-consensus
        # criterion (below MaskClustering's usual 0.8-0.9 merge threshold), so
        # positives without strong 3D overlap still reach the Qwen scorer.
        return (
            float(overlap["depth_iou"]) >= threshold
            or float(overlap["overlap_a_to_b"]) >= threshold
            or float(overlap["overlap_b_to_a"]) >= threshold
            or view_consensus_score >= vc_threshold
        )
    raise ValueError(f"Unknown overlap filter mode: {mode}")


def bool_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def nonnegative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    try:
        parsed = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def candidate_strength(row: dict) -> float:
    """Rank fallback candidates by their strongest consensus/overlap signal."""
    return max(
        float(row["view_consensus_score"]),
        float(row["depth_iou"]),
        float(row["overlap_a_to_b"]),
        float(row["overlap_b_to_a"]),
    )


def select_negatives(
    passing: list[dict], rejected: list[dict], target: int
) -> tuple[list[dict], int]:
    """Select exactly ``target`` negatives when available, preferring filtered pairs.

    Both pools are ordered by their strongest view-consensus or overlap score. Rejected
    negatives are only used when the overlap filter did not leave enough candidates.
    """
    passing = sorted(passing, key=candidate_strength, reverse=True)
    rejected = sorted(rejected, key=candidate_strength, reverse=True)
    selected = passing[:target]
    fallback_count = min(max(target - len(selected), 0), len(rejected))
    selected.extend(rejected[:fallback_count])
    return selected, fallback_count


def build_candidates(args: argparse.Namespace) -> tuple[list[dict], dict]:
    config = load_config(args.config)
    config.update(
        {
            "seq_name": args.scene,
            "config": args.config,
            "debug": args.debug,
        }
    )
    mc_args = SimpleNamespace(**config)

    data_link = MASKCLUSTERING_ROOT / "data"
    if not data_link.is_dir():
        raise FileNotFoundError(
            f"{data_link} is not a directory. MaskClustering expects data at "
            "MaskClustering/data; create a symlink to ../data first."
        )

    previous_cwd = Path.cwd()
    os.chdir(MASKCLUSTERING_ROOT)
    try:
        dataset = ReplicaDataset(args.scene)
        frame_list = parse_frame_list(args.frames, dataset, mc_args.step)
        if args.max_frames is not None:
            frame_list = frame_list[: args.max_frames]

        scene_points = dataset.get_scene_points()
        if args.debug:
            print("scene:", args.scene)
            print("frames:", len(frame_list), frame_list[:20])
            print("scene points:", scene_points.shape)

        boundary_points, point_in_mask_matrix, mask_point_clouds, point_frame_matrix, global_frame_mask_list = (
            build_point_in_mask_matrix(mc_args, scene_points, frame_list, dataset)
        )
        visible_frames, contained_masks, undersegment_mask_ids = process_masks(
            frame_list,
            global_frame_mask_list,
            point_in_mask_matrix,
            boundary_points,
            mask_point_clouds,
            mc_args,
        )
    finally:
        os.chdir(previous_cwd)

    gt_path = REPO_ROOT / "data" / "replica" / "ground_truth" / f"{args.scene}.txt"
    gt_ids = np.loadtxt(gt_path, dtype=np.int64) if gt_path.exists() else None
    if gt_ids is None:
        print(f"Warning: ground truth not found at {gt_path}; GT label columns will be empty.")

    valid_indices = list(range(len(global_frame_mask_list)))
    if not args.keep_undersegmented:
        invalid = set(undersegment_mask_ids)
        valid_indices = [i for i in valid_indices if i not in invalid]

    gt_by_index: dict[int, Optional[int]] = {}
    if gt_ids is not None:
        for idx in valid_indices:
            frame_id, mask_id = global_frame_mask_list[idx]
            point_ids = mask_point_clouds.get(f"{frame_id}_{mask_id}", set())
            gt_by_index[idx] = majority_gt_instance(point_ids, gt_ids, args.min_labeled_points)

    rows: list[dict] = []
    passing_negatives: list[dict] = []
    rejected_negatives: list[dict] = []
    total_pairs = 0
    kept_pairs = 0
    gt_positive_considered = 0
    gt_negative_considered = 0
    gt_positive_kept = 0
    gt_negative_kept = 0
    iterator = combinations(valid_indices, 2)
    if args.debug:
        pair_count = len(valid_indices) * (len(valid_indices) - 1) // 2
        iterator = tqdm(iterator, total=pair_count, desc="candidate pairs")

    for idx_a, idx_b in iterator:
        frame_a, mask_a = global_frame_mask_list[idx_a]
        frame_b, mask_b = global_frame_mask_list[idx_b]
        if not args.include_same_frame and frame_a == frame_b:
            continue

        total_pairs += 1

        # Resolved before the overlap filter (not after) so pairs the filter drops are
        # still counted by GT label -- that's what lets the summary report what fraction
        # of positives/negatives the overlap filter is throwing away.
        gt_a = gt_by_index.get(idx_a)
        gt_b = gt_by_index.get(idx_b)
        gt_label: Optional[int] = int(gt_a == gt_b) if gt_a is not None and gt_b is not None else None
        if gt_label == 1:
            gt_positive_considered += 1
        elif gt_label == 0:
            gt_negative_considered += 1

        points_a = mask_point_clouds.get(f"{frame_a}_{mask_a}", set())
        points_b = mask_point_clouds.get(f"{frame_b}_{mask_b}", set())
        overlap = compute_overlap(points_a, points_b)

        observer_num = bool_float(torch.dot(visible_frames[idx_a], visible_frames[idx_b]))
        supporter_num = bool_float(torch.dot(contained_masks[idx_a], contained_masks[idx_b]))
        view_consensus_score = supporter_num / (observer_num + 1e-7)

        passes_filter = passes_overlap_filter(
            overlap,
            args.depth_overlap_threshold,
            args.filter_mode,
            view_consensus_score=view_consensus_score,
            vc_threshold=args.vc_threshold,
        )

        row = {
            "scene": args.scene,
            "frame_a": frame_a,
            "mask_a": mask_a,
            "frame_b": frame_b,
            "mask_b": mask_b,
            "global_mask_idx_a": idx_a,
            "global_mask_idx_b": idx_b,
            **overlap,
            "num_observers": observer_num,
            "num_supporters": supporter_num,
            "view_consensus_score": view_consensus_score,
            "gt_instance_a": "" if gt_a is None else gt_a,
            "gt_instance_b": "" if gt_b is None else gt_b,
            "same_instance_label": "" if gt_label is None else gt_label,
        }

        # Keep rejected labeled negatives available for ratio backfilling. Other
        # rejected pairs retain the original behavior and are discarded.
        if gt_label == 0:
            (passing_negatives if passes_filter else rejected_negatives).append(row)
            if passes_filter:
                gt_negative_kept += 1
        elif passes_filter:
            rows.append(row)
            if gt_label == 1:
                gt_positive_kept += 1

        if passes_filter:
            kept_pairs += 1

    mode = getattr(args, "mode", "eval")
    negative_ratio = getattr(args, "negative_ratio", None)
    negative_fallback_count = 0
    negatives_before_ratio = len(passing_negatives)
    target_negatives: Optional[int] = None
    if mode == "eval" or negative_ratio is None:
        selected_negatives = passing_negatives
    else:
        target_negatives = gt_positive_kept * negative_ratio
        selected_negatives, negative_fallback_count = select_negatives(
            passing_negatives, rejected_negatives, target_negatives
        )
        gt_negative_kept = len(selected_negatives)
    rows.extend(selected_negatives)
    kept_pairs = len(rows)

    metadata = {
        "scene": args.scene,
        "config": args.config,
        "frames": int(len(frame_list)),
        "frame_list": [int(frame_id) for frame_id in frame_list],
        "global_masks": int(len(global_frame_mask_list)),
        "valid_masks": int(len(valid_indices)),
        "undersegmented_masks": int(len(undersegment_mask_ids)),
        "total_pairs_considered": int(total_pairs),
        "candidate_pairs_kept": int(kept_pairs),
        "gt_positive_pairs_considered": int(gt_positive_considered),
        "gt_positive_pairs_kept": int(gt_positive_kept),
        "gt_positive_kept_pct": round(100.0 * gt_positive_kept / gt_positive_considered, 2) if gt_positive_considered else None,
        "gt_negative_pairs_considered": int(gt_negative_considered),
        "gt_negative_pairs_kept": int(gt_negative_kept),
        "gt_negative_kept_pct": round(100.0 * gt_negative_kept / gt_negative_considered, 2) if gt_negative_considered else None,
        "depth_overlap_threshold": float(args.depth_overlap_threshold),
        "filter_mode": args.filter_mode,
        "vc_threshold": float(args.vc_threshold),
        "include_same_frame": bool(args.include_same_frame),
        "mode": mode,
        "negative_ratio": negative_ratio,
        "gt_negatives_passing_filter": int(negatives_before_ratio),
        "gt_negatives_added_by_ranked_fallback": int(negative_fallback_count),
        "gt_negatives_target": target_negatives,
        "gt_negatives_target_shortfall": (
            0 if target_negatives is None else max(target_negatives - len(selected_negatives), 0)
        ),
    }
    return rows, metadata


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "frame_a",
        "mask_a",
        "frame_b",
        "mask_b",
        "global_mask_idx_a",
        "global_mask_idx_b",
        "points_a",
        "points_b",
        "points_intersection",
        "depth_iou",
        "overlap_a_to_b",
        "overlap_b_to_a",
        "num_observers",
        "num_supporters",
        "view_consensus_score",
        "gt_instance_a",
        "gt_instance_b",
        "same_instance_label",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen candidate mask pairs using cheap 3D overlap."
    )
    parser.add_argument("--scene", default="room0", help="Replica scene name.")
    parser.add_argument("--config", default="replica", help="MaskClustering config name.")
    parser.add_argument(
        "--depth-overlap-threshold",
        type=float,
        default=0.05,
        help="Minimum 3D overlap required to keep a mask pair.",
    )
    parser.add_argument(
        "--filter-mode",
        choices=["iou", "any_asymmetric", "or_rule"],
        default="iou",
        help=(
            "iou: keep only if intersection/union passes threshold. "
            "any_asymmetric: also keep if either directional overlap passes. "
            "or_rule: any_asymmetric OR view consensus >= --vc-threshold."
        ),
    )
    parser.add_argument(
        "--vc-threshold",
        type=float,
        default=0.30,
        help="View consensus threshold for the or_rule filter mode.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--train",
        dest="mode",
        action="store_const",
        const="train",
        help="Balance negatives and backfill filtered-out negatives when necessary.",
    )
    mode_group.add_argument(
        "--eval",
        dest="mode",
        action="store_const",
        const="eval",
        help="Keep only candidates that pass the filter (default).",
    )
    parser.set_defaults(mode="eval")
    parser.add_argument(
        "--negative-ratio",
        type=nonnegative_int,
        default=None,
        metavar="N",
        help=(
            "In --train mode, number of GT-labeled negatives to keep per filtered "
            "positive. Default: None, meaning no cap is applied and all "
            "filter-passing negatives are kept (same negative selection as --eval "
            "mode). Set to an integer to cap negatives per positive instead: "
            "negatives passing the filter are preferred, and if there are too few, "
            "rejected negatives with the highest view-consensus or geometric-overlap "
            "score are added. Ignored in --eval mode."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument(
        "--frames",
        default=None,
        help="Optional comma-separated frame ids. Defaults to the config stride.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit on number of frames from the frame list.",
    )
    parser.add_argument(
        "--include-same-frame",
        action="store_true",
        help="Include pairs from the same RGB frame. Default excludes them.",
    )
    parser.add_argument(
        "--keep-undersegmented",
        action="store_true",
        help="Keep masks that MaskClustering marks as undersegmented.",
    )
    parser.add_argument(
        "--min-labeled-points",
        type=int,
        default=10,
        help="Minimum labeled 3D points needed to assign a GT instance id.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    rows, metadata = build_candidates(args)
    output_path = args.output
    if output_path is None:
        output_path = (
            REPO_ROOT
            / "Project"
            / "Milestone_3"
            / "data"
            / "candidates"
            / f"{args.scene}_qwen_candidates.csv"
        )
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    write_csv(rows, output_path)

    meta_path = output_path.with_suffix(".summary.json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    print("Candidate CSV written to:", output_path)
    print("Summary written to      :", meta_path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
