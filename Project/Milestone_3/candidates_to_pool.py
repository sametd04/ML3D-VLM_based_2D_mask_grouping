"""Convert build_qwen_candidates.py CSVs into Kate's pair-pool pickle format.

This bridges the clustering pipeline to the vision-classifier scorer
(kate_lora_fine_tuning: fine_tuning.py score), which consumes .pkl pair pools
of ViewPairMetadata instead of candidate CSVs.

The dataclass below mirrors load_training_data.ViewPairMetadata field-for-field.
Pickling it from __main__ is safe: load_pair_pool's _PoolUnpickler remaps
__main__.ViewPairMetadata to load_training_data.ViewPairMetadata on load.

Usage:
    python candidates_to_pool.py \
        --candidates data/candidates/room0_qwen_candidates.csv \
        --output data/pools/room0.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass
class ViewPairMetadata:
    scene: str
    frame_id_a: int
    mask_id_a: int
    frame_id_b: int
    mask_id_b: int
    label: int
    gt_id_a: int
    gt_id_b: int
    same_category: bool
    observer_frame_id: Optional[int] = None
    neg_strategy: Optional[str] = None
    angle_deg: Optional[float] = None
    observer_frame_ids: Optional[Tuple[int, ...]] = None
    depth_iou: Optional[float] = None
    overlap_a_to_b: Optional[float] = None
    overlap_b_to_a: Optional[float] = None


def row_to_meta(row: pd.Series) -> ViewPairMetadata:
    gt_a = int(row["gt_instance_a"]) if pd.notna(row["gt_instance_a"]) and row["gt_instance_a"] != "" else -1
    gt_b = int(row["gt_instance_b"]) if pd.notna(row["gt_instance_b"]) and row["gt_instance_b"] != "" else -1
    label = int(row["same_instance_label"]) if pd.notna(row["same_instance_label"]) and row["same_instance_label"] != "" else 0
    same_category = gt_a // 1000 == gt_b // 1000 if gt_a >= 0 and gt_b >= 0 else False
    return ViewPairMetadata(
        scene=str(row["scene"]),
        frame_id_a=int(row["frame_a"]),
        mask_id_a=int(row["mask_a"]),
        frame_id_b=int(row["frame_b"]),
        mask_id_b=int(row["mask_b"]),
        label=label,
        gt_id_a=gt_a,
        gt_id_b=gt_b,
        same_category=same_category,
        depth_iou=float(row["depth_iou"]),
        overlap_a_to_b=float(row["overlap_a_to_b"]),
        overlap_b_to_a=float(row["overlap_b_to_a"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert candidate CSV to pair-pool pickle.")
    parser.add_argument("--candidates", required=True, help="CSV from build_qwen_candidates.py.")
    parser.add_argument("--output", required=True, help="Output .pkl pool path.")
    args = parser.parse_args()

    df = pd.read_csv(args.candidates, keep_default_na=False, na_values=[])
    metas = [row_to_meta(row) for _, row in df.iterrows()]
    positives = [m for m in metas if m.label == 1]
    negatives = [m for m in metas if m.label != 1]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"positives": positives, "negatives": negatives}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{args.candidates}: {len(positives)} pos, {len(negatives)} neg -> {args.output}")


if __name__ == "__main__":
    main()
