"""Convert fine_tuning.py `score` JSON output into the qwen_scores CSV format.

Bridges the vision-classifier finetuning (Milestone 4, `fine_tuning.py score`)
to the clustering + AP evaluation harness (`run_qwen_clustering.py`), so
pair-level probabilities can be judged by the end metric instead of only
precision/recall/F1.

Usage:
    python convert_classifier_scores.py \
        --input results/score_run1.json \
        --scene room0 \
        --output data/qwen_scores/room0_classifier_scores.csv

    python run_qwen_clustering.py --scene room0 \
        --scores data/qwen_scores/room0_classifier_scores.csv \
        --output-name classifier --qwen-score-threshold 0.5

Runs anywhere (stdlib + pandas only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def convert(input_path: Path, scene: str | None) -> pd.DataFrame:
    with open(input_path) as f:
        results = json.load(f)

    samples = results["per_sample"]
    if scene is not None:
        samples = [s for s in samples if s["scene"] == scene]
    if not samples:
        raise ValueError(f"No samples found in {input_path}" + (f" for scene {scene!r}" if scene else ""))

    df = pd.DataFrame(
        {
            "frame_a": s["frame_id_a"],
            "mask_a": s["mask_id_a"],
            "frame_b": s["frame_id_b"],
            "mask_b": s["mask_id_b"],
            "qwen_same_instance_score": s["probability"],
        }
        for s in samples
    )

    # A pair can appear multiple times (one sample per observer/candidate
    # variant); average the probabilities into one score per pair.
    df = (
        df.groupby(["frame_a", "mask_a", "frame_b", "mask_b"], as_index=False)
        .agg(qwen_same_instance_score=("qwen_same_instance_score", "mean"))
    )
    return df


def merge_with_candidates(scores: pd.DataFrame, candidates_path: Path) -> pd.DataFrame:
    """Join classifier probabilities onto the candidate CSV so the output keeps
    the geometric feature and GT label columns required by fusion_mlp.py."""
    candidates = pd.read_csv(candidates_path)
    keys = ["frame_a", "mask_a", "frame_b", "mask_b"]
    merged = candidates.merge(scores, on=keys, how="left")
    n_missing = int(merged["qwen_same_instance_score"].isna().sum())
    if n_missing:
        print(f"Warning: {n_missing}/{len(merged)} candidate pairs have no classifier score; dropping them.")
        merged = merged.dropna(subset=["qwen_same_instance_score"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="JSON from fine_tuning.py score.")
    parser.add_argument("--scene", default=None, help="Optional scene filter (JSON may mix scenes).")
    parser.add_argument("--output", type=Path, required=True, help="CSV for run_qwen_clustering.py --scores.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Optional candidate CSV (build_qwen_candidates.py) to merge scores into, "
        "keeping geometric features and GT labels for fusion_mlp.py.",
    )
    args = parser.parse_args()

    df = convert(args.input, args.scene)
    if args.candidates is not None:
        df = merge_with_candidates(df, args.candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} pair scores to {args.output}")
    print(f"Score range: {df.qwen_same_instance_score.min():.3f} .. {df.qwen_same_instance_score.max():.3f}")


if __name__ == "__main__":
    main()
