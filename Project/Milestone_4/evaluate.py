"""Evaluation script for VLM-based mask-pair classification.

Loads a saved pair pool, runs batched Qwen3-VL inference, and computes binary
classification metrics (accuracy, precision, recall, F1) broken down by
condition, scene, same_category flag, and negative sampling strategy.

Usage:
    python evaluate.py \
        --pool-path /data/pairs/room0.pkl /data/pairs/office0.pkl \
        --masks-root /path/to/replica_masks \
        --rgb-root /path/to/replica \
        --conditions pair_only pair_with_context pair_with_candidate \
        --K 15 --neg-ratio 3 --batch-size 4 \
        --output results/eval_run1.json \
        --config config.yaml
"""

import argparse
import glob
import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from inference import run_inference_batch
from load_training_data import load_data_training
from prompt_templates import ConditionType
from vlm_utils import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_CONDITIONS: List[ConditionType] = ["pair_only", "pair_with_context", "pair_with_candidate"]


def compute_metrics(labels: List[int], predictions: List[int]) -> dict:
    """Compute binary classification metrics for same-object predictions.

    Args:
        labels: ground-truth labels (1 = same object, 0 = different).
        predictions: model decisions.

    Returns:
        Dict with accuracy, precision, recall, f1, n_samples, n_positive, n_negative.
    """
    if not labels:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None,
                "n_samples": 0, "n_positive": 0, "n_negative": 0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_samples": len(labels),
        "n_positive": int(sum(labels)),
        "n_negative": int(len(labels) - sum(labels)),
    }


def _group_metrics(all_samples: List[dict], all_preds: List[dict], key: str) -> dict:
    """Compute metrics broken down by a metadata key."""
    groups: Dict[str, tuple] = defaultdict(lambda: ([], []))
    for sample, pred in zip(all_samples, all_preds):
        group_val = str(sample.get(key))
        groups[group_val][0].append(sample["label"])
        groups[group_val][1].append(pred["decision"])
    return {k: compute_metrics(labels, preds) for k, (labels, preds) in groups.items()}


def run_evaluation(
    pool_paths: List[str],
    masks_root: str,
    rgb_root: str,
    conditions: List[ConditionType],
    K: int,
    neg_ratio: int,
    batch_size: int,
    shuffle: bool,
    output_path: str,
    seed: int,
    max_new_tokens: int,
    config: dict,
    max_positives: Optional[int] = None,
) -> dict:
    """Load data, run batch inference, compute metrics, save results."""

    # Step 1: load samples from all pool paths
    all_samples: List[dict] = []
    for path in pool_paths:
        logger.info("Loading pool: %s", path)
        samples = load_data_training(
            pool_path=path,
            masks_root=masks_root,
            rgb_root=rgb_root,
            K=K,
            neg_ratio=neg_ratio,
            conditions=conditions,
            seed=seed,
            max_positives=max_positives,
        )
        all_samples.extend(samples)
    logger.info("Total samples loaded: %d", len(all_samples))

    # Step 2: order
    if shuffle:
        random.Random(seed).shuffle(all_samples)
    else:
        # Group by condition so same-image-count samples batch together efficiently
        condition_order = {c: i for i, c in enumerate(ALL_CONDITIONS)}
        all_samples.sort(key=lambda s: condition_order.get(s["condition"], 99))

    # Step 3: load model
    logger.info("Loading model...")
    model, processor = load_model(config)

    # Step 4: batch inference
    all_preds = []
    with tqdm(total=len(all_samples), desc="Inference", unit="sample") as pbar:
        for i in range(0, len(all_samples), batch_size):
            chunk = all_samples[i : i + batch_size]
            preds = run_inference_batch(chunk, model, processor, max_new_tokens=max_new_tokens)
            all_preds.extend(preds)
            pbar.update(len(chunk))

    # Step 5: compute metrics
    y_true = [s["label"] for s in all_samples]
    y_pred = [p["decision"] for p in all_preds]

    results = {
        "config": {
            "pool_paths": pool_paths,
            "conditions": conditions,
            "K": K,
            "neg_ratio": neg_ratio,
            "max_positives": max_positives,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        },
        "overall": compute_metrics(y_true, y_pred),
        "by_condition": _group_metrics(all_samples, all_preds, "condition"),
        "by_scene": _group_metrics(all_samples, all_preds, "scene"),
        "by_same_category": _group_metrics(all_samples, all_preds, "same_category"),
        "by_neg_strategy": _group_metrics(all_samples, all_preds, "neg_strategy"),
        "per_sample": [
            {
                "scene": s["scene"],
                "condition": s["condition"],
                "label": s["label"],
                "prediction": p["decision"],
                "confidence": p["confidence"],
                "reasoning": p["reasoning"],
                "frame_id_a": s["frame_id_a"],
                "mask_id_a": s["mask_id_a"],
                "frame_id_b": s["frame_id_b"],
                "mask_id_b": s["mask_id_b"],
                "observer_frame_id": s.get("observer_frame_id"),
                "candidate_id": s.get("candidate_id"),
                "same_category": s["same_category"],
                "neg_strategy": s.get("neg_strategy"),
                "angle_deg": s.get("angle_deg"),
            }
            for s, p in zip(all_samples, all_preds)
        ],
    }

    # Step 6: save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Step 7: print summary
    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    ov = results["overall"]
    print("\n=== Overall ===")
    print(f"  Accuracy:  {ov['accuracy']:.4f}  ({ov['n_samples']} samples, "
          f"{ov['n_positive']} pos / {ov['n_negative']} neg)")
    print(f"  Precision: {ov['precision']:.4f}")
    print(f"  Recall:    {ov['recall']:.4f}")
    print(f"  F1:        {ov['f1']:.4f}")

    print("\n=== By Condition ===")
    print(f"  {'Condition':<26} {'Acc':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'N':>5}")
    for cond, m in sorted(results["by_condition"].items()):
        if m["accuracy"] is None:
            continue
        print(f"  {cond:<26} {m['accuracy']:>6.4f}  {m['precision']:>6.4f}  "
              f"{m['recall']:>6.4f}  {m['f1']:>6.4f}  {m['n_samples']:>5}")

    print("\n=== By Scene ===")
    for scene, m in sorted(results["by_scene"].items()):
        if m["accuracy"] is None:
            continue
        print(f"  {scene:<20} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  n={m['n_samples']}")

    print("\n=== By Neg Strategy (None = positives) ===")
    for strat, m in sorted(results["by_neg_strategy"].items()):
        if m["accuracy"] is None:
            continue
        print(f"  {strat:<14} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  n={m['n_samples']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM mask-pair classification on a saved pair pool."
    )
    parser.add_argument(
        "--pool-path", nargs="+", required=True,
        help="One or more .pkl pair pool paths (shell globs are expanded).",
    )
    parser.add_argument("--masks-root", required=True, help="Path to replica_masks directory.")
    parser.add_argument("--rgb-root", required=True, help="Path to Replica dataset root.")
    parser.add_argument(
        "--conditions", nargs="+", default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS, metavar="CONDITION",
        help="Which conditions to evaluate (default: all three).",
    )
    parser.add_argument(
        "--max-positives", type=int, default=None,
        help="Flat random sample of this many positives from the full pool "
             "(ignores --K; negatives = neg_ratio × max_positives).",
    )
    parser.add_argument("--K", type=int, default=15, help="Max positives per GT instance (ignored when --max-positives is set).")
    parser.add_argument("--neg-ratio", type=int, default=3, help="Negatives per positive.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle sample order (default: grouped by condition for batch efficiency).",
    )
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max tokens generated per sample (3-line output is short).",
    )
    args = parser.parse_args()

    # Expand globs in pool paths
    expanded_paths = []
    for pattern in args.pool_path:
        matches = glob.glob(pattern)
        if matches:
            expanded_paths.extend(sorted(matches))
        else:
            expanded_paths.append(pattern)  # keep as-is so the error is clear

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_evaluation(
        pool_paths=expanded_paths,
        masks_root=args.masks_root,
        rgb_root=args.rgb_root,
        conditions=args.conditions,
        K=args.K,
        neg_ratio=args.neg_ratio,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        output_path=args.output,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        config=config,
        max_positives=args.max_positives,
    )
