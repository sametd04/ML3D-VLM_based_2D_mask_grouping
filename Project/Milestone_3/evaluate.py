"""Evaluation script for VLM-based mask-pair classification.

Loads a candidate CSV, runs batched Qwen3-VL inference, and computes binary
classification metrics (accuracy, precision, recall, F1) broken down by
label (positive/negative), scene, and same_class flag.

Usage:
    python evaluate.py \
        --pool-path data/candidates/room0_qwen_candidates.csv data/candidates/office0_qwen_candidates.csv \
        --masks-root /path/to/replica_masks \
        --rgb-root /path/to/replica \
        --batch-size 4 \
        --output results/eval_run1.json \
        --config config_stage1.yaml
"""

import argparse
import glob
import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference import PreRenderedSampleDataset, generate_batch, make_collate_fn
from load_training_data import ViewPairMetadata, load_data_training, render_pairs
from prompt_templates import RenderMode
from vlm_utils import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


AggregationMode = Literal["mean", "confidence_weighted", "consensus_rate"]


def _pred_to_p_same(pred: dict) -> float:
    """Normalizes a per-variant prediction (VLM or LoRA format) to P(label==1).

    LoRA preds carry "probability" directly; VLM preds carry "confidence" in
    whichever decision was made, so it must be flipped when decision == 0.
    """
    if "probability" in pred:
        return float(pred["probability"])
    conf = float(pred["confidence"])
    return conf if pred["decision"] == 1 else 1.0 - conf


def _certainty(p_same: float) -> float:
    """0 = maximally uncertain (p_same=0.5), 1 = maximally certain (p_same=0 or 1)."""
    return abs(p_same - 0.5) * 2.0


def _pair_group_key(sample: dict) -> tuple:
    """Identifies the underlying pair within one scene and condition, so per-candidate/observer variants collapse without merging across conditions or scenes."""
    return (
        sample["scene"], sample["condition"],
        sample["frame_id_a"], sample["mask_id_a"],
        sample["frame_id_b"], sample["mask_id_b"],
    )


def aggregate_pair_predictions(
    samples: List[dict],
    preds: List[dict],
    mode: AggregationMode = "mean",
    connect_threshold: float = 0.5,
) -> Tuple[List[dict], List[dict]]:
    """Collapses per-variant predictions to one prediction per underlying pair.

    A no-op for pair_only (always exactly one sample per pair already). Kept for
    parity with fine_tuning.py's score(), which shares this function. "mean"/
    "confidence_weighted" average P(same) and threshold at 0.5. "consensus_rate"
    thresholds the fraction of variants voting "same" against connect_threshold,
    mirroring MaskClustering's supporter_nums/observer_nums ratio.

    Returns:
        (agg_samples, agg_preds), one entry per unique pair. agg_preds carries
        decision, p_same, n_variants, aggregation.
    """
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[_pair_group_key(s)].append(i)

    agg_samples: List[dict] = []
    agg_preds: List[dict] = []
    for key, idxs in groups.items():
        rep = samples[idxs[0]]

        labels = {samples[i]["label"] for i in idxs}
        if len(labels) > 1:
            logger.warning("Inconsistent labels within pair group %s: %s, using first variant's label.", key, labels)

        observers = {samples[i].get("observer_frame_id") for i in idxs}
        if len(observers) > 1:
            # Expected under consensus_rate (multiple observers per pair by design);
            # only warn for the other modes, where it would be unexpected.
            log_fn = logger.debug if mode == "consensus_rate" else logger.warning
            log_fn("Multiple observer_frame_id within pair group %s: %s", key, observers)

        if mode == "consensus_rate":
            decisions = [preds[i]["decision"] for i in idxs]
            p_pair = sum(decisions) / len(decisions)
            decision = 1 if p_pair >= connect_threshold else 0
        else:
            p_values = [_pred_to_p_same(preds[i]) for i in idxs]
            if mode == "mean" or len(idxs) == 1:
                p_pair = sum(p_values) / len(p_values)
            elif mode == "confidence_weighted":
                weights = [_certainty(p) for p in p_values]
                total_w = sum(weights)
                p_pair = (sum(w * p for w, p in zip(weights, p_values)) / total_w) if total_w > 0 \
                          else sum(p_values) / len(p_values)
            else:
                raise ValueError(f"Unknown aggregation mode {mode!r}")
            decision = 1 if p_pair >= 0.5 else 0

        agg_samples.append(rep)
        agg_preds.append({
            "decision": decision,
            "p_same": p_pair,
            "n_variants": len(idxs),
            "aggregation": mode,
        })
    return agg_samples, agg_preds


def _group_metrics(all_samples: List[dict], all_preds: List[dict], key) -> dict:
    """Compute metrics broken down by a metadata key (string) or a callable(sample) -> str."""
    groups: Dict[str, tuple] = defaultdict(lambda: ([], []))
    for sample, pred in zip(all_samples, all_preds):
        group_val = key(sample) if callable(key) else str(sample.get(key))
        groups[group_val][0].append(sample["label"])
        groups[group_val][1].append(pred["decision"])
    return {k: compute_metrics(labels, preds) for k, (labels, preds) in groups.items()}


def run_evaluation(
    masks_root: str,
    rgb_root: str,
    batch_size: int,
    shuffle: bool,
    output_path: str,
    seed: int,
    max_new_tokens: int,
    config: dict,
    num_workers: int = 2,
    pool_paths: Optional[List[str]] = None,
    pair_pools: Optional[List[Tuple[List[ViewPairMetadata], List[ViewPairMetadata]]]] = None,
    render_mode: RenderMode = "outline",
    alpha: float = 0.4,
    aggregation: AggregationMode = "mean",
    connect_threshold: float = 0.5,
) -> dict:
    """Load data, run batch inference, compute metrics, save results.

    Exactly one of pool_paths (candidate CSVs on disk) or pair_pools (in-memory
    (positives, negatives) lists, e.g. a pre-sampled subset already in memory) must
    be given.
    """
    if (pool_paths is None) == (pair_pools is None):
        raise ValueError(
            f"run_evaluation requires exactly one of pool_paths or pair_pools "
            f"(got pool_paths={pool_paths is not None}, pair_pools={pair_pools is not None})"
        )

    # Step 1: load samples, either from disk pool paths or from in-memory pair pools
    all_samples: List[dict] = []
    if pool_paths is not None:
        for path in pool_paths:
            logger.info("Loading pool: %s", path)
            samples = load_data_training(
                pool_path=path,
                masks_root=masks_root,
                rgb_root=rgb_root,
                render_mode=render_mode,
                alpha=alpha,
            )
            all_samples.extend(samples)
    else:
        for i, (positives, negatives) in enumerate(pair_pools):
            logger.info(
                "Rendering in-memory pair pool %d/%d (%d pos, %d neg)",
                i + 1, len(pair_pools), len(positives), len(negatives),
            )
            samples = render_pairs(
                positives, negatives,
                masks_root=masks_root,
                rgb_root=rgb_root,
                render_mode=render_mode,
                alpha=alpha,
            )
            all_samples.extend(samples)
    logger.info("Total samples loaded: %d", len(all_samples))

    # Step 2: order
    if shuffle:
        random.Random(seed).shuffle(all_samples)

    # Step 3: load model
    logger.info("Loading model...")
    model, processor = load_model(config)

    # Step 4: batch inference. DataLoader workers run the CPU-bound prompt
    # building + image preprocessing (collate_fn) for upcoming batches while the
    # main process runs model.generate() on the GPU for the current one, so the
    # GPU doesn't idle waiting on CPU preprocessing between batches.
    loader = DataLoader(
        PreRenderedSampleDataset(all_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(processor),
        num_workers=num_workers,
        prefetch_factor=1 if num_workers > 0 else None,
    )
    all_preds = []
    with tqdm(total=len(all_samples), desc="Inference", unit="sample") as pbar:
        for chunk, inputs in loader:
            preds = generate_batch(inputs, model, processor, max_new_tokens=max_new_tokens)
            all_preds.extend(preds)
            pbar.update(len(chunk))

    # Step 5: collapse per-variant predictions to one prediction per pair (no-op for
    # pair_only, kept for parity with fine_tuning.py's score()).
    agg_samples, agg_preds = aggregate_pair_predictions(
        all_samples, all_preds, mode=aggregation, connect_threshold=connect_threshold,
    )
    logger.info("Aggregated %d rendered samples into %d unique pairs", len(all_samples), len(agg_samples))

    y_true = [s["label"] for s in agg_samples]
    y_pred = [p["decision"] for p in agg_preds]

    results = {
        "config": {
            "pool_paths": pool_paths,
            "pair_pool_sizes": [[len(p), len(n)] for p, n in pair_pools] if pair_pools is not None else None,
            "render_mode": render_mode,
            "highlight_alpha": alpha,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": shuffle,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "aggregation": aggregation,
            "connect_threshold": connect_threshold,
        },
        "overall": compute_metrics(y_true, y_pred),
        "by_label": _group_metrics(agg_samples, agg_preds, lambda s: "positive" if s["label"] == 1 else "negative"),
        "by_scene": _group_metrics(agg_samples, agg_preds, "scene"),
        "by_same_class": _group_metrics(agg_samples, agg_preds, "same_class"),
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
                "observer_frame_id": int(s["observer_frame_id"]) if s.get("observer_frame_id") is not None else None,
                "candidate_id": s.get("candidate_id"),
                "same_class": s["same_class"],
            }
            for s, p in zip(all_samples, all_preds)
        ],
        "per_pair": [
            {
                "scene": s["scene"],
                "condition": s["condition"],
                "label": s["label"],
                "decision": p["decision"],
                "p_same": p["p_same"],
                "n_variants": p["n_variants"],
                "frame_id_a": s["frame_id_a"],
                "mask_id_a": s["mask_id_a"],
                "frame_id_b": s["frame_id_b"],
                "mask_id_b": s["mask_id_b"],
                "observer_frame_id": int(s["observer_frame_id"]) if s.get("observer_frame_id") is not None else None,
                "same_class": s["same_class"],
            }
            for s, p in zip(agg_samples, agg_preds)
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
    lines = ["", "=== Overall ==="]
    lines.append(f"  Accuracy:  {ov['accuracy']:.4f}  ({ov['n_samples']} samples, "
                  f"{ov['n_positive']} pos / {ov['n_negative']} neg)")
    lines.append(f"  Precision: {ov['precision']:.4f}")
    lines.append(f"  Recall:    {ov['recall']:.4f}")
    lines.append(f"  F1:        {ov['f1']:.4f}")

    lines.append("\n=== By Label ===")
    for lbl, m in sorted(results["by_label"].items()):
        if m["accuracy"] is None:
            continue
        lines.append(f"  {lbl:<14} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  n={m['n_samples']}")

    lines.append("\n=== By Scene ===")
    for scene, m in sorted(results["by_scene"].items()):
        if m["accuracy"] is None:
            continue
        lines.append(f"  {scene:<20} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  n={m['n_samples']}")

    lines.append("\n=== By Same Class ===")
    for val, m in sorted(results["by_same_class"].items()):
        if m["accuracy"] is None:
            continue
        lines.append(f"  {val:<14} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  n={m['n_samples']}")

    logger.info("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM mask-pair classification on a saved pair pool."
    )
    parser.add_argument(
        "--pool-path", nargs="+", required=True,
        help="One or more candidate CSV paths from build_qwen_candidates.py (shell globs are expanded).",
    )
    parser.add_argument("--masks-root", required=True, help="Path to replica_masks directory.")
    parser.add_argument("--rgb-root", required=True, help="Path to Replica dataset root.")
    parser.add_argument(
        "--render-mode", type=str, default="outline", choices=["outline", "filled", "raw_mask"],
        help="'outline' draws a colored contour on the frame (default); 'filled' highlights "
             "the region directly on the frame with a semi-transparent color fill; 'raw_mask' "
             "is not yet implemented.",
    )
    parser.add_argument(
        "--highlight-alpha", type=float, default=0.4,
        help="Fill opacity in [0, 1] for --render-mode filled (default: 0.4). Ignored otherwise.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader workers that prefetch/preprocess upcoming batches on CPU "
             "while the GPU generates the current one (default: 2). 0 disables prefetching.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle sample order.")
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config_stage1.yaml")
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max tokens generated per sample (3-line output is short).",
    )
    parser.add_argument(
        "--aggregation", type=str, default="mean",
        choices=["mean", "confidence_weighted", "consensus_rate"],
        help="How to collapse per-variant predictions into one prediction per pair "
             "before computing metrics (default: mean; a no-op for pair_only, which "
             "already renders exactly one sample per pair). 'consensus_rate' votes "
             "by decision fraction thresholded at --connect-threshold, mirroring "
             "MaskClustering's supporter_nums/observer_nums ratio.",
    )
    parser.add_argument(
        "--connect-threshold", type=float, default=0.5,
        help="Decision-fraction threshold for --aggregation consensus_rate (default: 0.5).",
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

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_evaluation(
        pool_paths=expanded_paths,
        masks_root=args.masks_root,
        rgb_root=args.rgb_root,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        output_path=args.output,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        config=config,
        num_workers=args.num_workers,
        render_mode=args.render_mode,
        alpha=args.highlight_alpha,
        aggregation=args.aggregation,
        connect_threshold=args.connect_threshold,
    )
