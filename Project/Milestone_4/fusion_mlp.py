"""Milestone 4: learned geometric-semantic fusion MLP.

Fuses MaskClustering's geometric per-pair signals (view consensus score,
observer count, 3D overlap) with the Qwen3-VL confidence score into a single
same-instance probability per candidate mask pair.

Input CSVs are the output of score_qwen_candidates.py (which preserves all
columns from build_qwen_candidates.py, including the Replica GT
`same_instance_label`, and adds `qwen_same_instance_score`). One file per
scene therefore contains features and labels together.

Usage:
    # train (6 scenes) + validate (office4)
    python fusion_mlp.py train \
        --train-scores scores/room1.csv scores/room2.csv scores/office0.csv \
                       scores/office1.csv scores/office2.csv scores/office3.csv \
        --val-scores scores/office4.csv \
        --feature-set fused \
        --model-out output/fusion_mlp_fused.pt

    # score test scene (room0) -> CSV for run_qwen_clustering.py --scores
    python fusion_mlp.py score \
        --scores scores/room0.csv \
        --model output/fusion_mlp_fused.pt \
        --output scores/room0_fused.csv

Feature sets (for the ablation):
    geometry : view_consensus_score, log1p(num_observers), depth_iou,
               overlap_a_to_b, overlap_b_to_a
    qwen     : qwen_same_instance_score
    fused    : geometry + qwen

Runs on CPU in seconds; works in either project environment (torch + pandas).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KEY_COLUMNS = ["frame_a", "mask_a", "frame_b", "mask_b"]
LABEL_COLUMN = "same_instance_label"

GEOMETRY_FEATURES = [
    "view_consensus_score",
    "log1p_num_observers",  # derived from num_observers
    "depth_iou",
    "overlap_a_to_b",
    "overlap_b_to_a",
]
QWEN_FEATURES = ["qwen_same_instance_score"]

FEATURE_SETS = {
    "geometry": GEOMETRY_FEATURES,
    "qwen": QWEN_FEATURES,
    "fused": GEOMETRY_FEATURES + QWEN_FEATURES,
}


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_frames(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["__source"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_features(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    df = df.copy()
    if "log1p_num_observers" in feature_names:
        df["log1p_num_observers"] = np.log1p(df["num_observers"].astype(float))
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing feature columns: {missing}")
    features = df[feature_names].astype(float)
    if features.isna().any().any():
        na_counts = features.isna().sum()
        raise ValueError(f"NaN feature values found:\n{na_counts[na_counts > 0]}")
    return features.to_numpy(dtype=np.float32)


def build_labels(df: pd.DataFrame) -> np.ndarray:
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Input CSV has no {LABEL_COLUMN!r} column.")
    labels = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    return labels.to_numpy(dtype=np.float32)  # NaN where GT was unavailable


def drop_unlabeled(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = ~np.isnan(labels)
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info("Dropping %d pairs without GT label.", n_dropped)
    return features[keep], labels[keep]


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U); avoids a sklearn dependency."""
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    df = pd.DataFrame({"score": y_score, "rank": ranks})
    ranks = df.groupby("score")["rank"].transform("mean").to_numpy()
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc(y_true, prob),
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_names = FEATURE_SETS[args.feature_set]

    train_df = load_frames(args.train_scores)
    x_train, y_train = drop_unlabeled(build_features(train_df, feature_names), build_labels(train_df))
    val_df = load_frames(args.val_scores)
    x_val, y_val = drop_unlabeled(build_features(val_df, feature_names), build_labels(val_df))

    # Standardize with training statistics only.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-8
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0))
    logger.info(
        "Train: %d pairs (%d pos / %d neg, pos_weight=%.2f) | Val: %d pairs (%d pos)",
        len(y_train), int(n_pos), int(n_neg), pos_weight.item(), len(y_val), int(y_val.sum()),
    )

    model = FusionMLP(len(feature_names), args.hidden_dim, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)

    best_auc = -1.0
    best_state = None
    patience_left = args.patience
    for epoch in range(args.max_epochs):
        model.train()
        perm = torch.randperm(len(x_train_t))
        epoch_loss = 0.0
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(x_val_t)).numpy()
        val_auc = roc_auc(y_val, val_prob)
        if epoch % 10 == 0 or val_auc > best_auc:
            logger.info(
                "epoch %3d  train_loss %.4f  val_auc %.4f", epoch, epoch_loss / len(perm), val_auc
            )
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d (best val AUC %.4f).", epoch, best_auc)
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(x_val_t)).numpy()
    metrics = binary_metrics(y_val, val_prob)
    logger.info("Validation metrics (thr 0.5): %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_set": args.feature_set,
            "feature_names": feature_names,
            "mean": mean,
            "std": std,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "val_metrics": metrics,
        },
        args.model_out,
    )
    logger.info("Model saved to %s", args.model_out)


def score(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    feature_names = checkpoint["feature_names"]
    model = FusionMLP(len(feature_names), checkpoint["hidden_dim"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    df = load_frames(args.scores)
    features = build_features(df, feature_names)
    features = (features - checkpoint["mean"]) / checkpoint["std"]
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(features))).numpy()

    out = df[[c for c in ["scene", *KEY_COLUMNS] if c in df.columns]].copy()
    out["qwen_same_instance_score"] = prob  # column name expected by run_qwen_clustering.py

    labels = build_labels(df)
    labeled = ~np.isnan(labels)
    if labeled.any():
        metrics = binary_metrics(labels[labeled], prob[labeled])
        logger.info("Metrics on labeled pairs: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.info("Wrote %d fused pair scores to %s", len(out), args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/score the geometric-semantic fusion MLP.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--train-scores", type=Path, nargs="+", required=True, help="Scored candidate CSVs for training scenes.")
    p_train.add_argument("--val-scores", type=Path, nargs="+", required=True, help="Scored candidate CSVs for validation scene(s).")
    p_train.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="fused")
    p_train.add_argument("--model-out", type=Path, required=True)
    p_train.add_argument("--hidden-dim", type=int, default=32)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--max-epochs", type=int, default=500)
    p_train.add_argument("--patience", type=int, default=30, help="Early-stopping patience in epochs (on val AUC).")
    p_train.add_argument("--seed", type=int, default=42)

    p_score = subparsers.add_parser("score")
    p_score.add_argument("--scores", type=Path, nargs="+", required=True, help="Scored candidate CSV(s) for the test scene.")
    p_score.add_argument("--model", type=Path, required=True)
    p_score.add_argument("--output", type=Path, required=True, help="CSV for run_qwen_clustering.py --scores.")

    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
