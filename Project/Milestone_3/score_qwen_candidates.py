"""Score geometric mask-pair candidates with a fine-tuned Qwen3-VL classifier.

This consumes the candidate CSV from build_qwen_candidates.py and writes the
same rows plus classifier scores.  The fine-tuned checkpoint is a PEFT LoRA
adapter plus a binary score head, so the base model is read from --config and
the trained weights are supplied separately with --adapter-path.

The output is append-only and safe to resume after a disconnect: rows already
present in the output CSV are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for path in (start, *start.parents):
        if (path / "MaskClustering").is_dir() and (path / "Project").is_dir():
            return path
    raise RuntimeError("Could not find repository root from current working directory.")


REPO_ROOT = find_repo_root()
MILESTONE_3 = REPO_ROOT / "Project" / "Milestone_3"

if str(MILESTONE_3) not in sys.path:
    sys.path.insert(0, str(MILESTONE_3))

from fine_tuning import build_classifier, make_collate_fn  # noqa: E402
from load_data import RenderedSample, load_frame_view  # noqa: E402


KEY_COLUMNS = ["scene", "frame_a", "mask_a", "frame_b", "mask_b"]
CLASSIFIER_REASONING = "<fine-tuned binary classifier; no text reasoning>"


def resolve_path(path: Path | None, default: Path) -> Path:
    resolved = default if path is None else path
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def row_key(row: pd.Series) -> tuple:
    return tuple(row[col] for col in KEY_COLUMNS)


def existing_keys(output_path: Path) -> set[tuple]:
    if not output_path.exists():
        return set()
    scored = pd.read_csv(output_path)
    if scored.empty:
        return set()
    return {
        tuple(row[col] for col in KEY_COLUMNS)
        for _, row in scored.iterrows()
    }


def append_rows(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        output_path,
        mode="a",
        index=False,
        header=not output_path.exists(),
    )


def build_sample(
    row: pd.Series,
    frame_cache: dict[tuple[str, int], object],
    args: argparse.Namespace,
) -> dict:
    scene = str(row["scene"])
    frame_a = int(row["frame_a"])
    frame_b = int(row["frame_b"])
    mask_a = int(row["mask_a"])
    mask_b = int(row["mask_b"])

    for frame_id in (frame_a, frame_b):
        cache_key = (scene, frame_id)
        if cache_key not in frame_cache:
            frame_cache[cache_key] = load_frame_view(
                scene,
                frame_id,
                masks_root=str(args.masks_root),
                rgb_root=str(args.rgb_root),
            )

    rendered = RenderedSample(
        condition="pair_only",
        frame_a=frame_cache[(scene, frame_a)],
        mask_id_a=mask_a,
        frame_b=frame_cache[(scene, frame_b)],
        mask_id_b=mask_b,
        thickness=args.thickness,
        render_mode=args.render_mode,
        alpha=args.highlight_alpha,
    )

    # make_collate_fn is shared with training and expects labels.  The classifier
    # ignores this dummy label because this script does not request or use loss.
    return {"condition": "pair_only", "images": rendered.render(), "label": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score candidate mask pairs with a fine-tuned Qwen3-VL LoRA classifier."
    )
    parser.add_argument("--scene", default="room0")
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=MILESTONE_3 / "config_stage1.yaml",
        help="Training YAML containing model_id, lora, and head settings.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Saved LoRA adapter + score_head directory.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--sort-by", default=None)
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--render-mode", choices=["outline", "filled"], default="outline")
    parser.add_argument("--highlight-alpha", type=float, default=0.4)
    parser.add_argument("--hf-home", type=Path, default=REPO_ROOT / "Project" / "env" / "hf_cache")
    parser.add_argument("--masks-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--rgb-root", type=Path, default=REPO_ROOT / "data" / "replica")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    args.candidates = resolve_path(
        args.candidates,
        MILESTONE_3 / "data" / "candidates" / f"{args.scene}_qwen_candidates.csv",
    )
    args.output = resolve_path(
        args.output,
        MILESTONE_3 / "data" / "qwen_scores" / f"{args.scene}_qwen_scores.csv",
    )
    args.config = resolve_path(args.config, MILESTONE_3 / "config_stage1.yaml")
    args.adapter_path = resolve_path(args.adapter_path, args.adapter_path)
    args.hf_home = resolve_path(args.hf_home, args.hf_home)
    args.masks_root = resolve_path(args.masks_root, args.masks_root)
    args.rgb_root = resolve_path(args.rgb_root, args.rgb_root)

    os.environ.setdefault("HF_HOME", str(args.hf_home))

    candidates = pd.read_csv(args.candidates)
    missing_columns = [col for col in KEY_COLUMNS if col not in candidates.columns]
    if missing_columns:
        raise ValueError(f"Candidate CSV is missing required columns: {missing_columns}")
    if args.sort_by:
        candidates = candidates.sort_values(args.sort_by, ascending=not args.descending)
    if args.limit is not None:
        candidates = candidates.head(args.limit)

    done = existing_keys(args.output)
    candidates = candidates[
        [row_key(row) not in done for _, row in candidates.iterrows()]
    ]

    print("Candidates input :", args.candidates)
    print("Scores output    :", args.output)
    print("Training config  :", args.config)
    print("Adapter path     :", args.adapter_path)
    print("Already scored   :", len(done))
    print("Remaining        :", len(candidates))
    print("CUDA available   :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU              :", torch.cuda.get_device_name(0))

    with args.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    classifier, processor = build_classifier(
        config,
        condition="pair_only",
        render_mode=args.render_mode,
        adapter_path=str(args.adapter_path),
    )
    classifier.eval()
    collate_fn = make_collate_fn(processor)
    device = next(classifier.score_head.parameters()).device

    frame_cache: dict[tuple[str, int], object] = {}
    pending_rows: list[dict] = []
    processed_since_save = 0

    progress = tqdm(range(0, len(candidates), args.batch_size), desc="Classifier batches")
    for start in progress:
        batch_df = candidates.iloc[start : start + args.batch_size]
        samples: list[dict] = []
        valid_rows: list[pd.Series] = []

        for _, row in batch_df.iterrows():
            try:
                samples.append(build_sample(row, frame_cache, args))
                valid_rows.append(row)
            except Exception as exc:
                out_row = row.to_dict()
                out_row.update(
                    {
                        "qwen_decision": 0,
                        "qwen_confidence": 0.0,
                        "qwen_reasoning": f"<render error: {exc}>",
                        "qwen_same_instance_score": 0.0,
                    }
                )
                pending_rows.append(out_row)

        if samples:
            inputs = collate_fn(samples)
            inputs = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in inputs.items()
                if key != "labels"
            }
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                probabilities = torch.sigmoid(classifier(**inputs).logits).float().cpu().tolist()

            for row, probability in zip(valid_rows, probabilities):
                probability = float(probability)
                out_row = row.to_dict()
                out_row.update(
                    {
                        "qwen_decision": int(probability >= args.threshold),
                        "qwen_confidence": probability,
                        "qwen_reasoning": CLASSIFIER_REASONING,
                        "qwen_same_instance_score": probability,
                    }
                )
                pending_rows.append(out_row)

        processed_since_save += len(batch_df)
        if processed_since_save >= args.save_every:
            append_rows(pending_rows, args.output)
            pending_rows = []
            processed_since_save = 0

    append_rows(pending_rows, args.output)
    print("Done. Scores written to:", args.output)


if __name__ == "__main__":
    main()
