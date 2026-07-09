"""Score geometric mask-pair candidates with Qwen3-VL.

This script consumes the candidate CSV from build_qwen_candidates.py and writes
the same rows plus Qwen scores. It is designed to be safe to resume after a
disconnect: existing scored rows are loaded and skipped.

Run in the qwen_env environment.
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
MILESTONE_4 = REPO_ROOT / "Project" / "Milestone_4"

if str(MILESTONE_4) not in sys.path:
    sys.path.insert(0, str(MILESTONE_4))

from inference import run_inference_batch  # noqa: E402
from load_data import RenderedSample, load_frame_view  # noqa: E402
from vlm_utils import load_model  # noqa: E402


KEY_COLUMNS = ["scene", "frame_a", "mask_a", "frame_b", "mask_b"]
QWEN_COLUMNS = [
    "qwen_decision",
    "qwen_confidence",
    "qwen_reasoning",
    "qwen_same_instance_score",
]


def resolve_path(path: Path | None, default: Path) -> Path:
    resolved = default if path is None else path
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def default_config_path() -> Path:
    milestone3_config = MILESTONE_3 / "config_qwen3vl_2b.yaml"
    if milestone3_config.exists():
        return milestone3_config
    return MILESTONE_4 / "config.yaml"


def row_key(row: pd.Series) -> tuple:
    return tuple(row[col] for col in KEY_COLUMNS)


def existing_keys(output_path: Path) -> set[tuple]:
    if not output_path.exists():
        return set()
    scored = pd.read_csv(output_path)
    if scored.empty:
        return set()
    return set(tuple(row[col] for col in KEY_COLUMNS) for _, row in scored.iterrows())


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


def build_sample(row: pd.Series, frame_cache: dict[int, object], args: argparse.Namespace):
    scene = str(row["scene"])
    frame_a = int(row["frame_a"])
    frame_b = int(row["frame_b"])
    mask_a = int(row["mask_a"])
    mask_b = int(row["mask_b"])

    for frame_id in (frame_a, frame_b):
        if frame_id not in frame_cache:
            frame_cache[frame_id] = load_frame_view(
                scene,
                frame_id,
                masks_root=str(args.masks_root),
                rgb_root=str(args.rgb_root),
            )

    return RenderedSample(
        condition=args.condition,
        frame_a=frame_cache[frame_a],
        mask_id_a=mask_a,
        frame_b=frame_cache[frame_b],
        mask_id_b=mask_b,
        thickness=args.thickness,
    )


def qwen_score(prediction: dict) -> float:
    if prediction["reasoning"] == "<parse error>":
        return 0.0
    if int(prediction["decision"]) == 1:
        return float(prediction["confidence"])
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Qwen candidate mask pairs.")
    parser.add_argument("--scene", default="room0")
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="Qwen YAML config.")
    parser.add_argument("--condition", default="pair_only", choices=["pair_only"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--sort-by", default=None, help="Optional score column for sorting, e.g. depth_iou.")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--hf-home", type=Path, default=REPO_ROOT / "Project" / "env" / "hf_cache")
    parser.add_argument("--masks-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--rgb-root", type=Path, default=REPO_ROOT / "data" / "replica")
    args = parser.parse_args()

    args.candidates = resolve_path(
        args.candidates,
        MILESTONE_3 / "data" / "candidates" / f"{args.scene}_qwen_candidates.csv",
    )
    args.output = resolve_path(
        args.output,
        MILESTONE_3 / "data" / "qwen_scores" / f"{args.scene}_qwen_scores.csv",
    )
    args.config = resolve_path(args.config, default_config_path())
    args.hf_home = resolve_path(args.hf_home, args.hf_home)
    args.masks_root = resolve_path(args.masks_root, args.masks_root)
    args.rgb_root = resolve_path(args.rgb_root, args.rgb_root)

    os.environ.setdefault("HF_HOME", str(args.hf_home))

    candidates = pd.read_csv(args.candidates)
    if args.sort_by:
        candidates = candidates.sort_values(args.sort_by, ascending=not args.descending)
    if args.limit is not None:
        candidates = candidates.head(args.limit)

    done = existing_keys(args.output)
    candidates = candidates[[row_key(row) not in done for _, row in candidates.iterrows()]]

    print("Candidates input :", args.candidates)
    print("Scores output    :", args.output)
    print("Qwen config      :", args.config)
    print("Already scored   :", len(done))
    print("Remaining        :", len(candidates))
    print("CUDA available   :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU              :", torch.cuda.get_device_name(0))

    with args.config.open() as f:
        config = yaml.safe_load(f)
    model, processor = load_model(config)

    frame_cache: dict[int, object] = {}
    pending_rows: list[dict] = []
    processed_since_save = 0

    progress = tqdm(range(0, len(candidates), args.batch_size), desc="Qwen batches")
    for start in progress:
        batch_df = candidates.iloc[start:start + args.batch_size]
        samples = []
        valid_rows = []

        for _, row in batch_df.iterrows():
            try:
                sample = build_sample(row, frame_cache, args)
                samples.append({"condition": sample.condition, "images": sample.render()})
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

        predictions = run_inference_batch(
            samples,
            model=model,
            processor=processor,
            max_new_tokens=int(config["inference"]["max_new_tokens"]),
        )

        for row, prediction in zip(valid_rows, predictions):
            out_row = row.to_dict()
            out_row.update(
                {
                    "qwen_decision": int(prediction["decision"]),
                    "qwen_confidence": float(prediction["confidence"]),
                    "qwen_reasoning": prediction["reasoning"],
                    "qwen_same_instance_score": qwen_score(prediction),
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
