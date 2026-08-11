# Overview

Replaces MaskClustering's geometric mask-merging check with a LoRA fine-tuned `Qwen3-VL` classifier for same-object mask-pair judgment. Only condition: `pair_only` (two mask views, no observer frame). Render modes: `outline` (default), `filled` (`--highlight-alpha`, default 0.4), `raw_mask` (not implemented).

Pipeline: **Build → Load → Fine-Tuning → Evaluation**.

# Build Phase — Candidate Pair CSV

Entry point: [build_qwen_candidates.py](build_qwen_candidates.py). Run once per scene, in the MaskClustering environment.

```bash
python build_qwen_candidates.py \
    --scene room0 \
    --config replica \
    --depth-overlap-threshold 0.05 \
    --train --negative-ratio 3 \
    --output data/candidates/room0_qwen_candidates.csv
```

- `--train`: keep every positive; cap negatives at `--negative-ratio` per positive (default: no cap).
- `--eval`: keep every filter-passing pair (default mode).
- `--filter-mode {iou, any_asymmetric, or_rule}` controls the overlap-pass rule (`--depth-overlap-threshold`, `--vc-threshold`).
- Output: `<scene>_qwen_candidates.csv` + `.summary.json`.

# Load Phase — Rendering Training Samples

Entry point: [load_training_data.py](load_training_data.py). Library only, no CLI.

```python
from load_training_data import load_data_training

samples = load_data_training(
    "data/candidates/room0_qwen_candidates.csv",
    masks_root="/path/to/replica_masks",
    rgb_root="/path/to/replica",
    render_mode="outline",
)
```

Loads the CSV and renders every row — no filtering, sampling, or ratio control. `build_pair_specs()` + `render_spec()` give the lazy-loading equivalent for large pools.

# Fine-Tuning Phase — LoRA Vision Classifier

Entry point: [fine_tuning.py](fine_tuning.py) (`train` / `score`), in the Qwen3-VL environment. Text decoder dropped; images embedded, pooled, concatenated; small trainable head scores same/different; LoRA on later vision-tower blocks only.

```bash
python fine_tuning.py train \
    --pool-path data/candidates/room0_qwen_candidates.csv data/candidates/office0_qwen_candidates.csv \
    --masks-root /path/to/replica_masks \
    --rgb-root /path/to/replica \
    --condition pair_only --render-mode outline \
    --config config_stage1.yaml

python fine_tuning.py score \
    --pool-path data/candidates/office1_qwen_candidates.csv \
    --masks-root /path/to/replica_masks \
    --rgb-root /path/to/replica \
    --adapter-path output/qwen3-vl-lora-stage1 \
    --output results/score_run1.json \
    --config config_stage1.yaml
```

Required flags: `--pool-path`, `--masks-root`, `--rgb-root`, `--condition pair_only`, `--config`. `train` also needs an output-producing config (`sft.output_dir`); `score` needs `--adapter-path` and `--output`.

Leave `--difficulty all` (default) — the curriculum filter is currently a no-op (depends on fields the CSV pipeline doesn't populate).

## Staged training

`train --init-adapter-path <dir>` resumes LoRA + head weights from a prior adapter. Use a separate config per stage (new `sft.output_dir`, smaller `sft.learning_rate`); `model_id`/`head:` must match the original config exactly.

```bash
python fine_tuning.py train \
    --pool-path data/candidates/room0_qwen_candidates.csv data/candidates/office0_qwen_candidates.csv \
    --masks-root /path/to/replica_masks \
    --rgb-root /path/to/replica \
    --condition pair_only --render-mode outline \
    --init-adapter-path output/qwen3-vl-lora-stage1 \
    --config config_stage2.yaml
```

# Evaluation Phase

`fine_tuning.py score` against a held-out CSV (scenes not in `train`'s `--pool-path`).

```bash
python fine_tuning.py score \
    --pool-path data/candidates/office1_qwen_candidates.csv \
    --masks-root /path/to/replica_masks \
    --rgb-root /path/to/replica \
    --adapter-path output/qwen3-vl-lora-stage1 \
    --output results/score_run1.json \
    --config config_stage1.yaml
```

Outputs one JSON at `--output`: `overall`, `by_label`, `by_scene`, `by_same_class`, plus `per_sample`/`per_pair` dumps.
