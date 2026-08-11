"""LoRA fine-tuning of the Qwen3-VL vision encoder for same-object mask-pair classification.

Bypasses the text decoder entirely: rendered images are embedded independently via the
vision tower's get_image_features(), mean-pooled per image, concatenated in a fixed
per-condition order, and scored by a small trainable MLP head (sigmoid -> 0/1).
Only the vision tower's later blocks (LoRA) and the head are trainable; the 8B decoder
is deleted from memory right after loading since it's never used in this design.

Run in the Qwen3-VL environment (qwen_env.sh).

Usage:
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
"""

import argparse
import glob
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from evaluate import _group_metrics, aggregate_pair_predictions
from evaluate import compute_metrics as binary_compute_metrics
from load_training_data import build_pair_specs, load_data_training, load_pool, render_spec
from prompt_templates import _EXPECTED_IMAGE_COUNTS, ConditionType, RenderMode
from vlm_utils import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_CONDITIONS: List[ConditionType] = ["pair_only"]

# Vision block submodule names, confirmed against transformers' modeling_qwen3_vl.py:
# Qwen3VLVisionAttention has a fused qkv Linear + a proj Linear (not separate q/k/v like
# the decoder); Qwen3VLVisionMLP and Qwen3VLVisionPatchMerger both use linear_fc1/linear_fc2.
_VISION_BLOCK_LORA_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")


class VisionScoreHead(nn.Module):
    """Binary classification MLP on top of concatenated per-image pooled embeddings."""

    def __init__(self, in_dim: int, mlp_hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AttentionPool(nn.Module):
    """Learned per-token attention weighting, replacing plain mean pooling over a single
    image's patch tokens. One shared instance is applied independently to each image's
    token set (not per-condition-slot, not per-pair)."""

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [T, D]
        scores = self.score(tokens)                  # [T, 1]
        weights = torch.softmax(scores, dim=0)       # [T, 1]
        pooled = torch.sum(weights * tokens, dim=0)  # [D]
        return pooled


def _condition_image_count(condition: ConditionType, render_mode: RenderMode) -> int:
    return _EXPECTED_IMAGE_COUNTS[condition][render_mode]


def _vision_lora_target_modules(vision_tower: nn.Module, layers: str, last_n: int) -> List[str]:
    """Relative-suffix module names for LoraConfig.target_modules.

    Plain suffixes (not regex) so "attn.proj" can't accidentally match
    patch_embed.proj (a Conv3d, unsupported/unwanted for LoRA here) — the full
    per-block relative path is required to match.
    """
    n_blocks = len(vision_tower.blocks)
    if layers == "all":
        indices = range(n_blocks)
    elif layers == "last_n":
        indices = range(max(0, n_blocks - last_n), n_blocks)
    else:
        raise ValueError(f"Unknown vision_lora_layers {layers!r}. Expected 'all' or 'last_n'.")

    targets = [f"blocks.{i}.{suffix}" for i in indices for suffix in _VISION_BLOCK_LORA_SUFFIXES]
    targets += ["merger.linear_fc1", "merger.linear_fc2"]
    return targets


class QwenVisionLoraClassifier(PeftModel):
    """PEFT-wrapped Qwen3-VL vision tower + a trainable classification head.

    Never touches the decoder/lm_head (both are deleted from the wrapped model before
    this is constructed). forward() takes pre-processed image tensors directly.
    """

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        per_sample_counts: List[int],
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        base = self.get_base_model()
        # get_image_features() returns a ModelOutput, not a plain per-image tensor list;
        # the actual (patch_merge_size-split) per-image token tensors are its
        # pooler_output field — see Qwen3VLModel.get_image_features(). Asserted explicitly
        # since this is transformers-version-sensitive Qwen3VL-specific behavior (unlike
        # ModelOutput's dict-like iteration, which is stable) — a bare AttributeError here
        # would otherwise be as opaque as the bug this replaced.
        image_outputs = base.get_image_features(pixel_values, image_grid_thw)
        if image_outputs.pooler_output is None:
            raise AttributeError(
                f"get_image_features() output has no pooler_output (keys: {list(image_outputs.keys())}). "
                "The installed transformers version may have changed Qwen3VLModel.get_image_features()'s "
                "return shape — inspect the keys above to find the per-image embeddings field."
            )
        pooled = torch.stack([base.attn_pool(tokens) for tokens in image_outputs.pooler_output])  # (total_images, out_hidden)

        n = per_sample_counts[0]
        if any(c != n for c in per_sample_counts):
            raise ValueError(
                f"All samples in a batch must have the same image count (fixed by condition/render_mode), "
                f"got {per_sample_counts}. Build the Dataset with a single condition/render_mode."
            )
        batch_size = len(per_sample_counts)
        grouped = pooled.reshape(batch_size, n * pooled.shape[-1])

        logits = base.score_head(grouped)

        loss = None
        if labels is not None:
            target = labels.to(logits.dtype)
            smoothing = getattr(self, "label_smoothing", 0.0)
            if smoothing:
                target = target * (1.0 - smoothing) + 0.5 * smoothing
            loss = F.binary_cross_entropy_with_logits(logits, target)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_classifier(
    config: dict,
    condition: ConditionType,
    render_mode: RenderMode,
    adapter_path: Optional[str] = None,
    is_trainable: bool = False,
) -> Tuple[QwenVisionLoraClassifier, "transformers.AutoProcessor"]:
    """Load Qwen3-VL, strip the decoder, attach the score head, and wrap the vision
    tower's later blocks with LoRA — or restore a previously trained adapter+head.

    is_trainable must be set when resuming a restored adapter for further training.
    """
    model, processor = load_model(config)

    n_images = _condition_image_count(condition, render_mode)
    out_hidden_size = model.config.vision_config.out_hidden_size
    head_cfg = config["head"]
    model.score_head = VisionScoreHead(
        in_dim=n_images * out_hidden_size,
        mlp_hidden_dim=head_cfg["mlp_hidden_dim"],
        dropout=head_cfg["dropout"],
    )
    model.attn_pool = AttentionPool(out_hidden_size)

    lora_cfg = config["lora"]
    target_modules = _vision_lora_target_modules(
        model.model.visual,
        layers=lora_cfg["vision_lora_layers"],
        last_n=lora_cfg["last_n_layers"],
    )
    if not target_modules:
        raise RuntimeError("No vision LoRA target modules resolved — check vision_lora_layers/last_n_layers.")

    # task_type is intentionally left unset: PeftModel.from_pretrained() only substitutes
    # a different wrapper class when config.task_type matches one of PEFT's built-in task
    # types, otherwise it instantiates `cls` (our QwenVisionLoraClassifier) directly.
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=target_modules,
        modules_to_save=["score_head", "attn_pool"],
    )

    # PEFT's adapter injection unconditionally calls model.get_input_embeddings() to check
    # for tied embedding layers (irrelevant here — target_modules/modules_to_save never
    # touch embeddings), which requires model.model.language_model to still exist. So the
    # decoder/lm_head deletion below must happen *after* wrapping, not before.
    if adapter_path is not None:
        classifier = QwenVisionLoraClassifier.from_pretrained(model, adapter_path, is_trainable=is_trainable)
    else:
        classifier = QwenVisionLoraClassifier(model, peft_config)

    # Decoder/lm_head are never used in this vision-only design — free them now that PEFT
    # is done with them. (load_model()'s device_map="auto" already placed them on the GPU
    # during from_pretrained(); this frees that memory back as soon as PEFT allows it.)
    base = classifier.get_base_model()
    del base.model.language_model
    del base.lm_head
    torch.cuda.empty_cache()

    # Only move score_head/attn_pool, not the whole classifier: load_model()'s device_map="auto"
    # already placed the base model's weights via accelerate's dispatch hooks, and calling
    # .to(device) on the full PEFT-wrapped classifier walks those too, raising "Cannot copy out
    # of meta tensor" for any parameter accelerate hasn't materialized (still a meta placeholder
    # from low_cpu_mem_usage lazy init). score_head/attn_pool are plain nn.Modules created fresh
    # above, so they default to CPU and genuinely need moving; LoRA's own injected weights
    # already match their target module's device, so they don't.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier.score_head.to(device)
    classifier.attn_pool.to(device)

    # Absent in older configs (e.g. an --init-adapter-path stage's own --config) -> no smoothing.
    classifier.label_smoothing = config.get("loss", {}).get("label_smoothing", 0.0)

    classifier.print_trainable_parameters()
    return classifier, processor


class _EpochTrainingMetricsCallback(TrainerCallback):
    """Flush a TrainingMetricsTrainer's accumulated counts at each epoch boundary."""

    def __init__(self, trainer: "TrainingMetricsTrainer"):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.log_epoch_training_metrics()
        return control


class TrainingMetricsTrainer(Trainer):
    """Trainer that logs epoch-level training metrics without another inference pass.

    Confusion-matrix counts are accumulated from the logits already produced for the
    training loss and reset at the end of each epoch. Consequently these metrics describe
    all batches seen during that epoch, evaluated by the changing model state that saw
    each batch; they are not a post-epoch evaluation using one frozen model state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_confusion: Optional[torch.Tensor] = None
        self.add_callback(_EpochTrainingMetricsCallback(self))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )

        if model.training and labels is not None:
            with torch.no_grad():
                predictions = outputs.logits.detach().reshape(-1) >= 0
                targets = labels.detach().reshape(-1).to(torch.bool)
                counts = torch.stack(
                    (
                        (predictions & targets).sum(),       # TP
                        (predictions & ~targets).sum(),      # FP
                        (~predictions & ~targets).sum(),     # TN
                        (~predictions & targets).sum(),      # FN
                    )
                )
                if self._training_confusion is None:
                    self._training_confusion = counts
                else:
                    self._training_confusion += counts

        return (loss, outputs) if return_outputs else loss

    def _training_metric_values(self, prefix: str) -> dict:
        if self._training_confusion is None:
            return {}

        # reduce() may perform an in-place collective depending on the Accelerate
        # version/backend. Reduce a clone so periodic snapshots never alter the local
        # epoch accumulator used by later batches.
        counts = self.accelerator.reduce(self._training_confusion.clone(), reduction="sum")
        tp, fp, tn, fn = (int(value) for value in counts.tolist())
        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        tpr = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        return {
            f"{prefix}_accuracy": (tp + tn) / total if total else 0.0,
            f"{prefix}_precision": precision,
            f"{prefix}_tpr": tpr,
            f"{prefix}_tnr": tnr,
            f"{prefix}_f1": (
                2.0 * precision * tpr / (precision + tpr)
                if precision + tpr
                else 0.0
            ),
        }

    def log(self, logs: dict, *args, **kwargs) -> None:
        # Periodic Trainer loss logs get a snapshot accumulated from the start of the
        # current epoch. Do not reset here: the same counts feed the final epoch metrics.
        if "loss" in logs:
            logs.update(self._training_metric_values("running"))
        super().log(logs, *args, **kwargs)

    def log_epoch_training_metrics(self) -> None:
        metrics = self._training_metric_values("epoch")
        if not metrics:
            return
        self.log(metrics)
        self._training_confusion = None


class PairImageDataset(torch.utils.data.Dataset):
    """Wraps load_data_training() output for a single (condition, render_mode)."""

    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class LazyPairImageDataset(torch.utils.data.Dataset):
    """Like PairImageDataset, but samples are metadata-only specs (from
    load_training_data.build_pair_specs()) -- __getitem__ renders on demand via
    render_spec() instead of expecting "images" to already be populated. This is what
    keeps memory bounded to a batch's worth of images for large (e.g. ~60k-pair) sample
    counts, where render_pairs()'s eager whole-set rendering would OOM.

    On a render failure, logs a warning and returns None rather than substituting a
    neighboring sample -- make_collate_fn()'s collate_fn filters None entries out of the
    batch. Not safe to use where predictions must positionally match an original sample
    list one-to-one (e.g. train()'s compute_metrics reading val_samples[i]) -- a dropped
    sample shifts every later prediction out of alignment silently. Use PairImageDataset
    (eager) for that case instead.
    """

    def __init__(self, specs: List[dict], masks_root: str, rgb_root: str):
        self.specs = specs
        self.masks_root = masks_root
        self.rgb_root = rgb_root

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> Optional[dict]:
        spec = self.specs[idx]
        images = render_spec(spec, self.masks_root, self.rgb_root)
        if images is None:
            logger.warning(
                "LazyPairImageDataset: render failed idx=%d (scene=%s, frame_a=%s, frame_b=%s) -> dropping",
                idx, spec.get("scene"), spec.get("frame_id_a"), spec.get("frame_id_b"),
            )
            return None
        return {**spec, "images": images}


def _make_dataset(samples: List[dict], masks_root: str, rgb_root: str) -> torch.utils.data.Dataset:
    """samples may be already-rendered dicts (has "images") or metadata-only specs (from
    build_pair_specs(), no "images") -- detected off the first element."""
    if samples and "images" not in samples[0]:
        return LazyPairImageDataset(samples, masks_root, rgb_root)
    return PairImageDataset(samples)


def make_collate_fn(processor):
    def collate_fn(batch: List[Optional[dict]]) -> Dict[str, object]:
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            raise RuntimeError(
                "collate_fn: every sample in this batch failed to render (LazyPairImageDataset "
                "returned None for all of them) -- this looks systemic (bad masks_root/rgb_root, "
                "corrupted pool), not an isolated missing frame."
            )

        flat_images: List[PILImage.Image] = []
        per_sample_counts: List[int] = []
        for sample in batch:
            flat_images.extend(sample["images"])
            per_sample_counts.append(len(sample["images"]))

        proc_out = processor(images=flat_images, return_tensors="pt")
        labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.float32)
        return {
            "pixel_values": proc_out["pixel_values"],
            "image_grid_thw": proc_out["image_grid_thw"],
            "per_sample_counts": per_sample_counts,
            "labels": labels,
        }

    return collate_fn


def _expand_pool_paths(patterns: List[str]) -> List[str]:
    expanded: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            expanded.extend(sorted(matches))
        else:
            expanded.append(pattern)  # keep as-is so the error is clear
    return expanded


def _log_gt_instance_coverage(
    pool_paths: List[str],
    samples: List[dict],
) -> None:
    """Compare distinct (scene, GT instance) ids that have >=1 valid positive pair in
    the pool against the ones that actually made it into the rendered samples.

    GT instance ids are only unique *within* a scene (Replica's label_id*1000 +
    instance_index scheme is re-derived per scene), so coverage is tracked by
    (scene, gt_id) pairs, not bare gt_id, to avoid conflating instances across scenes.

    This catches render()/missing-frame failures silently dropping a pair (logged only
    as a warning elsewhere). It can NOT catch instances that never had a valid pair in
    the pool to begin with — those were already filtered out by build_qwen_candidates.py.
    """
    pool_instances: set = set()
    for path in pool_paths:
        positives, _ = load_pool(path)
        pool_instances.update((m.scene, m.gt_id_a) for m in positives)

    sampled_instances = {
        (s["scene"], s["gt_id_a"]) for s in samples if s["label"] == 1
    }
    missing = sorted(pool_instances - sampled_instances)

    n_pool = len(pool_instances)
    n_sampled = len(sampled_instances)
    coverage_pct = 100.0 * n_sampled / n_pool if n_pool else 0.0
    logger.info(
        "GT instance coverage: %d/%d (%.1f%%) of instances with a valid positive pair "
        "made it into the rendered training set.",
        n_sampled, n_pool, coverage_pct,
    )
    if missing:
        preview = missing[:10]
        logger.warning(
            "%d instance(s) with a valid positive pair were NOT rendered "
            "(render() failed — check preceding warnings for the reason): %s%s",
            len(missing), preview, " ..." if len(missing) > 10 else "",
        )


def _is_easy(sample: dict, easy_min_angle_deg: float, easy_max_angle_deg: float) -> Optional[bool]:
    """Curriculum easiness check for the --difficulty train-time filter.

    positives: angle_deg within [easy_min_angle_deg, easy_max_angle_deg] (a band, not just
    a ceiling, so near-duplicate/degenerate small-angle views can be excluded from "easy"
    too if desired -- default easy_min_angle_deg=0.0 makes it a plain ceiling).
    negatives: neg_strategy == "co_visible" (the non-adversarial strategy) is easy,
    "both_views" (hard negative mining) is not.

    Returns None when positive angle_deg is missing -- unresolvable, so the sample is
    excluded from BOTH the "easy" and "hard" buckets rather than guessed into one.
    """
    if sample["label"] == 1:
        angle = sample.get("angle_deg")
        if angle is None:
            return None
        return easy_min_angle_deg <= angle <= easy_max_angle_deg
    return sample.get("neg_strategy") == "co_visible"


def _filter_by_difficulty(
    samples: List[dict],
    difficulty: str,
    easy_min_angle_deg: float,
    easy_max_angle_deg: float,
) -> List[dict]:
    """Curriculum filter for train()/score(): 'all' (default) is a no-op. 'easy' keeps
    only samples where _is_easy() is True; 'hard' keeps only samples where it's False
    (i.e. positives outside the easy angle band, negatives from 'both_views' mining).
    Samples with unresolvable easiness (missing angle_deg) are dropped from both."""
    if difficulty == "all":
        return samples
    want_easy = difficulty == "easy"
    filtered = [s for s in samples if _is_easy(s, easy_min_angle_deg, easy_max_angle_deg) is want_easy]
    n_pos = sum(1 for s in filtered if s["label"] == 1)
    n_neg = len(filtered) - n_pos
    logger.info(
        "Difficulty filter %r (angle band [%.1f, %.1f] deg): kept %d/%d samples (%d pos / %d neg).",
        difficulty, easy_min_angle_deg, easy_max_angle_deg, len(filtered), len(samples), n_pos, n_neg,
    )
    return filtered


def _load_samples(
    pool_paths: List[str],
    masks_root: str,
    rgb_root: str,
    render_mode: RenderMode,
    alpha: float = 0.4,
    difficulty: str = "all",
    easy_min_angle_deg: float = 0.0,
    easy_max_angle_deg: float = 45.0,
) -> List[dict]:
    all_samples: List[dict] = []
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
    logger.info("Total samples loaded: %d", len(all_samples))
    _log_gt_instance_coverage(pool_paths, all_samples)
    return _filter_by_difficulty(all_samples, difficulty, easy_min_angle_deg, easy_max_angle_deg)


def train(
    args: argparse.Namespace,
    train_samples: Optional[List[dict]] = None,
    val_samples: Optional[List[dict]] = None,
) -> None:
    """train_samples/val_samples let a caller pass pre-selected/curated data directly,
    bypassing _load_samples()'s CLI-arg-driven loading (e.g. args.pool_path is unused
    when train_samples is given)."""
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    classifier, processor = build_classifier(
        config, args.condition, args.render_mode,
        adapter_path=args.init_adapter_path, is_trainable=True,
    )
    collate_fn = make_collate_fn(processor)

    if train_samples is None:
        train_samples = _load_samples(
            args.pool_path, args.masks_root, args.rgb_root, args.render_mode,
            alpha=args.highlight_alpha,
            difficulty=args.difficulty,
            easy_min_angle_deg=args.easy_min_angle_deg, easy_max_angle_deg=args.easy_max_angle_deg,
        )
    train_dataset = _make_dataset(train_samples, args.masks_root, args.rgb_root)

    eval_dataset = None
    if val_samples is not None:
        eval_dataset = _make_dataset(val_samples, args.masks_root, args.rgb_root)
        if isinstance(eval_dataset, LazyPairImageDataset):
            logger.warning(
                "eval_dataset is a lazy spec dataset: a render failure drops that sample, "
                "which shifts every later prediction out of position-based alignment with "
                "val_samples in compute_metrics() below -- silently corrupting the "
                "pos/neg and hard/easy breakdowns, not just the overall metrics. Prefer "
                "eager val_samples (render_pairs()) unless render failures are known to be "
                "zero for this set."
            )
    elif args.val_pool_path:
        val_samples = _load_samples(
            args.val_pool_path, args.masks_root, args.rgb_root, args.render_mode,
            alpha=args.highlight_alpha,
            difficulty=args.difficulty,
            easy_min_angle_deg=args.easy_min_angle_deg, easy_max_angle_deg=args.easy_max_angle_deg,
        )
        eval_dataset = PairImageDataset(val_samples)

    sft_cfg = dict(config["sft"])
    if eval_dataset is None:
        # Trainer raises if eval_strategy isn't "no" and no eval_dataset was passed,
        # regardless of what config_stage1.yaml's sft.eval_strategy/eval_steps say.
        sft_cfg["eval_strategy"] = "no"
    training_args = TrainingArguments(**sft_cfg)

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
        preds = (probs >= 0.5).astype(int).tolist()
        metrics = binary_compute_metrics(np.asarray(labels).astype(int).tolist(), preds)

        # Breakdown mirroring score()'s by_label reporting, so per-epoch eval logs
        # surface positive/negative accuracy too, not just the aggregate numbers above.
        # Trainer's eval dataloader is sequential (no shuffling), so `preds` lines up
        # 1:1 with `val_samples`.
        pred_dicts = [{"decision": p} for p in preds]
        by_label = _group_metrics(val_samples, pred_dicts, lambda s: "positive" if s["label"] == 1 else "negative")
        for group_name, group_metrics in by_label.items():
            metrics[f"{group_name}_accuracy"] = group_metrics["accuracy"]
            metrics[f"{group_name}_n"] = group_metrics["n_samples"]

        return metrics

    trainer = TrainingMetricsTrainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )
    trainer.train()

    # Explicit save (not just trainer.save_model()) to guarantee only the LoRA
    # adapter + score_head are persisted, regardless of Trainer's PEFT auto-detection.
    classifier.save_pretrained(training_args.output_dir)
    logger.info("Saved adapter + head to %s", training_args.output_dir)


def score(args: argparse.Namespace, samples: Optional[List[dict]] = None) -> None:
    """samples lets a caller pass pre-selected/curated data directly, bypassing
    _load_samples()'s CLI-arg-driven loading (e.g. args.pool_path is unused when given)."""
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    classifier, processor = build_classifier(
        config, args.condition, args.render_mode, adapter_path=args.adapter_path,
    )
    classifier.eval()
    collate_fn = make_collate_fn(processor)

    if samples is None:
        samples = _load_samples(
            args.pool_path, args.masks_root, args.rgb_root, args.render_mode,
            alpha=args.highlight_alpha,
            difficulty=args.difficulty,
            easy_min_angle_deg=args.easy_min_angle_deg, easy_max_angle_deg=args.easy_max_angle_deg,
        )

    device = next(classifier.parameters()).device
    all_probs: List[float] = []
    # score() has no Dataset/DataLoader (unlike train()), so a lazy spec list needs its own
    # per-batch render step here -- mirrors LazyPairImageDataset.__getitem__, but drops
    # (rather than returning None for collate_fn to filter) failed specs, rebuilding
    # `samples` from what was actually scored so every zip(samples, preds) below the loop
    # stays correctly aligned. No-op when samples are already eager (rendered) dicts.
    scored_samples: List[dict] = []
    # Trainer wraps forward passes in bf16 autocast when sft.bf16 is set (config_stage1.yaml),
    # which is what lets train()'s per-epoch eval tolerate score_head's float32 weights against
    # the bf16 vision-tower activations (load_model() always loads Qwen3-VL in bfloat16). This
    # standalone loop has no Trainer, so it needs the same autocast or that mismatch raises
    # RuntimeError: mat1 and mat2 must have the same dtype.
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for i in tqdm(range(0, len(samples), args.batch_size), desc="Scoring", unit="batch"):
            batch = samples[i : i + args.batch_size]
            if batch and "images" not in batch[0]:
                rendered_batch = []
                for spec in batch:
                    images = render_spec(spec, args.masks_root, args.rgb_root)
                    if images is None:
                        logger.warning(
                            "score(): render failed (scene=%s frame_a=%s frame_b=%s) -> dropping",
                            spec.get("scene"), spec.get("frame_id_a"), spec.get("frame_id_b"),
                        )
                        continue
                    rendered_batch.append({**spec, "images": images})
                batch = rendered_batch
            if not batch:
                continue
            scored_samples.extend(batch)
            inputs = collate_fn(batch)
            inputs = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
                if k != "labels"
            }
            outputs = classifier(**inputs)
            all_probs.extend(torch.sigmoid(outputs.logits).float().cpu().tolist())

    samples = scored_samples  # no-op reassignment when samples were already eager dicts
    preds = [{"decision": 1 if p >= 0.5 else 0, "probability": p} for p in all_probs]

    # No-op for pair_only (one sample per pair already); kept so results stay directly
    # comparable to evaluate.py's zero-shot aggregation format.
    agg_samples, agg_preds = aggregate_pair_predictions(
        samples, preds, mode=args.aggregation, connect_threshold=args.connect_threshold,
    )
    logger.info("Aggregated %d rendered samples into %d unique pairs", len(samples), len(agg_samples))

    y_true = [s["label"] for s in agg_samples]
    y_pred = [p["decision"] for p in agg_preds]

    results = {
        "config": {
            "pool_paths": args.pool_path,
            "condition": args.condition,
            "render_mode": args.render_mode,
            "highlight_alpha": args.highlight_alpha,
            "adapter_path": args.adapter_path,
            "aggregation": args.aggregation,
            "connect_threshold": args.connect_threshold,
        },
        "overall": binary_compute_metrics(y_true, y_pred),
        "by_label": _group_metrics(agg_samples, agg_preds, lambda s: "positive" if s["label"] == 1 else "negative"),
        "by_scene": _group_metrics(agg_samples, agg_preds, "scene"),
        "by_same_class": _group_metrics(agg_samples, agg_preds, "same_class"),
        "per_sample": [
            {
                "scene": s["scene"],
                "condition": s["condition"],
                "label": s["label"],
                "probability": p["probability"],
                "decision": p["decision"],
                "frame_id_a": s["frame_id_a"],
                "mask_id_a": s["mask_id_a"],
                "frame_id_b": s["frame_id_b"],
                "mask_id_b": s["mask_id_b"],
                "observer_frame_id": s.get("observer_frame_id"),
                "candidate_id": s.get("candidate_id"),
                "same_class": s["same_class"],
            }
            for s, p in zip(samples, preds)
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
                "observer_frame_id": s.get("observer_frame_id"),
                "same_class": s["same_class"],
            }
            for s, p in zip(agg_samples, agg_preds)
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    ov = results["overall"]
    logger.info(
        "Overall: acc=%.4f prec=%.4f rec=%.4f f1=%.4f (%d samples, %d pos / %d neg) -> %s",
        ov["accuracy"], ov["precision"], ov["recall"], ov["f1"],
        ov["n_samples"], ov["n_positive"], ov["n_negative"], args.output,
    )


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--pool-path", nargs="+", required=True,
        help="One or more candidate CSV paths from build_qwen_candidates.py (shell globs expanded).",
    )
    p.add_argument("--masks-root", required=True, help="Path to replica_masks directory.")
    p.add_argument("--rgb-root", required=True, help="Path to Replica dataset root.")
    p.add_argument(
        "--condition", default="pair_only", choices=ALL_CONDITIONS,
        help="Which condition to train/score (fixes the head's input size). Default: pair_only.",
    )
    p.add_argument("--render-mode", default="outline", choices=["outline", "filled", "raw_mask"])
    p.add_argument(
        "--highlight-alpha", type=float, default=0.4,
        help="Fill opacity in [0, 1] for --render-mode filled (default: 0.4). Ignored otherwise.",
    )
    p.add_argument(
        "--difficulty", type=str, default="all", choices=["all", "easy", "hard"],
        help="Curriculum filter applied after loading (default: 'all', no-op). 'easy' keeps "
             "positives with angle_deg in [--easy-min-angle-deg, --easy-max-angle-deg] and "
             "negatives with neg_strategy == 'co_visible'; 'hard' keeps the complement "
             "(positives outside that angle band, negatives from 'both_views' mining). "
             "Positives with unresolvable angle_deg are dropped from both buckets.",
    )
    p.add_argument(
        "--easy-min-angle-deg", type=float, default=0.0,
        help="Lower bound (degrees) of the 'easy' positive angle band for --difficulty (default: 0.0).",
    )
    p.add_argument(
        "--easy-max-angle-deg", type=float, default=45.0,
        help="Upper bound (degrees) of the 'easy' positive angle band for --difficulty (default: 45.0).",
    )
    p.add_argument("--config", type=str, default="config_stage1.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune (LoRA) and score a vision-encoder-only same-object classifier for Qwen3-VL."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune the vision-encoder LoRA classifier.")
    _add_common_args(train_parser)
    train_parser.add_argument(
        "--val-pool-path", nargs="+", default=None,
        help="Optional held-out candidate CSV(s) for per-epoch validation. Same rules as --pool-path.",
    )
    train_parser.add_argument(
        "--init-adapter-path", default=None,
        help="Resume from a previously saved LoRA adapter + score_head (e.g. a prior "
             "stage's output_dir) instead of initializing fresh. LoRA hyperparameters "
             "come from that adapter's own config, not --config's lora: section.",
    )

    score_parser = subparsers.add_parser("score", help="Run the fine-tuned classifier over a pair pool.")
    _add_common_args(score_parser)
    score_parser.add_argument(
        "--adapter-path", required=True,
        help="Directory with the saved LoRA adapter + score_head (train's --output-dir / config_stage1.yaml sft.output_dir).",
    )
    score_parser.add_argument("--output", required=True, help="Path to write JSON results.")
    score_parser.add_argument("--batch-size", type=int, default=4)
    score_parser.add_argument(
        "--aggregation", type=str, default="mean",
        choices=["mean", "confidence_weighted", "consensus_rate"],
        help="How to collapse per-candidate/per-observer sample variants into one "
             "prediction per pair before computing metrics (default: mean). 'consensus_rate' "
             "votes by decision fraction thresholded at --connect-threshold.",
    )
    score_parser.add_argument(
        "--connect-threshold", type=float, default=0.9,
        help="Decision-fraction threshold for --aggregation consensus_rate (default: 0.9, "
             "matching MaskClustering's c(i,j) >= 0.9 graph-edge threshold).",
    )

    args = parser.parse_args()
    args.pool_path = _expand_pool_paths(args.pool_path)
    if getattr(args, "val_pool_path", None):
        args.val_pool_path = _expand_pool_paths(args.val_pool_path)

    if args.mode == "train":
        train(args)
    else:
        score(args)
