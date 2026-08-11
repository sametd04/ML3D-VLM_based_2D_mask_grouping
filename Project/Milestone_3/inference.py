"""Zero-shot VLM inference for a single mask pair (pair_only).

Usage:
    python inference.py \
        --scene room0 \
        --masks-root /path/to/replica_masks \
        --rgb-root /path/to/replica \
        --frame-a 120 --mask-a 4 \
        --frame-b 340 --mask-b 7 \
        --render-mode outline \
        --config config_stage1.yaml
"""

import argparse
import logging
from typing import List

import torch
import yaml
from torch.utils.data import Dataset

from load_data import FrameView, RenderedSample, load_frame_view
from prompt_templates import build_prompt
from vlm_utils import VLMPrediction, load_model, parse_vlm_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreRenderedSampleDataset(Dataset):
    """Wraps already-rendered samples (dicts with an "images" key) for DataLoader use.

    Pairs with `make_collate_fn` so DataLoader worker processes do the CPU-bound
    prompt building + image preprocessing for upcoming batches while the main
    process runs model.generate() on the GPU for the current one.
    """

    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def make_collate_fn(processor):
    """Builds a DataLoader collate_fn that runs the CPU side of inference.

    Returns (batch_samples, inputs): batch_samples is the raw sample dict list
    (needed downstream for metrics/aggregation), inputs is the processor output,
    still on CPU — move it to the model device in the eval loop after the
    DataLoader hands it back.
    """

    def collate(batch: List[dict]):
        texts: List[str] = []
        image_lists = []
        for s in batch:
            messages = build_prompt(s["condition"], s["images"], s.get("render_mode", "outline"))
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            image_lists.append(s["images"])

        inputs = processor(
            text=texts,
            images=image_lists,
            padding=True,
            return_tensors="pt",
        )
        return batch, inputs

    return collate


def generate_batch(inputs, model, processor, max_new_tokens: int = 64) -> List[VLMPrediction]:
    """GPU generation + decode for a batch already preprocessed by make_collate_fn's collate."""
    if inputs is None:
        return []

    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [parse_vlm_response(text) for text in decoded]


def run_inference(
    sample: RenderedSample,
    model,
    processor,
    max_new_tokens: int = 64,
) -> VLMPrediction:
    """Run VLM inference for a single RenderedSample.

    Args:
        sample: RenderedSample built from load_data.py — render() is called internally.
            condition determines which images are produced and which prompt is used.
        model: loaded Qwen3-VL model (bfloat16, device_map=auto).
        processor: matching AutoProcessor.
        max_new_tokens: token budget for generation.

    Returns:
        VLMPrediction with keys decision (int 0/1), confidence (float), reasoning (str).
        On any parse failure returns decision=0, confidence=0.0, reasoning="<parse error>".
    """
    images = sample.render()
    return run_inference_batch(
        [{"condition": sample.condition, "images": images, "render_mode": sample.render_mode}],
        model,
        processor,
        max_new_tokens,
    )[0]


def run_inference_batch(
    samples: List[dict],
    model,
    processor,
    max_new_tokens: int = 64,
) -> List[VLMPrediction]:
    """Run batched VLM inference over pre-rendered samples.

    Args:
        samples: list of dicts as returned by load_data_training() — each must have
            "images" (List[PIL.Image], already rendered), "condition" (ConditionType),
            and optionally "render_mode" (RenderMode, defaults to "outline").
        model: loaded Qwen3-VL model (bfloat16, device_map=auto).
        processor: matching AutoProcessor.
        max_new_tokens: token budget per sample.

    Returns:
        List[VLMPrediction] in the same order as samples.
    """
    if not samples:
        return []

    collate = make_collate_fn(processor)
    _, inputs = collate(samples)
    return generate_batch(inputs, model, processor, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VLM mask-pair inference for one Replica frame pair."
    )
    parser.add_argument("--config", type=str, default="config_stage1.yaml")
    parser.add_argument(
        "--condition",
        type=str,
        choices=["pair_only"],
        default="pair_only",
        help="VLM condition: pair_only=two mask views only.",
    )

    # Scene-level args
    parser.add_argument("--scene", type=str, required=True, help="Replica scene name, e.g. room0")
    parser.add_argument("--masks-root", type=str, required=True, help="Path to replica_masks directory")
    parser.add_argument("--rgb-root", type=str, required=True, help="Path to Replica dataset root (contains <scene>/color/)")

    # Mask pair args
    parser.add_argument("--frame-a", type=int, required=True, help="Frame id for mask A")
    parser.add_argument("--mask-a", type=int, required=True, help="Mask id within frame A")
    parser.add_argument("--frame-b", type=int, required=True, help="Frame id for mask B")
    parser.add_argument("--mask-b", type=int, required=True, help="Mask id within frame B")

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

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model, processor = load_model(config)

    # Load frames (deduplicate if frame-a == frame-b)
    frame_cache: dict[int, FrameView] = {}
    for fid in {args.frame_a, args.frame_b}:
        frame_cache[fid] = load_frame_view(
            args.scene, fid, masks_root=args.masks_root, rgb_root=args.rgb_root
        )

    sample = RenderedSample(
        condition=args.condition,
        frame_a=frame_cache[args.frame_a],
        mask_id_a=args.mask_a,
        frame_b=frame_cache[args.frame_b],
        mask_id_b=args.mask_b,
        render_mode=args.render_mode,
        alpha=args.highlight_alpha,
    )

    result = run_inference(sample, model, processor, max_new_tokens=config["inference"]["max_new_tokens"])
    print(f"decision:   {result['decision']}")
    print(f"confidence: {result['confidence']}")
    print(f"reasoning:  {result['reasoning']}")
