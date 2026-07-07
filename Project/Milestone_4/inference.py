import argparse
import logging
from typing import List

import torch
import yaml

from load_data import FrameView, RenderedSample, load_frame_view
from prompt_templates import build_prompt
from vlm_utils import VLMPrediction, load_model, parse_vlm_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        [{"condition": sample.condition, "images": images}],
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
            "images" (List[PIL.Image], already rendered) and "condition" (ConditionType).
        model: loaded Qwen3-VL model (bfloat16, device_map=auto).
        processor: matching AutoProcessor.
        max_new_tokens: token budget per sample.

    Returns:
        List[VLMPrediction] in the same order as samples.
    """
    if not samples:
        return []

    texts: List[str] = []
    image_lists = []
    for s in samples:
        messages = build_prompt(s["condition"], s["images"])
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
    ).to(model.device)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VLM mask-pair inference for one Replica frame pair."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--condition",
        type=str,
        choices=["pair_only", "pair_with_context", "pair_with_candidate"],
        required=True,
        help=(
            "VLM condition: "
            "pair_only=two mask views only | "
            "pair_with_context=mask views + full observer frame with all masks | "
            "pair_with_candidate=mask views + observer frame with one candidate highlighted"
        ),
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

    # Observer frame args (required for pair_with_context and pair_with_candidate)
    parser.add_argument("--observer-frame", type=int, default=None,
                        help="Observer frame id (required for pair_with_context and pair_with_candidate)")
    parser.add_argument("--candidate-id", type=int, default=None,
                        help="Candidate mask id in the observer frame (required for pair_with_candidate)")

    args = parser.parse_args()

    # Validate condition-dependent args
    if args.condition in ("pair_with_context", "pair_with_candidate") and args.observer_frame is None:
        parser.error(f"--observer-frame is required for condition '{args.condition}'")
    if args.condition == "pair_with_candidate" and args.candidate_id is None:
        parser.error("--candidate-id is required for condition 'pair_with_candidate'")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model, processor = load_model(config)

    # Load frames (deduplicate if frame-a == frame-b or observer == one of them)
    frame_cache: dict[int, FrameView] = {}
    for fid in {args.frame_a, args.frame_b, args.observer_frame} - {None}:
        frame_cache[fid] = load_frame_view(
            args.scene, fid, masks_root=args.masks_root, rgb_root=args.rgb_root
        )

    sample = RenderedSample(
        condition=args.condition,
        frame_a=frame_cache[args.frame_a],
        mask_id_a=args.mask_a,
        frame_b=frame_cache[args.frame_b],
        mask_id_b=args.mask_b,
        observer=frame_cache.get(args.observer_frame),
        candidate_id=args.candidate_id,
    )

    result = run_inference(sample, model, processor, max_new_tokens=config["inference"]["max_new_tokens"])
    print(f"decision:   {result['decision']}")
    print(f"confidence: {result['confidence']}")
    print(f"reasoning:  {result['reasoning']}")
