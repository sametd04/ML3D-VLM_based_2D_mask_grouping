import logging
import re
from typing import TypedDict

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)

_RESPONSE_RE = re.compile(
    r"^\s*([01])\s*\n\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*\n\s*(.+?)\s*$",
    re.DOTALL,
)


class VLMPrediction(TypedDict):
    decision: int      # 1 = same object, 0 = different objects
    confidence: float  # in [0.0, 1.0]
    reasoning: str


def load_model(config: dict) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    """Load Qwen3-VL model and processor from a config dict.

    Args:
        config: parsed config.yaml dict; uses model_id and attn_implementation keys.

    Returns:
        (model, processor) ready for run_inference() and run_inference_batch().
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model_id"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config["attn_implementation"],
    )
    processor = AutoProcessor.from_pretrained(config["model_id"])
    return model, processor


def parse_vlm_response(text: str) -> VLMPrediction:
    """Parse the strict 3-line VLM output format into a VLMPrediction.

    Expected format:
        Line 1: 1 or 0
        Line 2: float in [0.0, 1.0]
        Line 3: one sentence of reasoning

    Falls back to decision=0 / confidence=0.0 on any parse failure and logs
    the malformed response — never silently swallows errors.
    """
    m = _RESPONSE_RE.match(text.strip())
    if not m:
        logger.warning("Malformed VLM response: %r", text)
        return VLMPrediction(decision=0, confidence=0.0, reasoning="<parse error>")
    return VLMPrediction(
        decision=int(m.group(1)),
        confidence=float(m.group(2)),
        reasoning=m.group(3),
    )
