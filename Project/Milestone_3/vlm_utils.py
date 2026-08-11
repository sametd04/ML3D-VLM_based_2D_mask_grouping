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

# Fallback for the common zero-shot failure mode: the model collapses the 3-line
# contract onto one line, e.g. "1.0, Region A and Region B are both parts...".
# Group 1 is a leading float that may double as decision+confidence merged together
# (e.g. "1.0"); group 2 is an optional second float when the model DID separate
# decision/confidence but joined lines with a comma instead of a newline.
_LOOSE_RESPONSE_RE = re.compile(
    r"^\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*[,:;\n-]+\s*(?:(0(?:\.\d+)?|1(?:\.0+)?)\s*[,:;\n-]+\s*)?(.+?)\s*$",
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

    Tries the strict 3-line format first. If that doesn't match, falls back to a
    lenient single-line parser (handles the model collapsing the 3-line contract
    into one comma-joined line, e.g. "1.0, <reasoning>"). Only if both fail to
    match does it give up (decision=0 / confidence=0.0) and log a warning.
    """
    stripped = text.strip()
    m = _RESPONSE_RE.match(stripped)
    if m:
        return VLMPrediction(
            decision=int(m.group(1)),
            confidence=float(m.group(2)),
            reasoning=m.group(3),
        )

    m = _LOOSE_RESPONSE_RE.match(stripped)
    if m:
        first_val = float(m.group(1))
        second_val = m.group(2)
        if second_val is not None:
            # Model separated decision/confidence but joined lines with a comma.
            decision = int(round(first_val))
            confidence = float(second_val)
        else:
            # Single merged token (e.g. "1.0") — treat it as both decision and confidence.
            decision = 1 if first_val >= 0.5 else 0
            confidence = first_val
        return VLMPrediction(decision=decision, confidence=confidence, reasoning=m.group(3))

    logger.warning("Malformed VLM response: %r", text)
    return VLMPrediction(decision=0, confidence=0.0, reasoning="<parse error>")
