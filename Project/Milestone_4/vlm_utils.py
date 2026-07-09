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
_LABELED_RESPONSE_RE = re.compile(
    r"decision\s*:\s*([01](?:\.0+)?)"
    r".*?confidence\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)"
    r".*?reasoning\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_ONE_LINE_RESPONSE_RE = re.compile(
    r"^\s*([01](?:\.0+)?)\s*[,;:\-]\s*(.+?)\s*$",
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


def _prediction(decision: str, confidence: str, reasoning: str) -> VLMPrediction:
    return VLMPrediction(
        decision=int(float(decision)),
        confidence=max(0.0, min(1.0, float(confidence))),
        reasoning=reasoning.strip(),
    )


def parse_vlm_response(text: str) -> VLMPrediction:
    """Parse the VLM output into a VLMPrediction.

    Preferred format:
        Line 1: 1 or 0
        Line 2: float in [0.0, 1.0]
        Line 3: one sentence of reasoning

    Also accepts common near-misses, such as labeled answers or one-line
    answers like "1.0, explanation".
    """
    stripped = text.strip()

    m = _RESPONSE_RE.match(stripped)
    if m:
        return _prediction(m.group(1), m.group(2), m.group(3))

    m = _LABELED_RESPONSE_RE.search(stripped)
    if m:
        return _prediction(m.group(1), m.group(2), m.group(3))

    m = _ONE_LINE_RESPONSE_RE.match(stripped)
    if m:
        decision = m.group(1)
        return _prediction(decision, decision, m.group(2))

    logger.warning("Malformed VLM response: %r", text)
    return VLMPrediction(decision=0, confidence=0.0, reasoning="<parse error>")
