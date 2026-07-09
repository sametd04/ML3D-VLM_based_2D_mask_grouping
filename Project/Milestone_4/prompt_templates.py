from typing import Literal

from PIL import Image


ConditionType = Literal["pair_only", "pair_with_context", "pair_with_candidate"]

_EXPECTED_IMAGE_COUNTS: dict[ConditionType, int] = {
    "pair_only": 2,
    "pair_with_context": 3,
    "pair_with_candidate": 3,
}

_OUTPUT_FORMAT = """

Output format is strict. Respond with exactly 3 lines and nothing else.
Do not write labels such as "Decision:" or "Confidence:".
Do not use bullet points.
Do not put multiple fields on one line.

Line 1: 1 if the two regions belong to the same physical object, otherwise 0
Line 2: your confidence as a decimal number between 0.0 and 1.0
Line 3: one short sentence of reasoning"""

_SYSTEM_PROMPTS: dict[ConditionType, str] = {
    "pair_only": (
        """You are a visual analysis assistant for 3D scene reconstruction.
Task: decide whether region A (outlined in red) and region B (outlined in blue) are the same physical object.
You will see two scene views, each with one region outlined, no additional context frame is provided.
Base your decision solely on the visual appearance of the two outlined regions."""
        + _OUTPUT_FORMAT
    ),
    "pair_with_context": (
        """You are a visual analysis assistant for 3D scene reconstruction.
Task: decide whether region A (outlined in red) and region B (outlined in blue) are the same physical object.
You will see a full observer frame (a raw RGB image captured from a camera viewpoint where both regions are visible) followed by individual views of each region.
Use the spatial layout and surroundings in the observer frame to support your decision."""
        + _OUTPUT_FORMAT
    ),
    "pair_with_candidate": (
        """You are a visual analysis assistant for 3D scene reconstruction.
Task: decide whether region A (outlined in red) and region B (outlined in blue) are the same physical object.
You will see a full observer frame (a raw RGB image from a camera viewpoint where both regions are visible) with one 2D detection candidate outlined in green, followed by individual views of each region.
If both region A and region B fall within that single candidate detection, they belong to the same object."""
        + _OUTPUT_FORMAT
    ),
}

_DECISION_PROMPT = """Do region A and region B belong to the same physical object in this 3D scene?

Answer with exactly 3 lines:
1 or 0
confidence decimal
one short sentence

Return no other text."""


def _img(image: Image) -> dict:
    return {"type": "image", "image": image}


def _txt(text: str) -> dict:
    return {"type": "text", "text": text}


def _build_condition_a(images: list[Image]) -> list[dict]:
    """pair_only: [view_a, view_b] — each mask on its own frame."""
    view_a, view_b = images
    return [
        _txt("Region A (outlined in red):"),
        _img(view_a),
        _txt("Region B (outlined in blue):"),
        _img(view_b),
        _txt(_DECISION_PROMPT),
    ]


def _build_condition_b(images: list[Image]) -> list[dict]:
    """pair_with_context: [view_a, view_b, context_view] — pair + all masks in observer frame."""
    view_a, view_b, context_view = images
    return [
        _txt("Region A (outlined in red):"),
        _img(view_a),
        _txt("Region B (outlined in blue):"),
        _img(view_b),
        _txt("Observer frame with all detected masks outlined (scene context):"),
        _img(context_view),
        _txt(_DECISION_PROMPT),
    ]


def _build_condition_c(images: list[Image]) -> list[dict]:
    """pair_with_candidate: [view_a, view_b, candidate_view] — pair + one candidate in observer frame."""
    view_a, view_b, candidate_view = images
    return [
        _txt("Region A (outlined in red):"),
        _img(view_a),
        _txt("Region B (outlined in blue):"),
        _img(view_b),
        _txt("Observer frame with one candidate detection outlined in green:"),
        _img(candidate_view),
        _txt(_DECISION_PROMPT),
    ]


_BUILDERS = {
    "pair_only": _build_condition_a,
    "pair_with_context": _build_condition_b,
    "pair_with_candidate": _build_condition_c,
}


def build_prompt(condition: ConditionType, images: list[Image]) -> list[dict]:
    """Build the system and user prompts for Qwen3-VL's apply_chat_template.

    images order by condition:
        pair_only          -> [view_a, view_b]
        pair_with_context  -> [view_a, view_b, context_view]
        pair_with_candidate -> [view_a, view_b, candidate_view]
    """
    if condition not in _EXPECTED_IMAGE_COUNTS:
        raise ValueError(
            f"Unknown condition '{condition}'. Expected one of {list(_EXPECTED_IMAGE_COUNTS)}"
        )

    expected = _EXPECTED_IMAGE_COUNTS[condition]
    if len(images) != expected:
        raise ValueError(
            f"Condition {condition} requires exactly {expected} images, got {len(images)}."
        )

    content = _BUILDERS[condition](images)

    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[condition]},
        {"role": "user", "content": content},
    ]
