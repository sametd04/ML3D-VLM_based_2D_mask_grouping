from typing import Literal

from PIL import Image


ConditionType = Literal["pair_only"]
RenderMode = Literal["outline", "filled", "raw_mask"]

_EXPECTED_IMAGE_COUNTS: dict[ConditionType, dict[RenderMode, int]] = {
    "pair_only": {"outline": 2, "filled": 2},
}

_OUTPUT_FORMAT = """
Respond with exactly 3 lines and nothing else:
Line 1: 1 if the two regions belong to the same physical object, 0 if they do not
Line 2: your confidence as a float between 0.0 and 1.0
Line 3: one sentence of reasoning"""

_SYSTEM_PROMPTS: dict[ConditionType, dict[RenderMode, str]] = {
    "pair_only": {
        "outline": (
            """You are a visual analysis assistant for 3D scene reconstruction.
Task: decide whether region A (outlined in red) and region B (outlined in blue) are the same physical object.
You will see two scene views, each with one region outlined, no additional context frame is provided.
Base your decision solely on the visual appearance of the two outlined regions."""
            + _OUTPUT_FORMAT
        ),
        "filled": (
            """You are a visual analysis assistant for 3D scene reconstruction.
Task: decide whether region A (highlighted with a semi-transparent red fill) and region B (highlighted with a semi-transparent blue fill) are the same physical object.
You will see two scene views, each with one region filled, no additional context frame is provided.
Base your decision solely on the visual appearance of the two highlighted regions."""
            + _OUTPUT_FORMAT
        ),
    },
}

_DECISION_PROMPT = ("""Do region A and region B belong to the same physical object in this 3D scene?
                    Remember: answer with exactly 3 lines: decision (1/0), confidence (float), reasoning (one sentence)."""
                    )


def _img(image: Image) -> dict:
    return {"type": "image", "image": image}


def _txt(text: str) -> dict:
    return {"type": "text", "text": text}


def _build_pair_only(images: list[Image], render_mode: RenderMode) -> list[dict]:
    """pair_only: each mask rendered on its own frame."""
    view_a, view_b = images
    if render_mode == "filled":
        return [
            _txt("Region A (highlighted in red):"),
            _img(view_a),
            _txt("Region B (highlighted in blue):"),
            _img(view_b),
            _txt(_DECISION_PROMPT),
        ]
    return [
        _txt("Region A (outlined in red):"),
        _img(view_a),
        _txt("Region B (outlined in blue):"),
        _img(view_b),
        _txt(_DECISION_PROMPT),
    ]


def build_prompt(condition: ConditionType, images: list[Image], render_mode: RenderMode = "outline") -> list[dict]:
    """Build the system and user prompts for Qwen3-VL's apply_chat_template.

    images order: pair_only -> [view_a, view_b] (same for both outline and filled render modes).
    """
    if condition != "pair_only":
        raise ValueError(f"Unknown condition '{condition}'. Expected 'pair_only'")
    if render_mode == "raw_mask":
        raise NotImplementedError("raw_mask rendering is not yet implemented")
    if render_mode not in ("outline", "filled"):
        raise ValueError(f"Unknown render_mode '{render_mode}'. Expected 'outline', 'filled', or 'raw_mask'.")

    expected = _EXPECTED_IMAGE_COUNTS[condition][render_mode]
    if len(images) != expected:
        raise ValueError(
            f"Condition {condition} (render_mode={render_mode}) requires exactly {expected} images, got {len(images)}."
        )

    content = _build_pair_only(images, render_mode)

    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[condition][render_mode]},
        {"role": "user", "content": content},
    ]
