"""Compose the poster's 2D figures: mask overlays and the Qwen input pair.

Inputs are Replica RGB frames and CropFormer label maps (greyscale PNG where
pixel value = mask id). Outputs:
  masks_<fid>.png     RGB frame with all masks as colored overlays (motivation)
  qwen_pair.png       two frames side by side, one mask outlined in red / blue
                      (what the VLM actually sees, minus the prompt)

Usage:
    python make_figures.py --frames frame0.jpg frame40.jpg frame80.jpg \
        --masks mask0.png mask40.png mask80.png \
        --pair-a 1:frame0.jpg:mask0.png --pair-b 1:frame40.jpg:mask40.png \
        --out-dir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_PALETTE = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors


def mask_overlay(frame_path: Path, mask_path: Path, out_path: Path, alpha: float = 0.55) -> None:
    rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float64) / 255.0
    labels = np.asarray(Image.open(mask_path))
    out = rgb.copy()
    for i, mask_id in enumerate(np.unique(labels)):
        if mask_id == 0:
            continue
        color = np.array(_PALETTE[i % len(_PALETTE)][:3])
        sel = labels == mask_id
        out[sel] = (1 - alpha) * rgb[sel] + alpha * color
    Image.fromarray((out * 255).astype(np.uint8)).save(out_path)
    print(f"wrote {out_path}")


def outline(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], width: int = 6) -> np.ndarray:
    """Draw a thick contour of `mask` on `rgb` without cv2 (binary dilation via shifts)."""
    edge = mask ^ _erode(mask)
    thick = edge.copy()
    for _ in range(width - 1):
        thick = _dilate(thick)
    out = rgb.copy()
    out[thick] = color
    return out


def _shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(m)
    h, w = m.shape
    ys = slice(max(dy, 0), h + min(dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    ys_src = slice(max(-dy, 0), h + min(-dy, 0))
    xs_src = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = m[ys_src, xs_src]
    return out


def _dilate(m: np.ndarray) -> np.ndarray:
    return m | _shift(m, 1, 0) | _shift(m, -1, 0) | _shift(m, 0, 1) | _shift(m, 0, -1)


def _erode(m: np.ndarray) -> np.ndarray:
    return m & _shift(m, 1, 0) & _shift(m, -1, 0) & _shift(m, 0, 1) & _shift(m, 0, -1)


def qwen_pair(spec_a: str, spec_b: str, out_path: Path) -> None:
    panels = []
    for spec, color in ((spec_a, (1.0, 0.1, 0.1)), (spec_b, (0.1, 0.3, 1.0))):
        mask_id_str, frame_path, mask_path = spec.split(":")
        rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float64) / 255.0
        labels = np.asarray(Image.open(mask_path))
        mask = labels == int(mask_id_str)
        if not mask.any():
            raise ValueError(f"mask id {mask_id_str} empty in {mask_path}")
        panels.append(outline(rgb, mask, color))
    gap = np.ones((panels[0].shape[0], 24, 3))
    combined = np.concatenate([panels[0], gap, panels[1]], axis=1)
    Image.fromarray((combined * 255).astype(np.uint8)).save(out_path)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frames", type=Path, nargs="*", default=[])
    parser.add_argument("--masks", type=Path, nargs="*", default=[])
    parser.add_argument("--pair-a", help="maskid:frame.jpg:mask.png for region A (red)")
    parser.add_argument("--pair-b", help="maskid:frame.jpg:mask.png for region B (blue)")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for frame, mask in zip(args.frames, args.masks):
        mask_overlay(frame, mask, args.out_dir / f"masks_{frame.stem}.png")
    if args.pair_a and args.pair_b:
        qwen_pair(args.pair_a, args.pair_b, args.out_dir / "qwen_pair.png")


if __name__ == "__main__":
    main()
