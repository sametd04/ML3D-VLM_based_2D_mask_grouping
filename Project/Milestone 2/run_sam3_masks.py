"""
SAM 3 mask generation for Replica scenes using Ultralytics API.

Uses text prompts from REPLICA_LABELS to detect and segment all instances of each
object category per frame. Outputs uint16 instance maps compatible with MaskClustering.

Usage:
    # Full run on all scenes:
    python run_sam3_masks.py

    # Quick test: first frame only, all scenes:
    python run_sam3_masks.py --test

    # Specific scene:
    python run_sam3_masks.py --scene room0

    # Adjust confidence threshold:
    python run_sam3_masks.py --conf 0.1

Environment variables:
    REPLICA_ROOT: path to replica data (default: /cluster/52/go93coc/MaskClustering/data/replica)
    SAM3_CHECKPOINT: path to sam3.pt (default: /cluster/52/go93coc/checkpoints/sam3.pt)
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Disable cuDNN to avoid version mismatch with older drivers
torch.backends.cudnn.enabled = False

REPLICA_ROOT = Path(os.environ.get(
    "REPLICA_ROOT",
    "/cluster/52/go93coc/MaskClustering/data/replica"
))

SAM3_CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT",
    "/cluster/52/go93coc/checkpoints/sam3.pt"
)

REPLICA_LABELS = (
    "basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle",
    "box", "bowl", "camera", "cabinet", "candle", "chair", "clock", "cloth",
    "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant",
    "lamp", "monitor", "nightstand", "panel", "picture", "pillar", "pillow",
    "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa",
    "stool", "switch", "table", "tablet", "tissue-paper", "tv-screen",
    "tv-stand", "vase", "vent", "wall-plug", "window", "rug",
)

MASKCLUSTERING_STRIDE = 10


def get_scenes(replica_root):
    """Find all valid Replica scenes."""
    return sorted([
        d for d in os.listdir(replica_root)
        if os.path.isdir(replica_root / d)
        and d != "ground_truth"
        and os.path.isdir(replica_root / d / "color")
    ])


def get_frame_ids(scene_dir, stride=MASKCLUSTERING_STRIDE):
    """Get sorted frame IDs at the given stride."""
    color_dir = scene_dir / "color"
    all_ids = sorted([int(p.stem) for p in color_dir.glob("*.jpg")])
    return all_ids[::stride]


BATCH_SIZE = 4  # number of text prompts per inference call (to avoid OOM)


def run_sam3_on_scene(predictor, scene_name, replica_root, frame_ids,
                      output_dir_name="mask_sam3", conf=0.05):
    """
    Run SAM 3 on a single Replica scene using Ultralytics API.

    For each frame:
      1. Set image
      2. Query labels in small batches (to fit in GPU memory)
      3. Collect all masks and combine into uint16 instance map
    """
    scene_dir = replica_root / scene_name
    color_dir = scene_dir / "color"
    output_dir = scene_dir / "output" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split labels into batches
    label_batches = [
        list(REPLICA_LABELS[i:i + BATCH_SIZE])
        for i in range(0, len(REPLICA_LABELS), BATCH_SIZE)
    ]

    print(f"\n{'='*60}")
    print(f"Scene: {scene_name} | Frames: {len(frame_ids)} | Labels: {len(REPLICA_LABELS)}")
    print(f"Label batches: {len(label_batches)} (batch size {BATCH_SIZE})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    total_instances = 0
    t_start = time.time()

    for frame_idx, frame_id in enumerate(frame_ids):
        image_path = str(color_dir / f"{frame_id}.jpg")

        predictor.set_image(image_path)

        # Collect all masks across label batches
        all_masks = []

        for batch in label_batches:
            results = predictor(text=batch)
            r = results[0]
            if r.masks is not None and len(r.masks) > 0:
                batch_masks = r.masks.data.cpu().numpy()
                all_masks.append(batch_masks)

        if not all_masks:
            img = Image.open(image_path)
            w, h = img.size
            instance_map = np.zeros((h, w), dtype=np.uint16)
        else:
            all_masks = np.concatenate(all_masks, axis=0)
            h, w = all_masks.shape[1], all_masks.shape[2]
            instance_map = np.zeros((h, w), dtype=np.uint16)

            for i, mask in enumerate(all_masks):
                instance_id = i + 1
                instance_map[mask > 0.5] = instance_id

            total_instances += len(all_masks)

        out_path = output_dir / f"{frame_id}.png"
        Image.fromarray(instance_map).save(out_path)

        n_masks = len(all_masks) if isinstance(all_masks, np.ndarray) else 0
        elapsed = time.time() - t_start
        print(f"  Frame {frame_id:4d} [{frame_idx+1}/{len(frame_ids)}]: "
              f"{n_masks} instances ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t_start
    print(f"\n  Done: {scene_name} → {len(frame_ids)} frames, "
          f"{total_instances} total instances, {elapsed:.1f}s")
    return total_instances


def main():
    parser = argparse.ArgumentParser(description="SAM 3 mask generation for Replica (Ultralytics)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: first frame of each scene only")
    parser.add_argument("--scene", type=str, default=None,
                        help="Run on a specific scene only")
    parser.add_argument("--stride", type=int, default=MASKCLUSTERING_STRIDE,
                        help="Frame stride (default: 10)")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="Confidence threshold (default: 0.05)")
    parser.add_argument("--output-dir", type=str, default="mask_sam3",
                        help="Output directory name under output/ (default: mask_sam3)")
    args = parser.parse_args()

    # Build SAM 3 predictor
    print("Loading SAM 3 model...")
    from ultralytics.models.sam import SAM3SemanticPredictor
    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=SAM3_CHECKPOINT,
        verbose=False,
    ))
    print(f"Model loaded. Confidence threshold: {args.conf}\n")

    # Determine scenes to process
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = get_scenes(REPLICA_ROOT)

    print(f"Scenes: {scenes}")
    print(f"Frame stride: {args.stride}")
    print(f"Output dir: output/{args.output_dir}/")

    total_instances = 0
    t_start = time.time()

    for scene_name in scenes:
        scene_dir = REPLICA_ROOT / scene_name
        if args.test:
            frame_ids = get_frame_ids(scene_dir, stride=args.stride)[:1]
        else:
            frame_ids = get_frame_ids(scene_dir, stride=args.stride)

        if not frame_ids:
            print(f"  {scene_name}: no frames found, skipping")
            continue

        num_instances = run_sam3_on_scene(
            predictor, scene_name, REPLICA_ROOT, frame_ids,
            output_dir_name=args.output_dir, conf=args.conf,
        )
        total_instances += num_instances

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE | {len(scenes)} scenes | {total_instances} total instances | {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
