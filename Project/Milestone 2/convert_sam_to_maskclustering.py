"""
Convert SAM per-mask binary PNGs into the combined instance-ID maps
that MaskClustering expects.

Input:  data/replica/{scene}/masks/{frame_id}/000001.png, 000002.png, ...
        data/replica/{scene}/masks/{frame_id}/metadata.json
Output: data/replica/{scene}/output/mask/{frame_id}.png  (uint16, pixel value = instance ID)

Masks are painted in order of highest stability_score first, so SAM's
most confident masks get pixel priority over less confident ones.
"""

from pathlib import Path
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

REPLICA_ROOT = Path(os.environ.get(
    "REPLICA_ROOT",
    "/cluster/52/go93coc/MaskClustering/data/replica"
))

scenes = sorted([
    d for d in os.listdir(REPLICA_ROOT)
    if os.path.isdir(REPLICA_ROOT / d)
    and d != "ground_truth"
    and os.path.isdir(REPLICA_ROOT / d / "masks")
])

print(f"Found {len(scenes)} scenes with SAM masks: {scenes}")

for scene in scenes:
    masks_root = REPLICA_ROOT / scene / "masks"
    output_dir = REPLICA_ROOT / scene / "output" / "mask_sam"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_dirs = sorted(
        [d for d in masks_root.iterdir() if d.is_dir()],
        key=lambda p: int(p.name)
    )

    for frame_dir in tqdm(frame_dirs, desc=scene):
        metadata_path = frame_dir / "metadata.json"
        mask_files = sorted(frame_dir.glob("*.png"))

        if not mask_files:
            continue

        first_mask = np.array(Image.open(mask_files[0]))
        h, w = first_mask.shape[:2]
        instance_map = np.zeros((h, w), dtype=np.uint16)

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            order = sorted(metadata, key=lambda m: m["stability_score"], reverse=True)
        else:
            order = [{"mask_id": i + 1} for i in range(len(mask_files))]

        for instance_id, entry in enumerate(order, start=1):
            mask_path = frame_dir / f"{entry['mask_id']:06d}.png"
            if not mask_path.exists():
                continue
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            new_pixels = (mask > 127) & (instance_map == 0)
            instance_map[new_pixels] = instance_id

        out_path = output_dir / f"{frame_dir.name}.png"
        Image.fromarray(instance_map).save(out_path)

    print(f"  {scene}: {len(frame_dirs)} frames converted → {output_dir}")

print("\nDone! Masks are ready for MaskClustering.")
