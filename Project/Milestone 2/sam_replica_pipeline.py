from pathlib import Path
import os
import json
import gc
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

class SamReplicaMaskPipeline:
    def __init__(
            self,
            replica_root,
            mask_generator,
            min_stability_score=0.0,
            min_area=0,
            overwrite=False,
            save_metadata=True
    ):
        self.replica_root = Path(replica_root)
        self.mask_generator = mask_generator
        self.min_area = min_area
        self.min_stability_score = min_stability_score
        self.overwrite = overwrite
        self.save_metadata = save_metadata
    
    def process_scene(self, scene_name, frame_ids=None):
        """
        Takes the scene name, iterates over all images within the scene, creates a folder for all masks and
        then creates a separate folder for the masks of each image
        """
        scene_dir = self.replica_root / scene_name
        color_dir = scene_dir / "color"
        masks_dir = scene_dir / "masks"

        masks_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(color_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if frame_ids is not None:
            requested = {str(frame_id) for frame_id in frame_ids}
            image_paths = [p for p in image_paths if p.stem in requested]
            found = {p.stem for p in image_paths}
            missing = sorted(requested - found, key=int)
            if missing:
                raise FileNotFoundError(
                    f"Frames not found in {color_dir}: {', '.join(missing)}"
                )

        for image_path in tqdm(image_paths, desc=scene_name):
            frame_id = image_path.stem
            out_frame_dir = masks_dir / frame_id

            if out_frame_dir.exists(): 
                if not self.overwrite:
                    continue
                shutil.rmtree(out_frame_dir)

            out_frame_dir.mkdir(parents=True, exist_ok=True)

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"The image could not be loaded: {image_path}")
                continue
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            masks = self.mask_generator.generate(image_rgb)

            self.save_masks_individually(masks=masks, out_frame_dir=out_frame_dir)

            del image_bgr, image_rgb, masks
            gc.collect()
    
    def save_masks_individually(self, masks, out_frame_dir):
        """
        Takes all masks of an image, filters them by area and stability score,
        saves each mask as a grayscale PNG, and optionally stores metadata as JSON.
        """
        out_frame_dir = Path(out_frame_dir)
        out_frame_dir.mkdir(parents=True, exist_ok = True)

        masks_filtered = [
            m for m in masks
            if m["area"] >= self.min_area and m["stability_score"] >= self.min_stability_score
        ]
        # sort masks descending in terms of their size
        masks_sorted = sorted(
            masks_filtered,
            key=lambda x: x["area"],
            reverse=True
        )

        metadata = []
        for mask_id, mask_data in enumerate(masks_sorted, start=1):
            mask = mask_data["segmentation"].astype(np.uint8) * 255

            mask_path = out_frame_dir / f"{mask_id:06d}.png"
            Image.fromarray(mask).save(mask_path)

            metadata.append({
                "mask_id": mask_id,
                "area": int(mask_data["area"]),
                "bbox": [int(x) for x in mask_data["bbox"]],
                "predicted_iou": float(mask_data["predicted_iou"]),
                "stability_score": float(mask_data["stability_score"]),
            })
        if self.save_metadata:
            metadata_path = out_frame_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)


def replica_dataset_exists(replica_root):
    """This method checks whether there already exists a replica dataset with at least one scene and RGB images"""
    replica_root = Path(replica_root)

    if not replica_root.exists():
        return False

    scene_dirs = [
        p for p in replica_root.iterdir()
        if p.is_dir() and p.name != "ground_truth"
    ]

    return any((scene / "color").exists() for scene in scene_dirs)
