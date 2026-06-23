from pathlib import Path
import os
import urllib.request
import torch
import cv2
import numpy as np
from PIL import Image
import sys

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/cluster/52/go93coc"))
REPLICA_ROOT = Path(os.environ.get("REPLICA_ROOT", "/cluster/52/go93coc/MaskClustering/data/replica"))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/cluster/52/go93coc/checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from sam_replica_pipeline import SamReplicaMaskPipeline

SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = str(CHECKPOINT_DIR / "sam_vit_h_4b8939.pth")

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
if not os.path.exists(SAM_CHECKPOINT):
    print("Downloading SAM checkpoint (~2.5 GB)...")
    urllib.request.urlretrieve(url, SAM_CHECKPOINT)
    print("Download complete.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    pred_iou_thresh=0.0,
    stability_score_thresh=0.0,
    min_mask_region_area=0,
)

pipeline = SamReplicaMaskPipeline(
    replica_root=REPLICA_ROOT,
    mask_generator=mask_generator,
    min_area=15000,
    min_stability_score=0.5,
    save_metadata=True,
)

scenes = sorted([
    d for d in os.listdir(REPLICA_ROOT)
    if os.path.isdir(os.path.join(REPLICA_ROOT, d))
    and d != "ground_truth"
    and os.path.isdir(os.path.join(REPLICA_ROOT, d, "color"))
])

print(f"Found {len(scenes)} scenes: {scenes}")

for scene in scenes:
    pipeline.process_scene(scene)

print("Done! All masks generated.")
