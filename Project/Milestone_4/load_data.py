from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# MaskClustering has no package structure; add its root so bare imports (utils.*, graph.*) work.
_MASKCLUSTERING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MaskClustering"))
if _MASKCLUSTERING_ROOT not in sys.path:
    sys.path.insert(0, _MASKCLUSTERING_ROOT)

from prompt_templates import ConditionType

import cv2
import numpy as np
import torch
from PIL import Image as PILImage


@dataclass
class MaskRegion:                             # Represents a single segmented region within one camera frame
    mask_id: int                              # CropFormer greyscale pixel value; key in FrameView.masks
                                              # and used to form the "{frame_id}_{mask_id}" key in mask_point_clouds
    binary_mask: np.ndarray                   # (H, W) bool: which pixels belong to this mask
    gt_instance_id: Optional[int] = None      # Replica GT: label_id*1000 + instance_index; None until assign_gt() is called


@dataclass
class FrameView:  # Represents a whole frame
    scene: str
    frame_id: int
    label_map: np.ndarray           # (H, W) uint8: pixel value = mask_id, 0 = background (from segmentation model)
    rgb: Optional[np.ndarray]       # (H, W, 3) uint8 original camera frame, or None
    masks: Dict[int, MaskRegion]                          # mapping from mask_id to MaskRegion
    instance_to_masks: Optional[Dict[int, list]] = None  # gt_instance_id -> [mask_id, ...]; populated by assign_gt()
    _gt_assigned: bool = False                            # guard against redundant assign_gt calls

    def assign_gt(self) -> None:
        """Back-project this frame's masks onto the global mesh and assign Replica GT instance IDs.

        Derives dataset and scene_points from self.scene. Runs frame_backprojection (GPU)
        for this frame only, then does a majority-vote lookup against the per-vertex Replica
        GT file. Populates gt_instance_id on each MaskRegion; masks that project entirely
        onto unlabeled geometry keep None. No-op if called more than once on the same FrameView.
        """
        logger.debug("[assign_gt] frame=%d scene=%s", self.frame_id, self.scene)
        if self._gt_assigned:
            logger.debug("[assign_gt] skipping, GT already assigned for frame=%d", self.frame_id)
            return

        from dataset.replica import ReplicaDataset  # MaskClustering dep — build phase only
        from utils.mask_backprojection import frame_backprojection

        try:
            logger.debug("[assign_gt] loading ReplicaDataset for scene=%s", self.scene)
            dataset = ReplicaDataset(self.scene)
            # get_scene_points() returns np.ndarray; convert to CUDA tensor here because
            # crop_scene_points (called downstream) compares against CUDA tensors.
            scene_points: torch.Tensor = torch.tensor(
                dataset.get_scene_points(), dtype=torch.float32
            ).cuda()
            scene_gt_path =  f"data/replica/ground_truth/{self.scene}.txt"
        except AssertionError as e:
            logger.error("[assign_gt] failed to load dataset — %s. Run from the folder containing replica's data/", e)

        # Back-project this frame's 2D masks onto the global mesh — GPU call.
        # mask_info: {mask_id: set of global mesh vertex indices}
        try:
            logger.debug("[assign_gt] backprojecting masks for frame=%d", self.frame_id)
            mask_info, _ = frame_backprojection(dataset, scene_points, self.frame_id)

        except AssertionError as e:
            logger.error("[assign_gt] backprojection failed for frame=%d scene=%s — %s", self.frame_id, self.scene, e)

        gt_ids: np.ndarray = np.loadtxt(scene_gt_path, dtype=np.int64)
        logger.debug("[assign_gt] running majority-vote GT assignment for frame=%d", self.frame_id)
        for mask_id, region in self.masks.items():
            point_ids = mask_info.get(mask_id)
            if not point_ids:
                continue
            labels = gt_ids[list(point_ids)]
            labels = labels[labels > 0]  # drop GT=0 (walls / floor / ceiling)
            if len(labels) == 0:
                continue
            # Majority vote over vertex labels — boundary noise can bleed onto neighbors.
            unique_ids, counts = np.unique(labels, return_counts=True)
            region.gt_instance_id = int(unique_ids[np.argmax(counts)])
        
        
        # Build reverse mapping: gt_instance_id -> [mask_id, ...] for quick instance lookups.
        result: Dict[int, list] = {}
        for mask_id, region in self.masks.items():
            if region.gt_instance_id is not None:
                result.setdefault(region.gt_instance_id, []).append(mask_id)
        self.instance_to_masks = result
        self._gt_assigned = True
        logger.debug("[assign_gt] done for frame=%d (%d masks assigned)", self.frame_id, len(result))
        


def load_frame_view(
    scene: str,
    frame_id: int,
    masks_root: str,
    rgb_root: Optional[str] = None,
) -> FrameView:
    """Load one Replica frame: label map and per-mask binary regions.

    Args:
        scene:      Replica scene name, e.g. "room0".
        frame_id:   Frame index matching the filename in output/mask/.
        masks_root: Path to the replica_masks directory.
        rgb_root:   Path to the Replica scene directory (must contain color/). If None,
                    FrameView.rgb is None.

    Returns:
        FrameView with label_map, optional rgb, and mask dict keyed by mask_id.
    """
    
    # Load masks generated by the segmentation model
    mask_path = os.path.join(
        masks_root, "replica", scene, "output", "mask", f"{frame_id}.png"
    )
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # mask_id = value in mask
    label_map: np.ndarray = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # (H, W) uint8

    # Load RGB original iamges if provided
    rgb: Optional[np.ndarray] = None
    if rgb_root is not None:
        bgr = cv2.imread(os.path.join(rgb_root, scene, "color", f"{frame_id}.jpg"))
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Extract the individual masks and store them as MaskRegions
    masks: dict = {}
    for mask_id in np.unique(label_map):
        if mask_id == 0:
            continue
        binary = label_map == mask_id

        masks[int(mask_id)] = MaskRegion(
            mask_id=int(mask_id),
            binary_mask=binary,
        )

    return FrameView(scene=scene, frame_id=frame_id, label_map=label_map, rgb=rgb, masks=masks)


def _draw_outline(frame: np.ndarray, mask: np.ndarray, color: tuple, thickness: int) -> np.ndarray:
    """Return a copy of frame with a colored contour outline drawn around a binary mask."""
    out = frame.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(out, contours, -1, color, thickness=thickness)
    return out


def render_pair(
    frame_a: FrameView,
    mask_id_a: int,
    frame_b: FrameView,
    mask_id_b: int,
    thickness: int = 1,
) -> tuple[PILImage.Image, PILImage.Image]:
    """pair_only — each mask outlined on its own frame.

    Args:
        frame_a:    FrameView containing mask A; rgb must not be None.
        mask_id_a:  Key into frame_a.masks for region A (outlined red).
        frame_b:    FrameView containing mask B; rgb must not be None.
        mask_id_b:  Key into frame_b.masks for region B (outlined blue).
        thickness:  Contour line width in pixels.

    Returns:
        (view_a, view_b): view_a has mask A outlined red; view_b has mask B outlined blue.
    """
    if frame_a.rgb is None or frame_b.rgb is None:
        raise ValueError("FrameView.rgb must not be None for rendering.")
    view_a = PILImage.fromarray(_draw_outline(frame_a.rgb, frame_a.masks[mask_id_a].binary_mask, color=(255, 0, 0), thickness=thickness))
    view_b = PILImage.fromarray(_draw_outline(frame_b.rgb, frame_b.masks[mask_id_b].binary_mask, color=(0, 0, 255), thickness=thickness))
    return view_a, view_b


def _render_candidate_view(
    observer: FrameView,
    candidate_id: int,
    thickness: int = 1,
) -> PILImage.Image:
    """pair_with_candidate — observer frame with one candidate detection outlined green.

    Args:
        observer:      FrameView where both masks are visible; rgb must not be None.
        candidate_id:  Key into observer.masks for the candidate detection.
        thickness:     Contour line width in pixels.

    Returns:
        PIL RGB Image with the single candidate outlined green.
    """
    if observer.rgb is None:
        raise ValueError("FrameView.rgb must not be None for rendering.")
    return PILImage.fromarray(
        _draw_outline(observer.rgb, observer.masks[candidate_id].binary_mask, color=(0, 200, 0), thickness=thickness)
    )


def _render_context_view(
    observer: FrameView,
    thickness: int = 1,
) -> PILImage.Image:
    """pair_with_context — observer frame with all detected masks outlined white.

    Args:
        observer:  FrameView where both masks are visible; rgb must not be None.
        thickness: Contour line width in pixels.

    Returns:
        PIL RGB Image with every mask in observer.masks outlined white.
    """
    if observer.rgb is None:
        raise ValueError("FrameView.rgb must not be None for rendering.")
    context = observer.rgb
    for region in observer.masks.values():
        context = _draw_outline(context, region.binary_mask, color=(0, 200, 0), thickness=thickness)
    return PILImage.fromarray(context)


@dataclass
class RenderedSample:
    """Inputs for one VLM comparison call. Call render() to produce PIL images."""

    condition: ConditionType
    frame_a: FrameView
    mask_id_a: int
    frame_b: FrameView
    mask_id_b: int
    observer: Optional[FrameView] = None  # required for pair_with_context / pair_with_candidate
    candidate_id: Optional[int] = None    # required for pair_with_candidate
    thickness: int = 1

    def render(self) -> list[PILImage.Image]:
        """Render and return the image list for build_prompt(), based on condition."""
        view_a, view_b = render_pair(self.frame_a, self.mask_id_a, self.frame_b, self.mask_id_b, self.thickness)
        if self.condition == "pair_only":
            return [view_a, view_b]
        if self.condition == "pair_with_candidate" and self.observer is not None and self.candidate_id is not None:
            return [view_a, view_b, _render_candidate_view(self.observer, self.candidate_id, self.thickness)]
        if self.condition == "pair_with_context" and self.observer is not None:
            return [view_a, view_b, _render_context_view(self.observer, self.thickness)]
        raise ValueError(f"Condition {self.condition!r}: missing required observer / candidate_id.")

