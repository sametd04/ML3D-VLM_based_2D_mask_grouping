"""Training data pipeline for VLM mask-pair fine-tuning.

Both phases run in the MaskClustering env (Open3D + CUDA).

Step 1 — build pair pool metadata:
    python load_training_data.py --scenes office0 room0 \
        --masks-root /path/to/replica_masks \
        --rgb-root /path/to/replica \
        --out /path/to/pairs/

Step 2 — render training samples (same environment):
    from load_training_data import load_data_training
    samples = load_data_training(
        "pairs/room0.pkl",
        masks_root="/path/to/replica_masks",
        rgb_root="/path/to/replica",
    )
    # pickle `samples` for the training/fine-tuning environment
"""

import argparse
import itertools
import logging
import os
import pickle
import types
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MaskClustering path setup — only needed in the build phase.
# ---------------------------------------------------------------------------
_MASKCLUSTERING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MaskClustering"))
_MASKCLUSTERING_UTILS = os.path.join(_MASKCLUSTERING_ROOT, "utils")
for _p in (_MASKCLUSTERING_ROOT, _MASKCLUSTERING_UTILS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dataset.replica import ReplicaDataset
    from graph.construction import build_point_in_mask_matrix
    _BUILD_DEPS_AVAILABLE = True
except ImportError:
    _BUILD_DEPS_AVAILABLE = False

print(f'_BUILD_DEPS_AVAILABLE (required for obtaining training data): {_BUILD_DEPS_AVAILABLE}')
from load_data import FrameView, RenderedSample, load_frame_view
from prompt_templates import ConditionType


@dataclass
class ViewPairMetadata:
    """Metadata identifying one training sample — no image arrays stored.

    Serialized to disk (save_pair_pool) during build; loaded at training time
    to reconstruct RenderedSample objects from the FrameView pickle cache.
    neg_strategy is None for positives, "both_views" or "co_visible" for negatives.
    angle_deg is the inter-camera rotation angle between frame_a and frame_b.
    """
    scene: str
    frame_id_a: int
    mask_id_a: int
    frame_id_b: int
    mask_id_b: int
    label: int                           # 1 = same object, 0 = different
    gt_id_a: int
    gt_id_b: int
    same_category: bool                  # gt_id_a // 1000 == gt_id_b // 1000
    observer_frame_id: Optional[int] = None
    neg_strategy: Optional[str] = None   # None | "both_views" | "co_visible"
    angle_deg: Optional[float] = None    # inter-camera rotation angle in degrees


# ---------------------------------------------------------------------------
# Step-1 visibility (build phase)
# ---------------------------------------------------------------------------

def map_masks_to_visible_frames(
    frame_list: List[int],
    dataset: "ReplicaDataset",
    scene_points: np.ndarray,  # (N, 3) mesh vertices from dataset.get_scene_points()
    mask_visible_threshold: float = 0.3,
) -> Tuple[Dict[Tuple[int, int], torch.Tensor], Dict[int, int]]:
    """For every 2D mask in the scene, compute which frames its 3D object is visible in (Step 1 Criterion in Original MaskClustering).

    Returns:
        visibility: Dict[(frame_id, mask_id), FloatTensor(num_frames,)] — one vector per
            mask; index i is 1.0 if that mask's 3D object is visible in frame_list[i].
        frame_idx: Dict[frame_id, int] — maps frame_id to its column index in the visibility
            vectors (frame ids are arbitrary integers, columns are 0,1,2,...). To check
            "is mask B's 3D object visible in frame A?":
            visibility[(frame_id_b, mask_id_b)][frame_idx[frame_id_a]].
    """
    if not _BUILD_DEPS_AVAILABLE:
        raise RuntimeError(
            "map_masks_to_visible_frames requires the MaskClustering environment (Open3D, CUDA)."
        )
    scene_points_tensor = torch.tensor(scene_points, dtype=torch.float32).cuda()

    # Projects every frame's 2D masks onto the 3D mesh, producing a
    # (num_points, num_frames) lookup table shared across all masks.
    boundary_points, point_in_mask_matrix, mask_point_clouds, _, global_frame_mask_list = \
        build_point_in_mask_matrix(types.SimpleNamespace(debug=False), scene_points_tensor, frame_list, dataset)

    frame_idx: Dict[int, int] = {frame_id: i for i, frame_id in enumerate(frame_list)}
    visibility: Dict[Tuple[int, int], torch.Tensor] = {}

    for frame_id, mask_id in tqdm(global_frame_mask_list, desc="masks→visibility", unit="mask", leave=False):
        pc = mask_point_clouds.get(f"{frame_id}_{mask_id}", set())
        # For this mask's 3D points, check in which frames enough of them are visible.
        visibility[(frame_id, mask_id)] = get_visible_frames(
            point_in_mask_matrix, boundary_points, pc, frame_list, mask_visible_threshold
        )

    logger.info("Computed visibility for %d masks across %d frames", len(visibility), len(frame_list))
    return visibility, frame_idx


def get_visible_frames(
    point_in_mask_matrix: np.ndarray,
    boundary_points: set,
    mask_point_cloud: set,
    frame_list: List[int],
    mask_visible_threshold: float = 0.3,
) -> torch.Tensor:
    """Check in which frames this mask's 3D object is visible (Step-1, no containment check).

    Returns FloatTensor(num_frames,) — index i is 1.0 if ≥ mask_visible_threshold of
    the mask's non-boundary 3D points are visible in that frame, or ≥ 500 points visible
    regardless of ratio. All arguments are outputs of build_point_in_mask_matrix.
    """
    visible_frame = torch.zeros(len(frame_list))

    # Exclude boundary points — they appear in multiple masks and are unreliable.
    valid_pts = mask_point_cloud - boundary_points
    if not valid_pts:
        return visible_frame

    # Slice the shared matrix to rows for this mask's 3D points: (|valid_pts|, num_frames).
    # Entry [p, f] = mask_id of point p in frame f; 0 means the point is invisible there.
    mask_info = point_in_mask_matrix[list(valid_pts), :]

    # Only examine frames where at least one of these points has any projected mask_id.
    possibly_visible = np.where(np.sum(mask_info, axis=0) > 0)[0]

    for idx in possibly_visible:
        counts = np.bincount(mask_info[:, idx])  # counts[0] = number of valid_pts invisible in this frame (mask_id == 0)
        total = int(counts.sum())
        visible_pts = total - int(counts[0])
        invisible_ratio = counts[0] / total

        if 1 - invisible_ratio < mask_visible_threshold and visible_pts < 500:
            continue  # skip: visible ratio below threshold and fewer than 500 absolute points

        visible_frame[idx] = 1.0

    return visible_frame


def precompute_all_frames(
    scene: str,
    masks_root: str,
    rgb_root: Optional[str] = None,
    run_assign_gt: bool = True,
) -> List[FrameView]:
    """Load all frames for a scene into memory, optionally running GT assignment.

    Discovers available frames from the mask directory. Requires the MaskClustering
    environment when run_assign_gt=True (CUDA + ReplicaDataset).

    Args:
        scene:         Replica scene name, e.g. "room0".
        masks_root:    Path to the replica_masks directory.
        rgb_root:      Path to the Replica scene directory. If None, FrameView.rgb is None.
        run_assign_gt: If True, call assign_gt() on each frame.

    Returns:
        List of FrameView objects, one per successfully loaded frame.
    """
    mask_dir = os.path.join(masks_root, "replica", scene, "output", "mask")
    frame_ids = sorted(
        int(os.path.splitext(f)[0])
        for f in os.listdir(mask_dir)
        if f.endswith(".png")
    )
    logger.info("[precompute_all_frames] scene=%s: %d frames found", scene, len(frame_ids))

    frame_views: List[FrameView] = []
    for frame_id in tqdm(frame_ids, desc=f"loading frames [{scene}]", unit="frame", leave=False):
        try:
            fv = load_frame_view(scene, frame_id, masks_root, rgb_root)
        except FileNotFoundError as e:
            logger.warning("[precompute_all_frames] skipping frame %d: %s", frame_id, e)
            continue
        if run_assign_gt:
            fv.assign_gt()
        frame_views.append(fv)

    logger.info("[precompute_all_frames] loaded %d/%d frames for scene=%s", len(frame_views), len(frame_ids), scene)
    return frame_views


# ---------------------------------------------------------------------------
# Pair enumeration helpers
# ---------------------------------------------------------------------------

def _group_masks_by_gt_instance(
    frame_views: List[FrameView],
) -> Dict[int, List[Tuple[int, int]]]:
    """Map each GT instance id to all (frame_id, mask_id) detections across the scene."""
    groups: Dict[int, List[Tuple[int, int]]] = {}
    for fv in frame_views:
        for mask_id, region in fv.masks.items():
            if region.gt_instance_id is not None:
                groups.setdefault(region.gt_instance_id, []).append((fv.frame_id, mask_id))
    return groups


def _pick_observer(
    vis_a: torch.Tensor,
    vis_b: torch.Tensor,
    frame_list: List[int],
) -> Optional[int]:
    """Return the first frame_id where both masks are Step-1-visible, or None."""
    shared = torch.where(vis_a.bool() & vis_b.bool())[0]
    if len(shared) == 0:
        return None
    return frame_list[shared[0].item()]


def _camera_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Inter-camera rotation angle in degrees between two 3x3 rotation matrices."""
    trace_val = np.clip((np.trace(R_a.T @ R_b) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(trace_val)))


# ---------------------------------------------------------------------------
# Positive enumeration
# ---------------------------------------------------------------------------

def enumerate_positives(
    instance_groups: Dict[int, List[Tuple[int, int]]],
    visibility: Dict[Tuple[int, int], torch.Tensor],
    frame_list: List[int],
    dataset: "ReplicaDataset",
    scene: str,
    min_angle_deg: float = 15.0,
) -> List[ViewPairMetadata]:
    """Pairs of detections of the same GT instance from sufficiently different viewpoints.

    For each GT instance, generates all (mask_A, mask_B) pairs where the inter-camera
    rotation angle ≥ min_angle_deg and both masks share at least one observer frame.
    Returns List[ViewPairMetadata] with label=1.
    """
    positives: List[ViewPairMetadata] = []

    for gt_id, detections in tqdm(instance_groups.items(), desc=f"positives [{scene}]", unit="instance", leave=False):
        if len(detections) < 2:
            continue

        # Cache rotation matrices (single get_extrinsic call per unique frame)
        rotations: Dict[int, np.ndarray] = {}
        for frame_id, _ in detections:
            if frame_id not in rotations:
                rotations[frame_id] = dataset.get_extrinsic(frame_id)[:3, :3]

        # Enumerate all unique pairs of detections for this GT instance
        for (frame_id_a, mask_id_a), (frame_id_b, mask_id_b) in itertools.combinations(detections, 2):
            angle = _camera_angle_deg(rotations[frame_id_a], rotations[frame_id_b])
            if angle < min_angle_deg:  # Too similar viewpoints; skip this pair.
                continue

            vis_a = visibility.get((frame_id_a, mask_id_a))
            vis_b = visibility.get((frame_id_b, mask_id_b))
            if vis_a is None or vis_b is None:
                continue

            # Require a shared observer frame — needed for Condition B/C images.
            obs_fid = _pick_observer(vis_a, vis_b, frame_list)
            if obs_fid is None:
                continue

            positives.append(ViewPairMetadata(
                scene=scene,
                frame_id_a=frame_id_a, mask_id_a=mask_id_a,
                frame_id_b=frame_id_b, mask_id_b=mask_id_b,
                label=1,
                gt_id_a=gt_id, gt_id_b=gt_id,
                same_category=True,
                observer_frame_id=obs_fid,
                neg_strategy=None,
                angle_deg=angle,
            ))

    logger.info("scene=%s: %d positives from %d GT instances", scene, len(positives), len(instance_groups))
    return positives


# ---------------------------------------------------------------------------
# Negative enumeration
# ---------------------------------------------------------------------------

def enumerate_negatives(
    instance_groups: Dict[int, List[Tuple[int, int]]],
    visibility: Dict[Tuple[int, int], torch.Tensor],
    frame_list: List[int],
    frame_idx: Dict[int, int],
    scene: str,
    dataset: "ReplicaDataset",
) -> List[ViewPairMetadata]:
    """Full negative pool: both-views (hard) first, co-visible (relaxed) supplement.

    Both-views: each source frame must see both objects — the model cannot rely on
    context since both objects appear in each other's view.
    Co-visible: only requires a shared observer frame — used to supplement when the
    both-views pool is thin. Pairs already in both-views are excluded.

    Returns both pools concatenated; use neg_strategy field to distinguish.
    """
    rotation_cache: Dict[int, np.ndarray] = {}
    gt_ids = list(instance_groups.keys())

    def get_rotation(frame_id: int) -> np.ndarray:
        if frame_id not in rotation_cache:
            rotation_cache[frame_id] = dataset.get_extrinsic(frame_id)[:3, :3]
        return rotation_cache[frame_id]

    exclude: set = set()  # populated after pass 1; referenced by pass 2 via closure

    def run_loop(neg_strategy: str) -> List[ViewPairMetadata]:
        negatives: List[ViewPairMetadata] = []
        n_pairs = len(gt_ids) * (len(gt_ids) - 1) // 2
        for gt_a, gt_b in tqdm(itertools.combinations(gt_ids, 2), desc=f"negatives/{neg_strategy} [{scene}]", total=n_pairs, unit="pair", leave=False):
            # same_category: both instances share the same semantic class (gt_id // 1000 = class label).
            same_cat = (gt_a // 1000) == (gt_b // 1000)
            for frame_id_a, mask_id_a in instance_groups[gt_a]:
                vis_a = visibility.get((frame_id_a, mask_id_a))
                if vis_a is None:
                    continue
                for frame_id_b, mask_id_b in instance_groups[gt_b]:
                    vis_b = visibility.get((frame_id_b, mask_id_b))
                    if vis_b is None:
                        continue
                    if neg_strategy == "both_views":
                        # Both-views constraint: object B visible in frame_A AND object A visible in frame_B.
                        idx_a, idx_b = frame_idx.get(frame_id_a), frame_idx.get(frame_id_b)
                        if idx_a is None or idx_b is None or vis_b[idx_a] < 0.5 or vis_a[idx_b] < 0.5:
                            continue
                    else:
                        # Co-visible constraint: skip pairs already covered by both-views.
                        if frozenset({(frame_id_a, mask_id_a), (frame_id_b, mask_id_b)}) in exclude:
                            continue
                    obs_fid = _pick_observer(vis_a, vis_b, frame_list)
                    if obs_fid is None:
                        continue
                    negatives.append(ViewPairMetadata(
                        scene=scene,
                        frame_id_a=frame_id_a, mask_id_a=mask_id_a,
                        frame_id_b=frame_id_b, mask_id_b=mask_id_b,
                        label=0,
                        gt_id_a=gt_a, gt_id_b=gt_b,
                        same_category=same_cat,
                        observer_frame_id=obs_fid,
                        neg_strategy=neg_strategy,
                        angle_deg=_camera_angle_deg(get_rotation(frame_id_a), get_rotation(frame_id_b)),
                    ))
        return negatives

    # Pass 1: both-views hard negatives.
    both_views = run_loop("both_views")

    # Pass 2: co-visible relaxed negatives — exclude already-covered pairs.
    exclude = {frozenset({(n.frame_id_a, n.mask_id_a), (n.frame_id_b, n.mask_id_b)}) for n in both_views}
    co_visible = run_loop("co_visible")

    logger.info(
        "scene=%s negatives: %d both-views + %d co-visible = %d total",
        scene, len(both_views), len(co_visible), len(both_views) + len(co_visible),
    )
    return both_views + co_visible


# ---------------------------------------------------------------------------
# Pool persistence
# ---------------------------------------------------------------------------

def save_pair_pool(
    positives: List[ViewPairMetadata],
    negatives: List[ViewPairMetadata],
    path: str,
) -> None:
    """Pickle the full enumerated pair pool to disk.

    Args:
        positives: list of positive ViewPairMetadata (label=1).
        negatives: list of negative ViewPairMetadata (label=0).
        path: destination file; parent directory is created if needed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"positives": positives, "negatives": negatives}, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved pair pool to %s (%d pos, %d neg)", path, len(positives), len(negatives))


class _PoolUnpickler(pickle.Unpickler):
    """Remap __main__.ViewPairMetadata to load_training_data.ViewPairMetadata.

    Pools built by running load_training_data.py directly pickle ViewPairMetadata
    under __main__. This unpickler resolves it to the correct module regardless
    of which script is currently __main__.
    """
    def find_class(self, module, name):
        if module == "__main__" and name == "ViewPairMetadata":
            module = "load_training_data"
        return super().find_class(module, name)


def load_pair_pool(path: str) -> Tuple[List[ViewPairMetadata], List[ViewPairMetadata]]:
    """Load a pair pool saved by save_pair_pool.

    No MaskClustering imports required — safe to call in the Qwen3-VL environment.

    Returns:
        (positives, negatives) lists of ViewPairMetadata.
    """
    with open(path, "rb") as f:
        data = _PoolUnpickler(f).load()
    logger.info("Loaded pair pool from %s: %d pos, %d neg", path, len(data["positives"]), len(data["negatives"]))
    return data["positives"], data["negatives"]


# ---------------------------------------------------------------------------
# Build-phase orchestrator
# ---------------------------------------------------------------------------

def build_pair_pool(
    scenes: List[str],
    masks_root: str,
    rgb_root: str,
    out_dir: str,
    min_angle_deg: float = 15.0,
    mask_visible_threshold: float = 0.3,
) -> None:
    """Run the full build phase: GT assignment, visibility, pair enumeration.

    Requires the MaskClustering environment (Open3D, CUDA). Writes one pair
    pool pickle per scene to out_dir.

    Args:
        scenes: Replica scene names, e.g. ["office0", "room0"].
        masks_root: root containing replica/<scene>/output/mask/.
        rgb_root: root containing <scene>/color/ RGB images.
        out_dir: directory where per-scene pickles are written (<scene>.pkl).
        min_angle_deg: minimum inter-camera angle for positive pairs.
        mask_visible_threshold: τ_vis for Step-1 visibility.
    """
    # ReplicaDataset hardcodes ./data/replica/ relative to CWD. Resolve all
    # caller-supplied paths to absolute before switching to the directory that
    # contains data/replica/ (one level up from this file = Project/).
    masks_root = os.path.abspath(masks_root)
    rgb_root = os.path.abspath(rgb_root)
    out_dir = os.path.abspath(out_dir)
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    for scene in tqdm(scenes, desc="scenes", unit="scene"):
        logger.info("Starting scene: %s", scene)

        dataset = ReplicaDataset(scene)
        # ReplicaDataset hardcodes './data/replica/{scene}'; override with the
        # caller-supplied rgb_root so the script works regardless of CWD.
        dataset.root = os.path.join(rgb_root, scene)
        dataset.rgb_dir = os.path.join(dataset.root, "color")
        dataset.depth_dir = os.path.join(dataset.root, "depth")
        dataset.point_cloud_path = os.path.join(dataset.root, f"{scene}_mesh.ply")
        dataset.mesh_path = dataset.point_cloud_path
        dataset.extrinsics_dir = os.path.join(dataset.root, "poses")
        scene_points = dataset.get_scene_points()
        frame_list = dataset.get_frame_list(stride=1)
        logger.info("[build_pair_pool] %d frames loaded", len(frame_list))

        frame_views = precompute_all_frames(
            scene=scene,
            masks_root=masks_root,
            rgb_root=rgb_root,
            run_assign_gt=True,
        )
        logger.info("[precompute_all_frames] %d frame views precomputed", len(frame_views))

        visibility, frame_idx = map_masks_to_visible_frames(
            frame_list=frame_list,
            dataset=dataset,
            scene_points=scene_points,
            mask_visible_threshold=mask_visible_threshold,
        )
        logger.info("[map_masks_to_visible_frames] %d (frame, mask) pairs", len(visibility))

        instance_groups = _group_masks_by_gt_instance(frame_views)
        logger.info("[group_masks_by_gt_instance] %d GT instances", len(instance_groups))

        pos = enumerate_positives(
            instance_groups, visibility, frame_list, dataset, scene, min_angle_deg
        )
        neg = enumerate_negatives(
            instance_groups, visibility, frame_list, frame_idx, scene, dataset
        )
        logger.info("[enumerate_pairs] %d positives, %d negatives", len(pos), len(neg))

        save_pair_pool(pos, neg, os.path.join(out_dir, f"{scene}.pkl"))
        logger.info("Finished scene: %s", scene)


# ---------------------------------------------------------------------------
# Train-phase loader
# ---------------------------------------------------------------------------

def _metadata_to_dict(
    meta: ViewPairMetadata,
    condition: ConditionType,
    images: List[PILImage.Image],
    candidate_id: Optional[int],
) -> dict:
    return {
        "images": images,
        "label": meta.label,
        "condition": condition,
        "gt_id_a": meta.gt_id_a,
        "gt_id_b": meta.gt_id_b,
        "scene": meta.scene,
        "same_category": meta.same_category,
        "neg_strategy": meta.neg_strategy,
        "angle_deg": meta.angle_deg,
        "frame_id_a": meta.frame_id_a,
        "mask_id_a": meta.mask_id_a,
        "frame_id_b": meta.frame_id_b,
        "mask_id_b": meta.mask_id_b,
        "observer_frame_id": meta.observer_frame_id,
        "candidate_id": candidate_id,
    }


def load_data_training(
    pool_path: str,
    masks_root: str,
    rgb_root: str,
    K: int = 15,
    neg_ratio: int = 3,
    conditions: Optional[List[ConditionType]] = None,
    seed: int = 42,
    same_category_only: bool = False,
    max_positives: Optional[int] = None,
) -> List[dict]:
    """Sample from a pre-built pair pool and render images for SFTTrainer.

    Runs in the MaskClustering environment (same as build_pair_pool).
    Loads raw frames via load_frame_view, builds RenderedSample objects,
    and calls render() to produce PIL images. The returned list can be
    pickled and handed off to the fine-tuning environment.

    Args:
        pool_path: path to a pair pool pickle written by build_pair_pool.
        masks_root: root containing replica/<scene>/output/mask/.
        rgb_root: root containing <scene>/color/ RGB images.
        K: maximum positives to sample per GT instance (ignored when max_positives is set).
        neg_ratio: number of negatives per positive.
        conditions: conditions to produce samples for; defaults to all three.
        seed: random seed for reproducible sampling.
        same_category_only: if True, restrict negatives to same-category pairs.
        max_positives: if set, draw this many positives as a flat random sample
            from the full positive pool instead of K per GT instance.

    Returns:
        List of dicts with keys: images, label, condition, gt_id_a, gt_id_b,
        scene, same_category, neg_strategy, angle_deg, frame_id_a, mask_id_a,
        frame_id_b, mask_id_b, observer_frame_id, candidate_id.
    """
    if conditions is None:
        conditions = ["pair_only", "pair_with_context", "pair_with_candidate"]

    logger.info("Loading pair pool from %s (conditions=%s, K=%d, neg_ratio=%d)", pool_path, conditions, K, neg_ratio)
    rng = random.Random(seed)
    positives, negatives = load_pair_pool(pool_path)

    if max_positives is not None:
        n = min(max_positives, len(positives))
        sampled_pos = rng.sample(positives, n)
        logger.info("Sampled %d positives (flat random, max_positives=%d)", len(sampled_pos), max_positives)
    else:
        pos_by_instance: Dict[int, List[ViewPairMetadata]] = {}
        for meta in positives:
            pos_by_instance.setdefault(meta.gt_id_a, []).append(meta)

        sampled_pos: List[ViewPairMetadata] = []
        for specs in pos_by_instance.values():
            rng.shuffle(specs)
            sampled_pos.extend(specs[:K])

        logger.info("Sampled %d positives (K=%d per instance)", len(sampled_pos), K)

    neg_pool = [m for m in negatives if m.same_category] if same_category_only else list(negatives)
    rng.shuffle(neg_pool)
    sampled_neg = neg_pool[: neg_ratio * len(sampled_pos)]
    logger.info("Sampled %d negatives from pool of %d (same_category_only=%s)", len(sampled_neg), len(neg_pool), same_category_only)

    frame_cache: Dict[Tuple[str, int], Optional[FrameView]] = {}

    def get_frame(scene: str, frame_id: int) -> Optional[FrameView]:
        key = (scene, frame_id)
        if key not in frame_cache:
            try:
                frame_cache[key] = load_frame_view(scene, frame_id, masks_root, rgb_root)
            except FileNotFoundError:
                frame_cache[key] = None
        return frame_cache[key]

    results: List[dict] = []

    for meta in tqdm(sampled_pos + sampled_neg, desc="rendering samples", unit="pair"):
        frame_a = get_frame(meta.scene, meta.frame_id_a)
        frame_b = get_frame(meta.scene, meta.frame_id_b)
        if frame_a is None or frame_b is None:
            logger.warning("Missing cached FrameView for %s — skipping", meta)
            continue

        for condition in conditions:
            if condition != "pair_only" and meta.observer_frame_id is None:
                continue

            observer: Optional[FrameView] = None
            if meta.observer_frame_id is not None:
                observer = get_frame(meta.scene, meta.observer_frame_id)
                if observer is None:
                    logger.warning(
                        "Missing observer frame %d for condition=%s — skipping",
                        meta.observer_frame_id, condition,
                    )
                    continue

            if condition == "pair_with_candidate":
                for cand_id in observer.masks:
                    rendered = RenderedSample(
                        condition=condition,
                        frame_a=frame_a, mask_id_a=meta.mask_id_a,
                        frame_b=frame_b, mask_id_b=meta.mask_id_b,
                        observer=observer,
                        candidate_id=cand_id,
                    )
                    try:
                        images = rendered.render()
                    except Exception as exc:
                        logger.warning("render() failed cand=%d: %s", cand_id, exc)
                        continue
                    results.append(_metadata_to_dict(meta, condition, images, cand_id))
            else:
                rendered = RenderedSample(
                    condition=condition,
                    frame_a=frame_a, mask_id_a=meta.mask_id_a,
                    frame_b=frame_b, mask_id_b=meta.mask_id_b,
                    observer=observer,
                )
                try:
                    images = rendered.render()
                except Exception as exc:
                    logger.warning("render() failed condition=%s: %s", condition, exc)
                    continue
                results.append(_metadata_to_dict(meta, condition, images, None))

    logger.info("Done: %d samples rendered (conditions=%s)", len(results), conditions)
    return results


# ---------------------------------------------------------------------------
# Build-phase CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build a ViewPairMetadata pool for VLM training.")
    parser.add_argument("--scenes", nargs="+", required=True, help="Replica scene names")
    parser.add_argument("--masks-root", required=True, help="Root of replica_masks/")
    parser.add_argument("--rgb-root", required=True, help="Root of Replica dataset (<scene>/color/)")
    parser.add_argument("--out", required=True, help="Output directory for per-scene pair pool pickles (<scene>.pkl)")
    parser.add_argument("--min-angle-deg", type=float, default=15.0)
    parser.add_argument("--mask-visible-threshold", type=float, default=0.3)
    args = parser.parse_args()

    build_pair_pool(
        scenes=args.scenes,
        masks_root=args.masks_root,
        rgb_root=args.rgb_root,
        out_dir=args.out,
        min_angle_deg=args.min_angle_deg,
        mask_visible_threshold=args.mask_visible_threshold,
    )
