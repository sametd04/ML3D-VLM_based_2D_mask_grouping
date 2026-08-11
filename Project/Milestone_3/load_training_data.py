"""Training sample rendering for VLM mask-pair fine-tuning (pair_only).

Pool building -- geometry, GT labeling, overlap filtering, and positive/negative ratio
control -- all happens in build_qwen_candidates.py, which writes a candidate CSV. This
module just loads that CSV and renders every row into training samples; it does no
further filtering or resampling.

    from load_training_data import load_data_training
    samples = load_data_training(
        "candidates/room0_qwen_candidates.csv",
        masks_root="/path/to/replica_masks",
        rgb_root="/path/to/replica",
    )
    # pickle `samples` for the training/fine-tuning environment
"""

import csv
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image as PILImage
from tqdm import tqdm

logger = logging.getLogger(__name__)

from load_data import FrameView, RenderedSample, load_frame_view
from prompt_templates import ConditionType, RenderMode


@dataclass
class ViewPairMetadata:
    """One training pair's metadata, with no image arrays. Loaded from a build_qwen_candidates.py
    candidate CSV and resolved into RenderedSample images at load time."""
    scene: str
    frame_id_a: int
    mask_id_a: int
    frame_id_b: int
    mask_id_b: int
    label: int                           # 1 = same object, 0 = different
    gt_id_a: int
    gt_id_b: int
    same_class: bool                     # gt_id_a // 1000 == gt_id_b // 1000 (same semantic class, not necessarily same instance)


def load_pool(csv_path: str) -> Tuple[List[ViewPairMetadata], List[ViewPairMetadata]]:
    """Loads a build_qwen_candidates.py candidate CSV into (positives, negatives).

    Rows with same_instance_label == "" (GT unresolved for one or both masks in
    build_candidates()) are dropped -- their positive/negative status is unknown.
    """
    positives: List[ViewPairMetadata] = []
    negatives: List[ViewPairMetadata] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["same_instance_label"] == "":
                continue
            label = int(row["same_instance_label"])
            gt_a, gt_b = int(row["gt_instance_a"]), int(row["gt_instance_b"])
            meta = ViewPairMetadata(
                scene=row["scene"],
                frame_id_a=int(row["frame_a"]), mask_id_a=int(row["mask_a"]),
                frame_id_b=int(row["frame_b"]), mask_id_b=int(row["mask_b"]),
                label=label,
                gt_id_a=gt_a, gt_id_b=gt_b,
                same_class=(gt_a // 1000) == (gt_b // 1000),
            )
            (positives if label == 1 else negatives).append(meta)
    logger.info("Loaded candidates CSV %s: %d pos, %d neg", csv_path, len(positives), len(negatives))
    return positives, negatives


def _spec_fields(
    meta: ViewPairMetadata,
    condition: ConditionType,
    render_mode: RenderMode,
    alpha: float,
) -> dict:
    """Every sample-dict field except "images" -- entirely derivable from meta plus the
    condition/render_mode/alpha already resolved by the caller, no frame/image data needed.
    This is also the metadata-only "spec" shape consumed by render_spec()/LazyPairImageDataset
    (see build_pair_specs())."""
    return {
        "label": meta.label,
        "condition": condition,
        "render_mode": render_mode,
        "alpha": alpha,
        "gt_id_a": meta.gt_id_a,
        "gt_id_b": meta.gt_id_b,
        "scene": meta.scene,
        "same_class": meta.same_class,
        "frame_id_a": meta.frame_id_a,
        "mask_id_a": meta.mask_id_a,
        "frame_id_b": meta.frame_id_b,
        "mask_id_b": meta.mask_id_b,
    }


def _metadata_to_dict(
    meta: ViewPairMetadata,
    condition: ConditionType,
    images: List[PILImage.Image],
    render_mode: RenderMode,
    alpha: float,
) -> dict:
    return {"images": images, **_spec_fields(meta, condition, render_mode, alpha)}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_pairs(
    positives: List[ViewPairMetadata],
    negatives: List[ViewPairMetadata],
    masks_root: str,
    rgb_root: str,
    render_mode: RenderMode = "outline",
    alpha: float = 0.4,
) -> List[dict]:
    """Renders every positive/negative pair in the pool -- no filtering or resampling;
    build_qwen_candidates.py already curated overlap filtering and the positive/negative
    ratio when it wrote the CSV.

    Loads frames, builds RenderedSample objects, and calls render() to produce PIL images
    ready to pickle for training. This is the render half of load_data_training, factored
    out so callers that already have positives/negatives in memory (e.g. an in-memory pool)
    can skip the disk round-trip through load_pool.

    Args:
        positives: positive ViewPairMetadata, typically from load_pool.
        negatives: negative ViewPairMetadata, same provenance as positives.
        masks_root: root containing replica/<scene>/output/mask/.
        rgb_root: root containing <scene>/color/ RGB images.
        render_mode: "outline" (default) or "filled".
        alpha: fill opacity in [0, 1] for "filled" render_mode (ignored otherwise).

    Returns:
        List of sample dicts (images, label, condition, and pair metadata).
    """
    condition: ConditionType = "pair_only"
    logger.info(
        "Rendering pairs (render_mode=%s): %d positives, %d negatives",
        render_mode, len(positives), len(negatives),
    )

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
    for meta in tqdm(positives + negatives, desc="rendering samples", unit="pair"):
        frame_a = get_frame(meta.scene, meta.frame_id_a)
        frame_b = get_frame(meta.scene, meta.frame_id_b)
        if frame_a is None or frame_b is None:
            logger.warning("Missing cached FrameView for %s -> skipping", meta)
            continue

        rendered = RenderedSample(
            condition=condition,
            frame_a=frame_a, mask_id_a=meta.mask_id_a,
            frame_b=frame_b, mask_id_b=meta.mask_id_b,
            render_mode=render_mode,
            alpha=alpha,
        )
        try:
            images = rendered.render()
        except Exception as exc:
            logger.warning("render() failed: %s", exc)
            continue
        results.append(_metadata_to_dict(meta, condition, images, render_mode, alpha))

    logger.info("Done: %d samples rendered", len(results))
    return results


def build_pair_specs(
    positives: List[ViewPairMetadata],
    negatives: List[ViewPairMetadata],
    render_mode: RenderMode = "outline",
    alpha: float = 0.4,
) -> List[dict]:
    """Same pool/no-filtering semantics as render_pairs(), but returns metadata-only spec
    dicts (no "images" key) instead of rendered samples -- pair_only needs zero frame I/O
    to build a spec.

    Render each spec on demand with render_spec() -- e.g. inside a Dataset.__getitem__
    (see fine_tuning.py's LazyPairImageDataset) -- instead of eagerly rendering everything
    upfront like render_pairs() does. This is the memory-safe choice for large sample
    counts (tens of thousands of pairs), where holding every rendered image in memory at
    once is what causes an OOM.
    """
    condition: ConditionType = "pair_only"
    logger.info(
        "Building pair specs (render_mode=%s): %d positives, %d negatives",
        render_mode, len(positives), len(negatives),
    )

    specs = [_spec_fields(meta, condition, render_mode, alpha) for meta in positives + negatives]

    logger.info("Built %d specs -- no eager frame/image materialization", len(specs))
    return specs


def render_spec(spec: dict, masks_root: str, rgb_root: str) -> Optional[List[PILImage.Image]]:
    """Renders one metadata-only spec (from build_pair_specs()) into its image list.

    Reloads every needed frame from disk on every call -- no cache, by design: simplicity
    over redundant I/O for the lazy per-sample rendering path (see fine_tuning.py's
    LazyPairImageDataset and score()'s inline lazy-batch step, both of which call this).

    Returns None (after logging a warning) on any failure -- missing frame or render()
    raising -- instead of raising, so callers can uniformly treat "couldn't render this
    one" as a drop, matching how render_pairs() already just logs+skips on the same
    failures today.
    """
    render_mode = spec["render_mode"]
    alpha = spec.get("alpha", 0.4)

    try:
        frame_a = load_frame_view(spec["scene"], spec["frame_id_a"], masks_root, rgb_root)
        frame_b = load_frame_view(spec["scene"], spec["frame_id_b"], masks_root, rgb_root)
    except FileNotFoundError as e:
        logger.warning("render_spec: missing frame_a/frame_b: %s", e)
        return None

    rendered = RenderedSample(
        condition=spec["condition"],
        frame_a=frame_a, mask_id_a=spec["mask_id_a"],
        frame_b=frame_b, mask_id_b=spec["mask_id_b"],
        render_mode=render_mode,
        alpha=alpha,
    )

    try:
        return rendered.render()
    except Exception as exc:
        logger.warning("render_spec: render() failed: %s", exc)
        return None


def load_data_training(
    pool_path: str,
    masks_root: str,
    rgb_root: str,
    render_mode: RenderMode = "outline",
    alpha: float = 0.4,
) -> List[dict]:
    """Loads a candidate CSV from disk and renders every row for SFTTrainer. Thin wrapper
    around render_pairs; see that docstring for the rendering behavior. Use render_pairs
    directly if positives/negatives are already in memory.

    Args:
        pool_path: candidate CSV written by build_qwen_candidates.py.
        masks_root: root containing replica/<scene>/output/mask/.
        rgb_root: root containing <scene>/color/ RGB images.
        render_mode: "outline" (default) or "filled".
        alpha: fill opacity in [0, 1] for "filled" render_mode (ignored otherwise).

    Returns:
        List of sample dicts (images, label, condition, and pair metadata).
    """
    logger.info("Loading pair pool from %s", pool_path)
    positives, negatives = load_pool(pool_path)
    return render_pairs(positives, negatives, masks_root, rgb_root, render_mode=render_mode, alpha=alpha)
