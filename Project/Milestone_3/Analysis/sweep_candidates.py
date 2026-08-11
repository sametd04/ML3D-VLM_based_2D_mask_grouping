"""Sweep build_qwen_candidates.py's depth-overlap/view-consensus thresholds across scenes.

Run in the MaskClustering environment (myenv), from Project/Milestone_3/:

    python sweep_candidates.py

Calls build_candidates() in-process per (scene, depth_overlap_threshold, vc_threshold)
combo, writes each combo's CSV + summary.json under OUTPUT_DIR, and logs a combined
table (candidate_pairs_kept, gt_positive_kept_pct, gt_negative_kept_pct) across the
whole sweep so combos can be compared side by side.

vc_threshold only affects results when FILTER_MODE == "or_rule" -- "iou" and
"any_asymmetric" never read it (see passes_overlap_filter in build_qwen_candidates.py).

Caveat: build_candidates() bundles the (GPU) point-in-mask geometry computation together
with threshold filtering, so each combo below reruns that geometry from scratch for its
scene -- there's no caching across thresholds. Fine for a handful of combos per scene;
shrink SCENES/DEPTH_OVERLAP_THRESHOLDS/VC_THRESHOLDS if a full grid is too slow.
"""

import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from build_qwen_candidates import build_candidates, write_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCENES = ["office0", "office1", "office2", "room0"]  # training rooms -- edit as needed
DEPTH_OVERLAP_THRESHOLDS = [0.0, 0.025, 0.05, 0.1]
VC_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]
FILTER_MODE = "or_rule"  # only mode where VC_THRESHOLDS has any effect
CONFIG = "replica"
OUTPUT_DIR = Path("data/candidates/sweep").resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_rows = []

for scene in SCENES:
    for depth_thr in DEPTH_OVERLAP_THRESHOLDS:
        for vc_thr in VC_THRESHOLDS:
            tag = f"{scene}_depth{depth_thr}_vc{vc_thr}"
            logger.info("Running %s (filter_mode=%s)", tag, FILTER_MODE)

            args = SimpleNamespace(
                scene=scene, config=CONFIG, debug=False,
                depth_overlap_threshold=depth_thr, filter_mode=FILTER_MODE, vc_threshold=vc_thr,
                frames=None, max_frames=None, include_same_frame=False,
                keep_undersegmented=False, min_labeled_points=10,
            )
            rows, metadata = build_candidates(args)

            write_csv(rows, OUTPUT_DIR / f"{tag}.csv")
            (OUTPUT_DIR / f"{tag}.summary.json").write_text(json.dumps(metadata, indent=2))

            log_rows.append({
                "scene": scene,
                "depth_overlap_threshold": depth_thr,
                "vc_threshold": vc_thr,
                "filter_mode": FILTER_MODE,
                "candidate_pairs_kept": metadata["candidate_pairs_kept"],
                "gt_positive_pairs_kept": metadata["gt_positive_pairs_kept"],
                "gt_positive_kept_pct": metadata["gt_positive_kept_pct"],
                "gt_negative_pairs_kept": metadata["gt_negative_pairs_kept"],
                "gt_negative_kept_pct": metadata["gt_negative_kept_pct"],
            })

            logger.info(
                "  kept=%d  gt_pos_kept_pct=%s  gt_neg_kept_pct=%s",
                metadata["candidate_pairs_kept"],
                metadata["gt_positive_kept_pct"],
                metadata["gt_negative_kept_pct"],
            )

log_path = OUTPUT_DIR / "sweep_log.csv"
with log_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
    writer.writeheader()
    writer.writerows(log_rows)
logger.info("Sweep log written to %s", log_path)