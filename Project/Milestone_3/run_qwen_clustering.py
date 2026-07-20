"""Run Qwen-only clustering from precomputed Qwen mask-pair scores.

This keeps the original MaskClustering preprocessing and postprocessing, but
replaces the graph edges with Qwen scores:

    candidate if depth overlap passed earlier
    edge if qwen_same_instance_score >= threshold

Run in the maskclustering environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import torch


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for path in (start, *start.parents):
        if (path / "MaskClustering").is_dir() and (path / "Project").is_dir():
            return path
    raise RuntimeError("Could not find repository root from current working directory.")


REPO_ROOT = find_repo_root()
MASKCLUSTERING_ROOT = REPO_ROOT / "MaskClustering"
MILESTONE_3 = REPO_ROOT / "Project" / "Milestone_3"

if str(MASKCLUSTERING_ROOT) not in sys.path:
    sys.path.insert(0, str(MASKCLUSTERING_ROOT))

from graph.construction import mask_graph_construction  # noqa: E402
from graph.iterative_clustering import cluster_into_new_nodes  # noqa: E402
from utils.config import get_dataset  # noqa: E402
from utils.post_process import post_process  # noqa: E402


def resolve_path(path: Path | None, default: Path) -> Path:
    resolved = default if path is None else path
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def load_config(config_name: str) -> dict:
    path = MASKCLUSTERING_ROOT / "configs" / f"{config_name}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def build_qwen_graph(nodes, scores: pd.DataFrame, threshold: float) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))

    node_by_mask = {}
    for node_idx, node in enumerate(nodes):
        if len(node.mask_list) == 1:
            frame_id, mask_id = node.mask_list[0]
            node_by_mask[(int(frame_id), int(mask_id))] = node_idx

    edges_added = 0
    for _, row in scores.iterrows():
        score = float(row["qwen_same_instance_score"])
        if score < threshold:
            continue
        key_a = (int(row["frame_a"]), int(row["mask_a"]))
        key_b = (int(row["frame_b"]), int(row["mask_b"]))
        node_a = node_by_mask.get(key_a)
        node_b = node_by_mask.get(key_b)
        if node_a is None or node_b is None or node_a == node_b:
            continue
        graph.add_edge(node_a, node_b, weight=score)
        edges_added += 1

    print("Qwen graph nodes:", graph.number_of_nodes())
    print("Qwen graph edges:", edges_added)
    print("Qwen threshold  :", threshold)
    return graph


def build_score_matrices(
    nodes, scores: pd.DataFrame
) -> tuple[dict[tuple[int, int], int], np.ndarray, np.ndarray]:
    """Dense pairwise score matrix over the original (singleton) masks.

    Returns (key_to_idx, S, C): S[i, j] is the precomputed pair score, C[i, j]
    marks pairs that were actually scored (candidate pairs). Non-candidates
    contribute nothing to cluster-level aggregation.
    """
    key_to_idx: dict[tuple[int, int], int] = {}
    for node in nodes:
        for frame_id, mask_id in node.mask_list:
            key_to_idx.setdefault((int(frame_id), int(mask_id)), len(key_to_idx))
    n = len(key_to_idx)
    S = np.zeros((n, n), dtype=np.float32)
    C = np.zeros((n, n), dtype=np.float32)
    for row in scores.itertuples():
        a = key_to_idx.get((int(row.frame_a), int(row.mask_a)))
        b = key_to_idx.get((int(row.frame_b), int(row.mask_b)))
        if a is None or b is None or a == b:
            continue
        S[a, b] = S[b, a] = float(row.qwen_same_instance_score)
        C[a, b] = C[b, a] = 1.0
    return key_to_idx, S, C


def iterative_qwen_clustering(
    nodes,
    scores: pd.DataFrame,
    thresholds: list[float],
    min_pairs: int,
):
    """MaskClustering-style iterative clustering driven by precomputed pair scores.

    Mirrors graph/iterative_clustering.py, but instead of recomputing view
    consensus between merged nodes, the cluster-level edge weight is the mean
    of the precomputed pairwise scores between the two clusters' member masks
    (non-candidate pairs excluded). Progressively lower thresholds let small
    confident clusters snowball, exactly like the geometric baseline. No new
    VLM calls are needed: every pairwise score is already on disk.
    """
    from graph.iterative_clustering import cluster_into_new_nodes as merge_nodes

    key_to_idx, S, C = build_score_matrices(nodes, scores)
    n_masks = len(key_to_idx)

    for iteration, threshold in enumerate(thresholds, start=1):
        membership = np.zeros((len(nodes), n_masks), dtype=np.float32)
        for i, node in enumerate(nodes):
            for frame_id, mask_id in node.mask_list:
                idx = key_to_idx.get((int(frame_id), int(mask_id)))
                if idx is not None:
                    membership[i, idx] = 1.0

        pair_sums = membership @ S @ membership.T
        pair_counts = membership @ C @ membership.T
        mean_scores = np.divide(
            pair_sums,
            np.maximum(pair_counts, 1.0),
            out=np.zeros_like(pair_sums),
            where=pair_counts > 0,
        )

        adjacency = (mean_scores >= threshold) & (pair_counts >= min_pairs)
        np.fill_diagonal(adjacency, False)
        graph = nx.from_numpy_array(adjacency)
        nodes = merge_nodes(iteration, nodes, graph)
        print(
            f"Iteration {iteration}: threshold={threshold:.2f} "
            f"edges={int(adjacency.sum() // 2)} -> {len(nodes)} clusters"
        )
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen-only MaskClustering graph.")
    parser.add_argument("--scene", default="room0")
    parser.add_argument("--config", default="replica", help="MaskClustering config to load.")
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--qwen-score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Name used under data/prediction/<name>_class_agnostic.",
    )
    parser.add_argument("--copy-output", type=Path, default=None)
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="MaskClustering-style iterative clustering with cluster-level "
        "mean-aggregated pair scores instead of one-shot connected components.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.9,0.8,0.7,0.6,0.5",
        help="Comma-separated descending score thresholds for --iterative.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=1,
        help="Minimum number of scored member pairs required to link two clusters (--iterative).",
    )
    parser.add_argument("--debug", action="store_true")
    args_cli = parser.parse_args()

    scores_path = resolve_path(
        args_cli.scores,
        MILESTONE_3 / "data" / "qwen_scores" / f"{args_cli.scene}_qwen_scores.csv",
    )
    copy_output = resolve_path(
        args_cli.copy_output,
        MILESTONE_3
        / "data"
        / "qwen_clustering"
        / f"{args_cli.scene}_qwen_thr_{args_cli.qwen_score_threshold:.2f}.npz",
    )
    output_name = args_cli.output_name or f"qwen_{args_cli.config}_thr_{args_cli.qwen_score_threshold:.2f}"

    config = load_config(args_cli.config)
    config.update(
        {
            "seq_name": args_cli.scene,
            "seq_name_list": args_cli.scene,
            "config": output_name,
            "dataset": config["dataset"],
            "debug": args_cli.debug,
        }
    )
    mc_args = SimpleNamespace(**config)

    scores = pd.read_csv(scores_path)
    if "qwen_same_instance_score" not in scores.columns:
        raise ValueError(f"{scores_path} has no qwen_same_instance_score column.")

    previous_cwd = Path.cwd()
    os.chdir(MASKCLUSTERING_ROOT)
    try:
        dataset = get_dataset(mc_args)
        scene_points = dataset.get_scene_points()
        frame_list = dataset.get_frame_list(mc_args.step)

        with torch.no_grad():
            nodes, _, mask_point_clouds, point_frame_matrix = mask_graph_construction(
                mc_args, scene_points, frame_list, dataset
            )
            if args_cli.iterative:
                thresholds = [float(t) for t in args_cli.thresholds.split(",") if t.strip()]
                object_list = iterative_qwen_clustering(
                    nodes, scores, thresholds, args_cli.min_pairs
                )
            else:
                graph = build_qwen_graph(nodes, scores, args_cli.qwen_score_threshold)
                object_list = cluster_into_new_nodes(1, nodes, graph)
            print("Qwen clusters:", len(object_list))
            post_process(
                dataset,
                object_list,
                mask_point_clouds,
                scene_points,
                point_frame_matrix,
                frame_list,
                mc_args,
            )
    finally:
        os.chdir(previous_cwd)

    pred_path = (
        REPO_ROOT
        / "data"
        / "prediction"
        / f"{output_name}_class_agnostic"
        / f"{args_cli.scene}.npz"
    )
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    copy_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pred_path, copy_output)

    print("Prediction npz:", pred_path)
    print("Copied to     :", copy_output)


if __name__ == "__main__":
    main()
