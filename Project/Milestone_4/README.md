# Milestone 4 - Geometric-Semantic Fusion MLP

Milestone 4 combines geometric evidence from MaskClustering with the semantic same-instance score produced by the fine-tuned Qwen3-VL model. A small fully connected neural network maps these signals to one final same-instance probability for every candidate mask pair.

Candidate generation and Qwen scoring are implemented in `Project/Milestone_3`. Milestone 4 starts from the resulting scored candidate CSV files.

## Repository layout

The examples assume this layout:

```text
<repo>/
|-- data/replica/
|-- checkpoints/
|   |-- qwen/checkpoint-625/
|   `-- fusion_mlp/fusion_mlp_fused.pt
`-- Project/
    |-- Milestone_3/
    `-- Milestone_4/
```

Run all commands from the repository root. On Linux or macOS:

```bash
cd "$(git rev-parse --show-toplevel)"
```

No user-specific `/home`, `/rhome`, or `/cluster` path is required by `fusion_mlp.py`.

## Input features

The fused model uses six input features:

| Feature | Meaning |
|---|---|
| `view_consensus_score` | Fraction of shared observations supporting a connection |
| `log1p_num_observers` | Log-transformed number of shared observers, computed from `num_observers` |
| `depth_iou` | Symmetric 3D depth-overlap score |
| `overlap_a_to_b` | Directional overlap from mask A to mask B |
| `overlap_b_to_a` | Directional overlap from mask B to mask A |
| `qwen_same_instance_score` | Same-instance probability from the fine-tuned Qwen classifier |

Input CSV files must also contain `frame_a`, `mask_a`, `frame_b`, and `mask_b`. Training and validation CSV files require `same_instance_label`. Unlabeled rows are ignored during training and validation.

## Architecture

The default fused MLP is:

```text
6 input features
  -> Linear(6, 32)
  -> ReLU
  -> Dropout(0.1)
  -> Linear(32, 32)
  -> ReLU
  -> Dropout(0.1)
  -> Linear(32, 1)
  -> Sigmoid at inference time
  -> P(same instance)
```

Thus, the network has two hidden layers with 32 units each. Training uses Adam and weighted binary cross-entropy (`BCEWithLogitsLoss`) to account for class imbalance.

Features are standardized using only the training-set mean and standard deviation. The same statistics are stored in the checkpoint and reused for validation and inference.

## Feature-set ablations

`fusion_mlp.py` supports three feature sets:

| Feature set | Included features |
|---|---|
| `geometry` | View consensus, observer count, depth IoU, and both directional overlaps |
| `qwen` | Qwen same-instance score only |
| `fused` | All geometric features plus the Qwen score |

The ablations are useful for analyzing the MLP. The final method uses `--feature-set fused`.

## Train the Fusion MLP

First generate candidate CSV files and Qwen scores with the Milestone-3 pipeline. Then provide disjoint training and validation scenes:

```bash
python Project/Milestone_4/fusion_mlp.py train \
    --train-scores Project/Milestone_3/data/qwen_scores_final/room0_scored.csv \
                   Project/Milestone_3/data/qwen_scores_final/office0_scored.csv \
    --val-scores Project/Milestone_3/data/qwen_scores_final/room1_scored.csv \
    --feature-set fused \
    --model-out checkpoints/fusion_mlp/fusion_mlp_fused.pt \
    --hidden-dim 32 \
    --dropout 0.1 \
    --lr 0.001 \
    --batch-size 256 \
    --max-epochs 500 \
    --patience 30 \
    --seed 42
```

The scene names above illustrate the command format. Use the documented project split and do not use test scenes for training, feature normalization, hyperparameter selection, or early stopping.

Early stopping monitors validation ROC-AUC. The checkpoint contains:

- model weights;
- selected feature set and feature names;
- training-set normalization statistics;
- hidden dimension and dropout rate;
- validation metrics.

## Score a test scene

Apply the trained model to a scored candidate CSV:

```bash
python Project/Milestone_4/fusion_mlp.py score \
    --scores Project/Milestone_3/data/qwen_scores_final/room2_scored.csv \
    --model checkpoints/fusion_mlp/fusion_mlp_fused.pt \
    --output Project/Milestone_3/data/qwen_scores_final/room2_mlp_fused.csv
```

The output contains one final fused probability per mask pair. For compatibility with `run_qwen_clustering.py`, this probability is written to a column named `qwen_same_instance_score`, although it is now the Fusion-MLP score.

## Create final clusters

Pass the fused score CSV to the common clustering implementation in Milestone 3:

```bash
python Project/Milestone_3/run_qwen_clustering.py \
    --scene room2 \
    --scores Project/Milestone_3/data/qwen_scores_final/room2_mlp_fused.csv \
    --output-name final_mlp_fused \
    --qwen-score-threshold 0.5
```

Despite the historical argument name `--qwen-score-threshold`, the threshold is applied to the Fusion-MLP edge probability in this call. Mask pairs above the threshold become graph edges; connected components form the predicted 3D instances.

## Room0 end-to-end helper

`run_room0_mlp_inference.sh` runs candidate generation, fine-tuned Qwen scoring, Fusion-MLP scoring, and clustering for Room0. It uses the consolidated Qwen code in `Project/Milestone_3` and does not require a separate branch checkout.

Required model artifacts:

```text
checkpoints/qwen/checkpoint-625/adapter_model.safetensors
checkpoints/qwen/checkpoint-625/adapter_config.json
checkpoints/fusion_mlp/fusion_mlp_fused.pt
```

## Cluster jobs

The `.sbatch` files document the original university-cluster workflow. Cluster-specific account, partition, environment, and filesystem settings must be configured locally before submission. They should not contain personal paths in a public version of the repository.

## Main files

| File | Purpose |
|---|---|
| `fusion_mlp.py` | Train and score geometry-only, Qwen-only, or fused MLP models |
| `fusion_mlp.ipynb` | Interactive example for MLP training and scoring |
| `run_room0_mlp_inference.sh` | Room0 end-to-end helper |
| `run_m4_final.sbatch` | Full cluster workflow; requires cluster configuration |

Poster-specific export utilities belong in `Project/Poster` rather than in the production Milestone-4 pipeline.
