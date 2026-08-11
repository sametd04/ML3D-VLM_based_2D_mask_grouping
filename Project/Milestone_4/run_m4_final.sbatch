#!/usr/bin/env bash
#SBATCH --job-name=m4_final
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7:55:00
#SBATCH --output=m4_final_%j.log

# Portable Milestone-4 pipeline:
#   1. build high-recall candidates for every selected scene;
#   2. score candidates with the fine-tuned Qwen classifier;
#   3. convert Qwen outputs to per-pair CSV files;
#   4. train geometry, Qwen, and fused MLP ablations;
#   5. score and cluster the held-out test scenes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
M3="$REPO/Project/Milestone_3"
M4="$REPO/Project/Milestone_4"

ADAPTER="${ADAPTER:-$REPO/checkpoints/qwen/checkpoint-625}"
QWEN_PYTHON="${QWEN_PYTHON:-$REPO/Project/env/qwen_env/bin/python}"
MC_ENV="${MC_ENV:-maskclustering}"

CAND="$M3/data/candidates_orule"
SCORES="$M3/data/qwen_scores_final"
MODELS="${MODELS:-$REPO/checkpoints/fusion_mlp}"

# Defaults follow the documented split. Override any list when submitting:
#   sbatch --export=ALL,TRAIN_SCENES="...",VAL_SCENES="..." run_m4_final.sbatch
TRAIN_SCENES="${TRAIN_SCENES:-room1 office3 room0 office0}"
VAL_SCENES="${VAL_SCENES:-office1 office2}"
TEST_SCENES="${TEST_SCENES:-room2 office4}"
ALL_SCENES="$TRAIN_SCENES $VAL_SCENES $TEST_SCENES"

export HF_HOME="${HF_HOME:-$REPO/Project/env/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$M3:$M4${PYTHONPATH:+:$PYTHONPATH}"

for required in \
    "$QWEN_PYTHON" \
    "$M3/build_qwen_candidates.py" \
    "$M3/fine_tuning.py" \
    "$M3/config_stage1.yaml" \
    "$M4/fusion_mlp.py" \
    "$ADAPTER/adapter_model.safetensors" \
    "$ADAPTER/adapter_config.json"; do
    if [[ ! -s "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 1
    fi
done

mkdir -p "$CAND" "$SCORES" "$MODELS"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required for the MaskClustering environment." >&2
    exit 1
fi
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "$MC_ENV"
MC_PYTHON="$(command -v python)"

echo "=== STEP 1/5: build OR-rule candidates ==="
cd "$REPO"
for scene in $ALL_SCENES; do
    output="$CAND/${scene}_qwen_candidates.csv"
    if [[ -s "$output" ]]; then
        echo "Reusing $output"
        continue
    fi
    "$MC_PYTHON" "$M3/build_qwen_candidates.py" \
        --scene "$scene" --config replica \
        --depth-overlap-threshold 0.05 \
        --filter-mode or_rule --vc-threshold 0.6 \
        --output "$output"
done

echo "=== STEP 2/5: fine-tuned Qwen scoring ==="
for scene in $ALL_SCENES; do
    output="$SCORES/${scene}_classifier.json"
    if [[ -s "$output" ]]; then
        echo "Reusing $output"
        continue
    fi
    cd "$M3"
    "$QWEN_PYTHON" "$M3/fine_tuning.py" score \
        --pool-path "$CAND/${scene}_qwen_candidates.csv" \
        --masks-root "$REPO/data" \
        --rgb-root "$REPO/data/replica" \
        --condition pair_only --render-mode outline \
        --adapter-path "$ADAPTER" \
        --config "$M3/config_stage1.yaml" \
        --batch-size 16 \
        --output "$output"
done

echo "=== STEP 3/5: convert classifier outputs ==="
cd "$REPO"
for scene in $ALL_SCENES; do
    output="$SCORES/${scene}_scored.csv"
    if [[ -s "$output" ]]; then
        echo "Reusing $output"
        continue
    fi
    "$QWEN_PYTHON" "$M3/convert_classifier_scores.py" \
        --input "$SCORES/${scene}_classifier.json" \
        --output "$output" \
        --candidates "$CAND/${scene}_qwen_candidates.csv"
done

echo "=== STEP 4/5: train Fusion-MLP ablations ==="
train_csvs=()
val_csvs=()
for scene in $TRAIN_SCENES; do train_csvs+=("$SCORES/${scene}_scored.csv"); done
for scene in $VAL_SCENES; do val_csvs+=("$SCORES/${scene}_scored.csv"); done

for feature_set in geometry qwen fused; do
    model="$MODELS/fusion_mlp_${feature_set}.pt"
    "$QWEN_PYTHON" "$M4/fusion_mlp.py" train \
        --train-scores "${train_csvs[@]}" \
        --val-scores "${val_csvs[@]}" \
        --feature-set "$feature_set" \
        --model-out "$model"
done

echo "=== STEP 5/5: score and cluster test scenes ==="
for scene in $TEST_SCENES; do
    cp "$SCORES/${scene}_scored.csv" "$SCORES/${scene}_qwen.csv"
    for feature_set in geometry qwen fused; do
        "$QWEN_PYTHON" "$M4/fusion_mlp.py" score \
            --scores "$SCORES/${scene}_scored.csv" \
            --model "$MODELS/fusion_mlp_${feature_set}.pt" \
            --output "$SCORES/${scene}_mlp_${feature_set}.csv"
    done

    cd "$REPO"
    for variant in qwen mlp_geometry mlp_qwen mlp_fused; do
        "$MC_PYTHON" "$M3/run_qwen_clustering.py" \
            --scene "$scene" \
            --scores "$SCORES/${scene}_${variant}.csv" \
            --output-name "final_${variant}" \
            --qwen-score-threshold 0.5
    done
done

echo "Milestone-4 pipeline complete."
