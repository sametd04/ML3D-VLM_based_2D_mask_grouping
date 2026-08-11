#!/usr/bin/env bash
# Room0-only inference: candidate generation -> fine-tuned Qwen scores ->
# fused MLP scores -> Qwen and fused-MLP clustering.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
M3="$REPO/Project/Milestone_3"
M4="$REPO/Project/Milestone_4"
ADAPTER="$REPO/checkpoints/qwen/checkpoint-625"
QWEN_PYTHON="${QWEN_PYTHON:-$REPO/Project/env/qwen_env/bin/python}"
MC_PYTHON="${MC_PYTHON:-$(command -v python)}"

SCENE="room0"
CAND="$M3/data/candidates_orule/${SCENE}_qwen_candidates.csv"
SCORES="$M3/data/qwen_scores_final"
CLASSIFIER_JSON="$SCORES/${SCENE}_classifier.json"
SCORED_CSV="$SCORES/${SCENE}_scored.csv"
QWEN_CSV="$SCORES/${SCENE}_kate_raw.csv"
MLP_CSV="$SCORES/${SCENE}_mlp_fused.csv"
MLP_MODEL="$REPO/checkpoints/fusion_mlp/fusion_mlp_fused.pt"

export HF_HOME="${HF_HOME:-$REPO/Project/env/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$M3:$M4${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$(dirname "$CAND")" "$SCORES"

for required in \
    "$QWEN_PYTHON" \
    "$M3/fine_tuning.py" \
    "$M3/config_stage1.yaml" \
    "$ADAPTER/adapter_model.safetensors" \
    "$ADAPTER/adapter_config.json" \
    "$MLP_MODEL"; do
    if [[ ! -s "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 1
    fi
done

echo "=== 1/5: build Room0 candidate pairs ==="
if [[ ! -s "$CAND" ]]; then
    cd "$REPO"
    "$MC_PYTHON" "$M3/build_qwen_candidates.py" \
        --scene "$SCENE" --config replica \
        --depth-overlap-threshold 0.05 \
        --filter-mode or_rule --vc-threshold 0.6 \
        --output "$CAND"
else
    echo "Reusing $CAND"
fi

echo "=== 2/5: fine-tuned Qwen forward pass ==="
if [[ ! -s "$CLASSIFIER_JSON" ]]; then
    cd "$M3"
    "$QWEN_PYTHON" "$M3/fine_tuning.py" score \
        --pool-path "$CAND" \
        --masks-root "$REPO/data" \
        --rgb-root "$REPO/data/replica" \
        --condition pair_only --render-mode outline \
        --adapter-path "$ADAPTER" \
        --config "$M3/config_stage1.yaml" \
        --batch-size 16 \
        --output "$CLASSIFIER_JSON"
else
    echo "Reusing $CLASSIFIER_JSON"
fi

echo "=== 3/5: convert Qwen output and apply fused MLP ==="
if [[ ! -s "$SCORED_CSV" ]]; then
    cd "$REPO"
    "$QWEN_PYTHON" "$M3/convert_classifier_scores.py" \
        --input "$CLASSIFIER_JSON" \
        --output "$SCORED_CSV" \
        --candidates "$CAND"
fi

if [[ ! -s "$MLP_CSV" ]]; then
    "$QWEN_PYTHON" "$M4/fusion_mlp.py" score \
        --scores "$SCORED_CSV" \
        --model "$MLP_MODEL" \
        --output "$MLP_CSV"
fi
cp "$SCORED_CSV" "$QWEN_CSV"

echo "=== 4/5: Qwen-only clustering ==="
cd "$REPO"
if [[ ! -s "$REPO/data/replica/$SCENE/output/object/final_kate_raw/object_dict.npy" ]]; then
    "$MC_PYTHON" "$M3/run_qwen_clustering.py" \
        --scene "$SCENE" --scores "$QWEN_CSV" \
        --output-name final_kate_raw --qwen-score-threshold 0.5
else
    echo "Qwen object_dict already exists"
fi

echo "=== 5/5: fused-MLP clustering ==="
if [[ ! -s "$REPO/data/replica/$SCENE/output/object/final_mlp_fused/object_dict.npy" ]]; then
    "$MC_PYTHON" "$M3/run_qwen_clustering.py" \
        --scene "$SCENE" --scores "$MLP_CSV" \
        --output-name final_mlp_fused --qwen-score-threshold 0.5
else
    echo "Fused-MLP object_dict already exists"
fi

echo "Room0 inference complete."
echo "Qwen clusters: $REPO/data/replica/$SCENE/output/object/final_kate_raw/object_dict.npy"
echo "MLP clusters:  $REPO/data/replica/$SCENE/output/object/final_mlp_fused/object_dict.npy"
