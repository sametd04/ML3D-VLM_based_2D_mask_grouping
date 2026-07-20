#!/bin/bash
#SBATCH --job-name=m4_final
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7:55:00
#SBATCH --output=/cluster/52/go93coc/m4_final_%j.log

# Final E2E Milestone 4 run:
#   1. or_rule candidates (geometric overlap OR view consensus >= 0.6) for 8 scenes
#   2. candidate CSVs -> Kate-format pair pools
#   3. score all pairs with Kate's vision-classifier checkpoint (checkpoint-625)
#   4. classifier JSON -> per-scene score CSVs
#   5. train fusion MLP (geometry / qwen / fused) on 6 scenes, val office4
#   6. score room0, cluster, evaluate AP for all variants

set -euo pipefail
set -x

BASE=/cluster/52/go93coc
REPO=$BASE/ML3D-VLM_based_2D_mask_grouping
KATE=$BASE/kate_branch            # separate checkout of kate_lora_fine_tuning
ADAPTER=$BASE/checkpoint-625
M3=$REPO/Project/Milestone_3
M4=$REPO/Project/Milestone_4
CAND=$M3/data/candidates_orule
POOLS=$M3/data/pools_orule
SCORES=$M3/data/qwen_scores_final
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

TRAIN_SCENES="room1 room2 office0 office1 office2 office3"
VAL_SCENE="office4"
TEST_SCENE="room0"
ALL_SCENES="$TRAIN_SCENES $VAL_SCENE $TEST_SCENE"

cd $REPO && git pull origin flo/SAM

# Kate branch checkout (code only; data comes from the main repo)
if [ ! -d "$KATE" ]; then
  git clone --branch kate_lora_fine_tuning https://github.com/sametd04/ML3D-VLM_based_2D_mask_grouping $KATE
else
  cd $KATE && git pull origin kate_lora_fine_tuning
fi

echo "=== STEP 1: or_rule candidates for 8 scenes (maskclustering env) ==="
source $BASE/miniconda3/etc/profile.d/conda.sh
conda activate maskclustering
cd $REPO
for scene in $ALL_SCENES; do
  python Project/Milestone_3/build_qwen_candidates.py \
    --scene $scene --config replica \
    --depth-overlap-threshold 0.05 --filter-mode or_rule --vc-threshold 0.6 \
    --output $CAND/${scene}_qwen_candidates.csv
done
conda deactivate

echo "=== STEP 2: candidates -> pair pools ==="
source $BASE/qwen_env/bin/activate
for scene in $ALL_SCENES; do
  python $M3/candidates_to_pool.py \
    --candidates $CAND/${scene}_qwen_candidates.csv \
    --output $POOLS/${scene}.pkl
done

echo "=== STEP 3: score all candidate pairs with Kate's classifier ==="
cd $KATE/Project/Milestone_4
mkdir -p $SCORES
for scene in $ALL_SCENES; do
  python fine_tuning.py score \
    --pool-path $POOLS/${scene}.pkl \
    --masks-root $REPO/data \
    --rgb-root $REPO/data/replica \
    --condition pair_only --render-mode outline \
    --adapter-path $ADAPTER \
    --config config_stage1.yaml \
    --K 10000000 --neg-ratio 1000000 \
    --batch-size 16 \
    --output $SCORES/${scene}_classifier.json
done

echo "=== STEP 4: classifier JSON -> score CSVs ==="
cd $REPO
for scene in $ALL_SCENES; do
  python Project/Milestone_3/convert_classifier_scores.py \
    --input $SCORES/${scene}_classifier.json \
    --output $SCORES/${scene}_scored.csv \
    --candidates $CAND/${scene}_qwen_candidates.csv
done

echo "=== STEP 5: train fusion MLPs (3 feature sets) ==="
cd $M4
TRAIN_CSVS=""
for scene in $TRAIN_SCENES; do TRAIN_CSVS="$TRAIN_CSVS $SCORES/${scene}_scored.csv"; done
for fs in geometry qwen fused; do
  python fusion_mlp.py train \
    --train-scores $TRAIN_CSVS \
    --val-scores $SCORES/${VAL_SCENE}_scored.csv \
    --feature-set $fs \
    --model-out output_final/fusion_mlp_${fs}.pt
done

echo "=== STEP 6: score room0, cluster, evaluate ==="
for fs in geometry qwen fused; do
  python fusion_mlp.py score \
    --scores $SCORES/${TEST_SCENE}_scored.csv \
    --model output_final/fusion_mlp_${fs}.pt \
    --output $SCORES/${TEST_SCENE}_mlp_${fs}.csv
done
deactivate

conda activate maskclustering
cd $REPO
# kate_raw: Kate's classifier probabilities used directly as edge scores
cp $SCORES/${TEST_SCENE}_scored.csv $SCORES/${TEST_SCENE}_kate_raw.csv
for variant in kate_raw mlp_geometry mlp_qwen mlp_fused; do
  python Project/Milestone_3/run_qwen_clustering.py --scene $TEST_SCENE \
    --scores $SCORES/${TEST_SCENE}_${variant}.csv \
    --output-name final_${variant} --qwen-score-threshold 0.5
  echo "--- AP: $variant ---"
  cd $REPO/MaskClustering
  python -m evaluation.evaluate --pred_path data/prediction/final_${variant}_class_agnostic
  cd $REPO
done

echo "=== PIPELINE DONE ==="
