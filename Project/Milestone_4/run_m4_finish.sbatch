#!/bin/bash
#SBATCH --job-name=m4_finish
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/cluster/52/go93coc/m4_finish_%j.log

# Resumes run_m4_final.sbatch after its step-6 evaluate-arg failure: score CSVs,
# MLP models and the kate_raw prediction already exist; only one-shot clustering
# for the MLP variants and AP evaluation for all variants are (re)run here.

set -euo pipefail
set -x

BASE=/cluster/52/go93coc
REPO=$BASE/ML3D-VLM_based_2D_mask_grouping
SCORES=$REPO/Project/Milestone_3/data/qwen_scores_final
TEST_SCENE="room0"

source /rhome/go93coc/miniconda3/etc/profile.d/conda.sh
conda activate maskclustering
cd $REPO

for variant in kate_raw mlp_geometry mlp_qwen mlp_fused; do
  test -f $SCORES/${TEST_SCENE}_${variant}.csv
  if [ ! -f data/prediction/final_${variant}_class_agnostic/${TEST_SCENE}.npz ]; then
    python Project/Milestone_3/run_qwen_clustering.py --scene $TEST_SCENE \
      --scores $SCORES/${TEST_SCENE}_${variant}.csv \
      --output-name final_${variant} --qwen-score-threshold 0.5
  fi
  echo "--- AP: $variant ---"
  cd $REPO/MaskClustering
  python -m evaluation.evaluate --pred_path data/prediction/final_${variant}_class_agnostic \
    --gt_path data/replica/ground_truth --dataset replica --no_class
  cd $REPO
done

echo "=== FINISH DONE ==="
