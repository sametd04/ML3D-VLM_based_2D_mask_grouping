#!/bin/bash
#SBATCH --job-name=m4_iter
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/cluster/52/go93coc/m4_iter_%j.log

# Iterative-clustering ablation on top of run_m4_final.sbatch outputs:
# same edge scores, but MaskClustering-style progressive-threshold merging
# (cluster-level mean-aggregated pair scores) instead of one-shot connected
# components. Submit with --dependency=afterok:<final job id>.

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
  python Project/Milestone_3/run_qwen_clustering.py --scene $TEST_SCENE \
    --scores $SCORES/${TEST_SCENE}_${variant}.csv \
    --output-name iter_${variant} \
    --iterative --thresholds 0.9,0.8,0.7,0.6,0.5 --min-pairs 1
  echo "--- AP (iterative): $variant ---"
  cd $REPO/MaskClustering
  python -m evaluation.evaluate --pred_path data/prediction/iter_${variant}_class_agnostic \
    --gt_path data/replica/ground_truth --dataset replica --no_class
  cd $REPO
done

echo "=== ITERATIVE DONE ==="
