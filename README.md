# ML3D-VLM_based_2D_mask_grouping

This repository is for the course Machine Learning for 3D Geometry by Prof. Dr. Dai.

**Research question:** can a Vision-Language Model improve the grouping of 2D masks into 3D object instances?

The geometry-only [MaskClustering](https://github.com/PKU-EPIC/MaskClustering) baseline groups per-frame 2D
instance masks into 3D objects using multi-view *view consensus*. It is cheap and strong, but semantically
blind. We test whether semantic evidence from Qwen3-VL improves that grouping, either as a direct pairwise
edge scorer or fused with geometry through a small MLP. All variants feed the same downstream graph
clustering, so differences are attributable to the edge score rather than the clustering algorithm.

## Project structure

| Milestone | Directory | Contents |
|---|---|---|
| 1 - Baseline | `Project/Milestone 1/` | MaskClustering reproduction on Replica (notebooks) |
| 2 - Mask backbone | `Project/Milestone 2/` | CropFormer vs. SAM (ViT-H) vs. SAM3 comparison; CropFormer wins and is used from here on |
| 3 - Semantic edge scorer | `Project/Milestone_3/` | Candidate-pair filtering, Qwen3-VL scoring (zero-shot and LoRA fine-tuned), fine-tuning code |
| 4 - Fusion | `Project/Milestone_4/` | MLP fusing one semantic and five geometric features into a single edge score |
| Deliverable | `Project/poster/` | Poster (A0, LaTeX) and the qualitative render scripts |

`MaskClustering/` is the upstream baseline (graph construction, clustering, evaluation).

## Running the Milestone 1 pipeline end-to-end

Everything (CropFormer 2D masks → MaskClustering → CLIP features → semantic labels → evaluation) runs from a single notebook. Setup is a one-time thing per person; you don't need to repeat it every time you want to run the pipeline.

### 1. SSH into the cluster and grab a GPU

```bash
ssh <you>@ml3d.vc.in.tum.de
salloc --gpus=1
```
Requires TUM eduVPN if you're off-campus. `salloc` gives you an interactive session on one of the compute nodes (max 8h).

### 2. Clone the repo and run the setup script (one-time)

```bash
git clone https://github.com/sametd04/ML3D-VLM_based_2D_mask_grouping.git
cd ML3D-VLM_based_2D_mask_grouping
bash setup_env.sh
```

This creates the `maskclustering` conda env and installs everything needed: PyTorch (CUDA 11.8), pytorch3d, detectron2, CropFormer (cloned and built under `MaskClustering/third_party/`, which is gitignored — that's why every person has to build it locally instead of getting it from git), and the rest of `requirements.txt`. It takes a while (pytorch3d builds from source) — that's normal. It's safe to re-run if it gets interrupted, it skips whatever's already done.

### 3. Get access to the CropFormer checkpoint (one-time)

The checkpoint lives in a **gated** HuggingFace dataset, so this step can't be scripted — each person needs their own access grant:
1. Log in / sign up and request access at https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg (click to accept the terms)
2. Create a token at https://huggingface.co/settings/tokens
3. On the cluster, run `hf auth login` and paste the token

The CropFormer/Mask2Former weights used by the project are expected at:

```text
checkpoints/Mask2Former_hornet_3x_576d0b.pth
```

The `checkpoints/` directory should remain inside the repository working
directory because the CropFormer notebooks and scripts load the model from
this local path. The checkpoint is approximately 883 MB and is a local runtime
dependency, not source code. Keep it on every machine or cluster workspace on
which CropFormer is executed, but do not commit or upload the weight file to
Git. After cloning the repository, create `checkpoints/` and place the
downloaded checkpoint there if it is not downloaded automatically.

The expected layout is:

```text
ML3D-VLM_based_2D_mask_grouping/
|-- checkpoints/
|   |-- Mask2Former_hornet_3x_576d0b.pth
|   |-- qwen/
|   |   `-- checkpoint-822/
|   |       |-- adapter_config.json
|   |       `-- adapter_model.safetensors
|   `-- fusion_mlp/
|       `-- fusion_mlp_fused.pt
|-- MaskClustering/
|-- Project/
`-- README.md
```

If you skip this, you can still run the pipeline with `MASK_PREDICTOR = 'sam'` in the notebook, using pre-generated SAM masks instead.

### 3.1 Project-trained Qwen and Fusion-MLP checkpoints

In addition to the public CropFormer weights, the later milestones use two
checkpoints trained within this project. These are internal project artifacts
and currently do not have a public download location. Obtain them from a team
member or regenerate them with the corresponding Milestone 4 training code.

#### Fine-tuned Qwen3-VL adapter

The Qwen checkpoint contains the LoRA adapters for the final eight vision
blocks, the adapted vision patch merger, and the trained binary classification
head. Store the complete training checkpoint at:

```text
checkpoints/qwen/checkpoint-822/
```

For inference, the essential files are:

```text
checkpoints/qwen/checkpoint-822/adapter_config.json
checkpoints/qwen/checkpoint-822/adapter_model.safetensors
```

Keep the remaining files (`optimizer.pt`, `scheduler.pt`, `trainer_state.json`,
`training_args.bin`, and `rng_state.pth`) if training may be resumed. They are
not required for inference alone.

#### Fusion MLP

The Fusion MLP combines the fine-tuned Qwen same-instance score with geometric
features such as view consensus, observer count, depth IoU, and directional
overlap. Store its checkpoint at:

```text
checkpoints/fusion_mlp/fusion_mlp_fused.pt
```

The `.pt` file contains the trained state dictionary together with the feature
definition, normalization statistics, hidden dimension, and dropout setting
needed for inference.

The Room0 inference pipeline in
`Project/Milestone_4/run_room0_mlp_inference.sh` expects both project-trained
checkpoints at the locations above. When running on the cluster, reproduce the
same directory structure below the cluster-side repository root.

As with the CropFormer weights, these model artifacts are local runtime
dependencies and should not be committed to Git. Only the training and
inference code, lightweight configurations, and documentation should be
version controlled.

### 4. Make sure the Replica dataset is in place

Inside `MaskClustering/`, each scene folder (`office0`–`office4`, `room0`–`room2`) needs `color/`, `depth/`, `poses/`, `intrinsics.txt`, `{scene}_mesh.ply`, plus ground truth at `data/replica/ground_truth/`.

### 5. Launch Jupyter and tunnel in

On the cluster:
```bash
conda activate maskclustering
cd MaskClustering
jupyter notebook --no-browser --ip=0.0.0.0 --port 8888
```

On your **local** machine, open a new terminal and tunnel in (the notebook's Cell 2 prints the exact command with your actual node name — use that):
```bash
ssh -L 8888:<node>.vc.in.tum.de:8888 <you>@ml3d.vc.in.tum.de
```
Then open http://localhost:8888. Make sure the notebook's kernel is set to "Python (maskclustering)".

### 6. Run the notebook

Open `Project/Milestone 1/maskclustering_pipeline.ipynb` and run all cells top to bottom:

| Step | What it does |
|------|-------------|
| 0 | 2D mask prediction (CropFormer, auto-downloads the checkpoint) |
| 1 | Mask clustering — backproject 2D masks to 3D, build mask graph, cluster |
| 2 | Class-agnostic evaluation |
| 3 | CLIP visual feature extraction |
| 4 | CLIP text feature extraction |
| 5 | Semantic label assignment |
| 6 | Class-aware evaluation (baseline mAP) |

Every step checks its own prerequisites first and raises a clear error telling you exactly what's missing and which earlier step to run if something's off — if a cell fails, read the error message before asking around.

## Milestone 2: comparing 2D mask backbones

The clustering quality is bounded by the input 2D masks, so the backbone is fixed first. Generate masks with
each candidate segmenter and run the same clustering on top:

```bash
python "Project/Milestone 2/run_sam_masks.py"    # SAM (ViT-H)
python "Project/Milestone 2/run_sam3_masks.py"   # SAM3 (text-prompted)
python "Project/Milestone 2/convert_sam_to_maskclustering.py"
```

SAM3 needs its own environment because it requires a newer PyTorch than the rest of the project:

```bash
bash "Project/Milestone 2/setup_sam3_env.sh"
```

CropFormer produced the most instance-coherent masks and the best downstream AP, so it is the backbone for
Milestones 3 and 4.

## Milestone 3: Qwen3-VL as a semantic edge scorer

Scoring every mask pair with a VLM is quadratic in the number of masks, so a permissive geometric filter
selects candidate pairs first and the VLM is only queried on those:

```bash
python Project/Milestone_3/build_qwen_candidates.py   # candidate pairs + geometric features
python Project/Milestone_3/candidates_to_pool.py      # candidates -> pair pool
python Project/Milestone_3/score_qwen_candidates.py   # Qwen3-VL P(same instance)
```

Fine-tuning and its evaluation live in `fine_tuning.py` and `evaluate.py`; see
[Project/Milestone_3/README.md](Project/Milestone_3/README.md) for the environment setup (the geometry and
Qwen phases need separate virtual environments) and [Fine_Tuning.md](Project/Milestone_3/Fine_Tuning.md) for
the pair building, sampling and training design.

## Milestone 4: fusion MLP

`Project/Milestone_4/fusion_mlp.py` trains a small MLP that fuses the Qwen same-instance probability with
five geometric features (view consensus, observation count, depth IoU, and both directional overlaps) into a
single edge score, which then replaces view consensus in the same clustering step. Because it consumes a
pre-computed Qwen score, fusion adds no extra VLM inference cost at clustering time.

See [Project/Milestone_4/README.md](Project/Milestone_4/README.md) for training and inference details.

## Notes

- Anything using CUDA must run on a GPU node; the login node is CPU-only.
- Model weights (CropFormer, Qwen adapters, fusion MLP) are runtime dependencies and are deliberately not
  committed. See the checkpoint layout above.
- Cluster job scripts are intentionally not version controlled, since they contain machine-specific paths.
