# ML3D-VLM_based_2D_mask_grouping
This repository is for the course Machine Learning for 3D Geometry by Prof. Dr. Dai.

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
|   |   `-- checkpoint-625/
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
checkpoints/qwen/checkpoint-625/
```

For inference, the essential files are:

```text
checkpoints/qwen/checkpoint-625/adapter_config.json
checkpoints/qwen/checkpoint-625/adapter_model.safetensors
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
`Project/Milestone 4/run_room0_mlp_inference.sh` expects both project-trained
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
