# ML3D-VLM_based_2D_mask_grouping
This repository is for the course Machine Learning for 3D Geometry by Prof. Dr. Dai.

## Quick start (Milestone 1 pipeline)

1. SSH into the cluster and allocate a GPU: `ssh <user>@ml3d.vc.in.tum.de` then `salloc --gpus=1`.
2. Clone the repo and run the one-shot setup:
   ```bash
   git clone https://github.com/sametd04/ML3D-VLM_based_2D_mask_grouping.git
   cd ML3D-VLM_based_2D_mask_grouping
   bash setup_env.sh
   ```
   This creates the `maskclustering` conda env and installs everything (PyTorch, pytorch3d, detectron2, CropFormer, and the rest of `requirements.txt`).
3. Open `Project/Milestone 1/maskclustering_pipeline.ipynb` and run it top to bottom — it runs CropFormer, MaskClustering, CLIP feature extraction, semantic labeling, and evaluation end-to-end.

The only step that can't be scripted: the CropFormer checkpoint is hosted on a **gated** HuggingFace dataset, so each person needs to request access and log in once via `huggingface-cli login` (the notebook's Step 0 cell explains exactly how, and prints the error with next steps if you skip it).
