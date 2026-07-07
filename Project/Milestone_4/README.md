# VLM-Based 2D Mask Grouping


## Overall Goal

MaskClustering builds a graph over 2D masks and merges connected components. The original step uses a geometric containment check: for each observer frame, it asks whether ≥ 80 % of mask `i` and ≥ 80 % of mask `j` are both contained within the same mask `k` in an observer frame.

**This project replaces this logic with a VLM judgment** using `Qwen3-VL-8B-Instruct` with zero-shot learning and fine-tuning with LoRA. For each observer frame where both masks are visible, the model returns:

##### Zero-shot Learning
```
1        # 1 = same object, 0 = different objects
0.87     # confidence in [0, 1]
<one sentence of reasoning>
```

##### Fine-Tuning
Trained to do binary prediction only.
Idea: fine-tune decoder only part (LM) + MLP
* MLP takes the last hidden layer and combines it to produce score 1/0

### Original pipeline (Steps 1, 3, 4 unchanged)

| Step | What happens |
|---|---|
| Find observers | Views where ≥ 30 % of a mask's 3D points are visible (separately for each mask) |
| **Evaluate support (replaced)** | For each observer: check if both masks belong to the same object |
| Aggregate | Consensus rate `c(i,j) = n_supporters / n` |
| Graph edge | Hard threshold `c(i,j) ≥ 0.9` |


Originally, the check was based on geometric correspondence. We are replacing it with the Qwen3-VL prediction.

### Three experimental conditions

All three replace the geometric containment check. Masks are highlighted in colour on their respective views.

| Condition | Images sent to VLM | Goal |
|---|---|---|
| pair only | Mask A (red outline) + Mask B (blue outline), each on its own frame | Baseline: can the model decide from the views alone (no exstra observer is provided)? |
| pair + context | Both masks + full observer frame with all detected masks outlined | Does scene context improve disambiguation? |
| pair + candidate | Both crops + observer frame with one candidate mask outlined (iterated over all detections) | Does focused one-vs-one comparison outperform showing all detections at once? |

---

## Environment Setup

The build phase and the VLM phase need conflicting package versions, so they run in two separate venvs:

| Script | Environment | Used for |
|---|---|---|
| `start_env.sh` | Open3D + CUDA 11.8 + PyTorch3D | Building the training dataset (`load_training_data.py` build phase) |
| `qwen_env.sh` | torch (cu124) + `transformers` + `qwen-vl-utils` | Everything Qwen3-VL: `inference.py`, `evaluate.py`, loading samples at training time |

```bash
./start_env.sh        # creates ./myenv
./qwen_env.sh          # creates ./qwen_env
```

Each script also registers itself as a Jupyter kernel (`Python (myenv)` / `Python (qwen_env)`).

---

## How Data Loading Works

The pipeline is split into two phases because MaskClustering requires pinned package versions that conflict with the higher versions required by Qwen3-VL, they cannot share the same environment.

### What each file does

| File | Responsibility |
|---|---|
| `load_data.py` | Loading and caching frames and masks; GT assignment; image rendering |
| `load_training_data.py` | Computes which frames each mask is visible in; builds positive/negative training pairs; saves and loads the pair pool; entry point for the build phase |
| `prompt_templates.py` | System prompts and user message builders for all three conditions |
| `vlm_utils.py` | Loads the Qwen3-VL model/processor (`transformers`); parses the 3-line VLM response |
| `inference.py` | Single-pair VLM query (CLI) and the shared batched-inference routine used by `evaluate.py` |
| `evaluate.py` | Runs batched inference over a saved pair pool and reports accuracy/precision/recall/F1, broken down by condition, scene, category, and negative-sampling strategy |
| `fine_tuning.py` | *(planned, not yet implemented)* SFTTrainer + LoRA fine-tuning loop |
| `config.yaml` | Model ID, LoRA params (r=16, decoder projections only), SFT hyperparameters |

---

## Building the Training Dataset

Run in the **MaskClustering environment** (Open3D + CUDA). This assigns GT labels to every frame, caches them as pickles, then enumerates all valid positive and negative mask pairs.

```bash
cd /home/kate/Desktop/Github/3DML/ML3D-VLM_based_2D_mask_grouping

python Project/Milestone_4/load_training_data.py \
    --scenes office0 room0 \
    --masks-root Project/ \
    --rgb-root Project/data/replica/ \
    --out data/pair_pools/
```

| Argument | Description |
|---|---|
| `--scenes` | Replica scene names to process |
| `--masks-root` | Root of `replica_masks/` (contains `replica/<scene>/output/mask/`) |
| `--rgb-root` | Replica dataset root (expects `<scene>/color/<frame_id>.jpg`) |
| `--out` | Output directory for per-scene pair pool pickles (`<scene>.pkl`) |
| `--min-angle-deg` | Minimum inter-camera angle for positive pairs (default: 15.0) |
| `--mask-visible-threshold` | Step-1 visibility threshold (default: 0.3) |

### Loading samples at training time

Run in the **Qwen3-VL environment** (no Open3D needed) — frames are loaded directly from `masks_root`/`rgb_root`, there is no separate frame cache:

```python
from load_training_data import load_data_training

samples = load_data_training(
    pool_path="/path/to/pair_pool.pkl",
    masks_root="/path/to/replica_masks",
    rgb_root="/path/to/replica",
    K=15,        # max positives per GT instance
    neg_ratio=3, # negatives per positive
)
```

### How the training data is constructed

The pipeline runs in two phases separated by environment. The build phase (MaskClustering + CUDA) does all the 3D reasoning; the train phase (Qwen3-VL) only loads pre-rendered images.

**Labelling via 3D ground truth.** Each 2D mask is back-projected onto the scene's 3D mesh. Replica provides a per-vertex instance label, so a majority vote over the projected vertices assigns a GT instance ID to every mask. Two masks are a positive pair if and only if they belong to the same 3D instance.

**Positive pairs.** All detections of the same GT instance across different frames are paired. A pair is kept only if the two viewpoints are sufficiently different (inter-camera rotation ≥ 15°) and there exists at least one *observer frame* where both objects are simultaneously visible — the observer is needed for the context/candidate conditions.

**Negative pairs.** Two strategies produce negatives:
- *Hard negatives* (`both_views`): each source frame already sees the other's object, so the model cannot cheat by reasoning about absence.
- *Relaxed negatives* (`co_visible`): only a shared observer frame is required; used to supplement when hard negatives are scarce.

**Image rendering.** At training time, masks are drawn as coloured outlines on the original RGB frames (red / blue for the pair; green for observer context) and passed directly to the VLM. Images are rendered on the fly from cached metadata — no image arrays are stored on disk.

---

## Running Inference

Run in the **Qwen3-VL environment** (`qwen_env.sh`). `inference.py` queries the VLM on a single mask pair and prints its decision:

```bash
cd Project/Milestone_4

python inference.py \
    --config config.yaml \
    --condition pair_only \
    --scene room0 \
    --masks-root /path/to/replica_masks \
    --rgb-root /path/to/replica \
    --frame-a 12 --mask-a 3 \
    --frame-b 47 --mask-b 5
```

| Argument | Description |
|---|---|
| `--condition` | `pair_only` \| `pair_with_context` \| `pair_with_candidate` |
| `--scene`, `--masks-root`, `--rgb-root` | Same meaning as in the build phase |
| `--frame-a` / `--mask-a`, `--frame-b` / `--mask-b` | Frame + mask id for each region |
| `--observer-frame` | Required for `pair_with_context` and `pair_with_candidate` |
| `--candidate-id` | Required for `pair_with_candidate` |

---

## Evaluating the VLM

`evaluate.py` runs batched inference over one or more saved pair pools and reports accuracy, precision, recall, and F1 — overall and broken down by condition, scene, `same_category`, and negative-sampling strategy. Run in the **Qwen3-VL environment**:

```bash
cd Project/Milestone_4

python evaluate.py \
    --pool-path /path/to/pair_pools/room0.pkl /path/to/pair_pools/office0.pkl \
    --masks-root /path/to/replica_masks \
    --rgb-root /path/to/replica \
    --conditions pair_only pair_with_context pair_with_candidate \
    --K 15 --neg-ratio 3 --batch-size 4 \
    --output results/eval_run1.json \
    --config config.yaml
```

| Argument | Description |
|---|---|
| `--pool-path` | One or more `.pkl` pair pools from the build phase (shell globs are expanded) |
| `--masks-root`, `--rgb-root` | Same meaning as in the build phase |
| `--conditions` | Which conditions to evaluate (default: all three) |
| `--K` | Max positives per GT instance (ignored if `--max-positives` is set) |
| `--max-positives` | Flat random sample of this many positives instead of `K` per instance |
| `--neg-ratio` | Negatives sampled per positive |
| `--batch-size` | Inference batch size |
| `--shuffle` | Shuffle sample order (default: grouped by condition for batching efficiency) |
| `--output` | Path to write the JSON results (per-sample predictions + all metric breakdowns) |
| `--seed` | Sampling seed (default: 42) |
| `--max-new-tokens` | Generation token budget per sample (default: 64) |

Results are written to `--output` as JSON (`overall`, `by_condition`, `by_scene`, `by_same_category`, `by_neg_strategy`, and `per_sample` predictions) and a summary table is printed to stdout.
