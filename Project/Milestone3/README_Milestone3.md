# MaskClustering Pipeline - Code Changes & Navigation Guide


### Changes made to Samet's Code
**New:** Replaced traditional geometric-only mask clustering with **Qwen vision-language embeddings** to improve mask association by measuring visual similarity across frames, not just geometric overlap. changes can be viewed in the file maskclustering_pipeline.ipynb under section Step 1.

---

## 🗂️ Project Structure

The project strcuture should look as follows. Note, I have not uploaded the replica dataset to the repo (for obvious reasons) and the detectron2 as well. Apparently, you will need to download detectron2 for CropFormer to work. Maskclustering has instructions in its README file on how to correclty download CropFormer. For me this was the part that was creating errors in my code.
```
ML3D/
├── Project/Milestone3/
│   ├── load_qwen.py                 # Qwen model loading & embedding extraction
│   ├── prelim_graph.py              # Graph construction from masks
│   └── post_process_utils.py        # (referenced in pipeline)
├── MaskClustering/
│   ├── utils/
│   │   └── post_process.py          # Post-processing: DBSCAN, filtering, export
│   ├── graph/
│   │   ├── construction.py          # Build point-in-mask matrix
│   │   └── node.py                  # Node class for mask clusters
│   ├── datasets/
│   └── evaluation/
├── detectron2/
│   ├── projects/CropFormer/         # 2D mask prediction model
│   └── ...
├── data/
│   └── replica/                     # Replica dataset (scenes, masks, etc.)
├── .vscode/
│   └── settings.json                # Python path configuration
└── maskclustering_pipeline.ipynb    # Main pipeline notebook
```

---

## 📁 Modified & New Files

### 1. **`Project/Milestone3/load_qwen.py`** (NEW - Core Change)
**Purpose:** Load Qwen-VL embedding model and compute visual similarity between masks.

**Key Functions:**
- `load_qwen_model(model_name, device)` → Returns processor and model (bfloat16, auto device mapping)
- `get_embeddings(mask_image, processor, model)` → Extract embeddings from mask images
- `cosine_similarity(embeddings1, embeddings2)` → Compute normalized cosine similarity scores

**Why Changed:**
- Replaces pure geometric clustering with learned visual similarity
- Enables cross-frame mask association based on visual content, not just boundary overlap
- Similarity scores prune weak edges in the mask graph

**Code Example:**
```python
processor, model = load_qwen_model()
emb1 = get_embeddings(mask_image1, processor, model)
emb2 = get_embeddings(mask_image2, processor, model)
similarity = cosine_similarity(emb1, emb2)  # Score between -1 and 1
```

---

### 2. **`Project/Milestone3/prelim_graph.py`** (MODIFIED)
**Purpose:** Build initial graph connecting masks with boundary overlap. This graph will connect many masks that might not necessarily belong to the same object, but this is intentional in order to pass a lower number of pairs through the QWEN model but pairs that are related in some sense.

**Key Changes from Original:**
- Functions now called from `maskclustering_pipeline.ipynb` with Qwen refinement
- Returns: `graph`, `mask_point_clouds`, `point_frame_matrix`, `global_frame_mask_list`
- **New Usage:** Graph edges are later pruned by Qwen similarity scores (threshold: 0.9)

**Key Functions:**
- `init_nodes_edges()` → Create nodes & edges based on boundary overlap
- `build_graph(nodes, edges, threshold)` → Construct adjacency graph
- `prelim_graph(args, scene_points, frame_list, dataset)` → Full pipeline orchestration

**Data Flow:**
1. Build point-in-mask matrix from 2D masks + depth + poses
2. Create nodes for each mask (one node per mask initially)
3. Connect nodes with boundary overlap score
4. **→ Graph fed to Qwen refinement in notebook**

---

### 3. **`MaskClustering/utils/post_process.py`** (EXISTING - Used in Pipeline)
**Purpose:** Post-process clustered masks into final 3D objects.

**Key Functions:**
- `merge_overlapping_objects()` → Merge objects with >80% overlap
- `filter_point()` → Remove points with low detection ratio (<threshold)
- `dbscan_process()` → Split disconnected point clouds into separate objects
- `export_class_agnostic_mask()` → Save predictions as `.npz` files

**Pipeline Stage:** Called AFTER Qwen clustering completes (Step 1 → Step 2)

**Export Format:**
```python
{
    "pred_masks": (N_points, N_instances),    # Binary masks
    "pred_score": (N_instances,),              # All ones (class-agnostic)
    "pred_classes": (N_instances,)             # All zeros (no semantic labels yet)
}
```

---

### 4. **`.vscode/settings.json`** (CONFIGURATION)
**Purpose:** Configure VS Code Python language server paths.

**Changes:**
```json
{
    "python.analysis.extraPaths": [
        "./Project/Milestone3",
        "./MaskClustering"
    ]
}
```

**Effect:** Enables autocomplete and imports for:
- `from Project.Milestone3.load_qwen import ...`
- `from MaskClustering.utils.post_process import ...`

---

### 5. **`maskclustering_pipeline.ipynb`** (MAIN NOTEBOOK - Orchestrates Everything)
**Purpose:** End-to-end pipeline with 6 steps + visualization.

**Steps:**
| Step | Description | Output |
|------|-------------|--------|
| 0 | 2D Mask Prediction (CropFormer) | `data/replica/{scene}/output/mask/*.png` |
| **1 NEW** | **Qwen Clustering** | `data/prediction/replica_class_agnostic/*.npz` |
| 2 | Class-Agnostic Evaluation | mAP metrics (no labels) |
| 3 | CLIP Visual Features | Per-object embeddings |
| 4 | CLIP Text Features | Class label embeddings |
| 5 | Semantic Label Assignment | `data/prediction/replica/*.npz` |
| 6 | Class-Aware Evaluation | mAP metrics (with labels) |

**Key New Cell (Step 1 - Qwen Clustering):**
```python
def run_qwen_clustering(scene, config='replica', debug=False):
    # 1. Build preliminary graph (geometric overlap)
    graph = prelim_graph(...)
    
    # 2. Load Qwen model
    processor, model = load_qwen()
    
    # 3. For each mask edge: extract embeddings & compute similarity
    # 4. Prune edges with similarity < 0.9
    # 5. Find connected components in refined graph
    # 6. Convert to nodes & post-process
```

---

## 🔄 Data Flow Diagram

```
Input: 2D Masks from Step 0
        ↓
   [Preliminary Graph]
   - Boundary overlap edges
   - Create initial nodes
        ↓
   [Qwen Embeddings]
   - Load Qwen-VL model
   - Extract embeddings per mask image
   - Compute cosine similarity for each edge
   - Prune edges (similarity < 0.9)
        ↓
   [Connected Components]
   - Find connected components in refined graph
   - Each component = one 3D object
        ↓
   [Post-Processing]
   - DBSCAN: split disconnected point clouds
   - Filter: remove low-detection points
   - Merge: combine overlapping objects
        ↓
Output: 3D Object Instances (class-agnostic)
```

---

## 🚀 Quick Start Guide

### Prerequisites

I have uploaded the requirements.txt file for any dependencies that you might need, but i believe as long as u have the same env as for the MaskClustering, then it should work. But beware of the python requirements for QWEN. 


## 📊 Configuration Parameters that you can tune

### Qwen Clustering (`load_qwen.py`)
```python
SIMILARITY_THRESHOLD = 0.9  # Prune edges below this similarity
```
- **Why 0.9?** Masks must be visually similar (Qwen embeddings) to belong to same object
- Lower = more aggressive merging, higher = more conservative

### Graph Construction (`prelim_graph.py`)
```python
boundary_overlap_threshold = 0.3      # Min overlap ratio for edge creation
min_boundary_overlap_points = 50      # Min # points for edge
prelim_overlap_threshold = 0.3        # Edge weight threshold for graph connectivity
```

### Post-Processing (`post_process.py`)
```python
DBSCAN_THRESHOLD = 0.1               # Distance threshold for DBSCAN
point_filter_threshold = args.point_filter_threshold  # Detection ratio threshold
overlapping_ratio = 0.8              # Merge objects with >80% overlap
```

---

## 🔍 Navigating the Code: Key Entry Points

### **To Understand the Full Pipeline:**
1. Start: [`maskclustering_pipeline.ipynb`](maskclustering_pipeline.ipynb) - Read cells 1-6 (setup & steps)
2. Deep dive: `Step 1` cell → calls `run_qwen_clustering()`
3. Trace: `run_qwen_clustering()` → `prelim_graph()` → `load_qwen()` → `post_process()`

### **To Modify Qwen Clustering:**
1. Edit: [`Project/Milestone3/load_qwen.py`](Project/Milestone3/load_qwen.py)
   - Change `load_qwen_model()` to use different model
   - Adjust `cosine_similarity()` distance metric
   - Modify `get_embeddings()` preprocessing

2. Adjust threshold in notebook Step 1:
   ```python
   SIMILARITY_THRESHOLD = 0.85  # Lower to merge more
   ```

### **To Change Graph Construction:**
1. Edit: [`Project/Milestone3/prelim_graph.py`](Project/Milestone3/prelim_graph.py)
   - `init_nodes_edges()` - Modify boundary overlap logic
   - `build_graph()` - Change graph structure

### **To Modify Post-Processing:**
1. Edit: [`MaskClustering/utils/post_process.py`](MaskClustering/utils/post_process.py)
   - `dbscan_process()` - Change clustering algorithm
   - `filter_point()` - Adjust detection ratio filtering
   - `merge_overlapping_objects()` - Change merge threshold (currently 0.8)

---

## 📈 Output Files & Formats

### Class-Agnostic Predictions (Step 1-2 Output)
```
data/prediction/replica_class_agnostic/
├── office0.npz
├── office1.npz
└── ...
```
**Format (.npz file):**
```python
np.load('office0.npz')  # Returns dict with:
{
    'pred_masks': np.ndarray(N_points, N_instances),
    'pred_score': np.ndarray(N_instances,),          # All 1.0
    'pred_classes': np.ndarray(N_instances,)         # All 0
}
```

### 3D Object Metadata (Step 1 Internal)
```
data/replica/{scene}/output/object/replica/
├── object_dict.npy                   # Objects per cluster
└── open-vocabulary_features.npy      # CLIP embeddings (Step 3)
```

### Semantic Predictions (Step 5 Output)
```
data/prediction/replica/
├── office0.npz                       # With semantic labels
├── office1.npz
└── ...
```

---

## 🐛 Debugging Tips

### Check Qwen Model Loading:
```python
from Project.Milestone3.load_qwen import load_qwen_model
import torch

try:
    processor, model = load_qwen_model()
    print(f"✓ Model loaded: {type(model)}")
    print(f"✓ Device: {model.device}")
except Exception as e:
    print(f"✗ Error: {e}")
```

### Visualize Graph Before/After Qwen Pruning:
```python
# Before pruning
print(f"Initial edges: {sum(len(v) for v in graph.values()) // 2}")

# After pruning (in run_qwen_clustering)
print(f"Pruned edges: {sum(len(v) for v in graph.values()) // 2}")
```

### Check Embedding Shapes:
```python
emb = get_embeddings(mask_image, processor, model)
print(f"Embedding shape: {emb.shape}")  # Expected: (1, embedding_dim)
```

---

## 📝 Summary of Changes

| File | Change Type | What Changed |
|------|------------|---------------|
| `load_qwen.py` | **NEW** | Added Qwen-VL embeddings for visual similarity |
| `prelim_graph.py` | Modified | Now used with Qwen refinement (edge pruning) |
| `post_process.py` | No change | Used as-is for post-processing |
| `maskclustering_pipeline.ipynb` | **Major** | Step 1 now runs Qwen clustering instead of geometric |
| `settings.json` | Updated | Added Python paths for new modules |

---

## 🔗 Dependencies

```
Python 3.8+
torch >= 1.13
transformers (for Qwen)
open3d (for point cloud processing)
detectron2 (for CropFormer)
opencv-python (image processing)
numpy, scipy
```

---

## 📚 Additional Resources

- **Qwen Documentation**: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
- **MaskClustering Original**: See `README.md` in repo root
- **Replica Dataset**: https://replica-dataset.github.io/

---
**Last Updated:** July 8, 2026  

