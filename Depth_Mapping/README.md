# DepthWizard — Part 1: Single-View Relative Depth Estimation

**SIH26175 | Smart India Hackathon 2026**

Single-view monocular depth estimation for remote-sensing imagery using
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

---

## Problem Statement

Generating 3D terrain/surface information from aerial and satellite imagery
traditionally requires specialised hardware — LiDAR, InSAR, or overlapping
stereo acquisitions — that may be unavailable in rapid-response or
resource-constrained scenarios.

**Part 1** of DepthWizard addresses the first step: estimating a spatially
coherent relative depth map from a single RGB optical image, preserving all
available geospatial metadata so the output can be consumed by downstream
processing stages.

---

## Pipeline

```
RGB Remote-Sensing Image  (JPG / PNG / GeoTIFF)
            |
            v
   Global exposure normalisation
   (per-band 1-99th percentile stretch, applied once)
            |
            v
   Adaptive Tile Grid
   (near-500 px core tiles, no remainder)
            |
            v
   For each tile:
     Extract context crop  (core + 128 px margin on each side)
            |
            v
     Depth Anything V2 -- ViT-Large
     (inference at native crop resolution, NOT downscaled)
            |
            v
     Extract core prediction only
     (context margin discarded -- each core pixel predicted once)
            |
            v
     Shift alignment from 8-px feather zone
     (shift only, no scale change -- preserves local gradients)
            |
            v
   Composite into output canvas
   (narrow 8-px Hann feather at internal edges, no core-region blending)
            |
            v
   Relative Depth Map  (float32, H x W)
            |
     +------+------+
     |             |
  .npy          GeoTIFF
  (raw)     (co-registered,
           georef preserved)
```

---

## Input Formats

| Format | Notes |
|--------|-------|
| JPEG / PNG | Standard RGB images. Geo metadata: none. |
| GeoTIFF | Multi-band satellite/aerial raster. CRS, affine transform, pixel resolution preserved. |

---

## Output Formats

| File | Description |
|------|-------------|
| `<stem>.npy` | Raw float32 relative depth array, shape (H, W) |
| `<stem>_depth.png` | Viridis false-colour heatmap for quick inspection |
| `<stem>_depth.tif` | Co-registered float32 GeoTIFF *(GeoTIFF input only)* |
| `<stem>_depth_vis.png` | Multi-panel presentation figure |

> **Important:** Output values are **relative depth** — dimensionless model
> output. They are **not** metric elevation, DSM height, or height above ground.
> Conversion to absolute metric elevation requires external reference data
> (DEM, GCPs) and is out of scope for Part 1.

---

## Key Capabilities

- Monocular relative depth from a single RGB image
- JPEG, PNG, and GeoTIFF input
- GeoTIFF georeferencing (CRS, affine transform, bounds) fully preserved
- Large-image tiled inference — handles 5000 × 5000+ images on a 6 GB GPU
- **Core-only tiling**: each pixel predicted exactly once from a context-aware crop — no prediction averaging, no blurring
- GPU acceleration (tested on RTX 4050 Laptop, 6 GB VRAM)
- Automatic CPU fallback
- No Hugging Face download required — loads from local `.pth` checkpoints
- Optional GAMUS integration for above-ground height evaluation

---

## Novelty

1. **Bridging natural-image depth AI to remote-sensing workflows**
   Depth Anything V2 was trained on natural images; this pipeline adapts it
   to nadir-looking remote-sensing imagery with full geospatial metadata
   preservation, providing the foundation for metric elevation conversion
   via DEM/GCP alignment.

2. **Sensor-free depth representation**
   Produces a DSM-style relative depth map from a single RGB image without
   requiring LiDAR, InSAR, or stereo acquisitions — enabling rapid-response
   scenarios where only one image is available.

3. **Dual-mode input: georeferenced and non-georeferenced**
   A single pipeline handles both GeoTIFF (with CRS/transform) and
   JPG/PNG input, preserving all available spatial metadata for GIS
   integration.

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Depth_Mapping_Part_1

# 2. Install PyTorch with CUDA (adjust cu124 to your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Optional: GAMUS evaluation support
pip install scipy                # height metrics (MAE, RMSE, Spearman-r ...)
pip install datasets             # HuggingFace streaming (only needed for stream_gamus)
```

---

## Checkpoint Setup

Model weights are **not included in this repository** (ViT-Large is ~1.3 GB).

Download from the official [Depth Anything V2 releases](https://github.com/DepthAnything/Depth-Anything-V2):

| File | Purpose |
|------|---------|
| `depth_anything_v2_vitl.pth` | ViT-Large — production quality (~1.3 GB) |
| `depth_anything_v2_vits.pth` | ViT-Small — fast/development (~95 MB) |

Place the `.pth` file(s) in one of these locations (checked in priority order):

| Priority | Location |
|----------|----------|
| 1 | `--checkpoint-dir <path>` CLI flag |
| 2 | `DA_CHECKPOINT_DIR` environment variable |
| 3 | `<project_root>/checkpoints/` |
| 4 | `<project_root>/../checkpoints/` *(sibling directory — default SIH workspace layout)* |
| 5 | `<project_root>/../../checkpoints/` |

```bash
# Windows — set environment variable
set DA_CHECKPOINT_DIR=D:\path\to\checkpoints

# Or pass on the command line
python -m depth_mapping --image Input/test_1.jpeg \
    --checkpoint-dir D:\path\to\checkpoints
```

---

## Usage

### CLI

Run from the project root (`Depth_Mapping_Part_1/`):

```bash
# JPEG input -- relative depth only
python -m depth_mapping --image Input/test_1.jpeg --output-dir outputs/

# GeoTIFF -- relative depth + preserved georeferencing
python -m depth_mapping --image Input/test_2.tif --output-dir outputs/

# Fast mode (ViT-Small, no tiling -- useful for quick iteration)
python -m depth_mapping --image Input/test_1.jpeg --fast

# With explicit checkpoint directory
python -m depth_mapping --image Input/test_2.tif \
    --checkpoint-dir D:\Collaborate_Projects\SIH\checkpoints \
    --output-dir outputs/
```

### Python API

```python
from depth_mapping import get_depth, get_depth_with_meta, save_depth_outputs

# Simple relative depth array
depth = get_depth("Input/test_1.jpeg")
print(depth.shape, depth.dtype)    # (H, W), float32

# With GeoTIFF metadata
depth, geo_meta = get_depth_with_meta("Input/test_2.tif")
print(geo_meta["crs"])             # EPSG:26914 etc.

# Save all outputs (.npy, .png, and optionally .tif)
save_depth_outputs(depth, "outputs/my_run", "test_2", geo_meta=geo_meta)
```

---

## Large-Image Tiling

Images larger than ~650 px use a **core-only tiling** strategy that preserves
spatial detail.

For each tile:
1. A large **context crop** (core + 128 px margin on each side) is sent to the model.
2. Only the central **core** region of the prediction is written to the canvas.
   The 128 px border margin is discarded.
3. A narrow **8-px Hann feather** at internal edges removes hard depth jumps.
4. A **shift offset** (no scale change) is estimated from the feather zone
   to align each tile with its already-placed neighbours.

Each core pixel is contributed by exactly **one** forward pass — no averaging,
no blurring.

**Measured results on 5000 × 5000 px GeoTIFF (10 × 10 tile grid, RTX 4050):**

| Metric | Naive blending | Core-only tiling |
|--------|---------------|-----------------|
| Seam MAD (mean) | ~26–37 units | **0.000** |
| Gradient energy | baseline | **+190%** |
| Max inter-band diff | ~128 | **6.0** |
| Runtime | ~56 s | ~90 s |

---

## GAMUS Integration (Optional)

[GAMUS](https://huggingface.co/datasets/earthflow/GAMUS) provides aerial RGB
imagery paired with **nDSM** (normalised Digital Surface Model — above-ground
object height in metres).

**Scientific note:** GAMUS nDSM represents **above-ground height** (height of
buildings, vegetation above bare earth). It is *not* absolute geodetic terrain
elevation. Absolute metric elevation calibration requires DEM/GCP reference
data and is out of scope for Part 1.

GAMUS is useful for:
- Remote-sensing domain adaptation
- Above-ground height estimation evaluation
- Fine-tuning supervision

```python
from depth_mapping.gamus import GAMUSConfig, GAMUSDataset, evaluate_on_gamus

# Evaluate one sample
result = evaluate_on_gamus(
    rgb_path  = "path/to/aerial_rgb.png",
    ndsm_path = "path/to/ndsm.tif",
)
print(result["metrics"])   # MAE, RMSE, delta_1, Spearman-r, SSIM ...

# Iterate a local subset
cfg = GAMUSConfig(data_dir="gamus_subset/", max_samples=20)
for sample in GAMUSDataset(cfg):
    rgb  = sample["rgb"]   # PIL Image
    ndsm = sample["ndsm"]  # np.ndarray, metres above ground
```

See `depth_mapping/gamus/README.md` for full documentation.

---

## Testing

```bash
# From the project root
python -m pytest depth_mapping/test_depth.py -q
```

Expected: **69 passed** (~18 s, no checkpoint required — tests use a lightweight fake model)

### Validate an output directory

```bash
python -m depth_mapping.validate_output outputs/my_run

# With source GeoTIFF alignment check
python -m depth_mapping.validate_output outputs/my_run \
    --source-tif Input/test_2.tif \
    --write-report
```

---

## Project Structure

```
Depth_Mapping_Part_1/
|
+-- depth_mapping/              # Python package (core pipeline)
|   +-- depth.py                # Inference, tiling, core-only extraction
|   +-- model_loader.py         # Checkpoint discovery and model loading
|   +-- geo_utils.py            # GeoTIFF read/write, metadata preservation
|   +-- visualize.py            # Presentation figures
|   +-- validate_output.py      # Output validation and reporting
|   +-- tile_boundary_metric.py # Seam and structural quality diagnostics
|   +-- depth_anything_v2/      # Vendored DA V2 architecture (DINOv2 + DPT)
|   +-- gamus/                  # Optional GAMUS height-evaluation module
|   |   +-- __init__.py
|   |   +-- config.py
|   |   +-- dataset.py
|   |   +-- evaluate.py
|   |   +-- README.md
|   +-- sample_data/            # Tiny synthetic test image (0.5 KB)
|   +-- test_depth.py           # Test suite (69 tests)
|   +-- __init__.py
|   +-- __main__.py
|
+-- Input/
|   +-- test_1.jpeg             # Small demo image (0.2 MB) -- committed
|   +-- test_2.tif              # 5000x5000 GeoTIFF -- LOCAL ONLY, gitignored
|   +-- bellingham*.tif etc.    # Additional test TIFs -- LOCAL ONLY
|
+-- outputs/                    # Runtime output directory (gitignored)
|   +-- .gitkeep
|
+-- conftest.py                 # pytest sys.path configuration
+-- pyproject.toml              # Package metadata, pytest config, extras
+-- requirements.txt            # Runtime dependencies
+-- README.md                   # This file
+-- .gitignore
```

---

## GPU and Performance

| Image | Size | Device | Runtime |
|-------|------|--------|---------|
| `test_1.jpeg` | 562 x 1179 | CUDA (RTX 4050) | ~6 s |
| `test_2.tif` | 5000 x 5000 | CUDA (RTX 4050) | ~90 s |

Peak VRAM: ~4 GB for a 5000 x 5000 image with the 10 x 10 tile grid.

CPU fallback is automatic — expect 10-20x slower on CPU.

---

## Limitations

- **Relative depth only.** Output is dimensionless. Conversion to metric
  elevation requires external reference data (DEM, GCPs) — Part 2 scope.
- **Scale ambiguity.** Monocular models cannot recover true scale from a
  single image. The pipeline normalises tiles to a consistent relative scale
  but does not anchor to physical units.
- **Domain gap.** Depth Anything V2 was trained on natural images; performance
  on unusual imagery (thermal, SWIR, heavily processed) may be reduced.
- **No quantitative accuracy claim.** This project does not claim metric
  accuracy, superiority over LiDAR/InSAR/stereo, or validated DSM precision
  without supporting measurements against ground truth.
