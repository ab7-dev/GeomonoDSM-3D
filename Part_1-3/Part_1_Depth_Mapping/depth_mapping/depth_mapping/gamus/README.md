# GAMUS Support — Depth Mapping Part 1

## What is GAMUS?

GAMUS (*Ground-truth Aerial Monocular Urban Segmentation*) is a remote-sensing
dataset available at:

> https://huggingface.co/datasets/earthflow/GAMUS

It provides nadir-looking (top-down) aerial RGB imagery paired with:
- **nDSM** — normalised Digital Surface Model (height of objects above bare earth, in metres)
- Semantic segmentation labels

## Scientific note: nDSM ≠ absolute elevation

GAMUS nDSM values represent **above-ground height** — the height of buildings
and vegetation above the local ground surface.  This is **not**:
- absolute terrain elevation above datum (DTM/DSM)
- geodetic height (ellipsoidal or orthometric)

Therefore GAMUS is used here for:

| Use | Supported? |
|-----|-----------|
| Domain adaptation of depth models to aerial imagery | ✓ |
| Evaluation of above-ground height estimation | ✓ |
| Fine-tuning supervision (relative/structural depth) | ✓ |
| Absolute geodetic elevation calibration | ✗ (needs DEM/GCP — Part 2 scope) |

## Quick start

### Option 1: Local data

Download a subset from HuggingFace manually, then:

```python
from depth_mapping.gamus import GAMUSConfig, GAMUSDataset, evaluate_on_gamus

# Point to your local GAMUS data directory
cfg = GAMUSConfig(data_dir="path/to/gamus_subset", max_samples=20)

# Iterate over paired samples
ds = GAMUSDataset(cfg)
for sample in ds:
    rgb  = sample["rgb"]    # PIL Image (uint8 RGB)
    ndsm = sample["ndsm"]   # np.ndarray float32, shape (H, W), metres AGL

# Evaluate one sample
result = evaluate_on_gamus(
    rgb_path  = "path/to/rgb.png",
    ndsm_path = "path/to/ndsm.tif",
)
print(result["metrics"])
```

### Option 2: HuggingFace streaming (no full download)

```python
from depth_mapping.gamus import GAMUSConfig
from depth_mapping.gamus.dataset import stream_gamus

cfg = GAMUSConfig(streaming=True, max_samples=5)
for sample in stream_gamus(cfg):
    ...   # works without downloading the full ~80 GB
```

Requires: `pip install datasets`

## Metrics

`evaluate_on_gamus` computes (after least-squares scale+shift alignment):

| Metric | Description |
|--------|-------------|
| `mae` | Mean absolute error (aligned depth units vs nDSM metres) |
| `rmse` | Root mean squared error |
| `log10_mae` | Mean \|log10(pred) - log10(ref)\| (scale-invariant) |
| `abs_rel` | Mean \|pred - ref\| / ref |
| `delta_1` | % pixels with max(d/d*, d*/d) < 1.25 |
| `spearman_r` | Spearman rank correlation (fully scale-invariant) |
| `ssim` | Structural similarity |

## Fine-tuning preparation

The dataset structure is designed to support later domain adaptation:

```
RGB → Depth Anything V2 → predicted relative depth
GAMUS RGB + nDSM         → training supervision signal
```

A configuration-driven training entry point can use `GAMUSDataset` as the
data source.  No training is launched automatically by this module.

## Directory layout expected by GAMUSConfig

```
data_dir/
    rgb/
        image_001.png
        image_002.png
        ...
    ndsm/
        image_001.tif   ← must match RGB filename stem
        image_002.tif
        ...
```
