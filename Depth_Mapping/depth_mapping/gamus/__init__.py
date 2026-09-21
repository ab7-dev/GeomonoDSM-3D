"""GAMUS — Ground-truth Aerial Monocular Urban Segmentation dataset support.

Official dataset: https://huggingface.co/datasets/earthflow/GAMUS

What GAMUS provides
-------------------
GAMUS contains nadir-looking (top-down) RGB aerial images paired with:
  • nDSM  — normalised Digital Surface Model (height above ground, in metres)
  • Optionally: semantic segmentation labels

Scientific note on nDSM vs absolute elevation
----------------------------------------------
GAMUS height data is **nDSM-style above-ground height**, i.e. the height of
objects (buildings, vegetation) above the local bare-earth surface.  This is
NOT the same as:
  * absolute terrain elevation above datum (DTM)
  * geodetic height (ellipsoidal or orthometric)

Therefore GAMUS is appropriate for:
  ✓ Remote-sensing domain adaptation of depth models
  ✓ Evaluation of above-ground height estimation
  ✓ Depth model fine-tuning supervision
  ✗ Absolute geodetic elevation calibration (needs DEM/GCP — Part 2 scope)

This module provides:
  - GAMUSConfig         : configuration dataclass
  - GAMUSDataset        : lazy dataset loader (does NOT download the full ~80 GB)
  - load_sample         : load one (RGB, nDSM) pair from a local path
  - evaluate_on_gamus   : compare pipeline depth output against nDSM reference
  - compute_height_metrics : MAE / RMSE / Spearman correlation / SSIM

Quick start
-----------
# 1. Download a small subset manually from HuggingFace or use streaming:
#    https://huggingface.co/datasets/earthflow/GAMUS
#
# 2. Evaluate:
#    from depth_mapping.gamus import evaluate_on_gamus
#    results = evaluate_on_gamus(
#        rgb_path  = "path/to/rgb.png",
#        ndsm_path = "path/to/ndsm.tif",
#    )
#    print(results)

Public API
----------
    GAMUSConfig
    GAMUSDataset
    load_sample(rgb_path, ndsm_path)
    evaluate_on_gamus(rgb_path, ndsm_path, ...)
    compute_height_metrics(predicted, reference)
"""

from .config  import GAMUSConfig
from .dataset import GAMUSDataset, load_sample
from .evaluate import evaluate_on_gamus, compute_height_metrics

__all__ = [
    "GAMUSConfig",
    "GAMUSDataset",
    "load_sample",
    "evaluate_on_gamus",
    "compute_height_metrics",
]
