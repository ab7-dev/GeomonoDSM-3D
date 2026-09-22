"""GAMUS evaluation: compare pipeline depth output against nDSM reference.

GAMUS nDSM is above-ground height in metres (nDSM-style).
It is NOT absolute terrain elevation.

Metrics computed
----------------
mae         : mean absolute error (depth units vs nDSM metres, after alignment)
rmse        : root mean squared error
log10_mae   : mean |log10(d_pred) - log10(d_ref)| (scale-invariant)
abs_rel     : mean |d_pred - d_ref| / d_ref
sq_rel      : mean (d_pred - d_ref)^2 / d_ref
delta_1     : fraction of pixels where max(d/d*, d*/d) < 1.25
delta_2     : same with threshold 1.25^2
delta_3     : same with threshold 1.25^3
spearman_r  : Spearman rank correlation (scale-invariant)
ssim        : structural similarity index

Alignment note
--------------
Depth Anything V2 produces relative depth (dimensionless, arbitrary scale+shift).
Before computing MAE/RMSE the predicted depth is aligned to the nDSM reference
using a least-squares scale+shift fit over the valid pixel set.
This makes the metrics measure structural accuracy, not absolute scale.
For absolute scale calibration, see Part 2 (DEM/GCP-based metric conversion).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_height_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray | None = None,
    align: bool = True,
) -> dict[str, float]:
    """Compare predicted depth against nDSM reference.

    Parameters
    ----------
    predicted   : 2-D float32 relative depth (H, W) from the pipeline.
    reference   : 2-D float32 nDSM above-ground height in metres (H, W).
    valid_mask  : optional boolean mask (True = valid pixel).
                  NaN/Inf pixels and reference ≤ 0 are always excluded.
    align       : if True, fit a scale+shift to map predicted → reference
                  before computing error metrics.

    Returns
    -------
    dict of metric names to float values.
    """
    if predicted.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} vs reference {reference.shape}"
        )

    pred = predicted.astype(np.float64).ravel()
    ref  = reference.astype(np.float64).ravel()

    # Build validity mask
    mask = np.isfinite(pred) & np.isfinite(ref) & (ref > 0.0)
    if valid_mask is not None:
        mask &= valid_mask.ravel().astype(bool)

    n_valid = mask.sum()
    if n_valid < 16:
        logger.warning("Only %d valid pixels — metrics will be unreliable.", n_valid)
        return {"n_valid": int(n_valid), "error": "insufficient_valid_pixels"}

    p = pred[mask]
    r = ref[mask]

    # Least-squares scale+shift alignment (affine, not log-space)
    if align:
        A   = np.column_stack([p, np.ones_like(p)])
        res = np.linalg.lstsq(A, r, rcond=None)
        sc, sh = float(res[0][0]), float(res[0][1])
        # Clamp to prevent pathological transforms
        sc = float(np.clip(sc, 0.01, 100.0))
        p  = p * sc + sh
        logger.debug("Alignment: scale=%.4f shift=%.4f", sc, sh)

    # Core metrics
    abs_diff = np.abs(p - r)
    mae  = float(abs_diff.mean())
    rmse = float(np.sqrt((( p - r) ** 2).mean()))

    # Log-space (requires positive values — clip to small positive)
    p_pos = np.maximum(p, 1e-3)
    r_pos = np.maximum(r, 1e-3)
    log10_mae = float(np.abs(np.log10(p_pos) - np.log10(r_pos)).mean())

    # Relative errors
    abs_rel = float((abs_diff / r_pos).mean())
    sq_rel  = float((((p - r) ** 2) / r_pos).mean())

    # Threshold accuracy (delta metrics)
    ratio   = np.maximum(p_pos / r_pos, r_pos / p_pos)
    delta_1 = float((ratio < 1.25   ).mean())
    delta_2 = float((ratio < 1.25**2).mean())
    delta_3 = float((ratio < 1.25**3).mean())

    # Spearman rank correlation (scale-invariant)
    from scipy.stats import spearmanr  # type: ignore
    try:
        spearman_r, _ = spearmanr(p, r)
        spearman_r = float(spearman_r)
    except ImportError:
        # Fallback: manual rank correlation
        rank_p = np.argsort(np.argsort(p)).astype(np.float64)
        rank_r = np.argsort(np.argsort(r)).astype(np.float64)
        spearman_r = float(np.corrcoef(rank_p, rank_r)[0, 1])

    # SSIM (structural similarity)
    # Use the FULL 2D arrays (not the filtered 1D p/r vectors) so that the
    # reshape inside _ssim_2d works correctly even when valid_mask < all pixels.
    # Apply the valid_mask to zero-out invalid pixels before SSIM computation.
    _pred_2d = predicted.astype(np.float64).copy()
    _ref_2d  = reference.astype(np.float64).copy()
    if valid_mask is not None:
        _invalid = ~valid_mask.astype(bool)
        _pred_2d[_invalid] = 0.0
        _ref_2d[_invalid]  = 0.0
    try:
        ssim = _ssim_2d(_pred_2d, _ref_2d)
    except Exception:
        ssim = float(np.corrcoef(p, r)[0, 1]) if len(p) > 1 else 0.0

    return {
        "n_valid"    : int(n_valid),
        "mae"        : round(mae,        4),
        "rmse"       : round(rmse,       4),
        "log10_mae"  : round(log10_mae,  4),
        "abs_rel"    : round(abs_rel,    4),
        "sq_rel"     : round(sq_rel,     4),
        "delta_1"    : round(delta_1,    4),
        "delta_2"    : round(delta_2,    4),
        "delta_3"    : round(delta_3,    4),
        "spearman_r" : round(spearman_r, 4),
        "ssim"       : round(ssim,       4),
        "aligned"    : align,
    }


def evaluate_on_gamus(
    rgb_path: str | Path,
    ndsm_path: str | Path,
    fast: bool = False,
    align: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full Part 1 pipeline on one GAMUS sample and evaluate against nDSM.

    Parameters
    ----------
    rgb_path  : path to the RGB image.
    ndsm_path : path to the corresponding nDSM raster (.tif, metres AGL).
    fast      : use ViT-Small for speed (lower quality).
    align     : align predicted depth to nDSM before computing metrics.
    verbose   : print a formatted summary.

    Returns
    -------
    dict with keys:
        ``metrics``      – height metrics dict
        ``depth_map``    – np.ndarray float32 (H, W) raw predicted depth
        ``ndsm``         – np.ndarray float32 (H, W) nDSM reference
        ``rgb_path``     – Path
        ``ndsm_path``    – Path

    Notes
    -----
    GAMUS nDSM is **above-ground height** — it is not absolute elevation.
    Metrics here measure structural/relative height accuracy.
    Absolute scale calibration requires DEM/GCP data (Part 2).
    """
    from depth_mapping.depth import get_depth
    from .dataset import load_sample

    sample = load_sample(str(rgb_path), str(ndsm_path))
    rgb    = sample["rgb"]
    ndsm   = sample["ndsm"]

    # Run pipeline inference
    import tempfile, os
    from PIL import Image as PILImage
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        rgb.save(tmp_path)
        depth_map = get_depth(tmp_path, fast=fast)
    finally:
        os.unlink(tmp_path)

    # Resize depth to nDSM size if needed
    if depth_map.shape != ndsm.shape:
        from PIL import Image as PILImage
        dh, dw = ndsm.shape
        depth_pil = PILImage.fromarray(depth_map)
        depth_map = np.asarray(depth_pil.resize((dw, dh), PILImage.BILINEAR), dtype=np.float32)

    metrics = compute_height_metrics(depth_map, ndsm, align=align)

    if verbose:
        _print_metrics(metrics, rgb_path, ndsm_path)

    return {
        "metrics"   : metrics,
        "depth_map" : depth_map,
        "ndsm"      : ndsm,
        "rgb_path"  : Path(rgb_path),
        "ndsm_path" : Path(ndsm_path),
    }


def evaluate_dataset(
    config: "GAMUSConfig",  # noqa: F821
    fast: bool = False,
    align: bool = True,
) -> dict[str, Any]:
    """Evaluate over a full local GAMUS dataset and return aggregate metrics.

    Parameters
    ----------
    config : GAMUSConfig  — points to local GAMUS data directory.
    fast   : use ViT-Small encoder.
    align  : align depth to nDSM before metric computation.

    Returns
    -------
    dict with keys:
        ``per_sample``   – list of per-sample metric dicts
        ``aggregate``    – mean ± std for each metric across all samples
        ``n_samples``    – number of samples evaluated
    """
    from .dataset import GAMUSDataset

    ds       = GAMUSDataset(config)
    all_mets = []

    for i, sample in enumerate(ds):
        try:
            result   = evaluate_on_gamus(
                sample["rgb_path"],
                sample["ndsm_path"],
                fast=fast, align=align, verbose=False,
            )
            all_mets.append(result["metrics"])
            logger.info("Sample %d/%d: MAE=%.3f  delta_1=%.3f",
                        i + 1, len(ds),
                        result["metrics"]["mae"], result["metrics"]["delta_1"])
        except Exception as exc:
            logger.warning("Sample %d failed: %s", i, exc)

    if not all_mets:
        return {"per_sample": [], "aggregate": {}, "n_samples": 0}

    scalar_keys = [k for k in all_mets[0] if isinstance(all_mets[0][k], float)]
    aggregate   = {
        k: {
            "mean": round(float(np.mean([m[k] for m in all_mets])), 4),
            "std" : round(float(np.std( [m[k] for m in all_mets])), 4),
        }
        for k in scalar_keys
    }

    return {
        "per_sample": all_mets,
        "aggregate" : aggregate,
        "n_samples" : len(all_mets),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ssim_2d(pred: np.ndarray, ref: np.ndarray,
             window: int = 11, eps: float = 1e-8) -> float:
    """Simplified SSIM between two 2-D arrays (no scikit-image dependency)."""
    from scipy.ndimage import uniform_filter  # type: ignore
    try:
        p = pred.astype(np.float64)
        r = ref.astype(np.float64)
        mu_p  = uniform_filter(p, window)
        mu_r  = uniform_filter(r, window)
        sig_p = uniform_filter(p * p, window) - mu_p ** 2
        sig_r = uniform_filter(r * r, window) - mu_r ** 2
        sig_pr = uniform_filter(p * r, window) - mu_p * mu_r
        c1 = (0.01 * max(p.max() - p.min(), 1.0)) ** 2
        c2 = (0.03 * max(r.max() - r.min(), 1.0)) ** 2
        ssim_map = ((2 * mu_p * mu_r + c1) * (2 * sig_pr + c2)) / \
                   ((mu_p**2 + mu_r**2 + c1) * (sig_p + sig_r + c2) + eps)
        return float(ssim_map.mean())
    except ImportError:
        # scipy unavailable — return Pearson correlation as fallback
        return float(np.corrcoef(pred.ravel(), ref.ravel())[0, 1])


def _print_metrics(metrics: dict, rgb_path: Any, ndsm_path: Any) -> None:
    """Print a formatted evaluation summary."""
    print("=" * 60)
    print("GAMUS Evaluation")
    print("=" * 60)
    print(f"  RGB  : {rgb_path}")
    print(f"  nDSM : {ndsm_path}")
    print(f"  NOTE : nDSM = above-ground height (NOT absolute elevation)")
    print(f"  Valid pixels : {metrics.get('n_valid', '?')}")
    print(f"  Aligned      : {metrics.get('aligned', '?')}")
    print()
    for k in ("mae", "rmse", "log10_mae", "abs_rel", "sq_rel",
              "delta_1", "delta_2", "delta_3", "spearman_r", "ssim"):
        v = metrics.get(k, "n/a")
        if isinstance(v, float):
            print(f"  {k:<14}: {v:.4f}")
    print("=" * 60)
