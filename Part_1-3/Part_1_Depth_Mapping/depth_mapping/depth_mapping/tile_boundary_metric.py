"""Tile-boundary and structural-quality diagnostics for the depth pipeline.

Two families of metrics are provided:

1. Seam metrics (``measure_tile_boundaries``)
   Mean absolute depth difference across the 1-pixel line at each internal
   tile-core boundary.  Low values → no visible seams.

2. Structural / detail metrics (``measure_structural_quality``)
   Gradient magnitude, Laplacian variance, local-contrast, and edge-density.
   These catch the failure mode where seam MAD improves but detail is lost.

Usage
-----
Run from the project root:

    python -m depth_mapping.tile_boundary_metric \\
        --image Input/test_2.tif --output-dir outputs/boundary_test

Public API
----------
measure_tile_boundaries(depth, y_edges, x_edges)        → dict
measure_structural_quality(depth, *, reference=None)    → dict
run_boundary_test(image_path, output_dir, ...)          → dict
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seam metrics
# ---------------------------------------------------------------------------

def measure_tile_boundaries(
    depth_array: np.ndarray,
    y_edges: list[int],
    x_edges: list[int],
    half_band: int = 4,
) -> dict[str, Any]:
    """Mean-absolute-difference at each internal tile-core boundary line.

    Parameters
    ----------
    depth_array : 2-D float32 depth map (H × W).
    y_edges     : row-boundary list from ``_adaptive_edges(H)``.
    x_edges     : column-boundary list from ``_adaptive_edges(W)``.
    half_band   : pixels to compare on each side of the boundary.

    Returns
    -------
    dict with keys:
        horizontal_mads, vertical_mads, all_mads,
        mean_mad, max_mad, median_mad,
        n_h_boundaries, n_v_boundaries
    """
    H, W = depth_array.shape
    h_mads: list[float] = []
    v_mads: list[float] = []

    for by in y_edges[1:-1]:
        if 0 < by < H:
            mad = float(np.mean(np.abs(
                depth_array[by - 1, :].astype(np.float64) -
                depth_array[by,     :].astype(np.float64)
            )))
            h_mads.append(mad)

    for bx in x_edges[1:-1]:
        if 0 < bx < W:
            mad = float(np.mean(np.abs(
                depth_array[:, bx - 1].astype(np.float64) -
                depth_array[:, bx    ].astype(np.float64)
            )))
            v_mads.append(mad)

    all_mads = h_mads + v_mads
    return {
        "horizontal_mads" : [round(m, 4) for m in h_mads],
        "vertical_mads"   : [round(m, 4) for m in v_mads],
        "all_mads"        : [round(m, 4) for m in all_mads],
        "mean_mad"        : round(float(np.mean(all_mads)),   4) if all_mads else 0.0,
        "max_mad"         : round(float(np.max(all_mads)),    4) if all_mads else 0.0,
        "median_mad"      : round(float(np.median(all_mads)), 4) if all_mads else 0.0,
        "n_h_boundaries"  : len(h_mads),
        "n_v_boundaries"  : len(v_mads),
        "half_band_px"    : half_band,
    }


# ---------------------------------------------------------------------------
# Structural / detail metrics
# ---------------------------------------------------------------------------

def measure_structural_quality(
    depth: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    downsample_for_speed: int = 1,
) -> dict[str, Any]:
    """Lightweight structural-quality diagnostics.

    These metrics detect the failure mode:
        "seam MAD is low but roads/buildings are blurred away."

    Metrics computed
    ----------------
    grad_mean       : mean of |∇depth|   — overall edge/gradient strength
    grad_p90        : 90th percentile of |∇depth|  — strong-edge preservation
    laplacian_var   : variance of ∇²depth — texture/fine-detail energy
    local_contrast  : mean std of 9×9 local windows — local depth variance
    edge_density    : fraction of pixels where |∇depth| > 0.5 * grad_mean

    Comparison (when reference is provided)
    ----------------------------------------
    grad_ratio      : grad_mean(depth) / grad_mean(reference)
    lap_ratio       : laplacian_var(depth) / laplacian_var(reference)
    contrast_ratio  : local_contrast(depth) / local_contrast(reference)

    A ratio < 0.7 on any metric signals significant detail loss.

    Parameters
    ----------
    depth            : 2-D float32 array to evaluate.
    reference        : optional baseline depth array (same shape) for ratio computation.
    downsample_for_speed : factor by which to sub-sample before computing (1 = full res).
    """
    d = depth.astype(np.float64)
    if downsample_for_speed > 1:
        d = d[::downsample_for_speed, ::downsample_for_speed]

    gy, gx = np.gradient(d)
    gm = np.sqrt(gy ** 2 + gx ** 2)

    grad_mean  = float(gm.mean())
    grad_p90   = float(np.percentile(gm, 90))
    lap        = (np.gradient(gy, axis=0) + np.gradient(gx, axis=1))
    lap_var    = float(np.var(lap))
    edge_density = float((gm > 0.5 * grad_mean).mean())

    # Local contrast: std in 9×9 windows via block processing.
    block = 9
    H, W  = d.shape
    lc_vals = []
    for r in range(0, H - block, block):
        for c in range(0, W - block, block):
            lc_vals.append(float(d[r:r+block, c:c+block].std()))
    local_contrast = float(np.mean(lc_vals)) if lc_vals else 0.0

    result: dict[str, Any] = {
        "grad_mean"     : round(grad_mean,     5),
        "grad_p90"      : round(grad_p90,      5),
        "laplacian_var" : round(lap_var,        5),
        "local_contrast": round(local_contrast, 5),
        "edge_density"  : round(edge_density,   5),
    }

    if reference is not None:
        ref = measure_structural_quality(reference,
                                         downsample_for_speed=downsample_for_speed)
        eps = 1e-9
        result["grad_ratio"]     = round(grad_mean         / (ref["grad_mean"]      + eps), 4)
        result["lap_ratio"]      = round(lap_var            / (ref["laplacian_var"]  + eps), 4)
        result["contrast_ratio"] = round(local_contrast     / (ref["local_contrast"] + eps), 4)
        result["detail_ok"]      = (
            result["grad_ratio"]     >= 0.7 and
            result["lap_ratio"]      >= 0.5 and
            result["contrast_ratio"] >= 0.7
        )

    return result


# ---------------------------------------------------------------------------
# Full pipeline diagnostic runner
# ---------------------------------------------------------------------------

def run_boundary_test(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    fast: bool = False,
    save_report: bool = True,
) -> dict[str, Any]:
    """Full pipeline: load → infer → measure boundaries + structural quality."""
    from depth_mapping.depth import (
        _clip_exposure, _load_image_geo, _tiled_inference, _single_inference,
        load_model, _DEVICE, _adaptive_edges,
        MAX_CORE_SIZE, CONTEXT_MARGIN,
    )

    image_path = Path(image_path)
    raw, geo   = _load_image_geo(str(image_path))
    image      = _clip_exposure(raw)
    W, H       = image.size

    encoder = "vits" if fast else "vitl"
    model   = load_model(encoder=encoder, device=_DEVICE)

    t0 = time.perf_counter()
    if fast or max(W, H) <= MAX_CORE_SIZE + CONTEXT_MARGIN:
        depth_arr = _single_inference(image, model, _DEVICE)
    else:
        depth_arr = _tiled_inference(image, model, _DEVICE)
    elapsed = time.perf_counter() - t0

    y_edges = _adaptive_edges(H)
    x_edges = _adaptive_edges(W)
    n_y = len(y_edges) - 1
    n_x = len(x_edges) - 1

    seam  = measure_tile_boundaries(depth_arr, y_edges, x_edges)
    qual  = measure_structural_quality(depth_arr, downsample_for_speed=2)

    # Band analysis
    bw = H // n_y if n_y > 0 else H
    band_means = [float(depth_arr[i*bw:(i+1)*bw, :].mean()) for i in range(n_y)]
    max_inter  = (max(abs(band_means[i] - band_means[i+1])
                      for i in range(len(band_means) - 1))
                  if len(band_means) > 1 else 0.0)

    print(f"\nImage : {image_path.name}  ({W}x{H} px)")
    print(f"Grid  : {n_y}r x {n_x}c   device={_DEVICE}")
    print(f"Time  : {elapsed:.2f}s")
    print(f"Range : [{depth_arr.min():.2f}, {depth_arr.max():.2f}]  "
          f"std={depth_arr.std():.2f}")
    print(f"Seam  : mean={seam['mean_mad']:.4f}  "
          f"median={seam['median_mad']:.4f}  max={seam['max_mad']:.4f}")
    print(f"Detail: grad={qual['grad_mean']:.4f}  "
          f"lap_var={qual['laplacian_var']:.4f}  "
          f"contrast={qual['local_contrast']:.4f}  "
          f"edge_density={qual['edge_density']:.4f}")
    print(f"Bands : max_inter={max_inter:.2f}  "
          f"means={[round(m,1) for m in band_means]}")

    report: dict[str, Any] = {
        "image"       : str(image_path),
        "shape"       : list(depth_arr.shape),
        "elapsed_s"   : round(elapsed, 2),
        "depth_min"   : round(float(depth_arr.min()),  4),
        "depth_max"   : round(float(depth_arr.max()),  4),
        "depth_std"   : round(float(depth_arr.std()),  4),
        "all_finite"  : bool(np.all(np.isfinite(depth_arr))),
        "tile_grid"   : {"n_y": n_y, "n_x": n_x,
                         "y_edges": y_edges, "x_edges": x_edges},
        "seam_metrics": seam,
        "structural"  : qual,
        "band_means"  : [round(m, 2) for m in band_means],
        "max_inter_band_diff": round(max_inter, 2),
    }

    if save_report and output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"{image_path.stem}_depth.npy", depth_arr)
        with open(out / "boundary_metrics.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved: {out}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(prog="python -m depth_mapping.tile_boundary_metric")
    p.add_argument("--image",      required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--fast",       action="store_true")
    a = p.parse_args()
    run_boundary_test(a.image, output_dir=a.output_dir, fast=a.fast)
