"""
building_heights.py — Part 2/3 bridge: semantic building height estimation.

Replaces the scientifically incorrect global-5th-percentile building height
calculation with a per-building local ground estimation approach:

    DSM (calibrated elevation surface)
    + semantic building mask (from Part 3)
    → per-building local ground estimation (annular neighbourhood)
    → estimated_building_height = roof_elevation − local_ground_elevation

Memory safety notes (5000×5000 target)
---------------------------------------
A 5000×5000 float32 array is 100 MB.  This module is careful to:
  - Never store full-image arrays inside per-instance dicts
  - Work in local bounding-box windows wherever possible
  - Use a single labeled array computed once and indexed cheaply
  - Not duplicate the DSM or building_mask unnecessarily
  - Pre-compute the global ground fallback once rather than per-instance

Public API
----------
    estimate_building_heights(dsm_array, building_mask, instances=None)
        → dict with keys: "height_map", "instance_heights", "summary"

    save_building_height_geotiff(height_map, geo_meta, output_path)
        → Path

Scientific language
-------------------
Outputs are labelled "estimated building height" derived from an
"SRTM-calibrated approximate elevation surface".  These are NOT survey-grade
heights.  Accuracy is limited by:
  - relative depth model uncertainty
  - SRTM calibration residuals (~10–30 m typical)
  - semantic segmentation errors
  - local terrain estimation from a limited annular neighbourhood
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("building_heights")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Annular neighbourhood ring parameters
_INNER_MARGIN_PX: int = 3    # skip immediately adjacent pixels
_OUTER_MARGIN_PX: int = 20   # primary ring thickness

# Fallback: expand to larger ring if primary yields too few pixels
_MIN_GROUND_SAMPLES: int = 50
_EXPANDED_OUTER_PX: int = 50

# Robust statistics
_GROUND_PERCENTILE: float = 25.0   # lower pct → near terrain floor
_ROOF_PERCENTILE:   float = 75.0   # upper pct → roof surface

_MIN_BUILDING_HEIGHT_M: float =   0.0
_MAX_CREDIBLE_HEIGHT_M: float = 500.0


# ===========================================================================
# Internal helpers
# ===========================================================================

def _disk_struct(radius: int) -> np.ndarray:
    """Boolean disk structuring element."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2) <= radius ** 2


def _sample_local_ground(
    dsm: np.ndarray,
    building_mask: np.ndarray,
    row_min: int, row_max: int,
    col_min: int, col_max: int,
    outer_px: int,
) -> tuple[np.ndarray, int]:
    """Extract ground-elevation samples from an expanded bounding-box window.

    Works entirely in a local crop — never touches the full-image arrays
    beyond a single cheap slice.

    Returns (elevation_values_1d, n_samples).
    """
    H, W = dsm.shape
    r0 = max(0, row_min - outer_px)
    r1 = min(H, row_max + outer_px + 1)
    c0 = max(0, col_min - outer_px)
    c1 = min(W, col_max + outer_px + 1)

    # Cheap local slices — views, not copies
    crop_dsm  = dsm[r0:r1, c0:c1]
    crop_bldg = building_mask[r0:r1, c0:c1]

    # Ground = not any building, finite value
    ground_mask = (~crop_bldg) & np.isfinite(crop_dsm)
    vals = crop_dsm[ground_mask]
    return vals, len(vals)


# ===========================================================================
# Core estimation
# ===========================================================================

def estimate_building_heights(
    dsm_array: np.ndarray,
    building_mask: np.ndarray,
    instances: list[dict] | None = None,
) -> dict[str, Any]:
    """Estimate per-building heights using local ground estimation.

    Memory-safe for 5000×5000 images:
      - The labeled array is computed once (int32, 100 MB at 5000×5000) and
        released after instance processing.
      - No full-image bool array is stored per instance.
      - Ground sampling uses local bounding-box windows only.
      - The global fallback array is computed once and cached.

    Parameters
    ----------
    dsm_array     : (H, W) float32 — SRTM-calibrated approximate elevation (m)
    building_mask : (H, W) bool/uint8 — True/1 where building pixels exist
    instances     : list of building instance dicts from Part 3 JSON.
                    Each needs "instance_id" and "bbox": [x_min, y_min, x_max, y_max].
                    Pass None to auto-derive from the mask via connected components.

    Returns
    -------
    dict:
        "height_map"       : (H, W) float32 — 0 outside buildings, height inside
        "instance_heights" : list of per-instance result dicts
        "summary"          : aggregate stats dict
    """
    from scipy.ndimage import label as cc_label, find_objects, sum as ndi_sum

    if dsm_array.ndim != 2:
        raise ValueError(f"dsm_array must be 2-D, got shape {dsm_array.shape}")
    if building_mask.shape != dsm_array.shape:
        raise ValueError(
            f"building_mask shape {building_mask.shape} != dsm_array shape {dsm_array.shape}"
        )

    H, W = dsm_array.shape
    # Single bool view — no copy unless dtype conversion needed
    bldg_bool = building_mask.astype(bool, copy=False)

    # Output raster — one float32 array, zeroed
    height_map = np.zeros((H, W), dtype=np.float32)

    if not bldg_bool.any():
        logger.info("  No building pixels in mask — returning all-zero height map.")
        return {
            "height_map": height_map,
            "instance_heights": [],
            "summary": {"building_count": 0, "height_min_m": 0.0,
                        "height_max_m": 0.0, "height_mean_m": 0.0},
        }

    # ------------------------------------------------------------------
    # Pre-compute the global ground fallback ONCE.
    # Uses a boolean expression without materialising a full-image copy.
    # For 5000×5000: one ~25 MB bool mask operation, then immediate flatten
    # ------------------------------------------------------------------
    _global_ground_cache: np.ndarray | None = None

    def _get_global_ground() -> np.ndarray:
        nonlocal _global_ground_cache
        if _global_ground_cache is None:
            ground_mask = (~bldg_bool) & np.isfinite(dsm_array)
            _global_ground_cache = dsm_array[ground_mask]
        return _global_ground_cache

    # ------------------------------------------------------------------
    # Single labeled array — computed once, shared across all instances
    # int32 at 5000×5000 = 100 MB; released at end of function
    # ------------------------------------------------------------------
    labeled, n_feats = cc_label(bldg_bool)

    # ------------------------------------------------------------------
    # Build lightweight per-instance metadata (bbox + blob_id only).
    # No full-image arrays stored per instance.
    # ------------------------------------------------------------------
    if instances is None or len(instances) == 0:
        logger.info("  Auto-deriving building instances from mask CC …")
        blob_slices = find_objects(labeled)
        blob_ids    = np.arange(1, n_feats + 1)
        areas_all   = ndi_sum(bldg_bool, labeled, blob_ids)

        instances_meta: list[dict] = []
        for i, sl in enumerate(blob_slices):
            if sl is None:
                continue
            area = int(areas_all[i])
            if area < 50:
                continue
            blob_id = i + 1
            instances_meta.append({
                "instance_id": blob_id,
                "blob_id":     blob_id,
                "bbox":        [sl[1].start, sl[0].start,
                                sl[1].stop - 1, sl[0].stop - 1],  # [xmin,ymin,xmax,ymax]
                "area_pixels": area,
                "confidence":  None,
            })
    else:
        instances_meta = []
        for inst in instances:
            if inst.get("class", "building") not in ("building", ""):
                continue
            bbox = inst.get("bbox", [])
            if len(bbox) != 4:
                continue
            x_min, y_min, x_max, y_max = [int(v) for v in bbox]
            x_min = max(0, x_min); y_min = max(0, y_min)
            x_max = min(W - 1, x_max); y_max = min(H - 1, y_max)

            # Find the dominant blob ID in this bounding box — cheap crop
            bb_labels = labeled[y_min:y_max + 1, x_min:x_max + 1]
            nonzero   = bb_labels[bb_labels > 0]
            if len(nonzero) == 0:
                blob_id = -1
            else:
                ids, cnts = np.unique(nonzero, return_counts=True)
                blob_id = int(ids[np.argmax(cnts)])

            instances_meta.append({
                "instance_id": inst["instance_id"],
                "blob_id":     blob_id,
                "bbox":        [x_min, y_min, x_max, y_max],
                "area_pixels": inst.get("area_pixels",
                                        (y_max - y_min + 1) * (x_max - x_min + 1)),
                "confidence":  inst.get("confidence"),
            })

    logger.info("  Processing %d building instances …", len(instances_meta))

    instance_heights: list[dict] = []
    heights_for_summary: list[float] = []

    for inst in instances_meta:
        x_min, y_min, x_max, y_max = inst["bbox"]
        blob_id = inst["blob_id"]

        # ------------------------------------------------------------------
        # Roof elevation: work in the bounding-box window only.
        # Extract a local DSM crop and a matching blob mask crop.
        # No full-image bool array per instance.
        # ------------------------------------------------------------------
        if blob_id > 0:
            # Cheap local crop of the labeled array — view
            local_labeled = labeled[y_min:y_max + 1, x_min:x_max + 1]
            local_blob    = (local_labeled == blob_id)           # small bool array
        else:
            # Fallback: use building mask crop directly
            local_blob = bldg_bool[y_min:y_max + 1, x_min:x_max + 1]

        local_dsm   = dsm_array[y_min:y_max + 1, x_min:x_max + 1]
        local_valid = local_blob & np.isfinite(local_dsm)
        blob_dsm_vals = local_dsm[local_valid]

        if len(blob_dsm_vals) == 0:
            logger.debug("  Instance %d: no finite DSM values — skipping.",
                         inst["instance_id"])
            continue

        roof_elev = float(np.percentile(blob_dsm_vals, _ROOF_PERCENTILE))

        # ------------------------------------------------------------------
        # Local ground estimation — window-based, no full-image temp arrays
        # ------------------------------------------------------------------
        ground_vals, n_primary = _sample_local_ground(
            dsm_array, bldg_bool,
            y_min, y_max, x_min, x_max, _OUTER_MARGIN_PX,
        )
        if n_primary >= _MIN_GROUND_SAMPLES:
            ground_elev = float(np.percentile(ground_vals, _GROUND_PERCENTILE))
            method = "local_annulus"
        else:
            ground_vals, n_expanded = _sample_local_ground(
                dsm_array, bldg_bool,
                y_min, y_max, x_min, x_max, _EXPANDED_OUTER_PX,
            )
            if n_expanded >= _MIN_GROUND_SAMPLES:
                ground_elev = float(np.percentile(ground_vals, _GROUND_PERCENTILE))
                method = "expanded_local_annulus"
            else:
                # Global fallback — pre-computed once above
                gvals = _get_global_ground()
                if len(gvals) > 0:
                    ground_elev = float(np.percentile(gvals, _GROUND_PERCENTILE))
                    method = "global_fallback"
                    logger.debug("  Instance %d at (%d,%d): global fallback.",
                                 inst["instance_id"], y_min, x_min)
                else:
                    finite_all = dsm_array[np.isfinite(dsm_array)]
                    ground_elev = float(np.percentile(finite_all, 5)) \
                        if len(finite_all) > 0 else 0.0
                    method = "global_absolute_fallback"

        # ------------------------------------------------------------------
        # Estimated height
        # ------------------------------------------------------------------
        est_height = max(_MIN_BUILDING_HEIGHT_M, roof_elev - ground_elev)

        height_flag = None
        if est_height > _MAX_CREDIBLE_HEIGHT_M:
            height_flag = f"exceeds_{_MAX_CREDIBLE_HEIGHT_M}m_likely_calibration_error"
            logger.warning(
                "  Instance %d: height=%.1f m > %d m — possible calibration artefact.",
                inst["instance_id"], est_height, _MAX_CREDIBLE_HEIGHT_M,
            )

        # Write into the output raster — index only the local crop via blob_id
        if blob_id > 0:
            # Reuse the already-computed local_blob view
            row_slice = slice(y_min, y_max + 1)
            col_slice = slice(x_min, x_max + 1)
            height_map[row_slice, col_slice][local_blob] = est_height
        else:
            height_map[y_min:y_max + 1, x_min:x_max + 1][local_blob] = est_height

        heights_for_summary.append(est_height)

        entry: dict[str, Any] = {
            "instance_id":        inst["instance_id"],
            "bbox":               [x_min, y_min, x_max, y_max],
            "area_pixels":        inst["area_pixels"],
            "ground_elevation_m": round(ground_elev, 3),
            "roof_elevation_m":   round(roof_elev, 3),
            "estimated_height_m": round(est_height, 3),
            "height_method":      method,
        }
        if inst["confidence"] is not None:
            entry["confidence"] = inst["confidence"]
        if height_flag:
            entry["height_flag"] = height_flag
        instance_heights.append(entry)

    # Release the labeled array — no longer needed
    del labeled

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if heights_for_summary:
        valid_hs = [h for h in heights_for_summary if np.isfinite(h)]
        summary = {
            "building_count": len(valid_hs),
            "height_min_m":   round(float(min(valid_hs)), 3) if valid_hs else 0.0,
            "height_max_m":   round(float(max(valid_hs)), 3) if valid_hs else 0.0,
            "height_mean_m":  round(float(np.mean(valid_hs)), 3) if valid_hs else 0.0,
        }
    else:
        summary = {"building_count": 0, "height_min_m": 0.0,
                   "height_max_m": 0.0, "height_mean_m": 0.0}

    logger.info(
        "  Building heights: count=%d  min=%.1f m  max=%.1f m  mean=%.1f m",
        summary["building_count"], summary["height_min_m"],
        summary["height_max_m"],   summary["height_mean_m"],
    )

    return {
        "height_map":       height_map,
        "instance_heights": instance_heights,
        "summary":          summary,
    }


# ===========================================================================
# Output writers
# ===========================================================================

def save_building_height_geotiff(
    height_map: np.ndarray,
    geo_meta: dict,
    output_path: str | Path,
) -> Path:
    """Write building height raster as a single-band float32 GeoTIFF.

    Non-building pixels are 0.  CRS and affine transform are preserved.
    Output is tiled + LZW-compressed for efficient storage of large files.
    """
    import rasterio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver":     "GTiff",
        "dtype":      "float32",
        "width":      geo_meta["width"],
        "height":     geo_meta["height"],
        "count":      1,
        "crs":        geo_meta["crs"],
        "transform":  geo_meta["transform"],
        "compress":   "lzw",
        "predictor":  3,
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata":     0.0,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(height_map.astype(np.float32), 1)
        dst.update_tags(
            PRODUCT_TYPE="estimated_building_height",
            UNITS="metres_above_local_ground",
            METHOD="semantic_mask_local_ground_annulus",
            NOTE=(
                "Estimated building height from SRTM-calibrated approximate elevation "
                "surface and semantic building mask. NOT survey-grade. NOT LiDAR."
            ),
        )

    logger.info("Saved building height GeoTIFF: %s", output_path)
    return output_path.resolve()


def save_building_height_preview(
    height_map: np.ndarray,
    output_path: str | Path,
    colormap: str = "plasma",
) -> Path:
    """Write a false-colour preview PNG of the building height raster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bldg_pixels = height_map[height_map > 0]
    vmax = float(np.percentile(bldg_pixels, 98)) if len(bldg_pixels) > 0 else 1.0
    display = np.where(height_map > 0, height_map, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    cmap = plt.get_cmap(colormap).copy()
    cmap.set_bad(color=(0.1, 0.1, 0.1, 1.0))
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Estimated Height Above Ground (m)", fontsize=10)
    ax.set_title("Estimated Building Heights (metres above local ground)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info("Saved building height preview: %s", output_path)
    return output_path.resolve()
