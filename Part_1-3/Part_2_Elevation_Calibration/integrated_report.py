"""
integrated_report.py — Assemble the final per-image JSON report for Part 4.

Combines:
  - Part 1/2 calibration diagnostics
  - Part 3 terrain class pixel counts
  - Per-building estimated heights
  - Pipeline provenance metadata

Public API
----------
    build_integrated_report(stem, calib_diag, terrain_stats, building_result) -> dict
    save_integrated_report(report_dict, output_path) -> Path
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("integrated_report")


def build_integrated_report(
    stem: str,
    calib_diagnostics: dict,
    terrain_stats: dict | None,
    building_result: dict,
    depth_shape: tuple[int, int] | None = None,
    elapsed: dict | None = None,
) -> dict:
    """Assemble the integrated pipeline report dictionary.

    Parameters
    ----------
    stem               : image filename stem (e.g. "bellingham1")
    calib_diagnostics  : dict from compute_calibration_diagnostics()
    terrain_stats      : pixel count dict per terrain class, or None if Part 3 skipped
    building_result    : dict from estimate_building_heights()
    depth_shape        : (H, W) of the depth/DSM array
    elapsed            : dict of {"part1_s", "part2_s", "part3_s"} timing info

    Returns
    -------
    dict — structured integrated report
    """
    calib = calib_diagnostics or {}
    bldg  = building_result or {}
    bsum  = bldg.get("summary", {})

    report: dict[str, Any] = {
        "image_id": stem,
        "pipeline": {
            "part1": "Depth Anything V2 ViT-L relative depth",
            "part2": "SRTM-calibrated approximate elevation surface (RANSAC linear fit)",
            "part3": "LoveDA + ADE20K terrain segmentation (if available)",
            "note": (
                "Outputs are RAPID PROTOTYPE estimates. "
                "NOT survey-grade DSM. NOT LiDAR replacement. "
                "Building heights are LOCAL heights (above nearby ground), "
                "NOT absolute elevation above sea level."
            ),
        },
        "calibration": {
            "a":              calib.get("a"),
            "b":              calib.get("b"),
            "method":         "RANSAC linear regression: absolute_elevation = a·depth + b",
            "reference":      "SRTM30m via api.opentopodata.org",
            "interpretation": (
                "The calibrated DSM represents ABSOLUTE ELEVATION above the reference "
                "vertical datum (SRTM30m, approximately EGM96 geoid). "
                "Values are in metres. "
                "They represent the estimated height of the surface (terrain + buildings + "
                "vegetation canopy) above the datum at each pixel, NOT height above ground. "
                "Building heights in the building_heights raster are RELATIVE — "
                "they represent height above local ground, not absolute elevation."
            ),
            "n_gcps":         calib.get("n_gcps"),
            "n_valid_gcps":   calib.get("n_valid_gcps"),
            "n_inliers":      calib.get("n_inliers"),
            "inlier_ratio":   calib.get("inlier_ratio"),
            "ransac_fit_quality": (
                "RANSAC inlier ratio measures geometric fit consistency, NOT "
                "absolute accuracy.  Even with 16/16 inliers, the residual MAE "
                "reflects the combined uncertainty of the depth model and SRTM30m "
                "(SRTM30m has ~10–30 m absolute accuracy in flat terrain, "
                "~20–50 m in complex terrain)."
            ),
            "residual_mae_m": calib.get("residual_mae_m"),
            "residual_rmse_m":calib.get("residual_rmse_m"),
            "residual_median_m": calib.get("residual_median_m"),
            "depth_std":      calib.get("depth_std"),
            "srtm_std":       calib.get("srtm_std"),
            "depth_range":    calib.get("depth_range"),
            "srtm_range":     calib.get("srtm_range"),
            "fit_method":     calib.get("fit_method", "unknown"),
            "quality":        calib.get("quality", {}),
        },
    }

    if depth_shape is not None:
        report["image_shape"] = {"height": depth_shape[0], "width": depth_shape[1]}

    # Terrain class statistics
    if terrain_stats is not None:
        report["terrain"] = terrain_stats
    else:
        report["terrain"] = {"status": "skipped", "reason": "Part 3 segmentation not run"}

    # Building height summary
    report["buildings"] = {
        "count":                   bsum.get("building_count", 0),
        "estimated_height_min_m":  bsum.get("height_min_m", 0.0),
        "estimated_height_max_m":  bsum.get("height_max_m", 0.0),
        "estimated_height_mean_m": bsum.get("height_mean_m", 0.0),
        "instance_details":        bldg.get("instance_heights", []),
        "height_interpretation": (
            "These are LOCAL building heights (metres above nearby ground), "
            "NOT absolute elevation. "
            "Computed as: P75(building DSM pixels) - P25(surrounding non-building pixels). "
            "The DSM values themselves are ~AMSL (absolute elevation from SRTM calibration). "
            "Height = 0 means no building detected or building is at ground level."
        ),
        "accuracy_note": (
            "Building height accuracy is limited by SRTM calibration (~10–30 m RMSE) "
            "and semantic segmentation accuracy. Do not use for structural assessment."
        ),
    }

    # Timing info
    if elapsed:
        report["elapsed_seconds"] = elapsed

    return report


def save_integrated_report(report: dict, output_path: str | Path) -> Path:
    """Write the integrated report as a formatted JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=_json_default))
    logger.info("Saved integrated report: %s", output_path)
    return output_path.resolve()


def _json_default(obj: Any) -> Any:
    """Custom JSON serialiser for numpy scalars and similar types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
