"""
Batch Processing Script for GeoMonoDSM-3D Pipeline
===================================================

Processes all GeoTIFF images in an input folder through the full pipeline:

    Part 1: Depth inference (Depth Anything V2 ViT-L)
    Part 2: SRTM calibration → calibrated approximate elevation DSM
    Part 3: LoveDA+ADE20K terrain segmentation → building mask
    Fusion: building-specific local ground estimation → estimated building heights

Usage:
    python batch_process.py --input-dir Input --output-dir output

Or from anywhere:
    python batch_process.py \\
        --input-dir "path/to/input" \\
        --output-dir "path/to/output" \\
        --checkpoint-dir "D:/path/to/checkpoints"

Creates for each image:
    {stem}.npy                       — raw relative depth array (float32)
    {stem}_depth.png                 — depth preview (viridis)
    {stem}_depth.tif                 — georeferenced relative depth GeoTIFF
    {stem}_dsm.tif                   — SRTM-calibrated approximate elevation GeoTIFF
    {stem}_dsm_preview.png           — terrain colormap preview
    {stem}_terrain_labels.tif        — uint8 terrain class raster (0–4)
    {stem}_labelmap.png              — raw label map PNG for Part 4
    {stem}_overlay.png               — human-readable classification overlay
    {stem}_data.json                 — Part 3 segmentation JSON for Part 4
    {stem}_building_heights.tif      — estimated building height GeoTIFF (float32)
    {stem}_heights_preview.png       — building height false-colour preview
    {stem}_integrated_report.json    — full pipeline report for Part 4

Scientific note
---------------
Outputs are labelled "SRTM-calibrated approximate elevation" and
"estimated building height".  These are rapid-prototype estimates —
NOT survey-grade DSM, NOT LiDAR replacement.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PART1_DIR  = SCRIPT_DIR / "Depth_Mapping"
PART2_DIR  = SCRIPT_DIR / "Elevation_Calibration"
PART3_DIR  = SCRIPT_DIR / "Terrain_Classificartion"   # note: original folder name kept
DEFAULT_CHECKPOINT = Path(r"D:\Collaborate_Projects\SIH\checkpoints")

for p in (str(PART1_DIR), str(PART2_DIR), str(PART3_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_process")


# ===========================================================================
# Pipeline functions
# ===========================================================================

def _terrain_pixel_counts(label_map: np.ndarray) -> dict:
    """Return a dict of pixel counts per terrain class."""
    total = label_map.size
    counts = {
        "building_pixels":   int(np.sum(label_map == 2)),
        "vegetation_pixels": int(np.sum(label_map == 1)),
        "road_pixels":       int(np.sum(label_map == 3)),
        "water_pixels":      int(np.sum(label_map == 4)),
        "other_pixels":      int(np.sum(label_map == 0)),
        "total_pixels":      total,
    }
    # Percentages for readability
    for cls in ("building", "vegetation", "road", "water", "other"):
        px = counts[f"{cls}_pixels"]
        counts[f"{cls}_pct"] = round(100.0 * px / total, 2) if total > 0 else 0.0
    return counts


def process_one_image(
    input_path: Path,
    output_dir: Path,
    checkpoint_dir: Path | None = None,
    skip_part3: bool = False,
    use_dual_model: bool = False,
) -> dict[str, Any]:
    """Run Part 1 → Part 2 → Part 3 → Height fusion on a single GeoTIFF.

    Parameters
    ----------
    input_path     : source GeoTIFF
    output_dir     : where all outputs go
    checkpoint_dir : optional path to DA V2 checkpoint folder
    skip_part3     : skip terrain segmentation (Part 1+2 only)
    use_dual_model : enable ADE20K dual-model fusion in Part 3 (slower)

    Returns
    -------
    Result statistics dict.
    """
    stem = input_path.stem
    logger.info("=" * 70)
    logger.info("Processing: %s", input_path.name)
    logger.info("=" * 70)

    if checkpoint_dir:
        os.environ["DA_CHECKPOINT_DIR"] = str(checkpoint_dir)

    # ── PART 1: Depth inference ──────────────────────────────────────────
    logger.info("Part 1 — Depth inference …")
    from depth_mapping import get_depth_with_meta, save_depth_outputs

    t0 = time.perf_counter()
    depth_arr, geo_meta = get_depth_with_meta(str(input_path), fast=False)
    elapsed_p1 = time.perf_counter() - t0

    logger.info(
        "  Depth: shape=%s  range=[%.2f, %.2f]  (%.1fs)",
        depth_arr.shape, depth_arr.min(), depth_arr.max(), elapsed_p1,
    )

    save_depth_outputs(
        depth_arr,
        output_dir=str(output_dir),
        filename=stem,
        geo_meta=geo_meta,
        save_visualization=False,
    )
    logger.info("  Saved depth outputs: %s.npy / _depth.png / _depth.tif", stem)

    # ── PART 2: SRTM calibration ──────────────────────────────────────────
    logger.info("Part 2 — SRTM calibration …")
    from elevation_calibration import (
        fetch_reference_elevations,
        fit_calibration,
        compute_calibration_diagnostics,
        save_dsm_geotiff,
        save_dsm_preview,
    )

    t1 = time.perf_counter()

    depth_vals, srtm_vals = fetch_reference_elevations(
        depth_arr, geo_meta["transform"], geo_meta["crs"], n_gcps=16
    )
    a, b, fit_diag = fit_calibration(depth_vals, srtm_vals)

    calib_diag = compute_calibration_diagnostics(
        a, b, depth_vals, srtm_vals, fit_diag, n_gcps_requested=16
    )
    logger.info(
        "  Calibration: a=%.6f  b=%.4f m  quality=%s",
        a, b, calib_diag["quality"]["status"],
    )

    dsm_array = np.add(np.multiply(depth_arr, np.float32(a), dtype=np.float32),
                       np.float32(b), dtype=np.float32)
    logger.info("  DSM range: [%.1f, %.1f] m", dsm_array.min(), dsm_array.max())

    dsm_tif     = save_dsm_geotiff(dsm_array, geo_meta, output_dir / f"{stem}_dsm.tif")
    dsm_preview = save_dsm_preview(dsm_array, output_dir / f"{stem}_dsm_preview.png")
    elapsed_p2  = time.perf_counter() - t1

    # ── PART 3: Terrain segmentation ────────────────────────────────────
    label_map: np.ndarray | None = None
    part3_json_path: Path | None = None
    part3_overlay_path: Path | None = None
    building_instances: list[dict] = []
    terrain_stats: dict | None = None
    elapsed_p3 = 0.0

    if not skip_part3:
        logger.info("Part 3 — Terrain segmentation …")
        try:
            from segmentation import classify_terrain

            t2 = time.perf_counter()
            part3_json_path, part3_overlay_path = classify_terrain(
                input_path,             # use original RGB GeoTIFF (same as Part 1)
                output_dir=output_dir,
                use_dual_model=use_dual_model,
            )
            elapsed_p3 = time.perf_counter() - t2
            logger.info("  Part 3 done in %.1fs", elapsed_p3)

            # Load label map from the written PNG (uint8, values 0–4)
            from PIL import Image as PILImage
            labelmap_png = output_dir / f"{stem}_labelmap.png"
            if labelmap_png.is_file():
                label_map = np.array(PILImage.open(labelmap_png))
            else:
                # Fallback: load from terrain_labels.tif
                tif_labels = output_dir / f"{stem}_terrain_labels.tif"
                if tif_labels.is_file():
                    import rasterio
                    with rasterio.open(tif_labels) as src:
                        label_map = src.read(1)

            if label_map is not None:
                terrain_stats = _terrain_pixel_counts(label_map)
                logger.info(
                    "  Terrain: building=%.1f%%  vegetation=%.1f%%  "
                    "road=%.1f%%  water=%.1f%%",
                    terrain_stats["building_pct"], terrain_stats["vegetation_pct"],
                    terrain_stats["road_pct"],     terrain_stats["water_pct"],
                )

            # Load building instances from Part 3 JSON
            if part3_json_path and part3_json_path.is_file():
                with open(part3_json_path, encoding="utf-8") as fh:
                    part3_data = json.load(fh)
                building_instances = [
                    d for d in part3_data.get("detections", [])
                    if d.get("class") == "building"
                ]
                logger.info("  Building instances from Part 3: %d", len(building_instances))

        except Exception as exc:
            import traceback
            logger.warning("Part 3 failed — continuing without segmentation.")
            logger.warning("Error: %s", exc)
            logger.debug(traceback.format_exc())
    else:
        logger.info("Part 3 — Skipped (--skip-part3 flag).")

    # ── BUILDING HEIGHT FUSION ──────────────────────────────────────────
    logger.info("Height fusion — estimating building heights …")
    from building_heights import (
        estimate_building_heights,
        save_building_height_geotiff,
        save_building_height_preview,
    )
    from integrated_report import build_integrated_report, save_integrated_report

    building_mask = (label_map == 2) if label_map is not None else None

    t3 = time.perf_counter()
    if building_mask is not None:
        height_result = estimate_building_heights(
            dsm_array, building_mask, instances=building_instances or None
        )
    else:
        # No segmentation — produce all-zero height map
        logger.info(
            "  No building mask available — producing all-zero building height map."
        )
        height_result = {
            "height_map": np.zeros_like(dsm_array),
            "instance_heights": [],
            "summary": {
                "building_count": 0,
                "height_min_m": 0.0,
                "height_max_m": 0.0,
                "height_mean_m": 0.0,
            },
        }
    elapsed_fusion = time.perf_counter() - t3

    height_map = height_result["height_map"]

    # Save building height raster
    height_tif = save_building_height_geotiff(
        height_map, geo_meta, output_dir / f"{stem}_building_heights.tif"
    )
    height_preview = save_building_height_preview(
        height_map, output_dir / f"{stem}_heights_preview.png"
    )

    bsum = height_result["summary"]
    logger.info(
        "  Building heights: count=%d  min=%.1f m  max=%.1f m  mean=%.1f m",
        bsum["building_count"],
        bsum["height_min_m"],
        bsum["height_max_m"],
        bsum["height_mean_m"],
    )

    # ── INTEGRATED REPORT ──────────────────────────────────────────────
    elapsed_dict = {
        "part1_depth_s": round(elapsed_p1, 2),
        "part2_calib_s": round(elapsed_p2, 2),
        "part3_segm_s":  round(elapsed_p3, 2),
        "fusion_s":      round(elapsed_fusion, 2),
    }

    integrated = build_integrated_report(
        stem=stem,
        calib_diagnostics=calib_diag,
        terrain_stats=terrain_stats,
        building_result=height_result,
        depth_shape=tuple(depth_arr.shape),
        elapsed=elapsed_dict,
    )
    # Attach Part 3 instance heights into the report's building section
    integrated["buildings"]["instance_details"] = height_result["instance_heights"]

    report_path = save_integrated_report(
        integrated, output_dir / f"{stem}_integrated_report.json"
    )

    # ── Legacy JSON report (kept for backward compat) ──────────────────
    legacy_stats = {
        "input_file":         input_path.name,
        "depth_shape":        list(depth_arr.shape),
        "elapsed_p1_seconds": round(elapsed_p1, 2),
        "elapsed_p2_seconds": round(elapsed_p2, 2),
        "calibration": {
            "a":        round(a, 6),
            "b":        round(b, 4),
            "method":   "RANSAC/polyfit vs SRTM30m",
            "reference": "api.opentopodata.org",
            "quality":  calib_diag["quality"]["status"],
        },
        "dsm_metres": {
            "min":  round(float(dsm_array.min()), 2),
            "max":  round(float(dsm_array.max()), 2),
            "mean": round(float(dsm_array.mean()), 2),
        },
        "terrain": terrain_stats or {"status": "skipped"},
        "building_heights_metres": {
            "count":      bsum["building_count"],
            "min_height": bsum["height_min_m"],
            "max_height": bsum["height_max_m"],
            "mean_height": bsum["height_mean_m"],
            "method":     "semantic_mask_local_ground_annulus",
            "note":       (
                "Estimated from SRTM-calibrated elevation surface minus per-building "
                "local ground elevation. NOT survey-grade."
            ),
        },
        "outputs": {
            "depth_npy":         f"{stem}.npy",
            "depth_tif":         f"{stem}_depth.tif",
            "depth_png":         f"{stem}_depth.png",
            "dsm_tif":           f"{stem}_dsm.tif",
            "dsm_preview":       f"{stem}_dsm_preview.png",
            "terrain_labels_tif": f"{stem}_terrain_labels.tif",
            "labelmap_png":      f"{stem}_labelmap.png",
            "overlay_png":       f"{stem}_overlay.png",
            "heights_tif":       f"{stem}_building_heights.tif",
            "heights_preview":   f"{stem}_heights_preview.png",
            "integrated_report": f"{stem}_integrated_report.json",
        },
    }

    legacy_path = output_dir / f"{stem}_report.json"
    legacy_path.write_text(json.dumps(legacy_stats, indent=2))

    logger.info(
        "✓ Completed: %s  (P1: %.1fs  P2: %.1fs  P3: %.1fs  fusion: %.1fs)",
        input_path.name, elapsed_p1, elapsed_p2, elapsed_p3, elapsed_fusion,
    )

    return legacy_stats


def batch_process(
    input_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path | None = None,
    pattern: str = "*.tif",
    skip_part3: bool = False,
    use_dual_model: bool = False,
) -> None:
    """Process all GeoTIFF files in input_dir."""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Input dir : %s", input_dir.resolve())
    logger.info("Output dir: %s", output_dir.resolve())
    if checkpoint_dir:
        logger.info("Checkpoints: %s", checkpoint_dir.resolve())

    tif_files = sorted(
        list(input_dir.glob(pattern))
        + list(input_dir.glob(pattern.replace(".tif", ".tiff")))
    )

    if not tif_files:
        logger.warning("No .tif/.tiff files found in %s", input_dir)
        return

    logger.info("Found %d GeoTIFF file(s) to process:", len(tif_files))
    for i, f in enumerate(tif_files, 1):
        logger.info("  %d. %s  (%.1f MB)", i, f.name, f.stat().st_size / 1_048_576)

    results = []
    start_time = time.perf_counter()

    for idx, tif_path in enumerate(tif_files, 1):
        logger.info("")
        logger.info("[%d/%d] Starting: %s", idx, len(tif_files), tif_path.name)

        try:
            result = process_one_image(
                tif_path, output_dir, checkpoint_dir,
                skip_part3=skip_part3,
                use_dual_model=use_dual_model,
            )
            results.append(result)

        except Exception as exc:
            import traceback
            logger.error("FAILED: %s", tif_path.name)
            logger.error("Error: %s", exc)
            logger.error(traceback.format_exc())
            results.append({
                "input_file": tif_path.name,
                "status": "error",
                "error": str(exc),
            })

    total_elapsed = time.perf_counter() - start_time
    n_success = sum(1 for r in results if r.get("status") != "error")
    n_failed  = len(results) - n_success

    logger.info("")
    logger.info("=" * 70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info("Processed: %d / %d succeeded", n_success, len(results))
    if n_failed > 0:
        logger.warning("Failed: %d", n_failed)
        for r in results:
            if r.get("status") == "error":
                logger.warning("  - %s", r["input_file"])

    logger.info("Total time: %.1f minutes", total_elapsed / 60)
    logger.info("Output directory: %s", output_dir.resolve())

    summary_path = output_dir / "_batch_summary.json"
    summary = {
        "total_files": len(results),
        "succeeded":   n_success,
        "failed":      n_failed,
        "total_seconds": round(total_elapsed, 1),
        "results":     results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Batch summary: %s", summary_path.name)


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch process GeoTIFFs through GeoMonoDSM-3D pipeline "
            "(Part 1 depth + Part 2 SRTM calibration + Part 3 segmentation + "
            "building height fusion)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process.py --input-dir Input --output-dir output

  python batch_process.py \\
      --input-dir "C:/data/images" \\
      --output-dir "C:/data/results" \\
      --checkpoint-dir "D:/models/checkpoints"

  # Skip Part 3 segmentation (faster, Part 1+2 only):
  python batch_process.py --input-dir Input --output-dir output --skip-part3

  # Enable dual-model (LoveDA + ADE20K) fusion:
  python batch_process.py --input-dir Input --output-dir output --dual-model
        """,
    )
    parser.add_argument("--input-dir",  required=True,
                        help="Directory containing input GeoTIFF files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for all outputs (created if needed)")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT),
                        help=f"Path to DA V2 checkpoints (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--pattern",    default="*.tif",
                        help="File pattern to match (default: *.tif)")
    parser.add_argument("--skip-part3", action="store_true",
                        help="Skip terrain segmentation (Part 1+2 only)")
    parser.add_argument("--dual-model", action="store_true",
                        help="Enable ADE20K dual-model fusion in Part 3")

    args = parser.parse_args()

    batch_process(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        pattern=args.pattern,
        skip_part3=args.skip_part3,
        use_dual_model=args.dual_model,
    )


if __name__ == "__main__":
    main()
