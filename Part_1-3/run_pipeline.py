"""
run_pipeline.py — Single-image GeoMonoDSM-3D pipeline runner.

Usage (from D:\\SIH_2.0\\GeomonoDSM-3D-main):

  # Full pipeline (Part 1 + 2 + 3):
  python run_pipeline.py --input Input/sfo3.tif --output-dir output/sfo3_full

  # Skip Part 1 when depth .npy already exists:
  python run_pipeline.py --input Input/sfo3.tif --output-dir output/sfo3_full --depth output/sfo3.npy

  # Skip Part 3 (Part 1+2 only):
  python run_pipeline.py --input Input/sfo3.tif --output-dir output/sfo3_full --skip-part3

  # Custom checkpoint directory:
  python run_pipeline.py --input Input/sfo3.tif --output-dir output/sfo3_full \\
      --checkpoint-dir Depth_Mapping/checkpoints

  # Dual-model (LoveDA + ADE20K) for better road/water detection:
  python run_pipeline.py --input Input/sfo3.tif --output-dir output/sfo3_full --dual-model

Windows PowerShell examples:
  python run_pipeline.py --input Input\\sfo3.tif --output-dir output\\sfo3_result
  python run_pipeline.py --input Input\\sfo3.tif --output-dir output\\sfo3_result --depth output\\sfo3.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PART1_DIR  = SCRIPT_DIR / "Depth_Mapping"
PART2_DIR  = SCRIPT_DIR / "Elevation_Calibration"
PART3_DIR  = SCRIPT_DIR / "Terrain_Classificartion"
DEFAULT_CKPT = SCRIPT_DIR / "Depth_Mapping" / "checkpoints"

for p in (str(PART1_DIR), str(PART2_DIR), str(PART3_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def run(
    input_path: Path,
    output_dir: Path,
    depth_npy: Path | None = None,
    checkpoint_dir: Path | None = None,
    skip_part3: bool = False,
    dual_model: bool = False,
    skip_gamus: bool = True,
) -> dict:
    """Run the full pipeline on a single GeoTIFF.

    Returns the integrated result dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    if checkpoint_dir:
        os.environ["DA_CHECKPOINT_DIR"] = str(checkpoint_dir)

    t_total = time.perf_counter()

    # ── PART 1 ────────────────────────────────────────────────────────────
    if depth_npy and depth_npy.is_file():
        logger.info("Part 1 — Reusing existing depth: %s", depth_npy)
        depth_arr = np.load(depth_npy)
        # We still need geo_meta from the GeoTIFF
        import rasterio
        with rasterio.open(input_path) as src:
            geo_meta = {
                "crs": src.crs, "transform": src.transform,
                "width": src.width, "height": src.height,
                "nodata": src.nodata,
            }
        elapsed_p1 = 0.0
        logger.info("  Loaded depth %s  range=[%.2f, %.2f]",
                    depth_arr.shape, depth_arr.min(), depth_arr.max())
    else:
        logger.info("Part 1 — Running Depth Anything V2 …")
        from depth_mapping import get_depth_with_meta, save_depth_outputs
        t0 = time.perf_counter()
        depth_arr, geo_meta = get_depth_with_meta(str(input_path), fast=False)
        elapsed_p1 = time.perf_counter() - t0
        save_depth_outputs(depth_arr, output_dir=str(output_dir),
                           filename=stem, geo_meta=geo_meta, save_visualization=False)
        logger.info("  Depth done in %.1fs  shape=%s  range=[%.2f, %.2f]",
                    elapsed_p1, depth_arr.shape, depth_arr.min(), depth_arr.max())

    H, W = depth_arr.shape

    # ── PART 2 ────────────────────────────────────────────────────────────
    logger.info("Part 2 — SRTM calibration …")
    from elevation_calibration import (
        fetch_reference_elevations, fit_calibration,
        compute_calibration_diagnostics, save_dsm_geotiff, save_dsm_preview,
    )

    t2 = time.perf_counter()
    depth_vals, srtm_vals = fetch_reference_elevations(
        depth_arr, geo_meta["transform"], geo_meta["crs"], n_gcps=16,
    )
    a, b, fit_diag = fit_calibration(depth_vals, srtm_vals)
    calib_diag = compute_calibration_diagnostics(
        a, b, depth_vals, srtm_vals, fit_diag, n_gcps_requested=16,
    )
    quality = calib_diag["quality"]["status"]
    logger.info("  Calibration: a=%.6f  b=%.4f m  quality=%s", a, b, quality)
    if quality != "good":
        logger.warning("  Calibration reason: %s", calib_diag["quality"]["reason"])

    dsm_array = np.add(
        np.multiply(depth_arr, np.float32(a), dtype=np.float32),
        np.float32(b), dtype=np.float32,
    )
    elapsed_p2 = time.perf_counter() - t2
    logger.info("  DSM range: [%.1f, %.1f] m  (%.1fs)", dsm_array.min(), dsm_array.max(), elapsed_p2)

    dsm_tif     = save_dsm_geotiff(dsm_array, geo_meta, output_dir / f"{stem}_dsm.tif")
    dsm_preview = save_dsm_preview(dsm_array, output_dir / f"{stem}_dsm_preview.png")

    # ── PART 3 ────────────────────────────────────────────────────────────
    label_map: np.ndarray | None = None
    building_instances: list[dict] = []
    terrain_stats: dict | None = None
    elapsed_p3 = 0.0
    part3_status = "skipped"

    if not skip_part3:
        logger.info("Part 3 — Terrain segmentation (LoveDA) …")
        try:
            from segmentation import classify_terrain
            from PIL import Image as PILImage

            t3 = time.perf_counter()
            json_path, overlay_path = classify_terrain(
                input_path,
                output_dir=output_dir,
                use_dual_model=dual_model,
                dsm_path=dsm_tif,           # DSM for geometry-aware building separation
            )
            elapsed_p3 = time.perf_counter() - t3

            labelmap_png = output_dir / f"{stem}_labelmap.png"
            if labelmap_png.is_file():
                label_map = np.array(PILImage.open(labelmap_png))
            else:
                tif_lbl = output_dir / f"{stem}_terrain_labels.tif"
                if tif_lbl.is_file():
                    import rasterio
                    with rasterio.open(tif_lbl) as src:
                        label_map = src.read(1)

            if label_map is not None:
                total_px = label_map.size
                terrain_stats = {
                    "building_pixels":   int((label_map == 2).sum()),
                    "vegetation_pixels": int((label_map == 1).sum()),
                    "road_pixels":       int((label_map == 3).sum()),
                    "water_pixels":      int((label_map == 4).sum()),
                    "other_pixels":      int((label_map == 0).sum()),
                    "building_pct":      round(100.0 * (label_map == 2).sum() / total_px, 2),
                }
                logger.info(
                    "  Part 3 done in %.1fs  building=%.1f%%  vegetation=%.1f%%  road=%.1f%%",
                    elapsed_p3,
                    terrain_stats["building_pct"],
                    100.0 * terrain_stats["vegetation_pixels"] / total_px,
                    100.0 * terrain_stats["road_pixels"]       / total_px,
                )

            with open(json_path, encoding="utf-8") as fh:
                p3_data = json.load(fh)
            building_instances = [
                d for d in p3_data.get("detections", [])
                if d.get("class") == "building"
            ]
            logger.info("  Building instances detected: %d", len(building_instances))
            part3_status = "ok"

        except Exception as exc:
            import traceback
            logger.warning("Part 3 failed (non-fatal): %s", exc)
            logger.debug(traceback.format_exc())
            part3_status = f"failed: {exc}"

    # ── GAMUS-COMPATIBLE STRUCTURAL SIGNAL ───────────────────────────────
    # No pretrained GAMUS checkpoint is available locally.
    # Instead: compute_height_metrics() from gamus/evaluate.py is applied to
    # the depth-derived nDSM-proxy (height-above-minimum) vs. the building
    # height map.  This uses the GAMUS metric framework (MAE, Spearman-r,
    # delta-accuracy) for geometry-aware structural consistency scoring —
    # the same metrics GAMUS would compute against a ground-truth nDSM,
    # here applied as a self-consistency check within the pipeline.
    gamus_structural: dict = {}
    if label_map is not None and terrain_stats and terrain_stats["building_pixels"] > 100:
        try:
            sys.path.insert(0, str(PART1_DIR))
            from depth_mapping.gamus.evaluate import compute_height_metrics

            # nDSM-proxy: DSM minus terrain floor (non-building P10)
            non_bldg = (label_map != 2) & np.isfinite(dsm_array)
            terrain_floor = float(np.percentile(dsm_array[non_bldg], 10)) \
                if non_bldg.sum() > 0 else float(dsm_array.min())
            ndsm_proxy = np.clip(dsm_array - terrain_floor, 0, None)  # (H,W) above-ground

            # Reference: the semantic building mask as a binary height reference
            # (1.0 inside building, 0.0 outside) — structural, not metric
            building_ref = (label_map == 2).astype(np.float32)

            # Only evaluate over pixels where we have a real signal
            # Use full 2D arrays — compute_height_metrics expects (H,W) for SSIM
            valid = (building_ref > 0) | (ndsm_proxy > 0.5)
            if valid.sum() > 500:
                try:
                    metrics = compute_height_metrics(
                        ndsm_proxy, building_ref, valid_mask=valid, align=True
                    )
                except Exception:
                    # SSIM step may fail if filtered array can't be reshaped;
                    # compute Spearman directly from flattened valid pixels
                    from scipy.stats import spearmanr
                    p_vals = ndsm_proxy[valid].astype(np.float64)
                    r_vals = building_ref[valid].astype(np.float64)
                    try:
                        sr, _ = spearmanr(p_vals, r_vals)
                    except Exception:
                        sr = 0.0
                    metrics = {"spearman_r": float(sr), "delta_1": None,
                               "ssim": None, "n_valid": int(valid.sum())}
                gamus_structural = {
                    "note": (
                        "GAMUS-compatible structural consistency metrics. "
                        "No pretrained GAMUS checkpoint loaded — this is a "
                        "self-consistency check using gamus/evaluate.py metrics "
                        "on depth-derived nDSM proxy vs semantic building mask."
                    ),
                    "inference_type": "gamus_compatible_structural_check",
                    "actual_gamus_checkpoint_loaded": False,
                    "spearman_r":  None if (
                        metrics.get("spearman_r") is None or
                        (isinstance(metrics.get("spearman_r"), float) and
                         not np.isfinite(metrics["spearman_r"]))
                    ) else round(float(metrics["spearman_r"]), 4),
                    "delta_1":     metrics.get("delta_1"),
                    "ssim":        None if (
                        metrics.get("ssim") is None or
                        (isinstance(metrics.get("ssim"), float) and
                         not np.isfinite(metrics["ssim"]))
                    ) else round(float(metrics["ssim"]), 4),
                    "n_valid_px":  metrics.get("n_valid"),
                    "terrain_floor_m": round(terrain_floor, 2),
                }
                _sr = gamus_structural["spearman_r"]
                logger.info(
                    "  GAMUS structural check: spearman_r=%s  n_valid=%d",
                    f"{_sr:.3f}" if _sr is not None else "n/a",
                    metrics.get("n_valid") or 0,
                )
            del ndsm_proxy, building_ref, valid
        except Exception as exc:
            logger.debug("GAMUS structural check skipped: %s", exc)
            gamus_structural = {"note": f"skipped: {exc}",
                                "actual_gamus_checkpoint_loaded": False}
    else:
        gamus_structural = {
            "note": "Skipped — insufficient building pixels for structural check",
            "actual_gamus_checkpoint_loaded": False,
        }

    # ── HEIGHT FUSION ─────────────────────────────────────────────────────
    logger.info("Height fusion — building-specific local ground estimation …")
    from building_heights import (
        estimate_building_heights,
        save_building_height_geotiff,
        save_building_height_preview,
    )
    from integrated_report import build_integrated_report, save_integrated_report

    building_mask = (label_map == 2) if label_map is not None else None

    t_fuse = time.perf_counter()
    if building_mask is not None:
        height_result = estimate_building_heights(
            dsm_array, building_mask,
            instances=building_instances if building_instances else None,
        )
    else:
        height_result = {
            "height_map": np.zeros_like(dsm_array),
            "instance_heights": [],
            "summary": {"building_count": 0, "height_min_m": 0.0,
                        "height_max_m": 0.0, "height_mean_m": 0.0},
        }
    elapsed_fuse = time.perf_counter() - t_fuse

    bsum = height_result["summary"]
    height_map = height_result["height_map"]

    save_building_height_geotiff(height_map, geo_meta, output_dir / f"{stem}_building_heights.tif")
    save_building_height_preview(height_map, output_dir / f"{stem}_heights_preview.png")

    logger.info(
        "  Height fusion %.1fs — count=%d  min=%.1f  max=%.1f  mean=%.1f m",
        elapsed_fuse, bsum["building_count"],
        bsum["height_min_m"], bsum["height_max_m"], bsum["height_mean_m"],
    )

    # ── INTEGRATED REPORT ─────────────────────────────────────────────────
    integrated = build_integrated_report(
        stem=stem,
        calib_diagnostics=calib_diag,
        terrain_stats=terrain_stats,
        building_result=height_result,
        depth_shape=(H, W),
        elapsed={
            "part1_depth_s":  round(elapsed_p1, 2),
            "part2_calib_s":  round(elapsed_p2, 2),
            "part3_segm_s":   round(elapsed_p3, 2),
            "fusion_s":       round(elapsed_fuse, 2),
        },
    )
    # Add GAMUS structural signal to report
    integrated["gamus_structural_check"] = gamus_structural
    integrated["part3_status"] = part3_status

    report_path = save_integrated_report(
        integrated, output_dir / f"{stem}_integrated_report.json"
    )

    # ── VALIDATION SUMMARY ────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    total_px = H * W
    nonzero_height_px = int((height_map > 0).sum())

    print()
    print("=" * 65)
    print(f"  PIPELINE COMPLETE — {stem}")
    print("=" * 65)
    print(f"  Image        : {input_path.name}  ({W}×{H} px)")
    print(f"  Part 1       : {'reused' if elapsed_p1 == 0 else f'{elapsed_p1:.1f}s'}")
    print(f"  Part 2       : a={a:.6f}  b={b:.4f} m  quality={quality}  ({elapsed_p2:.1f}s)")
    print(f"  DSM range    : [{dsm_array.min():.1f}, {dsm_array.max():.1f}] m")
    print(f"  Part 3       : {part3_status}  ({elapsed_p3:.1f}s)")
    if terrain_stats:
        print(f"  Building px  : {terrain_stats['building_pixels']:,}  ({terrain_stats['building_pct']:.1f}%)")
        print(f"  Instances    : {len(building_instances)}")
    print(f"  Height fusion: count={bsum['building_count']}  "
          f"min={bsum['height_min_m']:.1f}  max={bsum['height_max_m']:.1f}  "
          f"mean={bsum['height_mean_m']:.1f} m  ({elapsed_fuse:.1f}s)")
    print(f"  Nonzero ht px: {nonzero_height_px:,}  ({100*nonzero_height_px/total_px:.2f}%)")
    if gamus_structural.get("spearman_r") is not None:
        sr = gamus_structural.get("spearman_r")
        print(f"  GAMUS struct : spearman_r={sr:.3f}  [structural consistency check only]")
    print(f"  CRS          : {geo_meta['crs']}")
    print(f"  Total time   : {total_elapsed:.1f}s")
    print(f"  Report       : {report_path}")
    print("=" * 65)
    print()

    return integrated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GeoMonoDSM-3D — Single-image pipeline runner (Part 1→2→3→Heights)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  required=True,
                        help="Path to input GeoTIFF (e.g. Input/sfo3.tif)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for all outputs")
    parser.add_argument("--depth",  default=None,
                        help="Path to existing depth .npy (skips Part 1 if provided)")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CKPT),
                        help=f"Depth Anything V2 checkpoint dir (default: {DEFAULT_CKPT})")
    parser.add_argument("--skip-part3", action="store_true",
                        help="Skip terrain segmentation (Part 1+2 only)")
    parser.add_argument("--dual-model", action="store_true",
                        help="Enable ADE20K dual-model fusion in Part 3")
    parser.add_argument("--no-gamus", action="store_true", default=True,
                        help="Skip GAMUS structural check (default: True — faster; set to False to enable)")
    args = parser.parse_args()

    run(
        input_path     = Path(args.input),
        output_dir     = Path(args.output_dir),
        depth_npy      = Path(args.depth) if args.depth else None,
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        skip_part3     = args.skip_part3,
        dual_model     = args.dual_model,
        skip_gamus     = args.no_gamus,
    )


if __name__ == "__main__":
    main()
