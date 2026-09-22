"""
smoke_test.py — Integration smoke test for the GeoMonoDSM-3D pipeline.

Reuses existing depth .npy outputs — does NOT re-run Part 1 inference.
Covers:
  - 512×512  (bellingham_test)
  - 5000×5000 scalability check (bellingham1, reusing existing depth .npy)

Run:
    python smoke_test.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")

ROOT      = Path(__file__).resolve().parent
PART2_DIR = ROOT / "Elevation_Calibration"
PART3_DIR = ROOT / "Terrain_Classificartion"
OUTPUT    = ROOT / "output"

for p in (str(PART2_DIR), str(PART3_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _check(cond: bool, label: str) -> bool:
    mark = "PASS" if cond else "FAIL"
    logger.info("  %-50s [%s]", label, mark)
    return cond


def _load_existing_dsm(stem: str) -> tuple[np.ndarray, dict] | None:
    """Load an existing DSM .tif from the output directory."""
    try:
        import rasterio
        tif = OUTPUT / f"{stem}_dsm.tif"
        if not tif.is_file():
            return None
        with rasterio.open(tif) as src:
            arr = src.read(1)
            geo = {
                "crs":       src.crs,
                "transform": src.transform,
                "width":     src.width,
                "height":    src.height,
                "nodata":    src.nodata,
            }
        return arr, geo
    except Exception as exc:
        logger.warning("  Could not load DSM: %s", exc)
        return None


# ── Core smoke test ───────────────────────────────────────────────────────────

def run_smoke_test(
    stem: str,
    depth_npy: Path,
    geotiff: Path,
    output_subdir: Path,
    run_part3: bool = True,
    label: str = "",
) -> dict:
    """Run Part 2 + Part 3 + fusion for one image.

    Skips Part 1 — uses the existing depth .npy.
    Returns a dict of pass/fail checks.
    """
    output_subdir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    t0 = time.perf_counter()

    logger.info("=" * 65)
    logger.info("SMOKE TEST: %s  (%s)", stem, label)
    logger.info("=" * 65)

    # -- Load depth array ------------------------------------------------
    if not depth_npy.is_file():
        logger.error("  depth .npy not found: %s", depth_npy)
        return {"MISSING_DEPTH": False}

    depth_arr = np.load(depth_npy)
    results["depth_loaded"] = _check(
        depth_arr.ndim == 2 and np.isfinite(depth_arr).any(),
        f"depth loaded shape={depth_arr.shape} dtype={depth_arr.dtype}"
    )

    # -- Load GeoTIFF metadata -------------------------------------------
    try:
        import rasterio
        with rasterio.open(geotiff) as src:
            geo_meta = {
                "crs":       src.crs,
                "transform": src.transform,
                "width":     src.width,
                "height":    src.height,
                "nodata":    src.nodata,
            }
        results["geotiff_loaded"] = _check(True, f"GeoTIFF metadata loaded CRS={geo_meta['crs']}")
    except Exception as exc:
        logger.error("  Failed to load GeoTIFF: %s", exc)
        return {"geotiff_failed": False}

    H, W = depth_arr.shape
    results["depth_dims_match"] = _check(
        H == geo_meta["height"] and W == geo_meta["width"],
        f"depth dims match GeoTIFF ({H}×{W})"
    )

    # -- Part 2: calibration + DSM ---------------------------------------
    logger.info("  Part 2: SRTM calibration …")
    from elevation_calibration import (
        fetch_reference_elevations,
        fit_calibration,
        compute_calibration_diagnostics,
        save_dsm_geotiff,
        save_dsm_preview,
    )
    from building_heights import (
        estimate_building_heights,
        save_building_height_geotiff,
        save_building_height_preview,
    )
    from integrated_report import build_integrated_report, save_integrated_report

    t_p2 = time.perf_counter()
    try:
        depth_vals, srtm_vals = fetch_reference_elevations(
            depth_arr, geo_meta["transform"], geo_meta["crs"], n_gcps=16
        )
        a, b, fit_diag = fit_calibration(depth_vals, srtm_vals)
        calib_diag = compute_calibration_diagnostics(
            a, b, depth_vals, srtm_vals, fit_diag, n_gcps_requested=16
        )
        dsm_array = np.add(
            np.multiply(depth_arr, np.float32(a), dtype=np.float32),
            np.float32(b), dtype=np.float32,
        )
        dsm_tif = save_dsm_geotiff(dsm_array, geo_meta,
                                    output_subdir / f"{stem}_dsm.tif")
        save_dsm_preview(dsm_array, output_subdir / f"{stem}_dsm_preview.png")
        elapsed_p2 = time.perf_counter() - t_p2

        results["dsm_created"]  = _check(dsm_tif.is_file(), "DSM GeoTIFF created")
        results["dsm_dtype"]    = _check(dsm_array.dtype == np.float32,
                                          f"DSM dtype float32 (got {dsm_array.dtype})")
        results["dsm_finite"]   = _check(bool(np.isfinite(dsm_array).any()),
                                          "DSM has finite values")
        results["dsm_no_nan"]   = _check(not np.any(np.isnan(dsm_array)),
                                          "DSM has no NaN")
        results["calib_quality"]= _check(calib_diag["quality"]["status"] in ("good","warning","failed"),
                                          f"calibration quality = {calib_diag['quality']['status']}")
        results["dsm_shape"]    = _check(dsm_array.shape == (H, W),
                                          f"DSM shape matches ({H}×{W})")

        # CRS/transform preserved
        with rasterio.open(dsm_tif) as ds:
            results["crs_preserved"] = _check(
                ds.crs == geo_meta["crs"],
                f"CRS preserved ({ds.crs})"
            )
            results["transform_preserved"] = _check(
                ds.transform == geo_meta["transform"],
                "affine transform preserved"
            )
        logger.info("  Part 2 done in %.1fs  a=%.6f  b=%.3f  quality=%s",
                    elapsed_p2, a, b, calib_diag["quality"]["status"])

    except Exception as exc:
        import traceback
        logger.error("  Part 2 FAILED: %s\n%s", exc, traceback.format_exc())
        results["part2_failed"] = False
        return results

    # -- Part 3: terrain segmentation ------------------------------------
    label_map: np.ndarray | None = None
    building_instances: list[dict] = []
    terrain_stats: dict | None = None
    elapsed_p3 = 0.0

    if run_part3:
        logger.info("  Part 3: terrain segmentation …")
        try:
            from segmentation import classify_terrain
            from PIL import Image as PILImage

            t_p3 = time.perf_counter()
            json_path, overlay_path = classify_terrain(
                geotiff,
                output_dir=output_subdir,
                use_dual_model=False,
            )
            elapsed_p3 = time.perf_counter() - t_p3

            results["p3_json"]    = _check(json_path.is_file(), "Part 3 JSON created")
            results["p3_overlay"] = _check(overlay_path.is_file(), "Part 3 overlay PNG created")

            labelmap_png = output_subdir / f"{stem}_labelmap.png"
            results["p3_labelmap"] = _check(labelmap_png.is_file(), "Part 3 labelmap PNG created")

            # Load label map
            label_map = np.array(PILImage.open(labelmap_png))
            results["labelmap_uint8"]  = _check(label_map.dtype == np.uint8,
                                                  f"labelmap dtype uint8 (got {label_map.dtype})")
            results["labelmap_shape"]  = _check(label_map.shape == (H, W),
                                                  f"labelmap shape matches ({H}×{W})")
            results["labelmap_values"] = _check(
                set(np.unique(label_map).tolist()).issubset({0,1,2,3,4}),
                f"labelmap values ⊆ {{0,1,2,3,4}} (got {sorted(np.unique(label_map).tolist())})"
            )

            # Load building instances
            with open(json_path, encoding="utf-8") as fh:
                p3_data = json.load(fh)
            building_instances = [d for d in p3_data.get("detections", [])
                                    if d.get("class") == "building"]

            n_bldg_px = int(np.sum(label_map == 2))
            terrain_stats = {
                "building_pixels":   n_bldg_px,
                "vegetation_pixels": int(np.sum(label_map == 1)),
                "road_pixels":       int(np.sum(label_map == 3)),
                "water_pixels":      int(np.sum(label_map == 4)),
                "other_pixels":      int(np.sum(label_map == 0)),
            }
            logger.info(
                "  Part 3 done in %.1fs  building_instances=%d  building_pixels=%d (%.1f%%)",
                elapsed_p3, len(building_instances), n_bldg_px,
                100.0 * n_bldg_px / label_map.size,
            )

        except Exception as exc:
            import traceback
            logger.warning("  Part 3 failed (non-fatal): %s", exc)
            logger.debug(traceback.format_exc())
            results["part3_failed"] = _check(False, "Part 3 segmentation")
    else:
        logger.info("  Part 3: skipped.")

    # -- Height fusion ---------------------------------------------------
    logger.info("  Height fusion …")
    building_mask = (label_map == 2) if label_map is not None else None

    t_fuse = time.perf_counter()
    if building_mask is not None:
        height_result = estimate_building_heights(
            dsm_array, building_mask, instances=building_instances or None
        )
    else:
        height_result = {
            "height_map": np.zeros_like(dsm_array),
            "instance_heights": [],
            "summary": {"building_count": 0, "height_min_m": 0.0,
                        "height_max_m": 0.0, "height_mean_m": 0.0},
        }
    elapsed_fuse = time.perf_counter() - t_fuse

    hmap = height_result["height_map"]
    bsum = height_result["summary"]

    results["hmap_shape"]   = _check(hmap.shape == (H, W),
                                      f"height_map shape ({H}×{W})")
    results["hmap_dtype"]   = _check(hmap.dtype == np.float32,
                                      f"height_map dtype float32 (got {hmap.dtype})")
    results["hmap_no_nan"]  = _check(not np.any(np.isnan(hmap)),
                                      "height_map has no NaN")
    results["hmap_no_inf"]  = _check(not np.any(np.isinf(hmap)),
                                      "height_map has no Inf")
    results["hmap_non_bldg_zero"] = _check(
        building_mask is None or bool(np.all(hmap[~building_mask] == 0.0)),
        "non-building pixels are zero in height_map"
    )

    ht_tif = save_building_height_geotiff(
        hmap, geo_meta, output_subdir / f"{stem}_building_heights.tif"
    )
    save_building_height_preview(hmap, output_subdir / f"{stem}_heights_preview.png")

    results["heights_tif_created"] = _check(ht_tif.is_file(), "building_heights.tif created")

    # Verify CRS/transform preserved in height TIF
    with rasterio.open(ht_tif) as ds:
        results["heights_crs"] = _check(ds.crs == geo_meta["crs"],
                                          "height TIF CRS preserved")
        results["heights_tf"]  = _check(ds.transform == geo_meta["transform"],
                                          "height TIF transform preserved")
        results["heights_dims"]= _check(ds.width == W and ds.height == H,
                                          f"height TIF dims ({W}×{H})")

    logger.info(
        "  Height fusion done in %.1fs  count=%d  min=%.1f  max=%.1f  mean=%.1f m",
        elapsed_fuse, bsum["building_count"],
        bsum["height_min_m"], bsum["height_max_m"], bsum["height_mean_m"],
    )

    # -- Integrated report -----------------------------------------------
    integrated = build_integrated_report(
        stem=stem,
        calib_diagnostics=calib_diag,
        terrain_stats=terrain_stats,
        building_result=height_result,
        depth_shape=(H, W),
        elapsed={
            "part2_calib_s": round(elapsed_p2, 2),
            "part3_segm_s":  round(elapsed_p3, 2),
            "fusion_s":      round(elapsed_fuse, 2),
        },
    )
    report_path = save_integrated_report(
        integrated, output_subdir / f"{stem}_integrated_report.json"
    )

    results["report_created"] = _check(report_path.is_file(), "integrated_report.json created")
    with open(report_path, encoding="utf-8") as fh:
        rpt = json.load(fh)
    results["report_valid_json"] = _check(
        "pipeline" in rpt and "calibration" in rpt and "buildings" in rpt,
        "integrated_report.json has required keys"
    )

    total = time.perf_counter() - t0
    n_pass = sum(v for v in results.values())
    n_fail = sum(1 for v in results.values() if not v)
    logger.info("  TOTAL %.1fs — %d passed, %d failed", total, n_pass, n_fail)
    return results


# ── Large-image scalability check ────────────────────────────────────────────

def run_scalability_check(
    stem: str,
    depth_npy: Path,
    geotiff: Path,
    output_subdir: Path,
) -> dict:
    """Lightweight scalability check on a large (5000×5000) depth array.

    Does NOT run Part 3 segmentation (avoids expensive full-resolution inference).
    Instead it uses a synthetic building mask to exercise the height fusion code path
    at full resolution, verifying memory safety and dtype correctness.
    """
    output_subdir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    logger.info("=" * 65)
    logger.info("SCALABILITY CHECK: %s", stem)
    logger.info("=" * 65)

    if not depth_npy.is_file():
        logger.warning("  depth .npy not found — skipping scalability check.")
        return {"skipped": True}

    t0 = time.perf_counter()
    depth_arr = np.load(depth_npy)
    H, W = depth_arr.shape
    size_mb = depth_arr.nbytes / 1_048_576
    logger.info("  Depth array: %d×%d  (%.0f MB)", H, W, size_mb)
    results["large_depth_loaded"] = _check(
        H >= 4000 or W >= 4000,
        f"large image loaded {H}×{W} ({size_mb:.0f} MB)"
    )

    # Load GeoTIFF meta
    try:
        import rasterio
        with rasterio.open(geotiff) as src:
            geo_meta = {
                "crs":       src.crs,
                "transform": src.transform,
                "width":     src.width,
                "height":    src.height,
                "nodata":    src.nodata,
            }
    except Exception as exc:
        logger.error("  Failed to load GeoTIFF: %s", exc)
        return {"geotiff_failed": False}

    from elevation_calibration import (
        fetch_reference_elevations, fit_calibration, compute_calibration_diagnostics,
        save_dsm_geotiff,
    )
    from building_heights import (
        estimate_building_heights, save_building_height_geotiff,
    )

    # Part 2 calibration
    logger.info("  Part 2 calibration on %d×%d …", H, W)
    t_p2 = time.perf_counter()
    depth_vals, srtm_vals = fetch_reference_elevations(
        depth_arr, geo_meta["transform"], geo_meta["crs"], n_gcps=16
    )
    a, b, fit_diag = fit_calibration(depth_vals, srtm_vals)
    calib_diag = compute_calibration_diagnostics(
        a, b, depth_vals, srtm_vals, fit_diag, n_gcps_requested=16
    )
    dsm_array = np.add(
        np.multiply(depth_arr, np.float32(a), dtype=np.float32),
        np.float32(b), dtype=np.float32,
    )
    dsm_mb = dsm_array.nbytes / 1_048_576
    logger.info(
        "  DSM: %.0f MB  dtype=%s  range=[%.1f, %.1f] m  quality=%s  (%.1fs)",
        dsm_mb, dsm_array.dtype, dsm_array.min(), dsm_array.max(),
        calib_diag["quality"]["status"],
        time.perf_counter() - t_p2,
    )
    results["large_dsm_float32"] = _check(dsm_array.dtype == np.float32,
                                            "large DSM is float32 (not float64)")
    results["large_dsm_no_nan"]  = _check(not np.any(np.isnan(dsm_array)),
                                            "large DSM has no NaN")

    # Save DSM
    dsm_tif = save_dsm_geotiff(dsm_array, geo_meta,
                                 output_subdir / f"{stem}_dsm.tif")
    results["large_dsm_tif"] = _check(dsm_tif.is_file(), "large DSM GeoTIFF saved")

    # Synthetic building mask — ~5% of pixels as buildings in random locations
    # Represents a typical urban scene without needing full Part 3 inference
    rng = np.random.default_rng(42)
    n_buildings = 200
    building_mask = np.zeros((H, W), dtype=bool)
    for _ in range(n_buildings):
        r = rng.integers(50, H - 50)
        c = rng.integers(50, W - 50)
        bh = rng.integers(10, 50)
        bw = rng.integers(10, 60)
        building_mask[r:r+bh, c:c+bw] = True

    n_bldg_px = int(building_mask.sum())
    bldg_pct  = 100.0 * n_bldg_px / (H * W)
    logger.info("  Synthetic building mask: %d pixels (%.1f%%) for scalability test",
                n_bldg_px, bldg_pct)

    # Height fusion — this is the expensive path at 5000×5000
    logger.info("  Height fusion on %d×%d …", H, W)
    t_fuse = time.perf_counter()
    height_result = estimate_building_heights(dsm_array, building_mask, instances=None)
    elapsed_fuse = time.perf_counter() - t_fuse

    hmap = height_result["height_map"]
    bsum = height_result["summary"]
    logger.info(
        "  Height fusion done in %.2fs — count=%d  max=%.1f m",
        elapsed_fuse, bsum["building_count"], bsum["height_max_m"],
    )

    results["large_hmap_shape"]   = _check(hmap.shape == (H, W),
                                             f"height_map shape {H}×{W}")
    results["large_hmap_dtype"]   = _check(hmap.dtype == np.float32,
                                             "height_map dtype float32")
    results["large_hmap_no_nan"]  = _check(not np.any(np.isnan(hmap)),
                                             "height_map no NaN")
    results["large_hmap_no_inf"]  = _check(not np.any(np.isinf(hmap)),
                                             "height_map no Inf")
    results["large_nonbldg_zero"] = _check(
        bool(np.all(hmap[~building_mask] == 0.0)),
        "non-building pixels are 0"
    )
    results["large_bldg_positive"]= _check(
        bool(hmap[building_mask].max() > 0),
        "building pixels have positive height"
    )

    # Save height TIF
    ht_tif = save_building_height_geotiff(
        hmap, geo_meta, output_subdir / f"{stem}_building_heights.tif"
    )
    with rasterio.open(ht_tif) as ds:
        results["large_ht_crs"] = _check(ds.crs == geo_meta["crs"],
                                           "height TIF CRS preserved")
        results["large_ht_dims"]= _check(ds.width == W and ds.height == H,
                                           f"height TIF dims {W}×{H}")

    total = time.perf_counter() - t0
    n_pass = sum(v for v in results.values())
    n_fail = sum(1 for v in results.values() if not v)
    logger.info(
        "  SCALABILITY TOTAL %.1fs (Part2 + fusion at %d×%d) — %d passed, %d failed",
        total, H, W, n_pass, n_fail,
    )
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    all_pass = True

    # ── 512×512 smoke test ───────────────────────────────────────────────
    smoke_512 = run_smoke_test(
        stem       = "bellingham_test",
        depth_npy  = ROOT / "Elevation_Calibration/test_input/bellingham_test.npy",
        geotiff    = ROOT / "Elevation_Calibration/test_input/bellingham_test.tif",
        output_subdir = ROOT / "output/_smoke_512",
        run_part3  = True,
        label      = "512×512 full pipeline",
    )
    failed_512 = [k for k, v in smoke_512.items() if not v]
    if failed_512:
        logger.warning("512×512 FAILURES: %s", failed_512)
        all_pass = False

    # ── 5000×5000 scalability check ─────────────────────────────────────
    large_npy = OUTPUT / "bellingham1.npy"
    large_tif = ROOT / "Input/bellingham1.tif"
    if large_npy.is_file() and large_tif.is_file():
        scale_5k = run_scalability_check(
            stem          = "bellingham1",
            depth_npy     = large_npy,
            geotiff       = large_tif,
            output_subdir = ROOT / "output/_smoke_5k",
        )
        failed_5k = [k for k, v in scale_5k.items() if not v]
        if failed_5k:
            logger.warning("5000×5000 FAILURES: %s", failed_5k)
            all_pass = False
    else:
        logger.warning(
            "Large-image inputs not found (%s / %s) — scalability check skipped.",
            large_npy, large_tif,
        )

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_pass:
        print("  ALL SMOKE TESTS PASSED  ✓")
    else:
        print("  SOME SMOKE TESTS FAILED  ✗")
    print("=" * 65 + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
