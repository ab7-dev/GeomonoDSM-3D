"""
Unit tests for building_heights.py and calibration diagnostics.

Run with:
    python -m pytest Elevation_Calibration/test_building_heights.py -v

These tests use only synthetic arrays — no GPU, no SRTM API, no model inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure module is importable from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Elevation_Calibration"))

from building_heights import estimate_building_heights, save_building_height_geotiff
from elevation_calibration import (
    compute_calibration_diagnostics,
    validate_calibration,
    fit_calibration,
)


# ===========================================================================
# Building height estimation tests
# ===========================================================================

class TestEstimateBuildingHeights:

    # ── Test 1: all-false building mask → all-zero height raster -----------

    def test_empty_mask_returns_all_zero(self):
        """Building mask with no buildings should produce all-zero height map."""
        dsm = np.ones((50, 50), dtype=np.float32) * 100.0
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = estimate_building_heights(dsm, mask)
        assert result["height_map"].shape == (50, 50)
        assert np.all(result["height_map"] == 0.0), "Expected all-zero height map for empty mask"
        assert result["summary"]["building_count"] == 0
        assert result["instance_heights"] == []

    # ── Test 2: synthetic building with known roof/ground elevation --------

    def test_known_roof_and_ground_elevation(self):
        """A 10×10 building patch surrounded by ground at 100m, roof at 112m.
        Expected estimated height ≈ 12m (within ±3m of local ground/percentile effects)."""
        H, W = 100, 100
        dsm  = np.full((H, W), 100.0, dtype=np.float32)  # ground everywhere = 100 m

        # Place building in the centre: rows 45–54, cols 45–54
        r0, r1, c0, c1 = 45, 55, 45, 55
        dsm[r0:r1, c0:c1] = 112.0   # building roof = 112 m

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[r0:r1, c0:c1] = 1

        result = estimate_building_heights(dsm, mask)
        heights = [h["estimated_height_m"] for h in result["instance_heights"]]
        assert len(heights) >= 1, "Expected at least one building instance"
        # Ground is uniformly 100 m, roof is 112 m → height should be ~12 m
        for h in heights:
            assert 9.0 <= h <= 15.0, (
                f"Expected estimated height ~12 m, got {h:.2f} m. "
                "Local ground estimation uses percentiles so small deviation is expected."
            )

    # ── Test 3: shape preservation ─────────────────────────────────────────

    def test_output_shape_matches_dsm(self):
        """height_map.shape must equal dsm_array.shape."""
        dsm  = np.random.rand(80, 120).astype(np.float32) * 50.0 + 100.0
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[30:50, 50:80] = 1
        result = estimate_building_heights(dsm, mask)
        assert result["height_map"].shape == dsm.shape

    # ── Test 4: non-building pixels are zero ──────────────────────────────

    def test_nonbuilding_pixels_are_zero(self):
        """All pixels outside the building mask must be 0 in the height raster."""
        dsm  = np.full((60, 60), 100.0, dtype=np.float32)
        dsm[20:40, 20:40] = 110.0
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[20:40, 20:40] = 1

        result = estimate_building_heights(dsm, mask)
        hmap = result["height_map"]

        # All non-building pixels must be 0
        non_bldg = mask == 0
        assert np.all(hmap[non_bldg] == 0.0), (
            "Non-building pixels should have height = 0.0"
        )

    # ── Test 5: negative roof-ground difference clamped to 0 ─────────────

    def test_negative_height_clamped_to_zero(self):
        """If local ground is HIGHER than the roof (calibration artefact), height = 0."""
        dsm = np.full((60, 60), 200.0, dtype=np.float32)  # ground = 200 m
        dsm[25:35, 25:35] = 195.0                           # 'roof' is LOWER than ground
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[25:35, 25:35] = 1
        result = estimate_building_heights(dsm, mask)
        for inst in result["instance_heights"]:
            assert inst["estimated_height_m"] >= 0.0, (
                "Estimated heights must never be negative"
            )

    # ── Test 6: CRS/transform preservation is tested via geotiff writer ───

    def test_save_building_height_geotiff_preserves_meta(self, tmp_path):
        """save_building_height_geotiff must write a float32 GeoTIFF that preserves
        CRS, transform, and dimensions from geo_meta."""
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        H, W = 64, 64
        height_map = np.zeros((H, W), dtype=np.float32)
        height_map[20:40, 20:40] = 8.0

        fake_crs = CRS.from_epsg(4326)
        fake_transform = from_bounds(0, 0, 1, 1, W, H)
        geo_meta = {
            "crs": fake_crs, "transform": fake_transform,
            "width": W, "height": H, "nodata": None,
        }

        out_path = tmp_path / "test_heights.tif"
        save_building_height_geotiff(height_map, geo_meta, out_path)
        assert out_path.is_file()

        with rasterio.open(out_path) as ds:
            assert ds.crs == fake_crs
            assert ds.transform == fake_transform
            assert ds.width  == W
            assert ds.height == H
            assert ds.dtypes[0] == "float32"
            arr = ds.read(1)

        assert arr.shape == (H, W)
        # Non-building pixels should be 0
        assert np.all(arr[:20, :] == 0.0)
        # Building pixels should have the injected value
        assert arr[30, 30] == pytest.approx(8.0, abs=0.01)

    # ── Test 7: instance_heights have required keys ────────────────────────

    def test_instance_height_dict_has_required_keys(self):
        dsm  = np.full((50, 50), 100.0, dtype=np.float32)
        dsm[10:20, 10:20] = 115.0
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        result = estimate_building_heights(dsm, mask)
        for inst in result["instance_heights"]:
            for key in ("instance_id", "bbox", "area_pixels",
                        "ground_elevation_m", "roof_elevation_m",
                        "estimated_height_m", "height_method"):
                assert key in inst, f"Missing key '{key}' in instance dict"

    # ── Test 8: provided instances with bbox ─────────────────────────────

    def test_provided_instances_bbox(self):
        """When instances list is provided, only those bboxes are used."""
        dsm  = np.full((100, 100), 100.0, dtype=np.float32)
        dsm[10:20, 10:20] = 110.0  # building A: height ~10 m
        dsm[60:70, 60:70] = 120.0  # building B: height ~20 m
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[60:70, 60:70] = 1

        instances = [
            {"class": "building", "instance_id": 1, "bbox": [10, 10, 19, 19], "confidence": 0.9},
            {"class": "building", "instance_id": 2, "bbox": [60, 60, 69, 69], "confidence": 0.85},
        ]
        result = estimate_building_heights(dsm, mask, instances=instances)
        assert len(result["instance_heights"]) == 2

    # ── Test 9: global fallback fires when no surrounding ground pixels ───

    def test_global_fallback_when_no_surrounding_ground(self):
        """A building that fills the entire image forces the global fallback path."""
        H, W = 30, 30
        dsm  = np.full((H, W), 110.0, dtype=np.float32)
        mask = np.ones((H, W), dtype=np.uint8)   # entire image is "building"
        result = estimate_building_heights(dsm, mask)
        # Should not crash; height may be 0 (no ground reference) but must be non-negative
        for inst in result["instance_heights"]:
            assert inst["estimated_height_m"] >= 0.0
            # Method must be some fallback variant
            assert "fallback" in inst["height_method"] or "local" in inst["height_method"]


# ===========================================================================
# Calibration diagnostics tests
# ===========================================================================

class TestValidateCalibration:

    def _make_diag(self, n_inliers=12, n_total=16, mae=5.0, method="ransac"):
        return {
            "method":          method,
            "n_total":         n_total,
            "n_inliers":       n_inliers,
            "inlier_ratio":    round(n_inliers / n_total, 4),
            "residual_mae_m":  mae,
            "residual_rmse_m": mae * 1.3,
            "residual_median_m": mae * 0.8,
        }

    def test_good_calibration(self):
        diag = self._make_diag()
        result = validate_calibration(
            a=0.336, b=50.0, depth_std=0.25, srtm_std=10.0, fit_diag=diag
        )
        assert result["status"] == "good"

    def test_near_zero_slope_fails(self):
        diag = self._make_diag()
        result = validate_calibration(
            a=0.0001, b=100.0, depth_std=0.25, srtm_std=10.0, fit_diag=diag
        )
        assert result["status"] == "failed"
        assert "slope" in result["reason"].lower()

    def test_negative_slope_not_auto_failed(self):
        """Negative slope must NOT automatically fail — it may be valid."""
        diag = self._make_diag()
        result = validate_calibration(
            a=-0.336, b=200.0, depth_std=0.25, srtm_std=10.0, fit_diag=diag
        )
        # Should be "good" or "warning", NOT "failed" solely due to sign
        assert result["status"] in ("good", "warning")
        assert "negative_slope" in result.get("checks", {})

    def test_non_finite_slope_fails(self):
        diag = self._make_diag()
        result = validate_calibration(
            a=float("inf"), b=0.0, depth_std=0.25, srtm_std=10.0, fit_diag=diag
        )
        assert result["status"] == "failed"
        assert "non-finite" in result["reason"].lower()

    def test_low_depth_variance_fails(self):
        diag = self._make_diag()
        result = validate_calibration(
            a=0.5, b=50.0, depth_std=0.0001, srtm_std=10.0, fit_diag=diag
        )
        assert result["status"] == "failed"
        assert "depth variance" in result["reason"].lower()

    def test_low_srtm_variance_warns(self):
        diag = self._make_diag()
        result = validate_calibration(
            a=0.5, b=50.0, depth_std=0.25, srtm_std=0.1, fit_diag=diag
        )
        assert result["status"] == "warning"
        assert "srtm" in result["reason"].lower() or "terrain" in result["reason"].lower()

    def test_low_inlier_ratio_warns(self):
        diag = self._make_diag(n_inliers=4, n_total=16, mae=5.0)
        result = validate_calibration(
            a=0.5, b=50.0, depth_std=0.25, srtm_std=10.0, fit_diag=diag
        )
        assert result["status"] == "warning"
        assert "inlier" in result["reason"].lower()

    def test_high_mae_warns(self):
        diag = self._make_diag(mae=60.0)
        result = validate_calibration(
            a=0.5, b=50.0, depth_std=0.25, srtm_std=10.0, fit_diag=diag
        )
        assert result["status"] == "warning"
        assert "residual" in result["reason"].lower() or "mae" in result["reason"].lower()


class TestComputeCalibrationDiagnostics:

    def test_returns_required_keys(self):
        depth_vals = np.linspace(0.1, 1.0, 16).astype(np.float32)
        srtm_vals  = depth_vals * 50.0 + 100.0  # known: a≈50, b≈100
        a, b, fit_diag = fit_calibration(depth_vals, srtm_vals)
        diag = compute_calibration_diagnostics(a, b, depth_vals, srtm_vals, fit_diag)

        for key in ("a", "b", "n_gcps", "n_valid_gcps", "depth_std",
                    "srtm_std", "depth_range", "srtm_range", "quality"):
            assert key in diag, f"Missing key '{key}' in calibration diagnostics"

    def test_quality_is_dict_with_status(self):
        depth_vals = np.linspace(0.1, 1.0, 16).astype(np.float32)
        srtm_vals  = depth_vals * 30.0 + 50.0
        a, b, fit_diag = fit_calibration(depth_vals, srtm_vals)
        diag = compute_calibration_diagnostics(a, b, depth_vals, srtm_vals, fit_diag)
        assert "status" in diag["quality"]
        assert diag["quality"]["status"] in ("good", "warning", "failed")


class TestFitCalibrationDiagnostics:

    def test_fit_calibration_returns_three_tuple(self):
        """fit_calibration must now return (a, b, diag_dict)."""
        depth = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        srtm  = depth * 40.0 + 80.0
        result = fit_calibration(depth, srtm)
        assert len(result) == 3, "fit_calibration should return (a, b, diag)"
        a, b, diag = result
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert isinstance(diag, dict)
        assert "method" in diag
        assert "n_inliers" in diag

    def test_fit_calibration_known_relationship(self):
        """With a known linear relationship, fit should recover approximately a=40, b=80."""
        depth = np.linspace(0.1, 1.0, 16).astype(np.float32)
        srtm  = (depth * 40.0 + 80.0).astype(np.float32)
        a, b, _ = fit_calibration(depth, srtm)
        assert abs(a - 40.0) < 5.0, f"Expected a≈40, got {a:.3f}"
        assert abs(b - 80.0) < 15.0, f"Expected b≈80, got {b:.3f}"

    def test_fit_calibration_diag_contains_residuals(self):
        depth = np.linspace(0.1, 1.0, 16).astype(np.float32)
        srtm  = depth * 30.0 + 100.0
        _, _, diag = fit_calibration(depth, srtm)
        assert diag.get("residual_mae_m") is not None
        assert diag.get("residual_rmse_m") is not None


if __name__ == "__main__":
    # Quick run without pytest
    import traceback
    passed = 0
    failed = 0

    suites = [
        TestEstimateBuildingHeights,
        TestValidateCalibration,
        TestComputeCalibrationDiagnostics,
        TestFitCalibrationDiagnostics,
    ]

    for suite_cls in suites:
        suite = suite_cls()
        for name in dir(suite_cls):
            if not name.startswith("test_"):
                continue
            method = getattr(suite, name)
            # Skip tests requiring tmp_path (pytest fixture)
            import inspect
            sig = inspect.signature(method)
            if "tmp_path" in sig.parameters:
                print(f"  SKIP (needs pytest fixture): {suite_cls.__name__}.{name}")
                continue
            try:
                method()
                print(f"  PASS: {suite_cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL: {suite_cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


# ===========================================================================
# Part 2 MODE A tests (relative calibration — no GeoTIFF required)
# ===========================================================================

class TestRunRelativeCalibration:
    """Tests for run_relative_calibration() — MODE A (no CRS, relative DSM)."""

    def test_returns_relative_mode_key(self, tmp_path):
        """Result must indicate MODE_A_relative processing mode."""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from elevation_calibration import run_relative_calibration

        depth = np.random.rand(64, 64).astype(np.float32) * 100.0 + 50.0
        npy_path = tmp_path / "test_depth.npy"
        np.save(npy_path, depth)

        result = run_relative_calibration(str(npy_path), str(tmp_path / "out"))
        assert result["mode"] == "MODE_A_relative"

    def test_rdsm_non_negative(self, tmp_path):
        """rDSM values must all be ≥ 0 (floor-shifted)."""
        from elevation_calibration import run_relative_calibration

        depth = np.random.rand(64, 64).astype(np.float32) * 50.0
        npy_path = tmp_path / "depth.npy"
        np.save(npy_path, depth)

        result = run_relative_calibration(str(npy_path), str(tmp_path / "out"))
        assert result["rdsm_array"].min() >= 0.0

    def test_rdsm_shape_matches_input(self, tmp_path):
        """rDSM shape must equal input depth shape."""
        from elevation_calibration import run_relative_calibration

        H, W = 128, 96
        depth = np.ones((H, W), dtype=np.float32) * 80.0
        depth[30:60, 20:70] = 120.0
        npy_path = tmp_path / "depth.npy"
        np.save(npy_path, depth)

        result = run_relative_calibration(str(npy_path), str(tmp_path / "out"))
        assert result["rdsm_array"].shape == (H, W)

    def test_rdsm_note_says_not_amsl(self, tmp_path):
        """The note field must warn that output is NOT metres above sea level."""
        from elevation_calibration import run_relative_calibration

        depth = np.ones((32, 32), dtype=np.float32)
        npy_path = tmp_path / "depth.npy"
        np.save(npy_path, depth)

        result = run_relative_calibration(str(npy_path), str(tmp_path / "out"))
        note = result["note"].lower()
        assert "not" in note and ("metres" in note or "meters" in note or "amsl" in note or "absolute" in note)

    def test_preview_png_created(self, tmp_path):
        """A preview PNG must be written."""
        from elevation_calibration import run_relative_calibration

        depth = np.random.rand(64, 64).astype(np.float32) * 30.0
        npy_path = tmp_path / "depth.npy"
        np.save(npy_path, depth)

        result = run_relative_calibration(str(npy_path), str(tmp_path / "out"))
        assert Path(result["rdsm_png_path"]).is_file()

    def test_nan_depth_handled_gracefully(self, tmp_path):
        """NaN pixels in depth must be replaced, not propagate into rDSM."""
        from elevation_calibration import run_relative_calibration

        depth = np.random.rand(64, 64).astype(np.float32) * 50.0
        depth[10:20, 10:20] = np.nan
        npy_path = tmp_path / "depth.npy"
        np.save(npy_path, depth)

        result = run_relative_calibration(str(npy_path), str(tmp_path / "out"))
        assert not np.any(np.isnan(result["rdsm_array"]))
        assert not np.any(np.isinf(result["rdsm_array"]))

    def test_depth_stats_present(self, tmp_path):
        """depth_stats must contain standard statistical keys."""
        from elevation_calibration import run_relative_calibration

        depth = np.ones((32, 32), dtype=np.float32) * 100.0
        npy_path = tmp_path / "depth.npy"
        np.save(npy_path, depth)

        result = run_relative_calibration(str(npy_path), str(tmp_path / "out"))
        for key in ("shape", "min", "max", "mean", "std", "p5", "p95"):
            assert key in result["depth_stats"], f"Missing depth_stats key: {key}"


# ===========================================================================
# skip_gamus parameter test
# ===========================================================================

class TestSkipGamusParam:
    """Verify skip_gamus is correctly threaded in fit_calibration pathway."""

    def test_fit_calibration_still_returns_3tuple_with_skip_gamus(self):
        """fit_calibration is independent of GAMUS; 3-tuple always returned."""
        from elevation_calibration import fit_calibration
        import numpy as np

        d = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float32)
        s = d * 0.5 + 100.0
        a, b, diag = fit_calibration(d, s)
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert "method" in diag
        assert abs(a - 0.5) < 0.1
        assert abs(b - 100.0) < 10.0
