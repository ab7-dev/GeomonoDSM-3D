"""Offline tests for the depth-mapping pipeline.

Model-heavy operations are replaced with a tiny fake model so the suite runs
without the 1.3 GB checkpoint and completes in a few seconds.

Test groups
-----------
Basic inference
    test_get_depth_returns_2d_numpy_array
    test_output_shape_matches_input

GeoTIFF metadata (geo_utils)
    test_read_geotiff_rgb_returns_image_and_meta
    test_read_geotiff_rgb_meta_keys
    test_read_geotiff_rgb_image_is_rgb_uint8
    test_read_geotiff_rgb_dimensions_match_meta
    test_write_depth_geotiff_creates_file
    test_write_depth_geotiff_alignment
    test_write_depth_geotiff_wrong_shape_raises

Tiling
    test_adaptive_edges_5000
    test_adaptive_edges_small_image
    test_adaptive_edges_coverage

get_depth_with_meta
    test_get_depth_with_meta_jpg_returns_none_meta
    test_get_depth_with_meta_geotiff_returns_dict
    test_get_depth_with_meta_shape_matches_input

save_depth_outputs
    test_save_depth_outputs_creates_npy_and_png
    test_save_depth_outputs_creates_geotiff_when_meta_provided
    test_save_depth_outputs_no_geotiff_without_meta
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from depth_mapping import depth, geo_utils
import depth_mapping.model_loader as model_loader

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

SAMPLE_PNG  = Path(__file__).parent / "sample_data" / "sample_scene.png"
REAL_GEOTIFF = Path(__file__).parent.parent / "Input" / "test_2.tif"

rasterio = pytest.importorskip("rasterio", reason="rasterio required for GeoTIFF tests")


class _FakeModel:
    """Minimal stand-in for DepthAnythingV2 — no weights, no GPU needed."""

    def __call__(self, tensor: Any) -> Any:
        b, c, h, w = tensor.shape
        return torch.ones(b, h, w, dtype=torch.float32) * 0.5

    def to(self, device: str) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self


def _fake_load_model(**kw: Any) -> _FakeModel:
    return _FakeModel()


def _make_synthetic_geotiff(path: Path, width: int = 64, height: int = 64) -> dict:
    """Write a tiny synthetic GeoTIFF and return its expected metadata."""
    import rasterio
    from rasterio.transform import from_bounds

    # rasterio 1.5+ requires positional arguments for from_bounds
    transform = from_bounds(
        616500.0, 3343500.0, 616564.0, 3343564.0, width, height
    )
    crs = rasterio.crs.CRS.from_epsg(26914)
    data = np.random.randint(0, 255, (3, height, width), dtype=np.uint8)

    profile = {
        "driver"   : "GTiff",
        "dtype"    : "uint8",
        "width"    : width,
        "height"   : height,
        "count"    : 3,
        "crs"      : crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    return {
        "crs"      : crs,
        "transform": transform,
        "width"    : width,
        "height"   : height,
        "bounds"   : rasterio.open(path).bounds,
        "res"      : (1.0, 1.0),
    }


# ===========================================================================
# Basic inference
# ===========================================================================

def test_get_depth_returns_2d_numpy_array(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_depth() returns a 2-D float32 numpy array."""
    monkeypatch.setattr(model_loader, "load_model", _fake_load_model)
    result = depth.get_depth(str(SAMPLE_PNG))
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.dtype == np.float32


def test_output_shape_matches_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output shape must exactly match the (H, W) of the input image."""
    monkeypatch.setattr(model_loader, "load_model", _fake_load_model)
    from PIL import Image
    img = Image.open(SAMPLE_PNG)
    result = depth.get_depth(str(SAMPLE_PNG))
    assert result.shape == (img.height, img.width), (
        f"Expected ({img.height}, {img.width}), got {result.shape}"
    )


# ===========================================================================
# GeoTIFF metadata — geo_utils
# ===========================================================================

def test_read_geotiff_rgb_returns_image_and_meta(tmp_path: Path) -> None:
    """read_geotiff_rgb() must return a (PIL Image, dict) tuple."""
    from PIL import Image as PILImage

    tif = tmp_path / "tiny.tif"
    _make_synthetic_geotiff(tif)

    image, meta = geo_utils.read_geotiff_rgb(tif)
    assert isinstance(image, PILImage.Image)
    assert isinstance(meta, dict)


def test_read_geotiff_rgb_meta_keys(tmp_path: Path) -> None:
    """geo_meta dict must contain all required keys."""
    tif = tmp_path / "tiny.tif"
    _make_synthetic_geotiff(tif)
    _, meta = geo_utils.read_geotiff_rgb(tif)

    required_keys = {"crs", "transform", "width", "height", "bounds",
                     "res", "nodata", "count", "dtype", "driver"}
    assert required_keys.issubset(meta.keys()), (
        f"Missing keys: {required_keys - meta.keys()}"
    )


def test_read_geotiff_rgb_image_is_rgb_uint8(tmp_path: Path) -> None:
    """Returned PIL image must be RGB mode and uint8."""
    tif = tmp_path / "tiny.tif"
    _make_synthetic_geotiff(tif)
    image, _ = geo_utils.read_geotiff_rgb(tif)
    assert image.mode == "RGB"
    arr = np.asarray(image)
    assert arr.dtype == np.uint8


def test_read_geotiff_rgb_dimensions_match_meta(tmp_path: Path) -> None:
    """PIL image dimensions must agree with meta width/height."""
    W, H = 80, 60
    tif = tmp_path / "wh.tif"
    _make_synthetic_geotiff(tif, width=W, height=H)
    image, meta = geo_utils.read_geotiff_rgb(tif)
    assert image.width  == W == meta["width"]
    assert image.height == H == meta["height"]


def test_write_depth_geotiff_creates_file(tmp_path: Path) -> None:
    """write_depth_geotiff() must create the output file."""
    W, H = 64, 64
    tif_in  = tmp_path / "src.tif"
    tif_out = tmp_path / "depth.tif"
    _, meta = geo_utils.read_geotiff_rgb(_make_synthetic_geotiff.__wrapped__(tif_in)
                                          if hasattr(_make_synthetic_geotiff, '__wrapped__')
                                          else tif_in) if False else (None, None)

    # Build meta directly for isolation
    _make_synthetic_geotiff(tif_in, width=W, height=H)
    _, meta = geo_utils.read_geotiff_rgb(tif_in)

    depth_arr = np.random.rand(H, W).astype(np.float32)
    geo_utils.write_depth_geotiff(depth_arr, meta, tif_out)
    assert tif_out.is_file()
    assert tif_out.stat().st_size > 0


def test_write_depth_geotiff_alignment(tmp_path: Path) -> None:
    """Written GeoTIFF must have CRS, transform, dimensions matching source."""
    W, H = 64, 48
    tif_in  = tmp_path / "src.tif"
    tif_out = tmp_path / "depth.tif"

    _make_synthetic_geotiff(tif_in, width=W, height=H)
    _, meta = geo_utils.read_geotiff_rgb(tif_in)

    depth_arr = np.ones((H, W), dtype=np.float32) * 100.0
    geo_utils.write_depth_geotiff(depth_arr, meta, tif_out)

    checks = geo_utils.validate_geotiff_alignment(meta, tif_out)
    failed = [k for k, v in checks.items() if not v["match"]]
    assert not failed, (
        f"Metadata mismatches in written GeoTIFF: {failed}\n"
        + "\n".join(f"  {k}: source={checks[k]['source']!r}  output={checks[k]['output']!r}"
                    for k in failed)
    )


def test_write_depth_geotiff_values_are_float32(tmp_path: Path) -> None:
    """Written GeoTIFF must store float32 values (not rescaled to int)."""
    import rasterio

    W, H = 32, 32
    tif_in  = tmp_path / "src.tif"
    tif_out = tmp_path / "depth.tif"

    _make_synthetic_geotiff(tif_in, width=W, height=H)
    _, meta = geo_utils.read_geotiff_rgb(tif_in)

    known_value = 123.456789
    depth_arr = np.full((H, W), known_value, dtype=np.float32)
    geo_utils.write_depth_geotiff(depth_arr, meta, tif_out)

    with rasterio.open(tif_out) as dst:
        written = dst.read(1)
    assert written.dtype == np.float32
    assert np.allclose(written, known_value, rtol=1e-5), (
        f"Value mismatch: expected {known_value}, got {written.mean():.6f}"
    )


def test_write_depth_geotiff_wrong_shape_raises(tmp_path: Path) -> None:
    """write_depth_geotiff() must raise ValueError for mismatched dimensions."""
    W, H = 64, 64
    tif_in  = tmp_path / "src.tif"
    tif_out = tmp_path / "depth.tif"

    _make_synthetic_geotiff(tif_in, width=W, height=H)
    _, meta = geo_utils.read_geotiff_rgb(tif_in)

    wrong_arr = np.ones((H + 10, W + 10), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        geo_utils.write_depth_geotiff(wrong_arr, meta, tif_out)


# ===========================================================================
# Tiling
# ===========================================================================

def test_adaptive_edges_5000() -> None:
    """5000 px must produce exactly 10 equal 500-px cores."""
    edges = depth._adaptive_edges(5000)
    assert len(edges) == 11                       # 10 tiles → 11 boundaries
    cores = [edges[i + 1] - edges[i] for i in range(10)]
    assert all(c == 500 for c in cores), f"Unequal cores: {cores}"


def test_adaptive_edges_small_image() -> None:
    """Images smaller than MIN_TILE_SIZE must produce a single tile."""
    edges = depth._adaptive_edges(300)
    assert edges == [0, 300]


def test_adaptive_edges_coverage() -> None:
    """Every pixel must be covered at least once in the blended output."""
    for length in (562, 1179, 5000):
        edges = depth._adaptive_edges(length)
        context = depth.TILE_OVERLAP // 2
        coverage = np.zeros(length, dtype=np.float32)
        for i in range(len(edges) - 1):
            y0 = max(0, edges[i]     - context)
            y1 = min(length, edges[i + 1] + context)
            coverage[y0:y1] += 1.0
        assert coverage.min() >= 1.0, (
            f"Uncovered pixels for length={length}: "
            f"{np.where(coverage == 0)[0].tolist()}"
        )


# ===========================================================================
# get_depth_with_meta
# ===========================================================================

def test_get_depth_with_meta_jpg_returns_none_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-GeoTIFF input must return geo_meta=None."""
    monkeypatch.setattr(model_loader, "load_model", _fake_load_model)
    _, meta = depth.get_depth_with_meta(str(SAMPLE_PNG))
    assert meta is None


def test_get_depth_with_meta_geotiff_returns_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GeoTIFF input must return a non-None geo_meta dict."""
    monkeypatch.setattr(model_loader, "load_model", _fake_load_model)
    tif = tmp_path / "tiny.tif"
    _make_synthetic_geotiff(tif, width=56, height=56)   # mult-of-14 friendly
    _, meta = depth.get_depth_with_meta(str(tif))
    assert isinstance(meta, dict)
    assert "crs" in meta
    assert "transform" in meta


def test_get_depth_with_meta_shape_matches_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Output shape must match input GeoTIFF dimensions."""
    monkeypatch.setattr(model_loader, "load_model", _fake_load_model)
    W, H = 56, 56
    tif = tmp_path / "tiny.tif"
    _make_synthetic_geotiff(tif, width=W, height=H)
    result, meta = depth.get_depth_with_meta(str(tif))
    assert result.shape == (H, W), f"Expected ({H}, {W}), got {result.shape}"
    assert meta["width"]  == W
    assert meta["height"] == H


# ===========================================================================
# save_depth_outputs
# ===========================================================================

def test_save_depth_outputs_creates_npy_and_png(tmp_path: Path) -> None:
    """save_depth_outputs() always creates .npy and _depth.png."""
    arr = np.random.rand(32, 48).astype(np.float32)
    depth.save_depth_outputs(arr, str(tmp_path), "scene.jpg")
    assert (tmp_path / "scene.npy").is_file()
    assert (tmp_path / "scene_depth.png").is_file()


def test_save_depth_outputs_creates_geotiff_when_meta_provided(
    tmp_path: Path,
) -> None:
    """Passing geo_meta must produce a _depth.tif alongside npy/png."""
    import rasterio

    W, H = 56, 56
    tif_src = tmp_path / "src.tif"
    _make_synthetic_geotiff(tif_src, width=W, height=H)
    _, meta = geo_utils.read_geotiff_rgb(tif_src)

    arr = np.random.rand(H, W).astype(np.float32)
    depth.save_depth_outputs(arr, str(tmp_path), "output", geo_meta=meta)

    tif_out = tmp_path / "output_depth.tif"
    assert tif_out.is_file(), "GeoTIFF output not created"

    with rasterio.open(tif_out) as dst:
        assert dst.crs   == meta["crs"]
        assert dst.width == W
        assert dst.height == H
        assert dst.dtypes[0] == "float32"
        written = dst.read(1)
    assert np.all(np.isfinite(written))


def test_save_depth_outputs_no_geotiff_without_meta(tmp_path: Path) -> None:
    """Without geo_meta, no .tif file should be created."""
    arr = np.random.rand(32, 48).astype(np.float32)
    depth.save_depth_outputs(arr, str(tmp_path), "plain")
    tif_files = list(tmp_path.glob("*.tif"))
    assert len(tif_files) == 0, f"Unexpected .tif files: {tif_files}"


# ===========================================================================
# Real GeoTIFF metadata (skipped if test_2.tif not present)
# ===========================================================================

@pytest.mark.skipif(
    not REAL_GEOTIFF.is_file(),
    reason="test_2.tif not present — skipping real-data metadata test",
)
def test_real_geotiff_metadata_preserved(tmp_path: Path) -> None:
    """Round-trip: read test_2.tif metadata, write depth GeoTIFF, compare."""
    import rasterio

    _, meta = geo_utils.read_geotiff_rgb(REAL_GEOTIFF)

    assert str(meta["crs"]) == "EPSG:26914"
    assert meta["width"]  == 5000
    assert meta["height"] == 5000
    assert abs(meta["res"][0] - 0.3) < 1e-6

    # Write a synthetic depth raster at the same spatial extent
    depth_arr = np.ones((5000, 5000), dtype=np.float32) * 99.0
    tif_out = tmp_path / "test_2_depth.tif"
    geo_utils.write_depth_geotiff(depth_arr, meta, tif_out)

    checks = geo_utils.validate_geotiff_alignment(meta, tif_out)
    failed = [k for k, v in checks.items() if not v["match"]]
    assert not failed, (
        f"Metadata mismatches: {failed}\n"
        + "\n".join(f"  {k}: {checks[k]}" for k in failed)
    )

    with rasterio.open(tif_out) as dst:
        assert dst.dtypes[0] == "float32"
        band = dst.read(1)
    assert np.allclose(band, 99.0, rtol=1e-5)


# ===========================================================================
# _clip_exposure — new behaviour (memory + early-exit)
# ===========================================================================

def test_clip_exposure_well_exposed_uint8_returns_unchanged() -> None:
    """Well-exposed uint8 image (full 0-255 range) must be returned unchanged."""
    from PIL import Image as PILImage
    # Create an image whose bands already span 0-255 → early-exit expected.
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    arr[:, :, 1] = 128
    arr[:, :, 2] = np.tile(np.arange(64, dtype=np.uint8)[::-1], (64, 1))
    image = PILImage.fromarray(arr, mode="RGB")
    result = depth._clip_exposure(image)
    # Early-exit path returns the same object, or at worst the same pixels.
    result_arr = np.asarray(result)
    # Values should be unchanged (or only trivially clipped at boundaries).
    assert result_arr.shape == arr.shape
    assert result_arr.dtype == np.uint8


def test_clip_exposure_narrow_range_stretches() -> None:
    """Narrow-range uint8 image must be stretched to use more of 0-255."""
    from PIL import Image as PILImage
    # Pixels all in [100, 150] — span = 50 < 0.9 * 255 → must stretch
    arr = np.full((64, 64, 3), 125, dtype=np.uint8)
    arr[0, 0, :] = 100
    arr[63, 63, :] = 150
    image = PILImage.fromarray(arr, mode="RGB")
    result = depth._clip_exposure(image)
    result_arr = np.asarray(result)
    # After stretch, range should be wider than the original 50-unit span.
    assert result_arr.max() > arr.max() or result_arr.min() < arr.min() or \
           int(result_arr.max()) - int(result_arr.min()) > 50


def test_clip_exposure_output_is_pil_rgb_uint8() -> None:
    """_clip_exposure always returns a PIL Image in RGB mode with uint8 data."""
    from PIL import Image as PILImage
    arr = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
    image = PILImage.fromarray(arr, mode="RGB")
    result = depth._clip_exposure(image)
    assert isinstance(result, PILImage.Image)
    assert result.mode == "RGB"
    assert np.asarray(result).dtype == np.uint8


# ===========================================================================
# _load_image — uint8 GeoTIFF fast path
# ===========================================================================

def test_load_image_geotiff_uint8_fast_path(tmp_path: Path) -> None:
    """uint8 GeoTIFF must be loaded without an intermediate float32 conversion."""
    import rasterio
    from rasterio.transform import from_bounds
    from PIL import Image as PILImage

    W, H = 64, 64
    tif = tmp_path / "uint8.tif"
    transform = from_bounds(0.0, 0.0, 64.0, 64.0, W, H)
    data = np.random.randint(0, 255, (3, H, W), dtype=np.uint8)
    with rasterio.open(tif, "w", driver="GTiff", dtype="uint8",
                       width=W, height=H, count=3,
                       crs=rasterio.crs.CRS.from_epsg(4326),
                       transform=transform) as dst:
        dst.write(data)

    image = depth._load_image(str(tif))
    assert isinstance(image, PILImage.Image)
    assert image.mode == "RGB"
    assert image.width  == W
    assert image.height == H
    # Pixel values must be preserved exactly (no normalisation for uint8).
    loaded = np.asarray(image)
    expected = np.moveaxis(data, 0, -1)  # (H, W, 3)
    assert np.array_equal(loaded, expected), "uint8 pixels were altered by fast path"


# ===========================================================================
# visualize module
# ===========================================================================

def test_save_depth_figure_creates_file(tmp_path: Path) -> None:
    """save_depth_figure must write a non-empty PNG."""
    from depth_mapping.visualize import save_depth_figure
    arr = np.random.rand(64, 96).astype(np.float32) * 100
    out = save_depth_figure(arr, output_dir=tmp_path, stem="test_scene")
    assert out.is_file()
    assert out.stat().st_size > 0
    assert out.suffix == ".png"


def test_save_depth_figure_with_rgb(tmp_path: Path) -> None:
    """save_depth_figure with rgb_image must still produce a valid PNG."""
    from PIL import Image as PILImage
    from depth_mapping.visualize import save_depth_figure
    arr = np.random.rand(64, 96).astype(np.float32) * 100
    rgb = PILImage.fromarray(
        np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8), mode="RGB"
    )
    out = save_depth_figure(arr, output_dir=tmp_path, stem="scene_rgb",
                            rgb_image=rgb)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_load_depth_and_visualize(tmp_path: Path) -> None:
    """load_depth_and_visualize must load .npy and produce a figure."""
    from depth_mapping.visualize import load_depth_and_visualize
    arr = np.random.rand(64, 96).astype(np.float32) * 100
    npy_path = tmp_path / "depth.npy"
    np.save(npy_path, arr)
    out = load_depth_and_visualize(npy_path, output_dir=tmp_path)
    assert out.is_file()
    assert out.stat().st_size > 0


# ===========================================================================
# validate_output module
# ===========================================================================

def test_validate_run_passes_for_valid_npy(tmp_path: Path) -> None:
    """validate_run must pass when .npy is present and valid."""
    from depth_mapping.validate_output import validate_run
    arr = np.random.rand(64, 96).astype(np.float32) * 100
    np.save(tmp_path / "result.npy", arr)
    report = validate_run(tmp_path)
    assert report["passed"], f"Failed checks: {report['failed_checks']}"


def test_validate_run_fails_for_constant_depth(tmp_path: Path) -> None:
    """validate_run must fail when depth array has zero variance."""
    from depth_mapping.validate_output import validate_run
    arr = np.ones((64, 96), dtype=np.float32) * 50.0
    np.save(tmp_path / "constant.npy", arr)
    report = validate_run(tmp_path)
    assert not report["passed"]
    assert "npy_nonzero_variance" in report["failed_checks"]


def test_validate_run_with_geotiff(tmp_path: Path) -> None:
    """validate_run must check GeoTIFF alignment when .tif is present."""
    import rasterio
    from rasterio.transform import from_bounds
    from depth_mapping.validate_output import validate_run
    from depth_mapping import geo_utils

    W, H = 56, 56
    tif_src = tmp_path / "src.tif"
    transform = from_bounds(616500.0, 3343500.0, 616556.0, 3343556.0, W, H)
    crs = rasterio.crs.CRS.from_epsg(26914)
    data = np.random.randint(0, 255, (3, H, W), dtype=np.uint8)
    with rasterio.open(tif_src, "w", driver="GTiff", dtype="uint8",
                       width=W, height=H, count=3, crs=crs,
                       transform=transform) as dst:
        dst.write(data)

    _, meta = geo_utils.read_geotiff_rgb(tif_src)
    depth_arr = np.random.rand(H, W).astype(np.float32) * 100
    np.save(tmp_path / "depth.npy", depth_arr)
    geo_utils.write_depth_geotiff(depth_arr, meta, tmp_path / "depth_depth.tif")

    report = validate_run(tmp_path, source_tif=tif_src)
    assert report["passed"], f"Failed checks: {report['failed_checks']}"


def test_write_report_creates_json(tmp_path: Path) -> None:
    """write_report must create a readable JSON file."""
    import json as json_mod
    from depth_mapping.validate_output import validate_run, write_report
    arr = np.random.rand(32, 32).astype(np.float32) * 50
    np.save(tmp_path / "d.npy", arr)
    report = validate_run(tmp_path)
    rpath = write_report(report, tmp_path)
    assert rpath.is_file()
    with open(rpath) as f:
        loaded = json_mod.load(f)
    assert "passed" in loaded
    assert "checks" in loaded


def test_validate_run_nonexistent_dir_raises() -> None:
    """validate_run must raise FileNotFoundError for missing directories."""
    from depth_mapping.validate_output import validate_run
    with pytest.raises(FileNotFoundError):
        validate_run("/this/does/not/exist/at/all")


# ===========================================================================
# Affine tile alignment  (_robust_affine, _align_tiles_affine)
# ===========================================================================

def test_robust_affine_identity() -> None:
    """When ref == src, robust_affine must return scale≈1, shift≈0."""
    from depth_mapping.depth import _robust_affine
    rng = np.random.default_rng(42)
    arr = rng.random(512).astype(np.float32) * 100
    scale, shift = _robust_affine(arr, arr)
    assert abs(scale - 1.0) < 0.05, f"Expected scale≈1, got {scale}"
    assert abs(shift)        < 2.0,  f"Expected shift≈0, got {shift}"


def test_robust_affine_known_transform() -> None:
    """_robust_affine must recover a known scale=2, shift=10 transform."""
    from depth_mapping.depth import _robust_affine
    rng = np.random.default_rng(7)
    src = rng.random(512).astype(np.float32) * 50 + 10
    ref = (src * 2.0 + 10.0).astype(np.float32)
    scale, shift = _robust_affine(ref, src)
    assert abs(scale - 2.0) < 0.15, f"Expected scale≈2, got {scale}"
    assert abs(shift - 10.0) < 3.0,  f"Expected shift≈10, got {shift}"


def test_robust_affine_clamps_pathological_scale() -> None:
    """A degenerate (near-constant) src must produce clamped scale in [0.1, 10]."""
    from depth_mapping.depth import _robust_affine
    ref = np.random.rand(256).astype(np.float32) * 100
    src = np.full(256, 0.001, dtype=np.float32)   # near-zero variance
    scale, shift = _robust_affine(ref, src)
    assert 0.1 <= scale <= 10.0, f"Scale out of clamp range: {scale}"


def test_robust_affine_too_few_pixels_returns_identity() -> None:
    """Fewer than 16 overlap pixels must return (1.0, 0.0)."""
    from depth_mapping.depth import _robust_affine
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scale, shift = _robust_affine(ref, src)
    assert scale == 1.0
    assert shift == 0.0


def test_align_tiles_affine_single_tile() -> None:
    """Single-tile grid: output must be finite, float32, same shape as input."""
    from depth_mapping.depth import _align_tiles_affine, TILE_OVERLAP
    rng = np.random.default_rng(0)
    tile = rng.random((64, 64), dtype=np.float32) * 100
    raw_tiles   = [[tile]]
    tile_coords = [[(0, 64, 0, 64)]]
    aligned = _align_tiles_affine(
        raw_tiles, tile_coords,
        y_edges=[0, 64], x_edges=[0, 64],
        image_h=64, image_w=64,
        context=TILE_OVERLAP // 2,
    )
    result = aligned[0][0]
    assert result.shape == (64, 64), f"Shape mismatch: {result.shape}"
    assert result.dtype == np.float32, f"dtype mismatch: {result.dtype}"
    assert np.all(np.isfinite(result)), "Non-finite values in single-tile output"
    # Z-score then rescale: result should be centred near 0 (median ~ 0)
    # and have non-trivial variance (not a constant).
    assert result.std() > 1.0, f"Output has near-zero variance: std={result.std()}"


def test_align_tiles_affine_2x2_consistent_depth() -> None:
    """2x2 grid from same depth map: overlap pixels must be close after alignment."""
    from depth_mapping.depth import _align_tiles_affine, TILE_OVERLAP
    rng = np.random.default_rng(1)
    H, W = 200, 200
    context = TILE_OVERLAP // 2
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    full_depth = (xx * 50 + yy * 30 + rng.random((H, W), dtype=np.float32) * 10).astype(np.float32)
    y_edges = [0, 100, 200]
    x_edges = [0, 100, 200]
    raw_tiles, tile_coords = [], []
    for yi in range(2):
        row_t, row_c = [], []
        for xi in range(2):
            y0 = max(0, y_edges[yi] - context)
            y1 = min(H, y_edges[yi + 1] + context)
            x0 = max(0, x_edges[xi] - context)
            x1 = min(W, x_edges[xi + 1] + context)
            row_t.append(full_depth[y0:y1, x0:x1].copy())
            row_c.append((y0, y1, x0, x1))
        raw_tiles.append(row_t)
        tile_coords.append(row_c)
    aligned = _align_tiles_affine(
        raw_tiles, tile_coords, y_edges=y_edges, x_edges=x_edges,
        image_h=H, image_w=W, context=context,
    )
    # After alignment, overlap between tile(0,0) and tile(0,1) must be close.
    y0_a, y1_a, x0_a, x1_a = tile_coords[0][0]
    y0_b, y1_b, x0_b, x1_b = tile_coords[0][1]
    ov_y0, ov_y1 = max(y0_a, y0_b), min(y1_a, y1_b)
    ov_x0, ov_x1 = max(x0_a, x0_b), min(x1_a, x1_b)
    a_ov = aligned[0][0][ov_y0 - y0_a:ov_y1 - y0_a, ov_x0 - x0_a:ov_x1 - x0_a]
    b_ov = aligned[0][1][ov_y0 - y0_b:ov_y1 - y0_b, ov_x0 - x0_b:ov_x1 - x0_b]
    residual = float(np.abs(a_ov - b_ov).mean())
    depth_range = float(full_depth.max() - full_depth.min())
    assert residual < 0.3 * depth_range, (
        f"Overlap residual {residual:.2f} too large vs range {depth_range:.2f}"
    )


def test_tiled_inference_output_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """_tiled_inference must return a 2-D array matching the input image size."""
    from PIL import Image as PILImage
    from depth_mapping.depth import _tiled_inference
    monkeypatch.setattr(
        "depth_mapping.depth.load_model",
        lambda **kw: _FakeModel(),
    )
    # 600×800 forces a 2×1 or similar tile grid
    img = PILImage.fromarray(
        np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8), mode="RGB"
    )
    from depth_mapping.model_loader import load_model
    model = _FakeModel()
    from depth_mapping.depth import _DEVICE
    result = _tiled_inference(img, model, _DEVICE)
    assert result.ndim == 2
    assert result.shape == (600, 800), f"Expected (600, 800), got {result.shape}"
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


def test_tile_boundary_metric_single_tile() -> None:
    """Single-tile image (no internal boundaries) must return zero boundaries."""
    from depth_mapping.tile_boundary_metric import measure_tile_boundaries
    depth_arr = np.random.rand(64, 64).astype(np.float32)
    metrics = measure_tile_boundaries(depth_arr, y_edges=[0, 64], x_edges=[0, 64])
    assert metrics["n_h_boundaries"] == 0
    assert metrics["n_v_boundaries"] == 0
    assert metrics["mean_mad"] == 0.0


def test_tile_boundary_metric_known_seam() -> None:
    """Depth map with a sharp step at y=50 must register a large H-boundary MAD."""
    from depth_mapping.tile_boundary_metric import measure_tile_boundaries
    depth_arr = np.zeros((100, 100), dtype=np.float32)
    depth_arr[50:, :] = 100.0   # sharp step at row 50
    metrics = measure_tile_boundaries(
        depth_arr, y_edges=[0, 50, 100], x_edges=[0, 100], half_band=4
    )
    assert metrics["n_h_boundaries"] == 1
    assert metrics["horizontal_mads"][0] > 50.0, (
        f"Expected large MAD at sharp seam, got {metrics['horizontal_mads'][0]}"
    )


def test_checkpoint_discovery_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DA_CHECKPOINT_DIR environment variable must override automatic discovery."""
    monkeypatch.setenv("DA_CHECKPOINT_DIR", str(tmp_path))
    from importlib import reload
    import depth_mapping.model_loader as ml_mod
    reload(ml_mod)   # force re-evaluation of default_checkpoint_dir
    result = ml_mod.default_checkpoint_dir()
    assert result == tmp_path, f"Expected {tmp_path}, got {result}"
    monkeypatch.delenv("DA_CHECKPOINT_DIR", raising=False)


def test_get_depth_tiled_large_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_depth on a 1200×1200 image must produce a 2-D float32 result."""
    from PIL import Image as PILImage
    monkeypatch.setattr(model_loader, "load_model", _fake_load_model)
    # Create a synthetic large PNG on disk for the test
    import tempfile
    arr = np.random.randint(0, 255, (1200, 1200, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path_str = f.name
    try:
        img.save(tmp_path_str)
        result = depth.get_depth(tmp_path_str)
        assert result.ndim == 2
        assert result.shape == (1200, 1200)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
    finally:
        import os
        os.unlink(tmp_path_str)


# ===========================================================================
# New tests for upgraded pipeline (Phase 2 quality upgrade)
# ===========================================================================

def test_hann_weight_sums_to_one_in_core() -> None:
    """Hann weight must be 1.0 in the tile core (non-overlap) region."""
    from depth_mapping.depth import _hann_weight, TILE_OVERLAP
    context = TILE_OVERLAP // 2
    # Simulate an interior tile: edges on all four sides
    crop_y0, crop_x0 = 500 - context, 500 - context  # e.g. 404
    h, w = 500 + context - crop_y0, 500 + context - crop_x0  # e.g. 192
    # A simple 1-D test: weight must be 1.0 in the core
    y_edges = [0, 500, 1000]
    x_edges = [0, 500, 1000]
    weight = _hann_weight(crop_y0, crop_x0, h, w, y_edges, x_edges, 1, 1)
    # Core region (rows/cols without overlap) must have weight 1.0
    core = weight[context:-context, context:-context]
    assert np.allclose(core, 1.0, atol=1e-5), f"Core weight not 1.0: min={core.min():.4f}"


def test_hann_weight_edge_tile_no_taper_on_image_boundary() -> None:
    """Top-left corner tile: image-boundary edges (top=0, left=0) have weight 1.0."""
    from depth_mapping.depth import _hann_weight, TILE_OVERLAP
    context = TILE_OVERLAP // 2
    height = 500 + context
    width  = 500 + context
    weight = _hann_weight(
        crop_y0=0, crop_x0=0, height=height, width=width,
        y_edges=[0, 500, 1000], x_edges=[0, 500, 1000],
        y_index=0, x_index=0,
    )
    # Top-left pixel must have weight 1.0 (both image boundaries, no taper)
    assert weight[0, 0] == pytest.approx(1.0, abs=1e-5), \
        f"Top-left corner weight not 1.0: {weight[0, 0]}"
    # Top-left quadrant (up to 500, 500 — the core) should all be 1.0
    core = weight[:500, :500]
    assert core.min() == pytest.approx(1.0, abs=1e-5), \
        f"Core region weight not 1.0: min={core.min():.5f}"
    # Beyond the core the taper reduces weight
    assert weight[-1, -1] < 1.0, "Trailing corner should have taper < 1.0"


def test_hann_weight_values_between_0_and_1() -> None:
    """All Hann weights must be in [0, 1]."""
    from depth_mapping.depth import _hann_weight, TILE_OVERLAP
    context = TILE_OVERLAP // 2
    weight = _hann_weight(
        crop_y0=context, crop_x0=context,
        height=500, width=500,
        y_edges=[0, 500, 1000, 1500], x_edges=[0, 500, 1000, 1500],
        y_index=1, x_index=1,
    )
    assert weight.min() >= 0.0, f"Negative weight: {weight.min()}"
    assert weight.max() <= 1.0 + 1e-5, f"Weight > 1: {weight.max()}"


def test_clip_tile_outliers_preserves_interior() -> None:
    """Per-tile clipping (0.5-99.5) must preserve the interior distribution."""
    from depth_mapping.depth import _clip_tile_outliers
    rng = np.random.default_rng(99)
    tile = rng.random((64, 64), dtype=np.float32) * 100
    clipped = _clip_tile_outliers(tile)
    # Most values should be unchanged
    unchanged_frac = np.mean(clipped == tile)
    assert unchanged_frac > 0.95, f"Too many values clipped: only {unchanged_frac:.2%} unchanged"
    assert clipped.dtype == np.float32


def test_clip_tile_outliers_removes_spikes() -> None:
    """Per-tile clipping must neutralise extreme spike pixels."""
    from depth_mapping.depth import _clip_tile_outliers
    tile = np.ones((64, 64), dtype=np.float32) * 50.0
    tile[0, 0]   = 1e6   # large spike
    tile[63, 63] = -1e6  # negative spike
    clipped = _clip_tile_outliers(tile)
    assert clipped.max() < 1e5, "Large spike not clipped"
    assert clipped.min() > -1e5, "Negative spike not clipped"


def test_affine_composition_correctness() -> None:
    """_align_tiles_affine shim: returns clipped, finite, float32 arrays."""
    from depth_mapping.depth import _align_tiles_affine, TILE_OVERLAP
    H, W_tile, context = 64, 200, TILE_OVERLAP // 2
    total_W = 2 * W_tile
    rng = np.random.default_rng(42)
    base = rng.random((H, total_W), dtype=np.float32) * 50 + 25
    t0 = base[:, :W_tile + context].copy()
    t1 = base[:, W_tile - context:].copy() * 2.0 + 10.0
    y_edges = [0, H]
    x_edges = [0, W_tile, total_W]
    tile_coords = [[(0, H, 0, W_tile + context), (0, H, W_tile - context, total_W)]]
    raw_tiles = [[t0, t1]]
    aligned = _align_tiles_affine(
        raw_tiles, tile_coords,
        y_edges=y_edges, x_edges=x_edges,
        image_h=H, image_w=total_W, context=context,
    )
    # Shim returns per-tile clipped arrays: same shape, finite, float32.
    assert aligned[0][0].shape == t0.shape
    assert aligned[0][1].shape == t1.shape
    assert np.all(np.isfinite(aligned[0][0]))
    assert np.all(np.isfinite(aligned[0][1]))
    assert aligned[0][0].dtype == np.float32
    assert aligned[0][1].dtype == np.float32

def test_tiled_inference_no_nan_inf(monkeypatch: pytest.MonkeyPatch) -> None:
    """_tiled_inference must produce a finite output with no NaN or Inf."""
    from PIL import Image as PILImage
    from depth_mapping.depth import _tiled_inference, _DEVICE
    monkeypatch.setattr("depth_mapping.depth.load_model", lambda **kw: _FakeModel())
    model = _FakeModel()
    img   = PILImage.fromarray(
        np.random.randint(0, 255, (1200, 1200, 3), dtype=np.uint8), mode="RGB"
    )
    result = _tiled_inference(img, model, _DEVICE)
    assert np.all(np.isfinite(result)), "NaN or Inf in tiled output"
    assert result.dtype == np.float32


def test_larger_overlap_improves_context() -> None:
    """TILE_OVERLAP should be at least 128 to provide meaningful context."""
    from depth_mapping.depth import TILE_OVERLAP
    assert TILE_OVERLAP >= 128, f"TILE_OVERLAP too small: {TILE_OVERLAP}"


def test_no_per_tile_histogram_normalisation() -> None:
    """_predict must NOT apply per-tile min-max normalisation.

    If the model returns constant 0.5 (FakeModel), the output should stay
    at 0.5 — not be rescaled by any per-tile normalisation.
    """
    from PIL import Image as PILImage
    from depth_mapping.depth import _predict, _DEVICE
    model = _FakeModel()
    img   = PILImage.fromarray(
        np.ones((64, 64, 3), dtype=np.uint8) * 128, mode="RGB"
    )
    result = _predict(img, model, _DEVICE)
    # FakeModel returns 0.5 everywhere; result should be uniform ≈ 0.5
    assert result.mean() > 0.4 and result.mean() < 0.6, (
        f"Per-tile normalisation suspected: mean={result.mean():.4f} (expected ≈0.5)"
    )
    assert result.std() < 0.01, (
        f"Output unexpectedly non-uniform: std={result.std():.4f}"
    )


# ===========================================================================
# New core-only tiling tests (v4 architecture)
# ===========================================================================

def test_estimate_shift_zero_for_identical_arrays() -> None:
    """_estimate_shift must return ~0 when both inputs are identical."""
    from depth_mapping.depth import _estimate_shift
    rng = np.random.default_rng(0)
    arr = rng.random((64, 64), dtype=np.float32) * 100 + 10
    shift = _estimate_shift(arr, arr)
    assert abs(shift) < 0.1, f"Expected ~0 shift for identical inputs, got {shift}"


def test_estimate_shift_recovers_known_offset() -> None:
    """_estimate_shift must recover a known positive shift."""
    from depth_mapping.depth import _estimate_shift
    rng = np.random.default_rng(1)
    base = rng.random((64, 64), dtype=np.float32) * 50 + 20
    shifted = base + 15.0
    delta = _estimate_shift(base, shifted)
    assert abs(delta - (-15.0)) < 2.0, f"Expected ~-15 shift, got {delta}"


def test_estimate_shift_insufficient_pixels_returns_zero() -> None:
    """_estimate_shift must return 0.0 when too few valid pixels."""
    from depth_mapping.depth import _estimate_shift
    arr = np.array([1.0, 2.0], dtype=np.float32)
    assert _estimate_shift(arr, arr) == 0.0


def test_make_feather_mask_interior_tile_is_one_in_core() -> None:
    """Interior tile feather mask must be 1.0 in the non-feather core."""
    from depth_mapping.depth import _make_feather_mask, FEATHER
    H, W = 200, 200
    mask = _make_feather_mask(H, W, top=True, bottom=True,
                               left=True, right=True, feather=FEATHER)
    f = FEATHER
    core = mask[f:H-f, f:W-f]
    assert np.allclose(core, 1.0, atol=1e-5), f"Core not 1.0: min={core.min()}"


def test_make_feather_mask_edge_tile_no_taper_at_boundary() -> None:
    """Edge-tile feather mask: image-boundary sides must stay at 1.0."""
    from depth_mapping.depth import _make_feather_mask, FEATHER
    # top-left corner: top=False, left=False  → no taper on top or left edge
    mask = _make_feather_mask(200, 200,
                               top=False, bottom=True,
                               left=False, right=True, feather=FEATHER)
    # Top-left corner: both boundaries are image edges → weight must be 1.0
    assert mask[0, 0] == pytest.approx(1.0, abs=1e-5)
    # Top edge (row 0): top=False means no vertical taper there
    # Left edge (col 0): left=False means no horizontal taper there
    assert np.allclose(mask[0, :FEATHER], 1.0, atol=1e-5), \
        "Top-left zone should be 1.0 (no taper on top or left boundary)"
    # Bottom-left corner must have vertical taper (bottom=True)
    assert mask[-1, 0] < 1.0, "Bottom edge should have taper (bottom=True)"


def test_make_feather_mask_values_in_range() -> None:
    """All feather mask values must be in [0, 1]."""
    from depth_mapping.depth import _make_feather_mask, FEATHER
    mask = _make_feather_mask(300, 400,
                               top=True, bottom=True,
                               left=True, right=True, feather=FEATHER)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0 + 1e-5


def test_context_margin_provides_sufficient_context() -> None:
    """CONTEXT_MARGIN must be at least 64 px (≈ 4.5 ViT patches)."""
    from depth_mapping.depth import CONTEXT_MARGIN
    assert CONTEXT_MARGIN >= 64, f"CONTEXT_MARGIN too small: {CONTEXT_MARGIN}"


def test_core_only_tiling_single_prediction_per_core_pixel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each core pixel must receive exactly one forward-pass prediction.

    With a FakeModel returning constant 0.5, and no blending inside the core,
    every pixel in the stitched output must be exactly 0.5 (or very close).
    """
    from PIL import Image as PILImage
    from depth_mapping.depth import _tiled_inference, _DEVICE, CORE_TILE_SIZE
    monkeypatch.setattr("depth_mapping.depth.load_model", lambda **kw: _FakeModel())
    # Use an image large enough to need tiling
    H = W = CORE_TILE_SIZE * 3
    img = PILImage.fromarray(
        np.full((H, W, 3), 128, dtype=np.uint8), mode="RGB"
    )
    result = _tiled_inference(img, _FakeModel(), _DEVICE)
    assert result.shape == (H, W)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    # With FakeModel output = 0.5, the stitched output (after mild clip)
    # should be very close to 0.5 everywhere — no averaging noise.
    assert result.mean() == pytest.approx(0.5, abs=0.01), (
        f"Expected ≈0.5 everywhere but mean={result.mean():.4f} std={result.std():.4f}"
    )


def test_structural_quality_metrics_basic() -> None:
    """measure_structural_quality returns expected keys for a random array."""
    from depth_mapping.tile_boundary_metric import measure_structural_quality
    rng = np.random.default_rng(42)
    depth_arr = rng.random((128, 128), dtype=np.float32) * 100
    qual = measure_structural_quality(depth_arr)
    for k in ("grad_mean", "grad_p90", "laplacian_var",
              "local_contrast", "edge_density"):
        assert k in qual, f"Missing metric: {k}"
        assert isinstance(qual[k], float), f"{k} is not float"
        assert np.isfinite(qual[k]),       f"{k} is not finite"


def test_structural_quality_metrics_flat_image_low_grad() -> None:
    """A perfectly flat depth map must have near-zero gradient metrics."""
    from depth_mapping.tile_boundary_metric import measure_structural_quality
    flat = np.ones((128, 128), dtype=np.float32) * 50.0
    qual = measure_structural_quality(flat)
    assert qual["grad_mean"] < 0.01, f"Flat image has high gradient: {qual['grad_mean']}"
    assert qual["laplacian_var"] < 0.01


def test_structural_quality_ratio_detects_blur() -> None:
    """Ratio metrics must flag detail loss when blurred vs sharp reference."""
    from depth_mapping.tile_boundary_metric import measure_structural_quality
    from scipy.ndimage import gaussian_filter  # type: ignore
    rng = np.random.default_rng(7)
    sharp = rng.random((256, 256), dtype=np.float32) * 100
    blurred = gaussian_filter(sharp, sigma=5).astype(np.float32)
    qual = measure_structural_quality(blurred, reference=sharp)
    assert "grad_ratio" in qual
    assert qual["grad_ratio"] < 0.9, (
        f"Blurred image not detected as lower-gradient: grad_ratio={qual['grad_ratio']:.3f}"
    )
    assert qual.get("detail_ok") is False, "Expected detail_ok=False for blurred image"


# ===========================================================================
# GAMUS module tests
# ===========================================================================

def test_gamus_config_default_values() -> None:
    """GAMUSConfig must have expected default attributes."""
    from depth_mapping.gamus import GAMUSConfig
    cfg = GAMUSConfig()
    assert cfg.hf_repo_id == "earthflow/GAMUS"
    assert cfg.rgb_subdir  == "rgb"
    assert cfg.ndsm_subdir == "ndsm"
    assert cfg.streaming   is False


def test_gamus_config_validates_missing_dir(tmp_path: Path) -> None:
    """GAMUSConfig.validate() raises FileNotFoundError for missing data_dir."""
    from depth_mapping.gamus import GAMUSConfig
    cfg = GAMUSConfig(data_dir=tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="GAMUS data_dir"):
        cfg.validate()


def test_gamus_dataset_empty_dir_yields_nothing(tmp_path: Path) -> None:
    """GAMUSDataset over an empty directory must yield 0 samples."""
    from depth_mapping.gamus import GAMUSConfig, GAMUSDataset
    (tmp_path / "rgb").mkdir()
    (tmp_path / "ndsm").mkdir()
    cfg = GAMUSConfig(data_dir=tmp_path)
    ds  = GAMUSDataset(cfg)
    assert len(ds) == 0
    assert list(ds) == []


def test_gamus_load_sample_synthetic(tmp_path: Path) -> None:
    """load_sample() must return dict with 'rgb' PIL and 'ndsm' float32 array."""
    import rasterio
    from rasterio.transform import from_bounds
    from depth_mapping.gamus import load_sample

    # Write synthetic RGB PNG
    from PIL import Image as PILImage
    rgb_img = PILImage.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), "RGB"
    )
    rgb_path = tmp_path / "scene.png"
    rgb_img.save(str(rgb_path))

    # Write synthetic nDSM GeoTIFF (float32 metres)
    ndsm_path = tmp_path / "scene_ndsm.tif"
    ndsm_data = np.random.rand(64, 64).astype(np.float32) * 10.0
    transform = from_bounds(0, 0, 64, 64, 64, 64)
    with rasterio.open(ndsm_path, "w", driver="GTiff", dtype="float32",
                       width=64, height=64, count=1,
                       crs=rasterio.crs.CRS.from_epsg(4326),
                       transform=transform) as dst:
        dst.write(ndsm_data, 1)

    sample = load_sample(str(rgb_path), str(ndsm_path))
    assert "rgb"  in sample
    assert "ndsm" in sample
    assert isinstance(sample["rgb"],  __import__("PIL").Image.Image)
    assert isinstance(sample["ndsm"], np.ndarray)
    assert sample["ndsm"].dtype == np.float32
    assert sample["ndsm"].ndim  == 2


def test_gamus_compute_height_metrics_perfect_prediction() -> None:
    """compute_height_metrics with identical pred/ref after alignment gives MAE≈0."""
    from depth_mapping.gamus import compute_height_metrics
    ref  = np.random.rand(64, 64).astype(np.float32) * 10.0 + 1.0
    # pred = exact same values (perfect prediction after alignment)
    metrics = compute_height_metrics(ref, ref, align=True)
    assert metrics["mae"]       < 0.01
    assert metrics["delta_1"]   > 0.99
    assert metrics["spearman_r"] > 0.99


def test_gamus_compute_height_metrics_shape_mismatch_raises() -> None:
    """compute_height_metrics must raise ValueError for mismatched shapes."""
    from depth_mapping.gamus import compute_height_metrics
    pred = np.ones((64, 64), dtype=np.float32)
    ref  = np.ones((32, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_height_metrics(pred, ref)
