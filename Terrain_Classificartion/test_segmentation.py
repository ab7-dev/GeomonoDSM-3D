from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image

from terrain_classification.segmentation import (
    TerrainClass,
    apply_greenness_fallback,
    classify_terrain,
    collapse_model_labels,
    save_segmentation_outputs,
    segment,
    visualize_confidence_heatmap,
    _OUTPUT_SCHEMA,
    _DETECTION_SCHEMA,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = PROJECT_ROOT / "work" / "content" / "india_geotiffs"
LOCAL_MODEL = PROJECT_ROOT / "work" / "models" / "loveda-segformer"
TEST_OUTPUTS = PROJECT_ROOT / "outputs" / "_test_terrain_classification"


def _model_name() -> str:
    """Use the locally cached checkpoint when available; otherwise Hugging Face."""
    return str(LOCAL_MODEL) if LOCAL_MODEL.is_dir() else "wu-pr-gw/segformer-b2-finetuned-with-LoveDA"


def _require_patch(name: str) -> Path:
    path = TEST_DATA / name
    if not path.is_file():
        pytest.skip(f"Optional SIH GeoTIFF test image is unavailable: {path}")
    return path


def _assert_label_contract(labels: np.ndarray, source_path: Path) -> None:
    with rasterio.open(source_path) as source:
        assert labels.shape == (source.height, source.width)
    assert labels.dtype == np.uint8
    assert set(np.unique(labels)).issubset({0, 1, 2, 3, 4})


def test_loveda_class_mapping_includes_water():
    raw = np.array([[0, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    labels = {
        0: "Ignore",
        2: "Building",
        3: "Road",
        4: "Water",
        5: "Barren",
        6: "Forest",
        7: "Agricultural",
    }

    result = collapse_model_labels(raw, labels)

    assert result.tolist() == [[0, 2, 3, 4, 0, 1, 1]]


@pytest.mark.parametrize("name", ["india_patch_53.tif", "india_patch_119.tif"])
def test_real_geotiff_segmentation_and_georeferenced_outputs(name: str):
    image_path = _require_patch(name)
    labels, overlay = segment(image_path, model_name=_model_name(), device="cpu")
    _assert_label_contract(labels, image_path)

    output_dir = TEST_OUTPUTS / image_path.stem
    paths = save_segmentation_outputs(image_path, labels, overlay, output_dir)

    assert paths["labels_npy"].is_file() and paths["labels_npy"].stat().st_size > 0
    assert paths["overlay_png"].is_file() and paths["overlay_png"].stat().st_size > 0
    assert paths["preview_png"].is_file() and paths["preview_png"].stat().st_size > 0
    assert paths["labels_tif"].is_file() and paths["labels_tif"].stat().st_size > 0

    with rasterio.open(image_path) as source, rasterio.open(paths["labels_tif"]) as output:
        assert output.crs == source.crs
        assert output.transform == source.transform
        assert output.width == source.width
        assert output.height == source.height
        assert output.dtypes == ("uint8",)


def test_plain_png_runs_and_skips_geotiff_output():
    image_path = _require_patch("india_patch_53.tif")
    output_dir = TEST_OUTPUTS / "plain_png"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "plain_input.png"
    with rasterio.open(image_path) as source:
        rgb = source.read([1, 2, 3]).transpose(1, 2, 0)
    Image.fromarray(rgb).save(png_path)

    labels, overlay = segment(png_path, model_name=_model_name(), device="cpu")
    assert labels.shape == (512, 512)
    assert labels.dtype == np.uint8
    assert set(np.unique(labels)).issubset(set(TerrainClass))

    paths = save_segmentation_outputs(png_path, labels, overlay, output_dir)
    assert "labels_tif" not in paths
    assert not (output_dir / "plain_input_terrain_labels.tif").exists()
    assert paths["overlay_png"].is_file() and paths["overlay_png"].stat().st_size > 0
    assert paths["preview_png"].is_file() and paths["preview_png"].stat().st_size > 0


# ---------------------------------------------------------------------------
# apply_greenness_fallback tests
# ---------------------------------------------------------------------------

class TestApplyGreennessGreennessallback:
    """Unit tests for apply_greenness_fallback.

    All tests use synthetic 1-D or small 2-D arrays so they run without any
    model or GeoTIFF dependency.
    """

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _make(h: int = 1, w: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (label_map, rgb, raw_model_labels) all zeros of given shape."""
        labels = np.zeros((h, w), dtype=np.uint8)
        rgb    = np.zeros((h, w, 3), dtype=np.uint8)
        raw    = np.zeros((h, w), dtype=np.int64)
        return labels, rgb, raw

    # ---- core promotion logic ----------------------------------------------

    def test_greenish_background_pixel_becomes_vegetation(self):
        """A pixel that is OTHER, came from Background, and is visually green
        must be promoted to VEGETATION."""
        labels, rgb, raw = self._make()
        labels[0, 0] = TerrainClass.OTHER        # collapsed to OTHER
        raw[0, 0]    = 1                          # LoveDA Background id
        rgb[0, 0]    = [50, 120, 40]             # G(120) > R(50), G > B(40), G > 70
        result = apply_greenness_fallback(labels, rgb, raw)
        assert result[0, 0] == TerrainClass.VEGETATION

    def test_non_background_other_pixel_not_promoted(self):
        """OTHER pixel that came from Barren (not Background) must stay OTHER
        even if it looks green — only Background→OTHER pixels are eligible."""
        labels, rgb, raw = self._make()
        labels[0, 0] = TerrainClass.OTHER
        raw[0, 0]    = 5                          # LoveDA Barren id
        rgb[0, 0]    = [50, 120, 40]             # greenish
        result = apply_greenness_fallback(labels, rgb, raw)
        assert result[0, 0] == TerrainClass.OTHER

    def test_non_green_background_pixel_not_promoted(self):
        """Background→OTHER pixel that fails the greenness test must stay OTHER."""
        labels, rgb, raw = self._make()
        labels[0, 0] = TerrainClass.OTHER
        raw[0, 0]    = 1                          # Background
        rgb[0, 0]    = [130, 90, 60]             # brownish: G < R
        result = apply_greenness_fallback(labels, rgb, raw)
        assert result[0, 0] == TerrainClass.OTHER

    def test_green_below_threshold_not_promoted(self):
        """G > R and G > B but G <= threshold must NOT be promoted."""
        labels, rgb, raw = self._make()
        labels[0, 0] = TerrainClass.OTHER
        raw[0, 0]    = 1                          # Background
        rgb[0, 0]    = [30, 65, 20]             # G(65) > R, G > B, but G <= 70
        result = apply_greenness_fallback(labels, rgb, raw, green_threshold=70)
        assert result[0, 0] == TerrainClass.OTHER

    def test_non_other_label_never_changed(self):
        """Pixels already labeled BUILDING, ROAD, WATER, or VEGETATION must
        never be reclassified regardless of RGB or raw model label."""
        for terrain in (TerrainClass.VEGETATION, TerrainClass.BUILDING,
                        TerrainClass.ROAD, TerrainClass.WATER):
            labels, rgb, raw = self._make()
            labels[0, 0] = terrain
            raw[0, 0]    = 1                      # Background
            rgb[0, 0]    = [10, 200, 10]         # very green
            result = apply_greenness_fallback(labels, rgb, raw)
            assert result[0, 0] == terrain, (
                f"{terrain.name} pixel was incorrectly modified"
            )

    # ---- input mutation / shape contracts ---------------------------------

    def test_original_label_map_not_mutated(self):
        """apply_greenness_fallback must return a new array and not modify
        the input label_map in-place."""
        labels, rgb, raw = self._make()
        labels[0, 0] = TerrainClass.OTHER
        raw[0, 0]    = 1
        rgb[0, 0]    = [50, 120, 40]
        original = labels.copy()
        _ = apply_greenness_fallback(labels, rgb, raw)
        np.testing.assert_array_equal(labels, original)

    def test_output_dtype_and_shape(self):
        """Result must be uint8 with the same H×W shape as the input."""
        labels, rgb, raw = self._make(4, 8)
        result = apply_greenness_fallback(labels, rgb, raw)
        assert result.dtype == np.uint8
        assert result.shape == (4, 8)

    def test_invalid_label_map_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            apply_greenness_fallback(
                np.zeros((2, 3, 1), dtype=np.uint8),
                np.zeros((2, 3, 3), dtype=np.uint8),
                np.zeros((2, 3), dtype=np.int64),
            )

    def test_invalid_rgb_shape_raises(self):
        with pytest.raises(ValueError, match="H × W × 3"):
            apply_greenness_fallback(
                np.zeros((2, 3), dtype=np.uint8),
                np.zeros((2, 3, 4), dtype=np.uint8),   # 4 channels, not 3
                np.zeros((2, 3), dtype=np.int64),
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            apply_greenness_fallback(
                np.zeros((2, 3), dtype=np.uint8),
                np.zeros((2, 3, 3), dtype=np.uint8),
                np.zeros((3, 2), dtype=np.int64),       # wrong shape
            )

    # ---- custom background_class_id and threshold -------------------------

    def test_custom_background_class_id(self):
        """background_class_id parameter must be honoured."""
        labels, rgb, raw = self._make()
        labels[0, 0] = TerrainClass.OTHER
        raw[0, 0]    = 99                         # non-default background id
        rgb[0, 0]    = [30, 100, 20]
        # With default id=1 this should NOT promote
        result_default = apply_greenness_fallback(labels, rgb, raw, background_class_id=1)
        assert result_default[0, 0] == TerrainClass.OTHER
        # With id=99 it should promote
        result_custom = apply_greenness_fallback(labels, rgb, raw, background_class_id=99)
        assert result_custom[0, 0] == TerrainClass.VEGETATION

    def test_custom_green_threshold(self):
        """Raising green_threshold must suppress promotion of marginally green
        pixels; lowering it must allow them."""
        labels, rgb, raw = self._make()
        labels[0, 0] = TerrainClass.OTHER
        raw[0, 0]    = 1
        rgb[0, 0]    = [40, 80, 30]              # G=80, just above 70, below 90
        assert apply_greenness_fallback(labels, rgb, raw, green_threshold=90)[0, 0] == TerrainClass.OTHER
        assert apply_greenness_fallback(labels, rgb, raw, green_threshold=70)[0, 0] == TerrainClass.VEGETATION

    # ---- multi-pixel sanity check -----------------------------------------

    def test_mixed_array_only_eligible_pixels_promoted(self):
        """3×1 array: first pixel eligible, second not greenish, third Barren.
        Only the first should change."""
        labels = np.array([[TerrainClass.OTHER],
                           [TerrainClass.OTHER],
                           [TerrainClass.OTHER]], dtype=np.uint8)
        rgb    = np.array([[[40, 120, 30]],    # greenish → promote
                           [[130,  80, 60]],    # brown → stay
                           [[40,  120, 30]]], dtype=np.uint8)   # green but Barren
        raw    = np.array([[1],                 # Background
                           [1],                 # Background
                           [5]], dtype=np.int64)  # Barren
        result = apply_greenness_fallback(labels, rgb, raw)
        assert result[0, 0] == TerrainClass.VEGETATION
        assert result[1, 0] == TerrainClass.OTHER
        assert result[2, 0] == TerrainClass.OTHER


# ---------------------------------------------------------------------------
# classify_terrain() tests
# ---------------------------------------------------------------------------

import json
import jsonschema


@pytest.fixture(scope="module")
def classify_outputs_119(tmp_path_factory):
    """Run classify_terrain() on india_patch_119.tif once; share across tests."""
    image_path = _require_patch("india_patch_119.tif")
    out_dir = tmp_path_factory.mktemp("classify_119")
    json_path, overlay_path = classify_terrain(
        image_path,
        output_dir=out_dir,
        model_name=_model_name(),
        device="cpu",
    )
    return {
        "image_path": image_path,
        "out_dir":    out_dir,
        "json_path":  json_path,
        "overlay":    overlay_path,
        "labelmap":   out_dir / "india_patch_119_labelmap.png",
    }


class TestClassifyTerrainGeoTIFF:
    """Tests for classify_terrain() on a real GeoTIFF input."""

    # ---- file presence -----------------------------------------------------

    def test_json_file_exists(self, classify_outputs_119):
        assert classify_outputs_119["json_path"].is_file()
        assert classify_outputs_119["json_path"].stat().st_size > 0

    def test_overlay_png_exists(self, classify_outputs_119):
        assert classify_outputs_119["overlay"].is_file()
        assert classify_outputs_119["overlay"].stat().st_size > 0

    def test_labelmap_png_exists(self, classify_outputs_119):
        assert classify_outputs_119["labelmap"].is_file()
        assert classify_outputs_119["labelmap"].stat().st_size > 0

    # ---- file naming convention --------------------------------------------

    def test_output_filenames_match_spec(self, classify_outputs_119):
        stem = "india_patch_119"
        assert classify_outputs_119["json_path"].name    == f"{stem}_data.json"
        assert classify_outputs_119["overlay"].name      == f"{stem}_overlay.png"
        assert classify_outputs_119["labelmap"].name     == f"{stem}_labelmap.png"

    # ---- JSON schema validation --------------------------------------------

    def test_json_validates_against_schema(self, classify_outputs_119):
        with open(classify_outputs_119["json_path"], encoding="utf-8") as fh:
            payload = json.load(fh)
        # Must not raise
        jsonschema.validate(instance=payload, schema=_OUTPUT_SCHEMA)

    def test_json_image_id_matches_filename(self, classify_outputs_119):
        with open(classify_outputs_119["json_path"], encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["image_id"] == "india_patch_119.tif"

    def test_json_dimensions_match_source(self, classify_outputs_119):
        with open(classify_outputs_119["json_path"], encoding="utf-8") as fh:
            payload = json.load(fh)
        with rasterio.open(classify_outputs_119["image_path"]) as src:
            assert payload["image_dimensions"]["width"]  == src.width
            assert payload["image_dimensions"]["height"] == src.height

    def test_json_label_map_path_field_matches_file(self, classify_outputs_119):
        with open(classify_outputs_119["json_path"], encoding="utf-8") as fh:
            payload = json.load(fh)
        lm_name = payload["segmentation"]["label_map_path"]
        assert (classify_outputs_119["out_dir"] / lm_name).is_file()

    def test_json_segmentation_classes_are_valid_names(self, classify_outputs_119):
        with open(classify_outputs_119["json_path"], encoding="utf-8") as fh:
            payload = json.load(fh)
        valid = {tc.name.lower() for tc in TerrainClass}
        for cls in payload["segmentation"]["classes"]:
            assert cls in valid, f"Unexpected class name: {cls!r}"

    def test_json_detections_all_valid(self, classify_outputs_119):
        with open(classify_outputs_119["json_path"], encoding="utf-8") as fh:
            payload = json.load(fh)
        for det in payload["detections"]:
            # Each detection validates individually
            jsonschema.validate(instance=det, schema=_DETECTION_SCHEMA)
            assert det["class"] == "building"
            assert det["instance_id"] >= 1
            x_min, y_min, x_max, y_max = det["bbox"]
            assert x_min <= x_max and y_min <= y_max
            assert 0.0 <= det["confidence"] <= 1.0
            if det.get("mask"):
                assert all(len(pt) == 2 for pt in det["mask"]), \
                    "Each mask point must be [x, y]"

    def test_json_pending_classes_present(self, classify_outputs_119):
        with open(classify_outputs_119["json_path"], encoding="utf-8") as fh:
            payload = json.load(fh)
        pc = payload["pending_classes"]
        # car is now detected via ADE20K when available — no longer pending.
        # tree (fine-grained distinction) is still pending.
        assert "tree" in pc["classes"]
        assert len(pc["reason"]) > 0

    def test_json_detections_instance_ids_unique_and_sequential(self, classify_outputs_119):
        with open(classify_outputs_119["json_path"], encoding="utf-8") as fh:
            payload = json.load(fh)
        ids = [d["instance_id"] for d in payload["detections"]]
        assert ids == list(range(1, len(ids) + 1)), \
            f"Instance IDs must be 1-based sequential, got: {ids}"

    # ---- overlay consistency: every detection bbox visible in overlay ------

    def test_overlay_has_non_black_pixels_at_every_detection_bbox(
        self, classify_outputs_119
    ):
        """Each detection bbox must contain at least one non-black pixel in
        the overlay — the building fill and/or instance outline ensures this."""
        with open(classify_outputs_119["json_path"], encoding="utf-8") as fh:
            payload = json.load(fh)
        overlay_np = np.array(Image.open(classify_outputs_119["overlay"]).convert("RGB"))
        for det in payload["detections"]:
            x_min, y_min, x_max, y_max = [int(v) for v in det["bbox"]]
            region = overlay_np[y_min:y_max + 1, x_min:x_max + 1]
            non_black = np.any(region > 0, axis=2)
            assert non_black.any(), (
                f"Building instance {det['instance_id']} bbox "
                f"[{x_min},{y_min},{x_max},{y_max}] has no visible mark in overlay"
            )

    # ---- label map PNG -----------------------------------------------------

    def test_labelmap_pixel_values_in_range(self, classify_outputs_119):
        lm = np.array(Image.open(classify_outputs_119["labelmap"]))
        valid_ids = {int(tc) for tc in TerrainClass}
        assert set(np.unique(lm).tolist()).issubset(valid_ids), \
            f"Unexpected pixel values in labelmap: {np.unique(lm).tolist()}"

    def test_labelmap_dimensions_match_source(self, classify_outputs_119):
        lm = np.array(Image.open(classify_outputs_119["labelmap"]))
        with rasterio.open(classify_outputs_119["image_path"]) as src:
            assert lm.shape == (src.height, src.width)

    # ---- overlay dimensions ------------------------------------------------

    def test_overlay_dimensions_match_source(self, classify_outputs_119):
        overlay = Image.open(classify_outputs_119["overlay"])
        with rasterio.open(classify_outputs_119["image_path"]) as src:
            assert overlay.width  == src.width
            assert overlay.height == src.height


class TestClassifyTerrainPlainPNG:
    """classify_terrain() on a plain PNG must not write a labels_tif and must
    still produce valid JSON, overlay, and labelmap."""

    def test_plain_png_produces_three_files_no_geotiff(self, tmp_path):
        # Build a plain PNG from the first GeoTIFF band
        src_tif = TEST_DATA / "india_patch_53.tif"
        if not src_tif.is_file():
            pytest.skip("Optional GeoTIFF unavailable")
        with rasterio.open(src_tif) as src:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        png_path = tmp_path / "plain_input.png"
        Image.fromarray(rgb).save(png_path)

        json_path, overlay_path = classify_terrain(
            png_path,
            output_dir=tmp_path / "out",
            model_name=_model_name(),
            device="cpu",
        )
        out_dir = tmp_path / "out"

        # Three expected files
        assert json_path.is_file()
        assert overlay_path.is_file()
        assert (out_dir / "plain_input_labelmap.png").is_file()

        # No GeoTIFF label raster
        assert not (out_dir / "plain_input_terrain_labels.tif").exists()

        # JSON is valid
        with open(json_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        jsonschema.validate(instance=payload, schema=_OUTPUT_SCHEMA)
        assert payload["image_id"] == "plain_input.png"


# ---------------------------------------------------------------------------
# Regression tests: confidence-gated apply_greenness_fallback (issue #1 fix)
# ---------------------------------------------------------------------------

class TestGreennessConfidenceGating:
    """Verify that the confidence gate prevents misclassification of
    car-like (dark/grey) pixels that happen to have a slight greenish cast,
    while still promoting genuinely ambiguous vegetation pixels.

    These tests use synthetic logits so they run fully offline without any
    model or GeoTIFF dependency.
    """

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _make_logits(h: int, w: int, n_classes: int = 8) -> "np.ndarray":
        """Return a (1, n_classes, H, W) float32 zero tensor (uniform = uncertain)."""
        import torch
        return torch.zeros(1, n_classes, h, w, dtype=torch.float32)

    @staticmethod
    def _set_confident(logits, row: int, col: int, class_id: int, value: float = 10.0):
        """Drive one pixel to near-certain confidence for class_id via a high logit."""
        logits[0, class_id, row, col] = value

    # ---- car / dark-green pixel tests --------------------------------------

    def test_car_dark_greenish_not_promoted_when_model_confident(self):
        """A dark-greenish pixel (car roof in shadow) where the model is
        CONFIDENT (Background softmax > ceiling) must NOT become VEGETATION
        after the fix."""
        labels = np.array([[TerrainClass.OTHER]], dtype=np.uint8)
        # slightly greenish dark pixel — passes G>R, G>B, G>70
        rgb    = np.array([[[68, 82, 60]]], dtype=np.uint8)   # G=82, R=68, B=60
        raw    = np.array([[1]], dtype=np.int64)               # Background
        # High logit for background → high softmax confidence (~1.0 after softmax)
        logits = self._make_logits(1, 1)
        self._set_confident(logits, 0, 0, class_id=1, value=12.0)

        result = apply_greenness_fallback(
            labels, rgb, raw,
            background_class_id=1,
            green_threshold=70,
            model_logits_full=logits,
            confidence_ceiling=0.60,
        )
        assert result[0, 0] == TerrainClass.OTHER, (
            "Confident Background pixel (dark car-like colour) must not be "
            "promoted to VEGETATION"
        )

    def test_car_dark_greenish_not_promoted_at_various_grey_tones(self):
        """Multiple dark grey-green tones that resemble car roofs / shadowed
        tarmac — none should be promoted when the model is confident."""
        import torch
        car_colours = [
            [60, 75, 55],   # dark olive-grey
            [70, 82, 65],   # shadowed tarmac near trees
            [55, 72, 50],   # very dark greenish shadow
            [80, 90, 72],   # medium-grey with slight green cast
        ]
        for colour in car_colours:
            h, w = 1, 1
            labels = np.array([[TerrainClass.OTHER]], dtype=np.uint8)
            rgb    = np.array([[colour]], dtype=np.uint8)
            raw    = np.array([[1]], dtype=np.int64)
            logits = self._make_logits(h, w)
            self._set_confident(logits, 0, 0, class_id=1, value=12.0)

            result = apply_greenness_fallback(
                labels, rgb, raw,
                background_class_id=1,
                green_threshold=70,
                model_logits_full=logits,
                confidence_ceiling=0.60,
            )
            assert result[0, 0] == TerrainClass.OTHER, (
                f"Car-like colour {colour} should not be promoted to VEGETATION "
                f"when model is confident"
            )

    # ---- genuine vegetation still promoted when model uncertain ------------

    def test_genuine_vegetation_promoted_when_model_uncertain(self):
        """A clearly green pixel where the model is UNCERTAIN (uniform logits →
        low softmax confidence) MUST still be promoted to VEGETATION."""
        labels = np.array([[TerrainClass.OTHER]], dtype=np.uint8)
        rgb    = np.array([[[50, 130, 40]]], dtype=np.uint8)   # bright green
        raw    = np.array([[1]], dtype=np.int64)                # Background
        # Uniform logits → softmax ≈ 1/8 = 0.125 per class → well below ceiling
        logits = self._make_logits(1, 1)

        result = apply_greenness_fallback(
            labels, rgb, raw,
            background_class_id=1,
            green_threshold=70,
            model_logits_full=logits,
            confidence_ceiling=0.60,
        )
        assert result[0, 0] == TerrainClass.VEGETATION, (
            "Genuinely green pixel where model is uncertain must be promoted"
        )

    def test_borderline_confidence_at_exact_ceiling_not_promoted(self):
        """A pixel whose max softmax probability equals the ceiling exactly
        must NOT be promoted (strictly less-than comparison)."""
        import torch, math
        labels = np.array([[TerrainClass.OTHER]], dtype=np.uint8)
        rgb    = np.array([[[50, 130, 40]]], dtype=np.uint8)
        raw    = np.array([[1]], dtype=np.int64)

        # Craft logits so Background softmax is exactly 0.60.
        # For 8 classes all-zero except background=x:
        #   softmax = e^x / (e^x + 7) = 0.60  →  e^x = 10.5  →  x = ln(10.5)
        exact_logit = math.log(10.5)   # ≈ 2.351375 → softmax ≈ 0.600000
        logits = self._make_logits(1, 1)
        logits[0, 1, 0, 0] = exact_logit

        result = apply_greenness_fallback(
            labels, rgb, raw,
            background_class_id=1,
            green_threshold=70,
            model_logits_full=logits,
            confidence_ceiling=0.60,
        )
        # softmax == 0.60 is NOT strictly less than ceiling → must NOT promote
        assert result[0, 0] == TerrainClass.OTHER

    def test_just_below_ceiling_is_promoted(self):
        """A pixel with max confidence just below the ceiling MUST be promoted."""
        import torch
        labels = np.array([[TerrainClass.OTHER]], dtype=np.uint8)
        rgb    = np.array([[[50, 130, 40]]], dtype=np.uint8)
        raw    = np.array([[1]], dtype=np.int64)

        # Background logit giving softmax ~0.50 (below 0.60 ceiling)
        # e^x/(e^x+7)=0.50 → e^x=7 → x=ln(7)≈1.946
        logits = self._make_logits(1, 1)
        logits[0, 1, 0, 0] = 1.946

        result = apply_greenness_fallback(
            labels, rgb, raw,
            background_class_id=1,
            green_threshold=70,
            model_logits_full=logits,
            confidence_ceiling=0.60,
        )
        assert result[0, 0] == TerrainClass.VEGETATION

    # ---- no logits supplied → backward-compatible colour-only behaviour ----

    def test_no_logits_still_works_colour_only(self):
        """When model_logits_full=None the confidence gate is disabled and
        the function behaves exactly as before the fix."""
        labels = np.array([[TerrainClass.OTHER]], dtype=np.uint8)
        rgb    = np.array([[[50, 130, 40]]], dtype=np.uint8)
        raw    = np.array([[1]], dtype=np.int64)

        result = apply_greenness_fallback(
            labels, rgb, raw,
            background_class_id=1,
            green_threshold=70,
            model_logits_full=None,   # no gating
        )
        assert result[0, 0] == TerrainClass.VEGETATION

    def test_no_logits_car_pixel_still_promoted_colour_only(self):
        """Without confidence gating a dark-greenish car pixel IS promoted —
        this documents the pre-fix behaviour and confirms the gate is necessary."""
        labels = np.array([[TerrainClass.OTHER]], dtype=np.uint8)
        rgb    = np.array([[[68, 82, 60]]], dtype=np.uint8)   # car-like colour
        raw    = np.array([[1]], dtype=np.int64)

        result = apply_greenness_fallback(
            labels, rgb, raw,
            background_class_id=1,
            green_threshold=70,
            model_logits_full=None,   # old behaviour
        )
        # Without the gate this IS promoted (the pre-fix bug)
        assert result[0, 0] == TerrainClass.VEGETATION, (
            "Without confidence gating the car pixel IS falsely promoted "
            "(documents the pre-fix bug)"
        )

    # ---- mixed array: car next to vegetation --------------------------------

    def test_mixed_scene_car_beside_vegetation(self):
        """2-pixel array: pixel 0 = car (confident Background), pixel 1 = real
        vegetation (uncertain Background).  Only pixel 1 must be promoted."""
        import torch
        labels = np.array([[TerrainClass.OTHER, TerrainClass.OTHER]], dtype=np.uint8)
        rgb    = np.array([[[68, 82, 60],   # car-like: slightly green, dark
                             [40, 140, 30]]], dtype=np.uint8)  # clearly green
        raw    = np.array([[1, 1]], dtype=np.int64)            # both Background

        logits = self._make_logits(1, 2)
        # pixel 0: confident Background
        self._set_confident(logits, 0, 0, class_id=1, value=12.0)
        # pixel 1: uniform logits (uncertain)

        result = apply_greenness_fallback(
            labels, rgb, raw,
            background_class_id=1,
            green_threshold=70,
            model_logits_full=logits,
            confidence_ceiling=0.60,
        )
        assert result[0, 0] == TerrainClass.OTHER,       "Car pixel must stay OTHER"
        assert result[0, 1] == TerrainClass.VEGETATION,  "Vegetation pixel must be promoted"


# ---------------------------------------------------------------------------
# visualize_confidence_heatmap tests (offline, no model required)
# ---------------------------------------------------------------------------

class TestVisualizeConfidenceHeatmap:
    """Smoke-test the function signature and output for invalid inputs."""

    def test_invalid_class_name_raises(self, tmp_path):
        """Requesting a non-existent class name must raise ValueError."""
        src = _require_patch("india_patch_53.tif")
        with pytest.raises(ValueError, match="Unknown class_name"):
            visualize_confidence_heatmap(
                src,
                class_name="unicorn",
                output_dir=tmp_path,
                model_name=_model_name(),
                device="cpu",
            )

    def test_valid_class_produces_png(self, tmp_path):
        """A valid class name must write a PNG at the expected path."""
        src = _require_patch("india_patch_53.tif")
        out = visualize_confidence_heatmap(
            src,
            class_name="Building",
            output_dir=tmp_path,
            model_name=_model_name(),
            device="cpu",
        )
        assert out.is_file()
        assert out.stat().st_size > 0
        assert out.name == "india_patch_53_confidence_building.png"

