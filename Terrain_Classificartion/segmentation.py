"""Pretrained terrain classification for the GeoMonoDSM-3D pipeline.

This stage labels the original RGB image independently of depth estimation:

    terrain_label_map, terrain_overlay = segment("satellite_image.tif")

The model is a LoveDA-finetuned SegFormer trained for remote-sensing land
cover.  It is not trained by this project.  Labels are uint8 values: other
(0), vegetation (1), building (2), road (3), and water (4).

At the supplied 10 m/pixel resolution, many Indian roads (typically 3--7 m
wide) are sub-pixel or only a few pixels wide.  Road predictions are therefore
useful as semantic context for the demo, but must not be presented as accurate
road mapping.

For the final pipeline handoff use ``classify_terrain(image_path)``, which
returns ``(json_path, overlay_png_path)`` and writes three files:
  - ``{stem}_data.json``       structured output for Role 4's mesh script
  - ``{stem}_overlay.png``     human-readable visualisation
  - ``{stem}_labelmap.png``    raw per-pixel class-ID image for texturing

# ---------------------------------------------------------------------------
# DOMAIN SHIFT DIAGNOSTIC NOTE — test_1.jpeg (Mediterranean dense urban)
# Model: wu-pr-gw/segformer-b2-finetuned-with-LoveDA
# Run: 2026-09-11  via visualize_confidence_heatmap()
# ---------------------------------------------------------------------------
#
# BUILDING (channel 2) — Max: 0.963  Mean: 0.229  >50%: 13.1%
#   Real signal exists on large flat-roof courtyard blocks. Misses all small
#   pitched terracotta rooftops (out-of-distribution for LoveDA, which trained
#   on flat Chinese urban roofs). Our warmth fallback recovers the obviously
#   orange ones. Further improvement requires a checkpoint trained on
#   European aerial imagery (e.g. ISPRS Potsdam/Vaihingen).
#   VERDICT: Missed buildings = domain shift, not a pipeline bug.
#
# WATER (channel 4) — Max: 0.004  Mean: 0.000  >50%: 0.0%
#   Absolute zero — model never considers water for any pixel in this scene.
#   Any water-coloured pixels in the overlay come from post-processing, not
#   the model. Not fixable by threshold tuning. Zero signal means no amount
#   of threshold adjustment will produce correct water predictions.
#   VERDICT: If blue pixels appear in the overlay, investigate the pipeline,
#   not the checkpoint.
#
# ROAD (channel 3) — Max: 0.032  Mean: 0.000  >50%: 0.0%
#   Effectively zero everywhere. Mediterranean streets are narrow, tree-lined,
#   partially occluded — visually unlike LoveDA's wide Chinese arterials.
#   VERDICT: Pure domain shift, not tunable. Requires a different checkpoint.
#
# BACKGROUND (channel 1) — Max: 0.995  Mean: 0.771  >50%: 86.9%
#   Model is extremely confident "Background" for 86.9% of all pixels. This
#   is the defining signature of severe domain shift: the catch-all class
#   absorbs everything the model has not seen before. The confidence gate in
#   apply_greenness_fallback (ceiling=0.60) correctly suppresses vegetation
#   promotions on these highly-confident Background pixels, leaving only
#   genuinely uncertain pixels eligible for the heuristic fallback.
# ---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


class TerrainClass(IntEnum):
    """Stable label values shared with mesh/integration stages."""

    OTHER = 0
    VEGETATION = 1
    BUILDING = 2
    ROAD = 3
    WATER = 4


DEFAULT_MODEL = "models/loveda-segformer"

# RGB values used by the human-readable preview.  The transparent overlay uses
# these same colours, letting the frontend place it over the original texture.
TERRAIN_COLORS: dict[TerrainClass, tuple[int, int, int]] = {
    TerrainClass.OTHER: (0, 0, 0),
    TerrainClass.VEGETATION: (47, 158, 68),
    TerrainClass.BUILDING: (214, 57, 57),
    TerrainClass.ROAD: (125, 125, 125),
    TerrainClass.WATER: (50, 120, 220),
}


@dataclass(frozen=True)
class GeoReference:
    """The georeferencing fields that must survive label-raster export."""

    crs: Any
    transform: Any


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a numeric raster into displayable RGB bytes, like Part 1."""
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.dtype == np.uint8:
        return image
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _load_image(image_path: str | Path) -> tuple[Image.Image, GeoReference | None]:
    """Load RGB pixels and retain GeoTIFF CRS/transform when they exist.

    GeoTIFFs use Rasterio rather than Pillow so their map metadata remains
    available for the output label raster.  PNG/JPG inputs intentionally have
    no georeference and use Pillow.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")

    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError(
                "GeoTIFF terrain classification needs Rasterio. Run "
                "`pip install -r terrain_classification/requirements.txt`."
            ) from exc

        with rasterio.open(path) as source:
            band_count = min(source.count, 3)
            if band_count == 0:
                raise ValueError(f"GeoTIFF has no raster bands: {path}")
            data = source.read(list(range(1, band_count + 1))).astype(np.float32)
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            elif data.shape[0] == 2:
                data = np.concatenate([data, data[:1]], axis=0)
            rgb = _to_uint8(np.moveaxis(data, 0, -1))
            georeference = GeoReference(source.crs, source.transform)
        return Image.fromarray(rgb, mode="RGB"), georeference

    with Image.open(path) as source:
        return source.convert("RGB"), None


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str | None):
    """Load and cache the pretrained remote-sensing model on first use."""
    try:
        import torch
        from transformers import (
            AutoImageProcessor,
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Terrain classification needs its optional dependencies. Run "
            "`pip install -r terrain_classification/requirements.txt`."
        ) from exc

    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
    except ValueError:
        # This checkpoint stores the legacy SegformerFeatureExtractor name;
        # the current image processor is compatible with its configuration.
        processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Use all logical CPU threads for inference.  Benchmarked on an Intel
    # Core 7 hybrid (P+E cores, 16 logical threads): os.cpu_count() is 9 %
    # faster than PyTorch's default of 10 and faster than any P-core-only
    # subset — E-cores contribute meaningfully to transformer inference.
    # This is a no-op when running on CUDA.
    torch.set_num_threads(os.cpu_count() or 1)
    model.to(resolved_device).eval()
    return processor, model, torch, resolved_device


def _terrain_for_model_label(label: str) -> TerrainClass:
    """Map LoveDA classes into the five stable GeoMonoDSM terrain classes."""
    normalised = label.lower().replace("_", " ").replace("-", " ")
    if "water" in normalised:
        return TerrainClass.WATER
    if "building" in normalised:
        return TerrainClass.BUILDING
    if "road" in normalised:
        return TerrainClass.ROAD
    if any(keyword in normalised for keyword in ("forest", "agricultur", "crop", "farmland")):
        return TerrainClass.VEGETATION
    # LoveDA Barren and Background intentionally remain OTHER.
    return TerrainClass.OTHER


def collapse_model_labels(model_labels: np.ndarray, id2label: dict[Any, str]) -> np.ndarray:
    """Collapse model-specific labels into a uint8 GeoMonoDSM label map."""
    if model_labels.ndim != 2:
        raise ValueError("model_labels must be a 2D array")
    result = np.zeros(model_labels.shape, dtype=np.uint8)
    for raw_id in np.unique(model_labels):
        label = id2label.get(int(raw_id), id2label.get(str(int(raw_id)), ""))
        result[model_labels == raw_id] = _terrain_for_model_label(str(label))
    return result


def apply_greenness_fallback(
    label_map: np.ndarray,
    rgb: np.ndarray,
    raw_model_labels: np.ndarray,
    background_class_id: int = 1,
    green_threshold: int = 70,
    model_logits_full: Any = None,
    confidence_ceiling: float = 0.60,
) -> np.ndarray:
    """Recover vegetation pixels lost to LoveDA's "Background" class.

    Context
    -------
    LoveDA was trained on Chinese land-cover imagery.  When applied to Indian
    patches it never predicts Forest (6) or Agricultural (7) — visually green
    farmland and scrub are absorbed into class 1 (Background), which
    ``collapse_model_labels`` then maps to OTHER (0).  This post-processing
    step recovers those pixels using an RGB greenness heuristic combined with
    a model-confidence gate on the original image.

    What it does
    ------------
    For every pixel that satisfies ALL of:
      1. currently labeled OTHER (0) in ``label_map``, AND
      2. was originally predicted as *Background* by the model
         (not Barren, Building, Road, or any other class), AND
      3. visually greenish in the source RGB:
            G > R  and  G > B  and  G > ``green_threshold``
      4. (when ``model_logits_full`` is supplied) the model's softmax
         confidence for its Background prediction is BELOW
         ``confidence_ceiling`` — i.e. the model was uncertain.

    … the pixel is reclassified as VEGETATION (1).

    Why confidence gating matters
    ------------------------------
    Without gating, the colour heuristic alone can promote dark car roofs or
    shadowed tarmac that happen to have a slight greenish cast (e.g. G=75,
    R=70, B=65 — passes G>70 but is plainly not vegetation).  When the model
    assigned that pixel Background with high softmax confidence (say 0.82) it
    had a real signal worth respecting.  The confidence gate suppresses those
    promotions, restricting the fallback to genuinely ambiguous pixels where
    the model itself was unsure.

    Pixels the model confidently called Barren, Building, Road, or Water are
    left untouched — only the ambiguous Background→OTHER pixels are eligible.

    Parameters
    ----------
    label_map:
        2-D uint8 array produced by ``collapse_model_labels``.
    rgb:
        H × W × 3 uint8 NumPy array of the original source image (the same
        pixels fed to the model).
    raw_model_labels:
        H × W int64/int array of the model's per-pixel argmax *before*
        collapsing — i.e., the LoveDA class IDs (0–7).
    background_class_id:
        LoveDA class index for "Background".  Default 1 matches the
        ``wu-pr-gw/segformer-b2-finetuned-with-LoveDA`` checkpoint.
    green_threshold:
        Minimum green-channel value (0–255) a pixel must have, in addition
        to G > R and G > B, to be considered vegetation.  Tuned empirically
        to 70: at this value the function recovers ~68 % of Background pixels
        in a vegetation-heavy Indian farmland patch while falsely promoting
        only ~8 % on a known-arid patch.
    model_logits_full:
        Optional torch.Tensor of shape (1, C, H, W) — the full-resolution
        upsampled logits from the model (before argmax).  When supplied,
        enables confidence gating: only pixels whose softmax max-probability
        is below ``confidence_ceiling`` are eligible for promotion.  Pass
        ``None`` (default) to disable confidence gating and use colour-only
        logic (backward-compatible behaviour).
    confidence_ceiling:
        Maximum softmax confidence the model may have for its prediction
        before the pixel becomes ineligible for promotion.  Only used when
        ``model_logits_full`` is provided.  Default 0.60: pixels where the
        model scored its Background prediction above 60 % are left as OTHER
        rather than overridden by the colour heuristic.

    Returns
    -------
    np.ndarray
        A *new* uint8 2-D array — ``label_map`` is not mutated.

    Notes
    -----
    This function is intentionally isolated and *not* called from
    ``segment()``.  Wire it in (or not) at the call site after reviewing
    before/after quality.
    """
    if label_map.ndim != 2:
        raise ValueError("label_map must be a 2-D array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an H × W × 3 array")
    if raw_model_labels.shape != label_map.shape:
        raise ValueError("raw_model_labels must have the same shape as label_map")

    R = rgb[:, :, 0].astype(np.int16)
    G = rgb[:, :, 1].astype(np.int16)
    B = rgb[:, :, 2].astype(np.int16)

    is_other      = label_map == TerrainClass.OTHER
    is_background = raw_model_labels == background_class_id
    is_greenish   = (G > R) & (G > B) & (G > green_threshold)

    eligible = is_other & is_background & is_greenish

    # Confidence gate: suppress promotions where the model was already sure.
    if model_logits_full is not None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Confidence gating requires torch.") from exc
        with torch.no_grad():
            # softmax over class dimension → max prob per pixel
            probs    = torch.nn.functional.softmax(model_logits_full[0], dim=0)
            max_prob = probs.max(dim=0).values.cpu().numpy()   # (H, W)
        is_uncertain = max_prob < confidence_ceiling
        eligible = eligible & is_uncertain

    result = label_map.copy()
    result[eligible] = TerrainClass.VEGETATION
    return result


def apply_warmth_fallback(
    label_map: np.ndarray,
    rgb: np.ndarray,
    raw_model_labels: np.ndarray,
    background_class_id: int = 1,
    barren_class_id: int = 5,
    red_over_green: int = 30,
    brightness_floor: int = 120,
) -> np.ndarray:
    """Recover building pixels lost to LoveDA's "Background" or "Barren" classes.

    Context
    -------
    LoveDA was trained on Chinese/Indian land cover.  On high-resolution
    urban aerial imagery (e.g. European cities with terracotta rooftops) the
    model has no equivalent training example and collapses warm-coloured
    rooftops into "Background" or "Barren", both of which map to OTHER (0).
    This post-processing step recovers those pixels using a warmth + brightness
    heuristic on the original RGB.

    What it does
    ------------
    For every pixel that satisfies ALL of:
      1. currently labeled OTHER (0) in ``label_map``, AND
      2. was originally predicted as *Background* OR *Barren* by the model, AND
      3. visually warm AND bright in the source RGB:
            R > G + ``red_over_green``
            R > B + ``red_over_green``
            R > ``brightness_floor``

    … the pixel is reclassified as BUILDING (2).

    NOTE — false-positive risk on arid/barren terrain
    -------------------------------------------------
    Sandy and dry barren land is also warm-toned.  The ``brightness_floor``
    (default 120) filters out most dim barren patches, but dense sunlit barren
    terrain at high resolution can still trigger this heuristic.  For GeoTIFF
    patches known to contain genuine barren land (e.g. Indian desert patches),
    either skip this fallback or raise ``red_over_green`` to ≥ 40.

    Parameters
    ----------
    label_map:
        2-D uint8 array from ``collapse_model_labels``.
    rgb:
        H × W × 3 uint8 source image array.
    raw_model_labels:
        H × W model argmax array before collapsing.
    background_class_id:
        LoveDA "Background" class index (default 1).
    barren_class_id:
        LoveDA "Barren" class index (default 5).  Also eligible because
        the model sometimes calls warm rooftops "Barren".
    red_over_green:
        How much redder than green/blue the pixel must be (default 30).
    brightness_floor:
        Minimum R channel value — filters dim warm colours that are more
        likely barren land than bright rooftops (default 120).

    Returns
    -------
    np.ndarray
        A *new* uint8 2-D array — ``label_map`` is not mutated.
    """
    if label_map.ndim != 2:
        raise ValueError("label_map must be a 2-D array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an H × W × 3 array")
    if raw_model_labels.shape != label_map.shape:
        raise ValueError("raw_model_labels must have the same shape as label_map")

    R = rgb[:, :, 0].astype(np.int16)
    G = rgb[:, :, 1].astype(np.int16)
    B = rgb[:, :, 2].astype(np.int16)

    is_other   = label_map == TerrainClass.OTHER
    is_eligible = (raw_model_labels == background_class_id) | (raw_model_labels == barren_class_id)
    is_warm    = (R > G + red_over_green) & (R > B + red_over_green) & (R > brightness_floor)

    result = label_map.copy()
    result[is_other & is_eligible & is_warm] = TerrainClass.BUILDING
    return result


def apply_brightness_fallback(
    label_map: np.ndarray,
    rgb: np.ndarray,
    raw_model_labels: np.ndarray,
    background_class_id: int = 1,
    barren_class_id: int = 5,
    brightness_floor: int = 160,
    max_saturation: int = 30,
    model_logits_full: Any = None,
    confidence_ceiling: float = 0.75,
) -> np.ndarray:
    """Recover white/grey flat-roof buildings missed by apply_warmth_fallback.

    Context
    -------
    ``apply_warmth_fallback`` catches warm-toned (terracotta/orange) rooftops.
    But many buildings — especially modern flat-roof structures, concrete
    blocks, and solar-panel rooftops — are white, light-grey, or neutral.
    These have high R, G, and B values with little colour bias, so the warmth
    heuristic (R > G+30) never fires on them.  This fallback catches them.

    This is a general fix, not city-specific: the same white/grey flat-roof
    pattern appears in industrial buildings, warehouses, and large structures
    across every terrain type.

    What it does
    ------------
    For every pixel that satisfies ALL of:
      1. currently labeled OTHER (0) in ``label_map``, AND
      2. was originally predicted as Background OR Barren by the model, AND
      3. bright and achromatic in the source RGB:
            R > ``brightness_floor``  (bright enough to be a light surface)
            max(R,G,B) - min(R,G,B) <= ``max_saturation``  (nearly grey/white)
      4. (when ``model_logits_full`` is supplied) model confidence below
         ``confidence_ceiling`` — suppresses high-confidence Background pixels

    … the pixel is reclassified as BUILDING (2).

    Parameters
    ----------
    label_map:
        2-D uint8 array from ``collapse_model_labels``.
    rgb:
        H × W × 3 uint8 source image array.
    raw_model_labels:
        H × W model argmax array before collapsing.
    background_class_id:
        LoveDA "Background" class index (default 1).
    barren_class_id:
        LoveDA "Barren" class index (default 5).
    brightness_floor:
        Minimum value of R channel for a pixel to be considered a bright
        surface (default 160).  Filters out dark grey roads and shadows.
    max_saturation:
        Maximum colour range (max - min across R,G,B) allowed.  Keeps only
        near-neutral pixels (default 30).  A terracotta roof has saturation
        ~80+, so this neatly excludes warm rooftops already handled by
        apply_warmth_fallback.
    model_logits_full:
        Optional (1, C, H, W) logits tensor.  When supplied, enables
        confidence gating — suppresses promotions where the model was already
        confident (above ``confidence_ceiling``).
    confidence_ceiling:
        Softmax confidence threshold for gating (default 0.75 — slightly
        more permissive than greenness fallback because grey pixels are more
        ambiguous than obviously green ones).

    Returns
    -------
    np.ndarray
        A *new* uint8 2-D array — ``label_map`` is not mutated.
    """
    if label_map.ndim != 2:
        raise ValueError("label_map must be a 2-D array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an H × W × 3 array")
    if raw_model_labels.shape != label_map.shape:
        raise ValueError("raw_model_labels must have the same shape as label_map")

    R = rgb[:, :, 0].astype(np.int16)
    G = rgb[:, :, 1].astype(np.int16)
    B = rgb[:, :, 2].astype(np.int16)

    sat = np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)

    is_other    = label_map == TerrainClass.OTHER
    is_eligible = (raw_model_labels == background_class_id) | (raw_model_labels == barren_class_id)
    is_bright_grey = (R > brightness_floor) & (sat <= max_saturation)

    eligible = is_other & is_eligible & is_bright_grey

    # Confidence gate: don't override pixels the model was already sure about.
    if model_logits_full is not None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Confidence gating requires torch.") from exc
        with torch.no_grad():
            probs    = torch.nn.functional.softmax(model_logits_full[0], dim=0)
            max_prob = probs.max(dim=0).values.cpu().numpy()
        eligible = eligible & (max_prob < confidence_ceiling)

    # Morphological closing: fill small gaps (shadows, skylights, edges).
    # Kernel size 3 closes 1-pixel gaps without merging separate buildings.
    from scipy.ndimage import binary_closing, label as cc_label, find_objects, sum as ndi_sum
    eligible_closed = binary_closing(eligible, structure=np.ones((3, 3), dtype=bool))
    eligible_closed = eligible_closed & is_other & is_eligible

    # Shape filter: reject road-like and oversized blobs.
    # Uses find_objects() for O(n_blobs) vectorised bounding-box lookup —
    # safe for large images where the blob loop would be catastrophically slow.
    labeled, n_features = cc_label(eligible_closed)
    if n_features == 0:
        return label_map.copy()

    MAX_ASPECT = 5.0
    MAX_AREA   = 40000
    MIN_AREA   = 16

    # Compute per-blob areas in one pass using ndi_sum
    blob_ids  = np.arange(1, n_features + 1)
    areas     = ndi_sum(eligible_closed, labeled, blob_ids)

    # find_objects returns a list of slice tuples for each blob's bounding box
    slices = find_objects(labeled)

    keep_mask = np.zeros(n_features + 1, dtype=bool)  # index 0 = background
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        blob_id = i + 1
        area    = int(areas[i])
        if area < MIN_AREA or area > MAX_AREA:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        aspect = max(h, w) / max(min(h, w), 1)
        if aspect > MAX_ASPECT:
            continue
        keep_mask[blob_id] = True

    filtered = keep_mask[labeled]  # (H, W) bool — vectorised lookup

    result = label_map.copy()
    result[filtered] = TerrainClass.BUILDING
    return result


def make_overlay(label_map: np.ndarray) -> Image.Image:
    """Return an RGBA overlay: terrain colours, transparent OTHER pixels."""
    if label_map.ndim != 2:
        raise ValueError("label_map must be a 2D array")
    overlay = np.zeros((*label_map.shape, 4), dtype=np.uint8)
    for terrain, colour in TERRAIN_COLORS.items():
        if terrain == TerrainClass.OTHER:
            continue
        mask = label_map == terrain
        overlay[mask, :3] = colour
        overlay[mask, 3] = 255
    return Image.fromarray(overlay, mode="RGBA")


def make_preview(label_map: np.ndarray) -> Image.Image:
    """Return an RGB class-map preview with black OTHER pixels."""
    if label_map.ndim != 2:
        raise ValueError("label_map must be a 2D array")
    preview = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for terrain, colour in TERRAIN_COLORS.items():
        preview[label_map == terrain] = colour
    return Image.fromarray(preview, mode="RGB")


def segment(
    image_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
) -> tuple[np.ndarray, Image.Image]:
    """Return ``(terrain_label_map, terrain_overlay)`` for an image path.

    The map is an H x W uint8 array with values 0--4 and exactly matches the
    source image dimensions.  The returned overlay is an RGBA PIL image with
    transparent OTHER pixels, ready for frontend composition.
    """
    rgb_image, _ = _load_image(image_path)
    processor, model, torch, resolved_device = _load_model(str(model_name), device)
    inputs = processor(images=rgb_image, return_tensors="pt")
    inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = torch.nn.functional.interpolate(
        logits,
        size=(rgb_image.height, rgb_image.width),
        mode="bilinear",
        align_corners=False,
    )
    model_labels = logits.argmax(dim=1)[0].cpu().numpy()
    labels = collapse_model_labels(model_labels, model.config.id2label)
    return labels, make_overlay(labels)


def save_segmentation_outputs(
    image_path: str | Path,
    terrain_label_map: np.ndarray,
    terrain_overlay: Image.Image,
    output_dir: str | Path = "outputs",
) -> dict[str, Path]:
    """Save Part 3 outputs and retain exact GeoTIFF georeferencing when present."""
    source_path = Path(image_path)
    _, georeference = _load_image(source_path)
    if terrain_label_map.dtype != np.uint8 or terrain_label_map.ndim != 2:
        raise ValueError("terrain_label_map must be a 2D uint8 array")
    expected_size = (terrain_label_map.shape[1], terrain_label_map.shape[0])
    if terrain_overlay.size != expected_size:
        raise ValueError("terrain_overlay dimensions must match terrain_label_map")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    paths = {
        "labels_npy": destination / f"{stem}_terrain_labels.npy",
        "overlay_png": destination / f"{stem}_terrain_overlay.png",
        "preview_png": destination / f"{stem}_terrain_preview.png",
    }
    np.save(paths["labels_npy"], terrain_label_map)
    terrain_overlay.save(paths["overlay_png"])
    make_preview(terrain_label_map).save(paths["preview_png"])

    if georeference is not None:
        import rasterio

        geotiff_path = destination / f"{stem}_terrain_labels.tif"
        with rasterio.open(
            geotiff_path,
            "w",
            driver="GTiff",
            height=terrain_label_map.shape[0],
            width=terrain_label_map.shape[1],
            count=1,
            dtype="uint8",
            crs=georeference.crs,
            transform=georeference.transform,
            compress="lzw",
        ) as output:
            output.write(terrain_label_map, 1)
        paths["labels_tif"] = geotiff_path

    return paths


# ---------------------------------------------------------------------------
# classify_terrain() — final pipeline handoff output
# ---------------------------------------------------------------------------

# JSON Schema for the _data.json output.  Validated before every write so
# callers receive an immediate error rather than a silently malformed file.
_DETECTION_SCHEMA: dict = {
    "type": "object",
    "required": ["class", "instance_id", "bbox"],
    "additionalProperties": False,
    "properties": {
        "class":       {"type": "string"},
        "instance_id": {"type": "integer", "minimum": 1},
        "bbox":        {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
        "mask": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "required": ["image_id", "image_dimensions", "segmentation", "detections",
                 "pending_classes"],
    "additionalProperties": False,
    "properties": {
        "image_id": {"type": "string"},
        "image_dimensions": {
            "type": "object",
            "required": ["width", "height"],
            "additionalProperties": False,
            "properties": {
                "width":  {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
            },
        },
        "segmentation": {
            "type": "object",
            "required": ["classes", "label_map_path"],
            "additionalProperties": False,
            "properties": {
                "classes":        {"type": "array", "items": {"type": "string"}},
                "label_map_path": {"type": "string"},
            },
        },
        "detections": {
            "type": "array",
            "items": _DETECTION_SCHEMA,
        },
        "pending_classes": {
            "type": "object",
            "required": ["classes", "reason"],
            "additionalProperties": False,
            "properties": {
                "classes": {"type": "array", "items": {"type": "string"}},
                "reason":  {"type": "string"},
            },
        },
    },
}

# Rotating palette for per-instance building outlines in the enriched overlay.
# Chosen to be mutually distinguishable and clearly separate from the red fill
# used by TERRAIN_COLORS[BUILDING].
_INSTANCE_OUTLINE_COLORS: list[tuple[int, int, int]] = [
    (255, 200,   0),   # yellow
    (0,   200, 255),   # cyan
    (200,   0, 255),   # violet
    (255, 128,   0),   # orange
    (0,   255, 128),   # mint
    (255,   0, 128),   # hot-pink
    (128, 255,   0),   # lime
    (0,   128, 255),   # sky-blue
]

# Minimum connected-component area (pixels) to emit as a building instance.
# Below this threshold the blob is almost certainly noise at 10 m/px.
_MIN_BUILDING_AREA_PX: int = 100  # ~9m² at 0.30m/px; filters noise fragments


def _extract_building_instances(
    label_map: np.ndarray,
    logits_full: Any,  # torch.Tensor  (1, C, H, W)  before argmax
    id2label: dict | None = None,
) -> list[dict]:
    """Return a list of building-instance dicts for the JSON detections array.

    Strategy
    --------
    Connected-component labeling (scipy.ndimage) on the building mask gives
    one component per contiguous blob.  Each blob becomes one instance.

    KNOWN LIMITATION: touching or adjacent buildings are merged into a single
    instance because we use semantic segmentation only.  A proper instance
    boundary requires Mask R-CNN / SpaceNet-trained instance segmentation —
    flagged as a follow-up task (swap in once base pipeline is end-to-end).

    Confidence
    ----------
    We use the mean softmax probability for the building class over the
    instance mask, converted to a float rounded to 4 d.p.  This is a
    reasonable proxy for detection confidence but is NOT equivalent to
    YOLO-style objectness scores.

    Polygon mask
    ------------
    cv2.findContours extracts the outer boundary of each blob as a polygon.
    The polygon is simplified (cv2.approxPolyDP, epsilon=1.5 px) to keep
    JSON size manageable for large buildings.  Points are stored as
    [[x, y], ...] integer pairs — same coordinate system as the image
    (origin top-left).
    """
    try:
        import cv2
        import torch
        from scipy.ndimage import label as cc_label
    except ImportError as exc:
        raise RuntimeError(
            "classify_terrain needs scipy and opencv-python. "
            "Run `pip install -r terrain_classification/requirements.txt`."
        ) from exc

    building_id = int(TerrainClass.BUILDING)
    building_mask = (label_map == building_id).astype(np.uint8)

    # Morphological dilation: expand the building mask by a few pixels so
    # nearby fragments of the same roof (separated by shadow lines, skylights,
    # or tile-boundary gaps) merge into one connected blob.  Erode back after
    # labeling so individual building footprints stay accurate.
    # Kernel 5×5 at 0.30m/px bridges gaps up to ~1.5m — within one rooftop.
    from scipy.ndimage import label as cc_label, binary_dilation, binary_erosion
    dilated = binary_dilation(building_mask, structure=np.ones((5, 5), dtype=bool))
    labeled_array, num_features = cc_label(dilated)
    # Map dilation labels back to original (un-dilated) building pixels
    labeled_array = labeled_array * building_mask

    # Softmax probabilities for the building channel over the full image.
    # Resolve the building channel dynamically from id2label so this remains
    # correct if the checkpoint is swapped for one with different channel order.
    # Assert it matches TerrainClass.BUILDING so any mismatch is caught
    # immediately rather than silently reading the wrong confidence channel.
    with torch.no_grad():
        probs = torch.nn.functional.softmax(logits_full[0], dim=0)  # (C, H, W)

    if id2label is not None:
        candidates = [
            k for k, v in id2label.items()
            if _terrain_for_model_label(str(v)) == TerrainClass.BUILDING
        ]
        if len(candidates) == 1:
            building_channel = int(candidates[0])
            assert building_channel == building_id, (
                f"Checkpoint id2label maps Building to channel {building_channel} "
                f"but TerrainClass.BUILDING == {building_id}. "
                "Update TerrainClass or the channel lookup if the checkpoint changed."
            )
        else:
            # Multiple or zero matches — fall back to TerrainClass value with a warning
            building_channel = building_id
    else:
        # No id2label supplied — use TerrainClass value (original behaviour)
        building_channel = building_id  # TerrainClass.BUILDING == 2 == LoveDA Building id

    building_prob = probs[building_channel].cpu().numpy()  # (H, W)

    # Use find_objects() for O(n_blobs) bounding-box lookup — safe for large
    # images with many blobs where a pixel-mask loop would be catastrophically slow.
    from scipy.ndimage import find_objects, sum as ndi_sum
    blob_slices = find_objects(labeled_array)
    blob_ids    = np.arange(1, num_features + 1)
    areas       = ndi_sum(building_mask, labeled_array, blob_ids)

    instances = []
    for i, sl in enumerate(blob_slices):
        if sl is None:
            continue
        area = int(areas[i])
        if area < _MIN_BUILDING_AREA_PX:
            continue

        blob_id = i + 1
        # Crop to bounding-box slice for fast per-blob ops
        sl_label = labeled_array[sl[0], sl[1]]
        local_mask = (sl_label == blob_id).astype(np.uint8)

        y_min = sl[0].start; y_max = sl[0].stop - 1
        x_min = sl[1].start; x_max = sl[1].stop - 1

        # Polygon contour within the local crop
        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon: list[list[int]] = []
        if contours:
            approx = cv2.approxPolyDP(contours[0], 1.5, closed=True)
            # Offset back to global image coords
            polygon = [[int(pt[0][0]) + x_min, int(pt[0][1]) + y_min] for pt in approx]

        # Mean building probability over the blob
        confidence = float(round(float(building_prob[sl[0], sl[1]][local_mask == 1].mean()), 4))

        instances.append({
            "class":       "building",
            "instance_id": len(instances) + 1,
            "mask":        polygon,
            "bbox":        [x_min, y_min, x_max, y_max],
            "confidence":  confidence,
        })

    return instances


def _make_label_map_png(label_map: np.ndarray, dest: Path) -> None:
    """Write a grayscale PNG where each pixel value is the TerrainClass ID (0–4).

    This is the texturing source for the Role 4 mesh script — it reads raw
    class IDs, not colours.  Mode "L" (8-bit greyscale) keeps values exact.
    """
    Image.fromarray(label_map, mode="L").save(dest)


def _make_enriched_overlay(
    label_map: np.ndarray,
    instances: list[dict],
    dest: Path,
    source_rgb: np.ndarray | None = None,
) -> None:
    """Write the human-readable overlay PNG.

    Layers (bottom to top)
    ----------------------
    1. Base: the original source RGB image (when provided) so context is
       always visible, or a black canvas when unavailable.
    2. Semi-transparent semantic colour wash: each non-OTHER pixel tinted
       with its terrain class colour at 55 % opacity over the base.
    3. Per-instance building outlines: 3-pixel border in a rotating palette
       so adjacent buildings are visually separable.  Instance ID number
       drawn near the centroid of each outline.
    4. Legend in the bottom-left corner showing class → colour mapping.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("classify_terrain needs opencv-python.") from exc

    H, W = label_map.shape

    # Layer 1: base — source RGB or black
    if source_rgb is not None and source_rgb.shape[:2] == (H, W):
        canvas = source_rgb.copy()
    else:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Layer 2: semi-transparent colour wash for each classified class
    ALPHA = 0.55  # opacity of the class colour over the base image
    for terrain, colour in TERRAIN_COLORS.items():
        if terrain == TerrainClass.OTHER:
            continue  # leave OTHER pixels as the raw source image
        mask = label_map == terrain
        if not mask.any():
            continue
        colour_layer = np.full((H, W, 3), colour, dtype=np.uint8)
        canvas[mask] = (
            colour_layer[mask] * ALPHA + canvas[mask] * (1 - ALPHA)
        ).astype(np.uint8)

    # ── Layer 3: per-instance building outlines + instance ID labels ──────────
    # All operations below work on `canvas` which is in RGB order.
    # cv2 expects BGR, so we convert to BGR, draw, then convert back.
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    for inst in instances:
        # White outline with thin black halo — visible on any background and
        # never confused with any class fill colour.
        outline_bgr = (255, 255, 255)  # white in BGR
        halo_bgr    = (0,   0,   0)    # black halo

        poly = inst["mask"]
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas_bgr, [pts], isClosed=True, color=halo_bgr,    thickness=5)
            cv2.polylines(canvas_bgr, [pts], isClosed=True, color=outline_bgr, thickness=2)

        # Instance ID number — white text with black halo
        x_min, y_min, x_max, y_max = inst["bbox"]
        cx = int((x_min + x_max) / 2)
        cy = int((y_min + y_max) / 2)
        label_text = str(inst["instance_id"])
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.30, min(0.55, (x_max - x_min) / 90))
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, 1)
        tx = max(0, min(cx - tw // 2, W - tw - 1))
        ty = max(th, min(cy + th // 2, H - 1))
        cv2.putText(canvas_bgr, label_text, (tx, ty), font, font_scale,
                    halo_bgr,    3, cv2.LINE_AA)
        cv2.putText(canvas_bgr, label_text, (tx, ty), font, font_scale,
                    outline_bgr, 1, cv2.LINE_AA)

    # ── Layer 4: legend (bottom-left) ─────────────────────────────────────
    # All colours are stored in RGB in TERRAIN_COLORS — convert to BGR for cv2.
    legend_entries = [
        ("other",      TERRAIN_COLORS[TerrainClass.OTHER]),
        ("vegetation", TERRAIN_COLORS[TerrainClass.VEGETATION]),
        ("building",   TERRAIN_COLORS[TerrainClass.BUILDING]),
        ("road",       TERRAIN_COLORS[TerrainClass.ROAD]),
        ("water",      TERRAIN_COLORS[TerrainClass.WATER]),
    ]
    swatch_w, swatch_h = 16, 14
    pad        = 4
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    legend_h   = len(legend_entries) * (swatch_h + pad) + pad
    legend_w   = 110
    lx         = pad
    ly_start   = H - legend_h - pad

    # Semi-transparent dark background
    overlay_legend = canvas_bgr.copy()
    cv2.rectangle(overlay_legend,
                  (lx - 2, ly_start - 2),
                  (lx + legend_w, ly_start + legend_h),
                  (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay_legend, 0.70, canvas_bgr, 0.30, 0, canvas_bgr)

    for i, (name, rgb_colour) in enumerate(legend_entries):
        row_y    = ly_start + pad + i * (swatch_h + pad)
        bgr_swatch = (int(rgb_colour[2]), int(rgb_colour[1]), int(rgb_colour[0]))
        cv2.rectangle(canvas_bgr,
                      (lx, row_y),
                      (lx + swatch_w, row_y + swatch_h),
                      bgr_swatch, cv2.FILLED)
        cv2.rectangle(canvas_bgr,
                      (lx, row_y),
                      (lx + swatch_w, row_y + swatch_h),
                      (200, 200, 200), 1)
        cv2.putText(canvas_bgr, name,
                    (lx + swatch_w + 4, row_y + swatch_h - 2),
                    font, font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    # Convert back to RGB for PIL
    canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(canvas, mode="RGB").save(dest)


def _build_json_payload(
    image_path: Path,
    label_map: np.ndarray,
    instances: list[dict],
    label_map_filename: str,
) -> dict:
    """Assemble the JSON payload dict.  Does not write to disk."""
    H, W = label_map.shape
    present_classes = sorted({
        TerrainClass(int(v)).name.lower()
        for v in np.unique(label_map)
    })

    return {
        "image_id":         image_path.name,
        "image_dimensions": {"width": W, "height": H},
        "segmentation": {
            "classes":        present_classes,
            "label_map_path": label_map_filename,
        },
        "detections": instances,
        # pending_classes: car is now detected via ADE20K when available.
        # tree (fine-grained tree vs crop distinction) still requires a
        # dedicated model.
        "pending_classes": {
            "classes": ["tree"],
            "reason": (
                "Fine-grained tree/crop distinction requires a dedicated model. "
                "Cars/vehicles are detected via ADE20K when that model is available "
                "locally (work/models/ade-segformer-b0)."
            ),
        },
    }


def _validate_json_payload(payload: dict) -> None:
    """Validate the payload against _OUTPUT_SCHEMA.  Raises on violation."""
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "classify_terrain needs jsonschema. "
            "Run `pip install -r terrain_classification/requirements.txt`."
        ) from exc
    jsonschema.validate(instance=payload, schema=_OUTPUT_SCHEMA)


# ---------------------------------------------------------------------------
# Tiled inference engine + dual-model helpers
# ---------------------------------------------------------------------------

ADE_DEFAULT_MODEL = "models/ade-segformer-b0"

# ADE20K class IDs that map to our TerrainClass values.
# Checked against the ade-segformer-b0 id2label.
_ADE_BUILDING_IDS:   frozenset[int] = frozenset({1, 25, 48, 84})    # building, house, skyscraper, tower
_ADE_VEGETATION_IDS: frozenset[int] = frozenset({4, 9, 17, 29, 66, 72})  # tree, grass, plant, field, flower, palm
_ADE_ROAD_IDS:       frozenset[int] = frozenset({6, 11, 52, 91})    # road, sidewalk, path, dirt track
_ADE_WATER_IDS:      frozenset[int] = frozenset({21, 26, 60, 109, 113, 128})  # water, sea, river, swimming pool, waterfall, lake
_ADE_CAR_IDS:        frozenset[int] = frozenset({20, 80, 83, 102})  # car, bus, truck, van


@lru_cache(maxsize=1)
def _load_ade_model(model_path: str, device: str | None):
    """Load and cache the ADE20K SegFormer complement model.

    Used alongside the LoveDA model to improve road, water, vegetation,
    and car detection on out-of-distribution imagery (e.g. European cities,
    high-resolution aerial GeoTIFFs).
    """
    try:
        import torch
        from transformers import (
            AutoImageProcessor,
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Terrain classification needs transformers. Run "
            "`pip install -r terrain_classification/requirements.txt`."
        ) from exc

    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
    except ValueError:
        processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved).eval()
    return processor, model, torch, resolved


def _run_tiled_inference(
    rgb_np: np.ndarray,
    processor: Any,
    model: Any,
    torch_mod: Any,
    device: str,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
) -> np.ndarray:
    """Run tiled sliding-window inference and return stitched probability array.

    For images ≤ tile_size × tile_size a single-pass is used.
    For larger images the full-resolution image is sliced into overlapping
    tiles, each is run through the model, and the per-class softmax
    probabilities are accumulated with a cosine-tapered blend weight so tile
    seams are invisible.

    Parameters
    ----------
    rgb_np : H×W×3 uint8 numpy array (source image).
    processor : HuggingFace image processor.
    model : HuggingFace SegFormer model (already on device, eval mode).
    torch_mod : the torch module reference from _load_model.
    device : resolved device string ("cuda" or "cpu").
    tile_size : each tile fed to the model (default 512).
    overlap : overlap in pixels between adjacent tiles (default 64).
    batch_size : tiles per GPU forward pass — increase if VRAM allows.

    Returns
    -------
    np.ndarray  shape (n_classes, H, W) float32 — blended softmax probabilities.
    """
    H, W = rgb_np.shape[:2]
    n_classes = model.config.num_labels

    # Single-pass shortcut for small images
    if H <= tile_size and W <= tile_size:
        tile_img = Image.fromarray(rgb_np)
        inputs   = processor(images=tile_img, return_tensors="pt")
        inputs   = {k: v.to(device) for k, v in inputs.items()}
        with torch_mod.no_grad():
            logits_raw = model(**inputs).logits
        logits_up = torch_mod.nn.functional.interpolate(
            logits_raw, size=(H, W), mode="bilinear", align_corners=False
        )
        probs = torch_mod.nn.functional.softmax(logits_up[0], dim=0).cpu().numpy()
        return probs  # (C, H, W)

    # Build tile grid
    step     = tile_size - overlap
    y_starts = list(range(0, H, step))
    x_starts = list(range(0, W, step))
    n_tiles  = len(y_starts) * len(x_starts)

    accum  = np.zeros((n_classes, H, W), dtype=np.float32)
    weight = np.zeros((H, W),            dtype=np.float32)

    # Pre-build all tile coordinates
    coords = []
    for y0 in y_starts:
        y1 = min(y0 + tile_size, H)
        y0 = max(0, y1 - tile_size)
        for x0 in x_starts:
            x1 = min(x0 + tile_size, W)
            x0 = max(0, x1 - tile_size)
            coords.append((y0, y1, x0, x1))

    # Process in batches for GPU efficiency
    done = 0
    for batch_start in range(0, len(coords), batch_size):
        batch_coords = coords[batch_start : batch_start + batch_size]
        tiles = [Image.fromarray(rgb_np[y0:y1, x0:x1]) for y0, y1, x0, x1 in batch_coords]

        inputs = processor(images=tiles, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch_mod.no_grad():
            logits_raw = model(**inputs).logits  # (B, C, H_s, W_s)

        for i, (y0, y1, x0, x1) in enumerate(batch_coords):
            th, tw = y1 - y0, x1 - x0
            logits_up = torch_mod.nn.functional.interpolate(
                logits_raw[i : i + 1], size=(th, tw), mode="bilinear", align_corners=False
            )
            probs_tile = torch_mod.nn.functional.softmax(
                logits_up[0], dim=0
            ).cpu().numpy()  # (C, th, tw)

            # Cosine-taper blend weight: 1 at centre, ~0 at edges
            wy = (np.hanning(th).reshape(-1, 1) + 1e-6).astype(np.float32)
            wx = (np.hanning(tw).reshape(1, -1) + 1e-6).astype(np.float32)
            w  = wy * wx  # (th, tw)

            accum[:, y0:y1, x0:x1]  += probs_tile * w
            weight[y0:y1, x0:x1]    += w

        done += len(batch_coords)
        if done % 20 == 0 or done == n_tiles:
            print(f"    [{done}/{n_tiles} tiles]", flush=True)

    eps = 1e-8
    accum /= (weight[np.newaxis, :, :] + eps)
    return accum  # (C, H, W)


def _fuse_dual_model_probs(
    loveda_probs: np.ndarray,
    ade_probs:    np.ndarray,
    id2label_loveda: dict,
    loveda_weight: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse LoveDA and ADE20K probability arrays into a unified (C=5, H, W) array.

    Strategy
    --------
    LoveDA is the primary model — it provides strong Building, Road, Water,
    Forest and Agricultural signals on aerial imagery.  ADE20K complements it
    for road, water, vegetation and car detection on out-of-distribution imagery
    (European cities, high-res urban GeoTIFFs).

    Both probability arrays are mapped independently to our 5-class TerrainClass
    space, then blended:
        fused = loveda_weight * loveda_mapped + (1-loveda_weight) * ade_mapped

    Parameters
    ----------
    loveda_probs : (C_loveda, H, W) float32 softmax probabilities.
    ade_probs    : (C_ade, H, W)    float32 softmax probabilities.
    id2label_loveda : LoveDA id→label dict.
    loveda_weight   : blend weight for the LoveDA model (default 0.65).

    Returns
    -------
    fused_5class : (5, H, W) float32 — per-class probability for TerrainClass 0-4.
    raw_argmax   : (H, W) int64      — argmax of fused probabilities.
    """
    C5, H, W = 5, loveda_probs.shape[1], loveda_probs.shape[2]

    # --- Map LoveDA → 5-class ------------------------------------------------
    loveda_5 = np.zeros((C5, H, W), dtype=np.float32)
    for raw_id, label in id2label_loveda.items():
        tc = _terrain_for_model_label(str(label))
        loveda_5[int(tc)] += loveda_probs[int(raw_id)]

    # --- Map ADE20K → 5-class ------------------------------------------------
    ade_5 = np.zeros((C5, H, W), dtype=np.float32)
    n_ade = ade_probs.shape[0]
    for ade_id in range(n_ade):
        if ade_id in _ADE_BUILDING_IDS:
            ade_5[int(TerrainClass.BUILDING)]   += ade_probs[ade_id]
        elif ade_id in _ADE_VEGETATION_IDS:
            ade_5[int(TerrainClass.VEGETATION)] += ade_probs[ade_id]
        elif ade_id in _ADE_ROAD_IDS:
            ade_5[int(TerrainClass.ROAD)]       += ade_probs[ade_id]
        elif ade_id in _ADE_WATER_IDS:
            ade_5[int(TerrainClass.WATER)]      += ade_probs[ade_id]
        else:
            ade_5[int(TerrainClass.OTHER)]      += ade_probs[ade_id]

    # Normalise each to sum to 1 per pixel
    loveda_5 /= (loveda_5.sum(axis=0, keepdims=True) + 1e-8)
    ade_5    /= (ade_5.sum(axis=0,    keepdims=True) + 1e-8)

    # Blend
    fused = loveda_weight * loveda_5 + (1 - loveda_weight) * ade_5
    raw_argmax = fused.argmax(axis=0).astype(np.int64)
    return fused, raw_argmax


def _extract_car_instances(
    ade_probs: np.ndarray,
    label_map: np.ndarray,
    car_confidence_threshold: float = 0.15,
    min_car_area_px: int = 9,
    max_car_area_px: int = 2000,
) -> list[dict]:
    """Extract car/vehicle instance detections from ADE20K probabilities.

    ADE20K car-related classes: car(20), bus(80), truck(83), van(102).
    Vehicles are small, compact objects — shape filter rejects oversized blobs.

    Returns list of detection dicts compatible with the existing JSON schema.
    The class field is 'car' — these are appended to the building detections
    list as additional entries with class='car'.
    """
    try:
        import cv2
        from scipy.ndimage import label as cc_label
    except ImportError:
        return []

    # Sum probabilities for all car-related ADE classes
    car_prob = np.zeros(ade_probs.shape[1:], dtype=np.float32)
    for cid in _ADE_CAR_IDS:
        if cid < ade_probs.shape[0]:
            car_prob += ade_probs[cid]

    car_mask  = (car_prob >= car_confidence_threshold).astype(np.uint8)
    labeled, n = cc_label(car_mask)

    detections = []
    for idx in range(1, n + 1):
        blob = labeled == idx
        area = int(blob.sum())
        if area < min_car_area_px or area > max_car_area_px:
            continue

        rows, cols = np.where(blob)
        h = int(rows.max() - rows.min() + 1)
        w = int(cols.max() - cols.min() + 1)
        aspect = max(h, w) / max(min(h, w), 1)
        if aspect > 5:
            continue  # too elongated to be a vehicle

        x_min, x_max = int(cols.min()), int(cols.max())
        y_min, y_max = int(rows.min()), int(rows.max())

        contours, _ = cv2.findContours(
            blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygon: list[list[int]] = []
        if contours:
            approx = cv2.approxPolyDP(contours[0], 1.5, closed=True)
            polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

        confidence = float(round(float(car_prob[blob].mean()), 4))
        detections.append({
            "class":       "car",
            "instance_id": len(detections) + 1,
            "bbox":        [x_min, y_min, x_max, y_max],
            "confidence":  confidence,
            **({"mask": polygon} if len(polygon) >= 3 else {}),
        })

    return detections


def classify_terrain(
    image_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
    ade_model_path: str | None = None,
    device: str | None = None,
    enable_fallbacks: bool = True,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    use_dual_model: bool = False,
) -> tuple[Path, Path]:
    """Run the full terrain classification pipeline and write final outputs.

    This is the primary entry point for Role 4 (mesh generation) handoff.

    For images larger than ``tile_size × tile_size``, tiled sliding-window
    inference is used automatically.  Each tile is run through the model on
    GPU (when available) and the per-class softmax probabilities are blended
    with a cosine-tapered weight so tile boundaries are invisible.

    When ``use_dual_model=True`` (default) and an ADE20K checkpoint is found
    at ``ade_model_path`` (defaults to ``work/models/ade-segformer-b0``),
    both LoveDA and ADE20K probabilities are fused before label assignment.
    This improves road, water, and vegetation detection on imagery outside
    LoveDA's training distribution, and adds car/vehicle instance detection.

    Writes three files into ``output_dir``:

    ``{stem}_data.json``       — Role 4 JSON: instances, classes, label_map_path
    ``{stem}_overlay.png``     — Human overlay: RGB base + class wash + outlines
    ``{stem}_labelmap.png``    — Raw uint8 PNG: pixel = TerrainClass ID (0-4)

    GeoTIFF inputs additionally produce ``{stem}_terrain_labels.tif`` with
    the original CRS and affine transform preserved pixel-for-pixel.

    Parameters
    ----------
    image_path : Path to GeoTIFF, PNG, or JPEG.
    output_dir : Output directory.  Defaults to image parent directory.
    model_name : LoveDA SegFormer checkpoint name or local path.
    ade_model_path : ADE20K SegFormer checkpoint path.  Defaults to
        ``work/models/ade-segformer-b0`` relative to the project root.
    device : Torch device (``"cuda"`` / ``"cpu"``).  Auto-detected when None.
    enable_fallbacks : Apply colour/confidence-gated heuristic fallbacks
        (greenness, warmth, brightness) after model inference.
    tile_size : Tile edge length in pixels fed to the model.  Default 512.
    overlap : Overlap in pixels between adjacent tiles.  Default 64.
    batch_size : Tiles per GPU forward pass.  Increase for more VRAM.
    use_dual_model : Fuse LoveDA + ADE20K probabilities.  Default ``False``
        (LoveDA-only, ~30s for 5000×5000 on GPU).  Set to ``True`` to add
        ADE20K complementary signal for roads, water and car detection
        (~90s additional on GPU with 6GB VRAM).

    Returns
    -------
    (json_path, overlay_png_path)
    """
    import torch
    import time as _time

    source = Path(image_path)
    dest   = Path(output_dir) if output_dir else source.parent
    dest.mkdir(parents=True, exist_ok=True)
    stem = source.stem

    # --- auto-detect device -------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t_start = _time.perf_counter()

    # --- load image ---------------------------------------------------------
    rgb_image, georef = _load_image(source)
    rgb_np = np.array(rgb_image)
    W, H   = rgb_image.size

    # --- load LoveDA model --------------------------------------------------
    processor, model, torch_mod, resolved_device = _load_model(str(model_name), device)
    id2label = model.config.id2label

    print(f"  classify_terrain: {W}×{H}px  device={resolved_device}  "
          f"tile={tile_size}  overlap={overlap}", flush=True)

    # --- LoveDA tiled inference ---------------------------------------------
    loveda_probs = _run_tiled_inference(
        rgb_np, processor, model, torch_mod, resolved_device,
        tile_size=tile_size, overlap=overlap, batch_size=batch_size,
    )  # (C_loveda, H, W)

    t_loveda = _time.perf_counter()
    print(f"  LoveDA inference: {t_loveda - t_start:.1f}s", flush=True)

    # --- ADE20K tiled inference (optional) ----------------------------------
    ade_probs: np.ndarray | None = None
    if use_dual_model:
        # Resolve ADE model path: explicit arg → project-relative default → skip
        if ade_model_path is None:
            candidate = Path(__file__).resolve().parents[1] / "work" / "models" / "ade-segformer-b0"
            ade_model_path = str(candidate) if candidate.is_dir() else None

        if ade_model_path is not None:
            try:
                # Free LoveDA GPU memory before loading ADE20K so both can
                # run on GPU without contention on a 6GB VRAM card.
                # We move LoveDA to CPU temporarily; it stays cached in
                # lru_cache so the CPU weights are available for reload.
                model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                ade_proc, ade_mod, ade_torch, ade_dev = _load_ade_model(
                    str(ade_model_path), device   # same device as LoveDA
                )
                ade_probs = _run_tiled_inference(
                    rgb_np, ade_proc, ade_mod, ade_torch, ade_dev,
                    tile_size=1024, overlap=128, batch_size=2,
                )
                t_ade = _time.perf_counter()
                print(f"  ADE20K inference: {t_ade - t_loveda:.1f}s", flush=True)

                # Restore LoveDA to GPU for any downstream use
                model.to(resolved_device)
            except Exception as exc:
                print(f"  ADE20K unavailable ({exc}), using LoveDA only.", flush=True)
                model.to(resolved_device)   # ensure LoveDA is back on device
                ade_probs = None

    # --- fuse probabilities → label map -------------------------------------
    if ade_probs is not None:
        fused_5, raw_argmax_5 = _fuse_dual_model_probs(
            loveda_probs, ade_probs, id2label
        )
        label_map  = raw_argmax_5.astype(np.uint8)
        # raw_labels for fallbacks: LoveDA raw argmax (used for Background/Barren IDs)
        raw_labels = loveda_probs.argmax(axis=0).astype(np.int64)
        # logits_full: use fused 5-class probs reshaped as (1,5,H,W) tensor
        logits_full = torch_mod.from_numpy(fused_5[np.newaxis]).to(resolved_device)
        # For fallbacks: Background id and Barren id from LoveDA id2label
        bg_id     = next(k for k, v in id2label.items() if v == "Background")
        barren_id = next(k for k, v in id2label.items() if v == "Barren")
    else:
        # LoveDA only
        raw_labels  = loveda_probs.argmax(axis=0).astype(np.int64)
        label_map   = collapse_model_labels(raw_labels, id2label)
        logits_full = torch_mod.from_numpy(loveda_probs[np.newaxis]).to(resolved_device)
        bg_id     = next(k for k, v in id2label.items() if v == "Background")
        barren_id = next(k for k, v in id2label.items() if v == "Barren")

    # --- post-processing fallbacks (opt-in) ---------------------------------
    if enable_fallbacks:
        label_map = apply_greenness_fallback(
            label_map, rgb_np, raw_labels,
            background_class_id=bg_id,
            green_threshold=60,
            model_logits_full=logits_full,
            confidence_ceiling=0.85,
        )
        label_map = apply_warmth_fallback(
            label_map, rgb_np, raw_labels,
            background_class_id=bg_id,
            barren_class_id=barren_id,
            red_over_green=30,
            brightness_floor=120,
        )
        label_map = apply_brightness_fallback(
            label_map, rgb_np, raw_labels,
            background_class_id=bg_id,
            barren_class_id=barren_id,
            brightness_floor=120,
            max_saturation=60,
            model_logits_full=logits_full,
            confidence_ceiling=0.95,
        )

    t_post = _time.perf_counter()

    # --- building instances -------------------------------------------------
    instances = _extract_building_instances(
        label_map, logits_full, id2label=id2label
    )

    # --- car instances (ADE20K only) ----------------------------------------
    car_instances: list[dict] = []
    if ade_probs is not None:
        car_instances = _extract_car_instances(ade_probs, label_map)
        # Re-number instance IDs so they don't clash with building IDs
        for i, ci in enumerate(car_instances):
            ci["instance_id"] = len(instances) + i + 1
        instances = instances + car_instances

    # --- GeoTIFF label raster (preserves CRS and transform) -----------------
    if georef is not None:
        import rasterio
        geotiff_path = dest / f"{stem}_terrain_labels.tif"
        with rasterio.open(
            geotiff_path, "w", driver="GTiff",
            height=label_map.shape[0], width=label_map.shape[1],
            count=1, dtype="uint8",
            crs=georef.crs, transform=georef.transform, compress="lzw",
        ) as out_raster:
            out_raster.write(label_map, 1)

    # --- output paths -------------------------------------------------------
    json_path     = dest / f"{stem}_data.json"
    overlay_path  = dest / f"{stem}_overlay.png"
    labelmap_path = dest / f"{stem}_labelmap.png"

    # --- label map PNG ------------------------------------------------------
    _make_label_map_png(label_map, labelmap_path)

    # --- enriched overlay PNG -----------------------------------------------
    _make_enriched_overlay(label_map, instances, overlay_path, source_rgb=rgb_np)

    # --- JSON ---------------------------------------------------------------
    payload = _build_json_payload(source, label_map, instances, labelmap_path.name)
    _validate_json_payload(payload)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    t_end = _time.perf_counter()

    # --- report -------------------------------------------------------------
    total = label_map.size
    print(f"  Total time: {t_end - t_start:.1f}s", flush=True)
    print(f"  Output: {W}×{H}px  building_instances={len([i for i in instances if i['class']=='building'])}  "
          f"car_instances={len(car_instances)}", flush=True)
    for v, n in {0:"other",1:"vegetation",2:"building",3:"road",4:"water"}.items():
        pct = 100 * np.sum(label_map == v) / total
        print(f"    {n:<12} {pct:.1f}%", flush=True)

    return json_path, overlay_path


def visualize_confidence_heatmap(
    image_path: str | Path,
    class_name: str,
    *,
    output_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
) -> Path:
    """Write a false-colour heatmap PNG showing per-pixel softmax confidence
    for a single LoveDA class, before any thresholding or label collapsing.

    Use this to diagnose misclassification issues:
    - High confidence everywhere the class IS present  → model knows it, pipeline bug
    - Low/zero confidence everywhere               → domain shift, wrong checkpoint
    - Patchy moderate confidence                   → borderline, threshold tunable

    The heatmap uses a cyan→yellow→red (jet-like) palette:
      deep-blue  = 0 % confidence for this class
      cyan/green = low-moderate confidence
      yellow     = moderate-high confidence
      red        = near-certain for this class

    The source image is blended at 40 % opacity behind the heatmap so spatial
    context is visible alongside the confidence values.

    Parameters
    ----------
    image_path:
        Input image (GeoTIFF, PNG, or JPEG).
    class_name:
        LoveDA class name to visualise, case-insensitive.
        Valid values: ``"Background"``, ``"Building"``, ``"Road"``,
        ``"Water"``, ``"Barren"``, ``"Forest"``, ``"Agricultural"``,
        ``"Ignore"``.
    output_dir:
        Where to write the output PNG.  Defaults to the parent directory of
        ``image_path``.
    model_name:
        SegFormer checkpoint name or local directory.
    device:
        Torch device string.  Auto-detected when ``None``.

    Returns
    -------
    Path
        Path to the written heatmap PNG, named
        ``{stem}_confidence_{class_name.lower()}.png``.

    Raises
    ------
    ValueError
        If ``class_name`` is not found in the model's id2label mapping.
    """
    import torch

    source = Path(image_path)
    dest   = Path(output_dir) if output_dir else source.parent
    dest.mkdir(parents=True, exist_ok=True)

    rgb_image, _ = _load_image(source)
    rgb_np        = np.array(rgb_image)
    W, H          = rgb_image.size

    processor, model, torch_mod, resolved_device = _load_model(
        str(model_name), device
    )
    id2label = model.config.id2label

    # Resolve class_name → channel index
    target_channel: int | None = None
    for cid, cname in id2label.items():
        if cname.lower() == class_name.lower():
            target_channel = int(cid)
            break
    if target_channel is None:
        valid = sorted(id2label.values())
        raise ValueError(
            f"Unknown class_name {class_name!r}. "
            f"Valid names for this checkpoint: {valid}"
        )

    # Run model
    inputs = processor(images=rgb_image, return_tensors="pt")
    inputs = {k: v.to(resolved_device) for k, v in inputs.items()}
    with torch_mod.no_grad():
        logits_raw = model(**inputs).logits
    logits_full = torch_mod.nn.functional.interpolate(
        logits_raw, size=(H, W), mode="bilinear", align_corners=False
    )
    with torch.no_grad():
        probs = torch_mod.nn.functional.softmax(logits_full[0], dim=0)
    confidence = probs[target_channel].cpu().numpy()  # (H, W) float32, 0–1

    # Convert confidence [0,1] → uint8 heat colour using cv2 colourmap
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("visualize_confidence_heatmap needs opencv-python.") from exc

    conf_uint8 = (confidence * 255).clip(0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)  # BGR
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend: 60 % heatmap + 40 % source image
    ALPHA = 0.60
    blended = (heatmap_rgb * ALPHA + rgb_np * (1 - ALPHA)).clip(0, 255).astype(np.uint8)

    # Annotate: class name, confidence stats, colour scale bar
    pct_high = float(np.mean(confidence > 0.5) * 100)
    pct_med  = float(np.mean((confidence > 0.2) & (confidence <= 0.5)) * 100)
    pct_low  = float(np.mean(confidence <= 0.2) * 100)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    label_bgr  = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    info_lines = [
        f"Class: {class_name}  (channel {target_channel})",
        f"Max conf: {confidence.max():.3f}   Mean: {confidence.mean():.3f}",
        f">50%: {pct_high:.1f}%   20-50%: {pct_med:.1f}%   <20%: {pct_low:.1f}%",
    ]
    for i, line in enumerate(info_lines):
        y = 22 + i * 22
        cv2.putText(label_bgr, line, (8, y), font, 0.55, (0, 0, 0),   3, cv2.LINE_AA)
        cv2.putText(label_bgr, line, (8, y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Compact colour-scale bar (bottom-right)
    bar_w, bar_h = 160, 14
    bar_x = W - bar_w - 8
    bar_y = H - bar_h - 24
    bar_grad = np.tile(np.arange(256, dtype=np.uint8), (bar_h, 1))      # (h, 256)
    bar_resized = cv2.resize(bar_grad, (bar_w, bar_h))
    bar_colour  = cv2.applyColorMap(bar_resized, cv2.COLORMAP_JET)
    label_bgr[bar_y:bar_y+bar_h, bar_x:bar_x+bar_w] = bar_colour
    cv2.rectangle(label_bgr, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (200,200,200), 1)
    cv2.putText(label_bgr, "0%", (bar_x, bar_y+bar_h+14),
                font, 0.4, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(label_bgr, "100%", (bar_x+bar_w-30, bar_y+bar_h+14),
                font, 0.4, (255,255,255), 1, cv2.LINE_AA)

    out_path = dest / f"{source.stem}_confidence_{class_name.lower()}.png"
    cv2.imwrite(str(out_path), label_bgr)
    return out_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate terrain labels and a color-coded overlay.")
    parser.add_argument("--image", required=True, type=Path, help="Path to a JPG, PNG, or GeoTIFF image.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SegFormer checkpoint or local model directory.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda.")
    args = parser.parse_args(argv)

    labels, overlay = segment(args.image, model_name=args.model, device=args.device)
    paths = save_segmentation_outputs(args.image, labels, overlay, args.output_dir)
    print(f"Terrain labels: shape={labels.shape}, classes={sorted(np.unique(labels).tolist())}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

