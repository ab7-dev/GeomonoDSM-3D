"""
gamus/inference.py — GAMUS-aligned pretrained land-cover segmentation inference.

This module loads a pretrained SegFormer fine-tuned on satellite land-cover imagery
(nave1616/SegFormer-landcover-FT) whose class taxonomy closely mirrors GAMUS:

    Model classes (9-class land-cover):
        0  background
        1  bareland
        2  rangeland
        3  developed space
        4  road
        5  tree
        6  water
        7  agriculture land
        8  buildings

    GAMUS reference classes (7-class):
        0  others/background
        1  ground
        2  low vegetation
        3  building
        4  water
        5  road
        6  tree

This is a GENUINE pretrained checkpoint loaded and used for real inference.
It is NOT the same model as the original GAMUS paper benchmark, but it uses
a semantically equivalent taxonomy for remote-sensing land-cover segmentation.
The checkpoint is publicly available on HuggingFace Hub.

Public API
----------
    run_gamus_inference(image_path, output_dir, device=None, tile_size=512)
        → dict with keys: json_path, labelmap_path, overlay_path, gamus_map_path,
                          inference_time_s, model_id, n_classes, class_distribution

    load_gamus_model(device=None) → (processor, model, device_str)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("gamus.inference")

# ── Model identifier ──────────────────────────────────────────────────────
GAMUS_MODEL_ID = "nave1616/SegFormer-landcover-FT"

# ── Model class labels ───────────────────────────────────────────────────
# From the checkpoint config.json id2label
_MODEL_LABELS: dict[int, str] = {
    0: "background",
    1: "bareland",
    2: "rangeland",
    3: "developed_space",
    4: "road",
    5: "tree",
    6: "water",
    7: "agriculture_land",
    8: "buildings",
}

# ── GAMUS-aligned class mapping ───────────────────────────────────────────
# Maps each model class → GAMUS semantic class (0-6)
# GAMUS: 0=others, 1=ground, 2=low_veg, 3=building, 4=water, 5=road, 6=tree
_MODEL_TO_GAMUS: dict[int, int] = {
    0: 0,   # background → GAMUS others
    1: 1,   # bareland   → GAMUS ground
    2: 2,   # rangeland  → GAMUS low vegetation
    3: 0,   # developed  → GAMUS others (parking lots etc.)
    4: 5,   # road       → GAMUS road
    5: 6,   # tree       → GAMUS tree
    6: 4,   # water      → GAMUS water
    7: 2,   # agriculture→ GAMUS low vegetation
    8: 3,   # buildings  → GAMUS building
}

# ── GAMUS class names ─────────────────────────────────────────────────────
GAMUS_CLASS_NAMES: dict[int, str] = {
    0: "others_background",
    1: "ground",
    2: "low_vegetation",
    3: "building",
    4: "water",
    5: "road",
    6: "tree",
}

# ── Pipeline TerrainClass mapping ─────────────────────────────────────────
# Our existing 5-class TerrainClass (0-4)
# OTHER=0, VEGETATION=1, BUILDING=2, ROAD=3, WATER=4
_GAMUS_TO_TERRAINCLASS: dict[int, int] = {
    0: 0,   # others/background → OTHER
    1: 0,   # ground            → OTHER
    2: 1,   # low vegetation    → VEGETATION
    3: 2,   # building          → BUILDING
    4: 4,   # water             → WATER
    5: 3,   # road              → ROAD
    6: 1,   # tree              → VEGETATION (merged)
}

TERRAIN_CLASS_NAMES: dict[int, str] = {
    0: "other",
    1: "vegetation",
    2: "building",
    3: "road",
    4: "water",
}

TERRAIN_COLORS: dict[int, tuple[int, int, int]] = {
    0: (60, 60, 70),       # other — dark grey
    1: (47, 158, 68),      # vegetation — green
    2: (214, 57, 57),      # building — red
    3: (125, 125, 125),    # road — grey
    4: (50, 120, 220),     # water — blue
}

GAMUS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (60, 60, 70),       # others — dark
    1: (180, 140, 90),     # ground — tan
    2: (100, 190, 100),    # low vegetation — light green
    3: (214, 57, 57),      # building — red
    4: (50, 120, 220),     # water — blue
    5: (180, 180, 80),     # road — yellow-grey
    6: (30, 120, 30),      # tree — dark green
}


@lru_cache(maxsize=1)
def load_gamus_model(device: str | None = None):
    """Load and cache the GAMUS-aligned pretrained model.

    Downloads nave1616/SegFormer-landcover-FT from HuggingFace on first call
    (cached locally thereafter).  Uses GPU if available.

    Returns
    -------
    (processor, model, resolved_device_str)
    """
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading GAMUS model: %s  device=%s", GAMUS_MODEL_ID, resolved)
    t0 = time.perf_counter()

    # The nave1616 model doesn't include a preprocessor_config.json,
    # so load the standard SegFormer processor from nvidia/mit-b0 (compatible)
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(GAMUS_MODEL_ID)
    except Exception:
        from transformers import SegformerImageProcessor
        processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        logger.info("  Using fallback SegformerImageProcessor (nvidia/mit-b0)")

    model = SegformerForSemanticSegmentation.from_pretrained(GAMUS_MODEL_ID)
    model.to(resolved).eval()

    elapsed = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs  num_labels=%d", elapsed, model.config.num_labels)
    return processor, model, resolved


def _run_inference_tiled(
    rgb_np: np.ndarray,
    processor: Any,
    model: Any,
    device: str,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Run tiled inference and return stitched argmax label map (H×W uint8).

    For images ≤ tile_size a single forward pass is used.
    Larger images are processed tile-by-tile with cosine-taper blending.
    """
    import torch

    H, W   = rgb_np.shape[:2]
    n_cls  = model.config.num_labels

    if H <= tile_size and W <= tile_size:
        img    = Image.fromarray(rgb_np)
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Tiled path
    accum  = np.zeros((n_cls, H, W), dtype=np.float32)
    weight = np.zeros((H, W),        dtype=np.float32)
    step   = tile_size - overlap

    ys = list(range(0, H, step))
    xs = list(range(0, W, step))

    for y0 in ys:
        y1 = min(y0 + tile_size, H); y0 = max(0, y1 - tile_size)
        for x0 in xs:
            x1 = min(x0 + tile_size, W); x0 = max(0, x1 - tile_size)
            tile  = Image.fromarray(rgb_np[y0:y1, x0:x1])
            inp   = processor(images=tile, return_tensors="pt")
            inp   = {k: v.to(device) for k, v in inp.items()}
            th, tw = y1-y0, x1-x0
            with torch.no_grad():
                lg = model(**inp).logits
            lg = torch.nn.functional.interpolate(
                lg, size=(th, tw), mode="bilinear", align_corners=False
            )
            probs = torch.nn.functional.softmax(lg[0], dim=0).cpu().numpy()
            wy = (np.hanning(th).reshape(-1, 1) + 1e-6).astype(np.float32)
            wx = (np.hanning(tw).reshape(1, -1) + 1e-6).astype(np.float32)
            w  = wy * wx
            accum[:, y0:y1, x0:x1]  += probs * w
            weight[y0:y1, x0:x1]    += w

    accum /= (weight[np.newaxis] + 1e-8)
    return accum.argmax(axis=0).astype(np.uint8)


def _colorize(label_map: np.ndarray, color_dict: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Convert integer label map to RGB image using a colour dictionary."""
    H, W   = label_map.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, rgb in color_dict.items():
        mask = (label_map == cls_id)
        canvas[mask] = rgb
    return canvas


def run_gamus_inference(
    image_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    max_size: int = 1024,
) -> dict:
    """Run GAMUS-aligned pretrained inference on an RGB image.

    Produces four outputs:
      {stem}_gamus_raw.png       — raw 9-class model prediction (colourised)
      {stem}_gamus_labels.png    — 7-class GAMUS label map (uint8 PNG)
      {stem}_terrain_gamus.png   — 5-class TerrainClass label map (uint8 PNG)
      {stem}_gamus_overlay.png   — GAMUS classes overlaid on source RGB
      {stem}_gamus_report.json   — inference metadata and class statistics

    Parameters
    ----------
    image_path : path to input RGB image (PNG, JPG, TIFF, GeoTIFF)
    output_dir : directory where all outputs are written
    device     : torch device string ('cuda'/'cpu'); auto-detected if None
    tile_size  : tile edge for large-image inference (default 512)
    overlap    : tile overlap in pixels (default 64)
    max_size   : maximum dimension for inference; image is downsampled if larger
                 (output maps are then upsampled back to original size)

    Returns
    -------
    dict with keys: json_path, labelmap_path, overlay_path, gamus_map_path,
                    terrain_map_path, inference_time_s, model_id, n_classes,
                    class_distribution, gamus_distribution, terrain_distribution
    """
    source   = Path(image_path)
    dest     = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    stem     = source.stem

    logger.info("GAMUS inference: %s", source)

    # ── Load image ────────────────────────────────────────────────────────
    if source.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
            with rasterio.open(source) as src:
                bands = min(src.count, 3)
                data  = src.read(list(range(1, bands + 1)))
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            hwc = np.moveaxis(data, 0, -1)
            if hwc.dtype != np.uint8:
                lo, hi = float(np.percentile(hwc, 1)), float(np.percentile(hwc, 99))
                hwc = np.clip((hwc - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
            orig_pil = Image.fromarray(hwc, "RGB")
        except Exception as e:
            logger.warning("rasterio failed (%s), trying PIL.", e)
            orig_pil = Image.open(source).convert("RGB")
    else:
        orig_pil = Image.open(source).convert("RGB")

    orig_w, orig_h = orig_pil.size

    # ── Downsample for inference if needed ───────────────────────────────
    if max(orig_w, orig_h) > max_size:
        scale    = max_size / max(orig_w, orig_h)
        inf_w    = int(orig_w * scale)
        inf_h    = int(orig_h * scale)
        inf_pil  = orig_pil.resize((inf_w, inf_h), Image.LANCZOS)
        logger.info("  Downsampled %dx%d → %dx%d for inference", orig_w, orig_h, inf_w, inf_h)
    else:
        inf_pil = orig_pil
        inf_w, inf_h = orig_w, orig_h

    rgb_np = np.array(inf_pil, dtype=np.uint8)

    # ── Load model and run inference ──────────────────────────────────────
    processor, model, resolved_device = load_gamus_model(device)

    t0    = time.perf_counter()
    label_map_raw = _run_inference_tiled(rgb_np, processor, model, resolved_device,
                                          tile_size=tile_size, overlap=overlap)
    elapsed = time.perf_counter() - t0
    logger.info("  Inference done in %.1fs  shape=%s", elapsed, label_map_raw.shape)

    # ── Upsample labels back to original resolution ───────────────────────
    if (inf_h, inf_w) != (orig_h, orig_w):
        label_map_raw = np.array(
            Image.fromarray(label_map_raw).resize((orig_w, orig_h), Image.NEAREST)
        )

    H, W = label_map_raw.shape

    # ── Map to GAMUS 7-class ──────────────────────────────────────────────
    gamus_map = np.zeros((H, W), dtype=np.uint8)
    for model_cls, gamus_cls in _MODEL_TO_GAMUS.items():
        gamus_map[label_map_raw == model_cls] = gamus_cls

    # ── Map to TerrainClass 5-class ───────────────────────────────────────
    terrain_map = np.zeros((H, W), dtype=np.uint8)
    for gamus_cls, terrain_cls in _GAMUS_TO_TERRAINCLASS.items():
        terrain_map[gamus_map == gamus_cls] = terrain_cls

    # ── Statistics ────────────────────────────────────────────────────────
    total = H * W
    raw_dist = {
        _MODEL_LABELS.get(int(v), str(v)): int((label_map_raw == v).sum())
        for v in range(model.config.num_labels)
        if (label_map_raw == v).any()
    }
    gamus_dist = {
        GAMUS_CLASS_NAMES.get(int(v), str(v)): round(100.0 * (gamus_map == v).sum() / total, 2)
        for v in range(7)
    }
    terrain_dist = {
        TERRAIN_CLASS_NAMES.get(int(v), str(v)): round(100.0 * (terrain_map == v).sum() / total, 2)
        for v in range(5)
    }

    # ── Save outputs ──────────────────────────────────────────────────────
    # 1. Raw 9-class colourised prediction
    raw_vis_rgb = _colorize(label_map_raw, {
        0: (60, 60, 70), 1: (200, 160, 100), 2: (100, 180, 100),
        3: (100, 100, 150), 4: (180, 180, 80), 5: (30, 120, 30),
        6: (50, 120, 220), 7: (140, 200, 80), 8: (214, 57, 57),
    })
    raw_path = dest / f"{stem}_gamus_raw.png"
    Image.fromarray(raw_vis_rgb).save(raw_path)

    # 2. GAMUS 7-class label map (uint8 PNG)
    gamus_lm_path = dest / f"{stem}_gamus_labels.png"
    Image.fromarray(gamus_map, mode="L").save(gamus_lm_path)

    # 3. GAMUS colourised overlay over source RGB
    orig_np  = np.array(orig_pil)
    overlay  = orig_np.copy().astype(np.float32)
    ALPHA    = 0.55
    for cls_id, colour in GAMUS_COLORS.items():
        mask = (gamus_map == cls_id)
        if not mask.any():
            continue
        overlay[mask] = (
            np.array(colour, dtype=np.float32) * ALPHA
            + overlay[mask] * (1 - ALPHA)
        )
    overlay_path = dest / f"{stem}_gamus_overlay.png"
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(overlay_path)

    # 4. TerrainClass label map (uint8 PNG — compatible with rest of pipeline)
    terrain_path = dest / f"{stem}_terrain_gamus.png"
    Image.fromarray(terrain_map, mode="L").save(terrain_path)

    # 5. TerrainClass colourised preview
    tc_vis = _colorize(terrain_map, TERRAIN_COLORS)
    tc_vis_path = dest / f"{stem}_terrain_gamus_vis.png"
    Image.fromarray(tc_vis).save(tc_vis_path)

    # 6. JSON report
    report = {
        "model_id":            GAMUS_MODEL_ID,
        "model_type":          "SegFormer-B4 fine-tuned on land-cover satellite imagery",
        "gamus_alignment":     "GAMUS-class-compatible taxonomy (building/road/water/tree/vegetation)",
        "checkpoint_loaded":   True,
        "actual_gamus_paper_model": False,
        "note": (
            "This model (nave1616/SegFormer-landcover-FT) is a REAL pretrained SegFormer "
            "loaded from HuggingFace Hub. It uses a 9-class land-cover taxonomy that maps "
            "directly to GAMUS 7-class and our pipeline 5-class TerrainClass schema. "
            "It is NOT the exact model from the GAMUS benchmark paper, but performs "
            "semantically equivalent remote-sensing land-cover segmentation."
        ),
        "input_image":         str(source),
        "input_size":          [orig_h, orig_w],
        "inference_size":      [inf_h, inf_w],
        "inference_time_s":    round(elapsed, 2),
        "device":              resolved_device,
        "n_model_classes":     model.config.num_labels,
        "model_class_distribution": raw_dist,
        "gamus_class_distribution_pct": gamus_dist,
        "terrain_class_distribution_pct": terrain_dist,
        "outputs": {
            "raw_prediction":      raw_path.name,
            "gamus_labels":        gamus_lm_path.name,
            "gamus_overlay":       overlay_path.name,
            "terrain_classes":     terrain_path.name,
            "terrain_vis":         tc_vis_path.name,
        },
    }
    json_path = dest / f"{stem}_gamus_report.json"
    json_path.write_text(json.dumps(report, indent=2))

    logger.info("  GAMUS output -> %s", dest)
    logger.info("  GAMUS class dist: %s",
                " | ".join(f"{k}={v:.1f}%" for k, v in gamus_dist.items() if v > 0))
    logger.info("  TerrainClass: %s",
                " | ".join(f"{k}={v:.1f}%" for k, v in terrain_dist.items() if v > 0))

    return {
        "json_path":            json_path,
        "raw_path":             raw_path,
        "labelmap_path":        gamus_lm_path,
        "overlay_path":         overlay_path,
        "terrain_map_path":     terrain_path,
        "inference_time_s":     elapsed,
        "model_id":             GAMUS_MODEL_ID,
        "n_classes":            model.config.num_labels,
        "class_distribution":   raw_dist,
        "gamus_distribution":   gamus_dist,
        "terrain_distribution": terrain_dist,
    }
