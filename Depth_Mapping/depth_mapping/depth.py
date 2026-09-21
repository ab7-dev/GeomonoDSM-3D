"""Single-image relative depth estimation with Depth Anything V2.

Depth Anything predicts *relative* depth, not surveyed elevation or distance in
metres.  Larger values in the returned array generally indicate pixels that are
closer to the camera (or taller in a nadir-looking image).

Tiling strategy — core-only extraction
---------------------------------------
Large images are split into a regular grid of non-overlapping **core** tiles.
For each core tile, a larger **context crop** (core + margin on every side) is
fed to the model.  Only the central core region of the model's output is written
into the final canvas.  The border margin is discarded.

Why this preserves detail
--------------------------
* Each core pixel is predicted exactly once, from a single model forward pass.
  There is no averaging of multiple predictions over the same pixel, so no
  blurring occurs inside core regions.

* The context margin gives the ViT enough receptive-field context at the tile
  edges so that core predictions are not affected by the hard crop boundary.
  A 196-px margin on each side (≈ 14 full ViT patches) is sufficient.

* A narrow feather (8 px) is applied at internal core edges only to remove
  hard depth jumps that would be visible as seam lines.  8 px is narrow enough
  that structural detail (roads, buildings) is unaffected.

* Relative depth is affine-invariant (each tile may have a different scale+shift).
  A per-core-tile shift is estimated from the narrow feather-overlap region
  between adjacent already-placed tiles and the current tile's edge pixels.
  Only a shift (no scale) is adjusted: scaling would change local gradients.

* No z-score normalisation is applied.  Normalising each tile to mean=0 std=1
  destroys inter-tile relative depth magnitudes and collapses the effective
  dynamic range of the final map.

Model backend
-------------
Uses the original Depth Anything V2 architecture (vendored under
``depth_anything_v2/``) loaded from local ``.pth`` checkpoints.
No Hugging Face download is required.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from depth_mapping.model_loader import load_model
from depth_mapping import geo_utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tiling constants
# ---------------------------------------------------------------------------

# Core tile target size.  DA V2 is trained at 518 px; a ~500 px core gives
# the model a natural field-of-view per tile.
CORE_TILE_SIZE: int = 500          # target core tile edge length (px)
MIN_CORE_SIZE:  int = 350          # smallest acceptable core
MAX_CORE_SIZE:  int = 650          # largest acceptable core

# Context margin added around every core tile before model inference.
# 128 px = ~9 full ViT patches — enough for the attention mechanism to
# see meaningful surrounding context at every core-edge pixel, without
# making the context crop unnecessarily large (which slows inference).
CONTEXT_MARGIN: int = 128

# Feather width at internal core edges for seam removal.
# Must be << CONTEXT_MARGIN so the feather region is well within the area
# where the model has received full context.
FEATHER: int = 8

# Kept for backward-compatibility with tests that import the old constant.
TILE_OVERLAP:     int = CONTEXT_MARGIN * 2
NATIVE_RESOLUTION: int = 518
MIN_TILE_SIZE:    int = MIN_CORE_SIZE
MAX_TILE_SIZE:    int = MAX_CORE_SIZE

# ImageNet normalisation (same as DA V2 training).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

if _DEVICE == "cuda":
    _CPU_THREADS = max(1, min(4, (os.cpu_count() or 4) // 4))
else:
    _CPU_THREADS = os.cpu_count() or 1

torch.set_num_threads(_CPU_THREADS)
logger.debug("Device: %s | CPU threads: %d", _DEVICE, _CPU_THREADS)

_ONNX_SESSIONS: dict[str, Any] = {}


# ===========================================================================
# Image I/O
# ===========================================================================

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Stretch arbitrary raster to uint8 via 1–99th percentile."""
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.dtype == np.uint8:
        return image
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _load_image(image_path: str) -> Image.Image:
    """Load JPG/PNG/GeoTIFF → uint8 RGB PIL image."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")

    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
            with rasterio.open(path) as src:
                n = min(src.count, 3)
                dtype = src.dtypes[0]
                data  = src.read(list(range(1, n + 1)))
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            elif data.shape[0] == 2:
                data = np.concatenate([data, data[:1]], axis=0)
            hwc = np.moveaxis(data, 0, -1)
            if dtype == "uint8":
                return Image.fromarray(hwc, "RGB")
            return Image.fromarray(_to_uint8(hwc.astype(np.float32)), "RGB")
        except ImportError:
            pass

    return Image.open(path).convert("RGB")


def _load_image_geo(image_path: str) -> tuple[Image.Image, dict | None]:
    """Load image and return (rgb, geo_meta).  geo_meta is None for non-GeoTIFF."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")

    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            img, meta = geo_utils.read_geotiff_rgb(path)
            return img, meta
        except ImportError:
            logger.warning("rasterio unavailable — GeoTIFF metadata will be lost.")

    return Image.open(path).convert("RGB"), None


# ===========================================================================
# Preprocessing
# ===========================================================================

def _clip_exposure(image: Image.Image) -> Image.Image:
    """Global per-band 1–99 percentile stretch.  Applied once before tiling.

    Early-exit when every band already spans ≥ 90 % of [0, 255].
    """
    arr = np.asarray(image)
    _, _, c = arr.shape
    lows  = np.empty(c, dtype=np.float32)
    highs = np.empty(c, dtype=np.float32)
    for i in range(c):
        lows[i], highs[i] = np.percentile(arr[:, :, i], (1, 99))
    span = highs - lows
    if np.all(span >= 0.9 * 255.0):
        return image
    out = np.empty_like(arr)
    for i in range(c):
        s  = span[i] if span[i] > 0 else 1.0
        ch = arr[:, :, i].astype(np.float32)
        out[:, :, i] = np.clip((ch - lows[i]) * 255.0 / s, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


# ===========================================================================
# Model inference
# ===========================================================================

def _pil_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    """PIL RGB → normalised float32 tensor (1, 3, H', W').  H',W' mult-of-14."""
    w, h = image.size
    pad_h = (14 - h % 14) % 14
    pad_w = (14 - w % 14) % 14
    if pad_h or pad_w:
        fill = tuple((np.array(_MEAN) * 255).astype(np.uint8).tolist())
        canvas = Image.new("RGB", (w + pad_w, h + pad_h), fill)
        canvas.paste(image, (0, 0))
        image = canvas
    arr    = np.asarray(image, dtype=np.float32) / 255.0
    arr    = (arr - _MEAN) / _STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor.unsqueeze(0).to(device)


def _predict(image: Image.Image, model: Any, device: str) -> np.ndarray:
    """Run one crop through DA V2.  Returns float32 array (orig H, orig W).

    The tile is passed at native resolution — NOT downscaled to 518 px.
    The model output is bilinearly up-sampled back to the original crop size.
    No per-tile normalisation is applied.
    """
    orig_h, orig_w = image.height, image.width
    tensor = _pil_to_tensor(image, device)
    with torch.inference_mode():
        out = model(tensor)
    out = functional.interpolate(
        out.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    return out.cpu().float().numpy()


# ===========================================================================
# Tile-grid planning
# ===========================================================================

def _adaptive_edges(length: int, target: int = CORE_TILE_SIZE,
                    lo: int = MIN_CORE_SIZE, hi: int = MAX_CORE_SIZE) -> list[int]:
    """Split axis into near-``target`` cores covering [0, length] exactly.

    Returns N+1 boundary positions.  Every pixel belongs to exactly one core.
    """
    ideal = max(1, round(length / target))
    n_min = max(1, int(np.ceil(length / hi)))
    n_max = int(np.floor(length / lo))
    if n_min <= n_max:
        count = min(max(ideal, n_min), n_max)
    else:
        count = ideal
    return [(i * length) // count for i in range(count + 1)]


# ===========================================================================
# Core-only tiled inference
# ===========================================================================

def _estimate_shift(
    canvas_slice: np.ndarray,
    tile_slice:   np.ndarray,
    trim_frac: float = 0.10,
) -> float:
    """Robust shift-only alignment: find delta so tile_slice + delta ≈ canvas_slice.

    Uses the trimmed median of (canvas - tile) over the feather zone.
    Shift-only (no scale) alignment preserves local depth gradients.

    Returns 0.0 when fewer than 16 valid pixels are available.
    """
    mask = np.isfinite(canvas_slice) & np.isfinite(tile_slice) & (canvas_slice != 0.0)
    if mask.sum() < 16:
        return 0.0
    diff = (canvas_slice[mask] - tile_slice[mask]).astype(np.float64)
    lo  = np.percentile(diff, trim_frac * 100)
    hi  = np.percentile(diff, (1.0 - trim_frac) * 100)
    tri = diff[(diff >= lo) & (diff <= hi)]
    if len(tri) < 4:
        tri = diff
    return float(np.median(tri))


def _make_feather_mask(h: int, w: int,
                       top: bool, bottom: bool,
                       left: bool, right: bool,
                       feather: int) -> np.ndarray:
    """Return a float32 mask in [0, 1] that tapers to 0 at requested edges.

    The taper uses a raised-cosine (Hann) ramp over ``feather`` pixels.
    Edges marked False (image boundary) stay at 1.0 — no taper needed.
    """
    def hann_rise(n: int) -> np.ndarray:
        t = np.linspace(0.0, 0.5, n, endpoint=False)
        return (0.5 * (1.0 - np.cos(2.0 * np.pi * t))).astype(np.float32)

    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)

    f = min(feather, h // 4, w // 4)   # guard against tiny tiles
    if f > 0:
        if top:
            wy[:f] = hann_rise(f)
        if bottom:
            wy[-f:] = hann_rise(f)[::-1]
        if left:
            wx[:f] = hann_rise(f)
        if right:
            wx[-f:] = hann_rise(f)[::-1]

    return np.outer(wy, wx)


def _tiled_inference(image: Image.Image, model: Any, device: str) -> np.ndarray:
    """Core-only tiled inference with shift alignment and narrow feathering.

    Algorithm
    ---------
    For each tile (yi, xi):

    1. Extract a context crop:
          [core_y0 - CONTEXT_MARGIN, core_y1 + CONTEXT_MARGIN] × (x similarly)
       clamped to image boundaries.

    2. Run the model on the full context crop.

    3. Slice out only the core prediction from the model output:
          ctx_pred[core_y0 - crop_y0 : core_y1 - crop_y0, ...]
       This is the region for which the model had full context on all sides.

    4. Estimate a shift offset between the current core prediction and the
       already-placed canvas in the FEATHER-pixel border of the core.
       (Only relevant for internal tiles where the canvas has already been
       filled by adjacent tiles.)

    5. Apply the shift to the core prediction (no scale change — scaling
       would change local gradients and blur depth transitions).

    6. Write the core prediction into the canvas using a narrow Hann-feather
       mask so that the transition at core edges is smooth but no more than
       FEATHER px wide.

    7. Accumulate the feather mask into a weight canvas.  Final depth =
       weighted_sum / weight_sum.

    Key property: each core pixel is contributed by EXACTLY ONE forward pass.
    The only averaging occurs in the narrow FEATHER-px border zone between
    adjacent cores.  Interior core pixels are written once, unmodified.
    """
    width, height = image.size
    y_edges = _adaptive_edges(height)
    x_edges = _adaptive_edges(width)
    n_y = len(y_edges) - 1
    n_x = len(x_edges) - 1

    canvas  = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)

    for yi in range(n_y):
        for xi in range(n_x):
            # ── Core boundaries ──────────────────────────────────────────
            core_y0 = y_edges[yi]
            core_y1 = y_edges[yi + 1]
            core_x0 = x_edges[xi]
            core_x1 = x_edges[xi + 1]
            core_h  = core_y1 - core_y0
            core_w  = core_x1 - core_x0

            # ── Context crop (clamped to image) ──────────────────────────
            crop_y0 = max(0, core_y0 - CONTEXT_MARGIN)
            crop_y1 = min(height, core_y1 + CONTEXT_MARGIN)
            crop_x0 = max(0, core_x0 - CONTEXT_MARGIN)
            crop_x1 = min(width,  core_x1 + CONTEXT_MARGIN)

            # ── Model inference ──────────────────────────────────────────
            crop_img = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            ctx_pred = _predict(crop_img, model, device)

            # ── Slice out the core prediction ─────────────────────────────
            cy0 = core_y0 - crop_y0   # offsets within ctx_pred
            cy1 = core_y1 - crop_y0
            cx0 = core_x0 - crop_x0
            cx1 = core_x1 - crop_x0
            core_pred = ctx_pred[cy0:cy1, cx0:cx1]   # (core_h, core_w) — EXACT

            # ── Shift alignment (internal edges only) ────────────────────
            # We estimate a shift from the already-placed canvas pixels in
            # the narrow feather zone on each internal edge.
            shift = 0.0
            f = FEATHER

            # Left internal edge: canvas columns [core_x0, core_x0+f)
            if xi > 0 and f < core_w:
                canvas_zone = canvas[core_y0:core_y1, core_x0:core_x0 + f]
                tile_zone   = core_pred[:, :f].astype(np.float64)
                shift = _estimate_shift(canvas_zone, tile_zone)

            # Top internal edge (only if left edge gave insufficient pixels)
            elif yi > 0 and f < core_h:
                canvas_zone = canvas[core_y0:core_y0 + f, core_x0:core_x1]
                tile_zone   = core_pred[:f, :].astype(np.float64)
                shift = _estimate_shift(canvas_zone, tile_zone)

            core_pred = core_pred.astype(np.float64) + shift

            # ── Feather mask ──────────────────────────────────────────────
            # Taper at internal edges only — image-boundary edges stay at 1.
            mask = _make_feather_mask(
                core_h, core_w,
                top    = (yi > 0),
                bottom = (yi < n_y - 1),
                left   = (xi > 0),
                right  = (xi < n_x - 1),
                feather = f,
            )

            # ── Accumulate into canvas ────────────────────────────────────
            canvas [core_y0:core_y1, core_x0:core_x1] += core_pred * mask
            weights[core_y0:core_y1, core_x0:core_x1] += mask

    # Normalise (weight > 0 everywhere by construction).
    eps = np.finfo(np.float64).eps
    result = (canvas / np.maximum(weights, eps)).astype(np.float32)

    # Very mild spike removal only (0.05–99.95%) — does not clip gradients.
    lo, hi = np.nanpercentile(result, (0.05, 99.95))
    return np.clip(result, lo, hi).astype(np.float32)


# ===========================================================================
# Single-tile inference (small images)
# ===========================================================================

def _single_inference(image: Image.Image, model: Any, device: str) -> np.ndarray:
    """Predict depth for an image that fits in a single forward pass."""
    raw = _predict(image, model, device)
    lo, hi = np.nanpercentile(raw, (0.1, 99.9))
    return np.clip(raw, lo, hi).astype(np.float32)


# ===========================================================================
# Backward-compatibility shims (used by tests and tile_boundary_metric)
# ===========================================================================

def _clip_outliers(depth: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(depth, (1, 99))
    return np.clip(depth, lo, hi).astype(np.float32)


def _clip_tile_outliers(tile: np.ndarray,
                        low_pct: float = 0.5,
                        high_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.nanpercentile(tile, (low_pct, high_pct))
    return np.clip(tile, lo, hi).astype(np.float32)


def _robust_affine(ref: np.ndarray, src: np.ndarray,
                   trim_frac: float = 0.10) -> tuple[float, float]:
    """Kept for test compatibility.  Trimmed-OLS affine estimator."""
    mask = np.isfinite(ref) & np.isfinite(src)
    if mask.sum() < 16:
        return 1.0, 0.0
    r = ref[mask].astype(np.float64)
    s = src[mask].astype(np.float64)
    if s.std() < 1e-6 * (abs(s.mean()) + 1.0):
        return 1.0, float(r.mean() - s.mean())
    res = np.abs(r - s)
    lo  = np.percentile(res, trim_frac * 100)
    hi  = np.percentile(res, (1.0 - trim_frac) * 100)
    keep = (res >= lo) & (res <= hi)
    if keep.sum() < 8:
        keep = np.ones(len(r), dtype=bool)
    A = np.column_stack([s[keep], np.ones(keep.sum())])
    try:
        result, *_ = np.linalg.lstsq(A, r[keep], rcond=None)
        scale, shift = float(result[0]), float(result[1])
    except np.linalg.LinAlgError:
        return 1.0, 0.0
    return float(np.clip(scale, 0.2, 5.0)), shift


def _robust_shift_only(ref: np.ndarray, src: np.ndarray,
                       trim_frac: float = 0.10) -> float:
    """Kept for test compatibility.  Trimmed-median shift estimator."""
    return _estimate_shift(ref, src, trim_frac)


def _tile_weight(crop_y0, crop_x0, height, width,
                 y_edges, x_edges, y_index, x_index):
    """Backward-compat alias: returns a Hann-style weight array."""
    return _make_feather_mask(
        height, width,
        top    = y_index > 0,
        bottom = y_index + 1 < len(y_edges) - 1,
        left   = x_index > 0,
        right  = x_index + 1 < len(x_edges) - 1,
        feather = FEATHER,
    )


def _hann_weight(crop_y0, crop_x0, height, width,
                 y_edges, x_edges, y_index, x_index):
    """Backward-compat alias."""
    return _tile_weight(crop_y0, crop_x0, height, width,
                        y_edges, x_edges, y_index, x_index)


def _align_tiles_affine(raw_tiles, tile_coords, y_edges, x_edges,
                        image_h, image_w, context):
    """Backward-compat shim: applies per-tile shift alignment only.

    Called by tests.  The production path now uses _tiled_inference directly.
    """
    n_y = len(y_edges) - 1
    n_x = len(x_edges) - 1
    result = []
    for yi in range(n_y):
        row = []
        for xi in range(n_x):
            row.append(_clip_tile_outliers(raw_tiles[yi][xi]))
        result.append(row)
    return result


# ===========================================================================
# Public API
# ===========================================================================

def get_depth(image_path: str, fast: bool = False,
              use_onnx: bool = False) -> np.ndarray:
    """Estimate a 2-D relative-depth map for one image.

    Parameters
    ----------
    image_path : path to JPG, PNG, or GeoTIFF.
    fast       : use ViT-Small without tiling (quick iteration).
    use_onnx   : not supported; raises NotImplementedError.
    """
    if use_onnx:
        raise NotImplementedError("--use-onnx is not supported with the .pth backend.")

    image   = _clip_exposure(_load_image(image_path))
    encoder = "vits" if fast else "vitl"
    model   = load_model(encoder=encoder, device=_DEVICE)

    if fast or max(image.size) <= MAX_CORE_SIZE + CONTEXT_MARGIN:
        return _single_inference(image, model, _DEVICE)

    return _tiled_inference(image, model, _DEVICE)


def get_depth_with_meta(
    image_path: str,
    fast: bool = False,
) -> tuple[np.ndarray, dict | None]:
    """Estimate relative depth and return GeoTIFF metadata when available."""
    raw_image, geo_meta = _load_image_geo(image_path)
    image   = _clip_exposure(raw_image)
    encoder = "vits" if fast else "vitl"
    model   = load_model(encoder=encoder, device=_DEVICE)

    if fast or max(image.size) <= MAX_CORE_SIZE + CONTEXT_MARGIN:
        return _single_inference(image, model, _DEVICE), geo_meta

    return _tiled_inference(image, model, _DEVICE), geo_meta


# ===========================================================================
# Output saving
# ===========================================================================

def save_depth_outputs(
    depth_array: np.ndarray,
    output_dir: str,
    filename: str,
    geo_meta: dict | None = None,
    rgb_image: Any | None = None,
    save_visualization: bool = True,
) -> None:
    """Save depth: .npy, viridis .png, optional GeoTIFF, optional vis figure.

    GeoTIFF values are relative depth — NOT metric elevation.
    """
    if depth_array.ndim != 2:
        raise ValueError("depth_array must be 2-D")
    dst  = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem

    np.save(dst / f"{stem}.npy", depth_array)

    import matplotlib.pyplot as plt
    plt.imsave(dst / f"{stem}_depth.png", depth_array, cmap="viridis")

    if geo_meta is not None:
        tif = dst / f"{stem}_depth.tif"
        geo_utils.write_depth_geotiff(depth_array, geo_meta, tif)
        logger.info("Wrote GeoTIFF: %s", tif.name)

    if save_visualization:
        try:
            from depth_mapping.visualize import save_depth_figure
            save_depth_figure(depth_array, output_dir=dst,
                              stem=f"{stem}_depth", rgb_image=rgb_image)
        except Exception as exc:
            logger.warning("Visualization skipped: %s", exc)


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate relative depth with Depth Anything V2.",
        prog="python -m depth_mapping",
    )
    parser.add_argument("--image",          required=True)
    parser.add_argument("--output-dir",     default="outputs")
    parser.add_argument("--fast",           action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    if args.checkpoint_dir:
        os.environ["DA_CHECKPOINT_DIR"] = args.checkpoint_dir

    t0 = time.perf_counter()
    depth_arr, geo_meta = get_depth_with_meta(args.image, fast=args.fast)
    elapsed = time.perf_counter() - t0

    save_depth_outputs(depth_arr, args.output_dir,
                       Path(args.image).stem, geo_meta=geo_meta)

    mode = "fast/Small" if args.fast else "tiled/Large"
    print(f"shape={depth_arr.shape}  "
          f"min={depth_arr.min():.3f}  max={depth_arr.max():.3f}  "
          f"std={depth_arr.std():.3f}")
    print(f"time ({mode}, {_DEVICE.upper()}): {elapsed:.2f}s")
    if geo_meta:
        print(f"GeoTIFF: {Path(args.output_dir) / (Path(args.image).stem + '_depth.tif')}")


if __name__ == "__main__":
    main()
