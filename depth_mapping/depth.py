"""Single-image relative depth estimation with Depth Anything V2.

Depth Anything predicts *relative* depth, not surveyed elevation or distance in
metres.  Larger values in the returned array generally indicate pixels that are
closer to the camera (or taller in a nadir-looking image).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
SMALL_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
NATIVE_RESOLUTION = 518
MIN_TILE_SIZE = 400
MAX_TILE_SIZE = 600
TILE_OVERLAP = 128

# CPU inference benefits substantially from allowing PyTorch to schedule work
# across every logical Ryzen core available to this process.
torch.set_num_threads(os.cpu_count() or 1)

# Model objects are cached after the first call so batch jobs do not reload them.
_MODEL: Any | None = None
_PROCESSOR: Any | None = None
_LOADED_MODEL_ID: str | None = None
_ONNX_SESSIONS: dict[str, Any] = {}


def _load_image(image_path: str) -> Image.Image:
    """Load a JPG, PNG, or GeoTIFF and return a three-channel RGB PIL image.

    Rasterio is used first for GeoTIFFs because it reliably handles multi-band
    satellite rasters.  Other formats use Pillow directly.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")

    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as source:
                # Use the first three bands; repeat a lone grayscale band as RGB.
                band_count = min(source.count, 3)
                data = source.read(list(range(1, band_count + 1))).astype(np.float32)
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            elif data.shape[0] == 2:
                data = np.concatenate([data, data[:1]], axis=0)
            data = np.moveaxis(data, 0, -1)
            data = _to_uint8(data)
            return Image.fromarray(data, mode="RGB")
        except ImportError:
            # Pillow may still open simple GeoTIFFs when rasterio is unavailable.
            pass

    return Image.open(path).convert("RGB")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale arbitrary numeric raster values into the uint8 image range."""
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.dtype == np.uint8:
        return image
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _clip_exposure(image: Image.Image) -> Image.Image:
    """Clip extreme brightness values to reduce over/under-exposure influence.

    The 1st and 99th percentiles are calculated independently per colour band,
    then stretched back to the normal 0--255 range.
    """
    pixels = np.asarray(image, dtype=np.float32)
    low = np.percentile(pixels, 1, axis=(0, 1), keepdims=True)
    high = np.percentile(pixels, 99, axis=(0, 1), keepdims=True)
    scale = np.where(high > low, high - low, 1.0)
    corrected = np.clip((pixels - low) * 255.0 / scale, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected, mode="RGB")


def _load_model(model_id: str | None = None) -> tuple[Any, Any]:
    """Load and cache one CPU-only Hugging Face image processor and model."""
    global _MODEL, _PROCESSOR, _LOADED_MODEL_ID
    selected_model = model_id or MODEL_ID
    if _MODEL is None or _PROCESSOR is None or _LOADED_MODEL_ID != selected_model:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        _PROCESSOR = AutoImageProcessor.from_pretrained(selected_model)
        _MODEL = AutoModelForDepthEstimation.from_pretrained(selected_model)
        _MODEL.to("cpu").eval()
        _LOADED_MODEL_ID = selected_model
    return _PROCESSOR, _MODEL


def _predict_pytorch(image: Image.Image, processor: Any, model: Any) -> np.ndarray:
    """Run one image/tile through the PyTorch model and resize its result."""
    import torch.nn.functional as functional

    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        predicted_depth = model(**inputs).predicted_depth
    depth = functional.interpolate(
        predicted_depth.unsqueeze(1), size=(image.height, image.width), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    return np.asarray(depth, dtype=np.float32)


def _predict_onnx(image: Image.Image, processor: Any, model: Any, model_id: str) -> np.ndarray:
    """Export a model once, then run one tile with ONNX Runtime's CPU provider."""
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("--use-onnx requires onnxruntime. Install it with pip install onnxruntime onnx.") from error

    inputs = processor(images=image, return_tensors="pt")
    cache_path = Path.home() / ".cache" / "depth_module" / f"{model_id.replace('/', '_')}.onnx"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        class DepthOutputWrapper(torch.nn.Module):
            """Expose only the tensor that ONNX Runtime should return."""

            def __init__(self, wrapped_model: Any) -> None:
                super().__init__()
                self.wrapped_model = wrapped_model

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                return self.wrapped_model(pixel_values=pixel_values).predicted_depth

        torch.onnx.export(
            DepthOutputWrapper(model),
            (inputs["pixel_values"],),
            cache_path,
            input_names=["pixel_values"],
            output_names=["predicted_depth"],
            dynamic_axes={"pixel_values": {2: "height", 3: "width"}, "predicted_depth": {1: "height", 2: "width"}},
            opset_version=17,
        )
    session = _ONNX_SESSIONS.get(str(cache_path))
    if session is None:
        session = ort.InferenceSession(str(cache_path), providers=["CPUExecutionProvider"])
        _ONNX_SESSIONS[str(cache_path)] = session
    predicted_depth = session.run(None, {"pixel_values": inputs["pixel_values"].numpy()})[0]
    depth = torch.nn.functional.interpolate(
        torch.from_numpy(predicted_depth).unsqueeze(1),
        size=(image.height, image.width), mode="bicubic", align_corners=False,
    ).squeeze().numpy()
    return np.asarray(depth, dtype=np.float32)


def _adaptive_edges(length: int) -> list[int]:
    """Split one image axis into near-518px cores with no remainder.

    The ideal tile count is ``round(length / 518)``.  When an integer tile
    count can keep the resulting core in the requested 400--600px interval,
    it is clamped to that feasible range.  Very small/narrow images cannot be
    split into 400px cores, so they correctly use one complete source tile.
    """
    ideal_count = max(1, round(length / NATIVE_RESOLUTION))
    min_count = max(1, int(np.ceil(length / MAX_TILE_SIZE)))
    max_count = int(np.floor(length / MIN_TILE_SIZE))
    if min_count <= max_count:
        count = min(max(ideal_count, min_count), max_count)
    else:
        count = ideal_count
    # Integer boundaries cover every source pixel exactly once.  There is no
    # short last core and therefore no undersized edge tile to pad.
    return [(index * length) // count for index in range(count + 1)]


def _tile_weight(
    crop_y0: int,
    crop_x0: int,
    height: int,
    width: int,
    y_edges: list[int],
    x_edges: list[int],
    y_index: int,
    x_index: int,
) -> np.ndarray:
    """Create linear blending weights for the real-image overlap context."""
    weight_y = np.ones(height, dtype=np.float32)
    weight_x = np.ones(width, dtype=np.float32)
    if y_index:
        overlap = y_edges[y_index] - crop_y0
        if overlap > 0:
            weight_y[:overlap] = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
    if y_index + 1 < len(y_edges) - 1:
        overlap = crop_y0 + height - y_edges[y_index + 1]
        if overlap > 0:
            weight_y[-overlap:] = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
    if x_index:
        overlap = x_edges[x_index] - crop_x0
        if overlap > 0:
            weight_x[:overlap] = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
    if x_index + 1 < len(x_edges) - 1:
        overlap = crop_x0 + width - x_edges[x_index + 1]
        if overlap > 0:
            weight_x[-overlap:] = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
    return np.outer(weight_y, weight_x)


def _clip_outliers(depth: np.ndarray) -> np.ndarray:
    """Remove the extreme 1% tails that often become depth spikes."""
    low, high = np.nanpercentile(depth, (1, 99))
    return np.clip(depth, low, high).astype(np.float32)


def get_depth(image_path: str, fast: bool = False, use_onnx: bool = False) -> np.ndarray:
    """Estimate a 2-D relative-depth map for one image.

    By default, adaptive near-518px tile cores get 128px of real-image context
    on internal edges and are blended linearly. ``fast=True`` uses the Small
    model on the complete image instead. ``use_onnx=True`` uses a cached ONNX
    export.
    """
    image = _clip_exposure(_load_image(image_path))
    selected_model = SMALL_MODEL_ID if fast else MODEL_ID
    # Keep the no-argument call for the default path so lightweight test doubles
    # can replace _load_model without needing knowledge of checkpoint names.
    processor, model = _load_model() if not fast else _load_model(selected_model)
    predict = _predict_onnx if use_onnx else _predict_pytorch
    if fast:
        return _clip_outliers(predict(image, processor, model, selected_model) if use_onnx else predict(image, processor, model))

    width, height = image.size
    stitched = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    y_edges, x_edges = _adaptive_edges(height), _adaptive_edges(width)
    context = TILE_OVERLAP // 2
    for y_index in range(len(y_edges) - 1):
        for x_index in range(len(x_edges) - 1):
            crop_y0 = max(0, y_edges[y_index] - context)
            crop_y1 = min(height, y_edges[y_index + 1] + context)
            crop_x0 = max(0, x_edges[x_index] - context)
            crop_x1 = min(width, x_edges[x_index + 1] + context)
            tile = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            tile_depth = (
                predict(tile, processor, model, selected_model)
                if use_onnx
                else predict(tile, processor, model)
            )
            weight = _tile_weight(crop_y0, crop_x0, tile.height, tile.width, y_edges, x_edges, y_index, x_index)
            stitched[crop_y0:crop_y1, crop_x0:crop_x1] += tile_depth * weight
            weights[crop_y0:crop_y1, crop_x0:crop_x1] += weight
    # A positive denominator is guaranteed by coverage, but clamp defensively.
    return _clip_outliers(stitched / np.maximum(weights, np.finfo(np.float32).eps))


def save_depth_outputs(depth_array: np.ndarray, output_dir: str, filename: str) -> None:
    """Save raw relative depth and a human-readable viridis heatmap.

    ``filename`` may include an extension; it is removed so the output names are
    always ``<filename>.npy`` and ``<filename>_depth.png``.
    """
    if depth_array.ndim != 2:
        raise ValueError("depth_array must be a 2-D array")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    np.save(destination / f"{stem}.npy", depth_array)

    import matplotlib.pyplot as plt

    plt.imsave(destination / f"{stem}_depth.png", depth_array, cmap="viridis")


def main() -> None:
    """Run depth inference from the command line."""
    parser = argparse.ArgumentParser(description="Estimate relative depth with Depth Anything V2.")
    parser.add_argument("--image", required=True, help="Path to a JPG, PNG, or GeoTIFF image.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for .npy and heatmap outputs.")
    parser.add_argument("--fast", action="store_true", help="Use the Small checkpoint without tiling for quicker iteration.")
    parser.add_argument("--use-onnx", action="store_true", help="Export/cache ONNX and use ONNX Runtime on CPU.")
    args = parser.parse_args()

    started = time.perf_counter()
    depth = get_depth(args.image, fast=args.fast, use_onnx=args.use_onnx)
    inference_seconds = time.perf_counter() - started
    save_depth_outputs(depth, args.output_dir, Path(args.image).stem)
    print(f"Depth map: shape={depth.shape}, min={depth.min():.4f}, max={depth.max():.4f}")
    print(f"Inference time ({'Small / fast' if args.fast else 'Large / tiled'}): {inference_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
