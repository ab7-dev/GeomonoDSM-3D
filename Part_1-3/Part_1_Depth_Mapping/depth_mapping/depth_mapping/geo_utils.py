"""GeoTIFF read/write utilities for the depth-mapping pipeline.

Stage 6 — metadata preservation
Stage 7 — georeferenced relative-depth output

This module is the single place where rasterio is used for I/O.  The rest of
the pipeline (model inference, tiling, blending) never touches rasterio so that
it remains format-agnostic.

Public API
----------
read_geotiff_rgb(path)
    Open a GeoTIFF, extract an RGB PIL image for model inference, and return
    all geospatial metadata in a plain dict.  No metadata is discarded.

write_depth_geotiff(depth_array, geo_meta, output_path)
    Write a float32 relative-depth array as a single-band GeoTIFF that is
    spatially co-registered with the source image.

GeoMeta dict schema
-------------------
The dict returned by read_geotiff_rgb() and accepted by write_depth_geotiff()
contains exactly these keys:

    crs         : rasterio.crs.CRS  – coordinate reference system
    transform   : affine.Affine     – pixel-to-world affine transform
    width       : int               – image width in pixels
    height      : int               – image height in pixels
    bounds      : rasterio.coords.BoundingBox
    res         : tuple(float, float) – (x_res, y_res) in CRS units/pixel
    nodata      : float | None      – NoData value from source (may be None)
    count       : int               – number of bands in source
    dtype       : str               – source band dtype string (e.g. "uint8")
    driver      : str               – source driver (typically "GTiff")

These are intentionally plain Python / rasterio types so the dict can be
passed across module boundaries without rasterio being imported at the call
site.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def read_geotiff_rgb(path: str | Path) -> tuple[Image.Image, dict[str, Any]]:
    """Open a GeoTIFF and return (RGB PIL image, geo_meta dict).

    The PIL image is suitable for direct input to the depth model.
    All geospatial metadata is captured in ``geo_meta`` before the rasterio
    file handle closes — nothing is silently discarded.

    For multi-band rasters, bands 1–3 are used as R, G, B.
    A single-band (greyscale) raster is replicated to three identical channels.
    Float / uint16 rasters are contrast-stretched to uint8 using the 1–99th
    percentile of the combined pixel distribution.

    Parameters
    ----------
    path:
        Filesystem path to the GeoTIFF.

    Returns
    -------
    image : PIL.Image.Image
        Three-channel uint8 RGB image.
    geo_meta : dict
        Geospatial metadata dict (see module docstring for schema).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If rasterio is not installed.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for GeoTIFF support. "
            "Install it with: pip install rasterio"
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    with rasterio.open(path) as src:
        # ── Capture all metadata before the handle closes ──────────────────
        geo_meta: dict[str, Any] = {
            "crs"       : src.crs,
            "transform" : src.transform,
            "width"     : src.width,
            "height"    : src.height,
            "bounds"    : src.bounds,
            "res"       : src.res,
            "nodata"    : src.nodata,
            "count"     : src.count,
            "dtype"     : src.dtypes[0],
            "driver"    : src.driver,
        }

        # ── Read pixel data (up to 3 bands) ────────────────────────────────
        band_count = min(src.count, 3)
        data = src.read(list(range(1, band_count + 1)))  # (bands, H, W)

    # ── Band handling ───────────────────────────────────────────────────────
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)   # greyscale → RGB
    elif data.shape[0] == 2:
        data = np.concatenate([data, data[:1]], axis=0)  # pad to 3 bands

    # ── dtype normalisation → uint8 ────────────────────────────────────────
    data_hwc = np.moveaxis(data, 0, -1)   # (H, W, 3)
    data_hwc = _to_uint8(data_hwc)

    image = Image.fromarray(data_hwc, mode="RGB")

    logger.debug(
        "read_geotiff_rgb: %s  size=%dx%d  CRS=%s  res=%.4f m/px",
        path.name, geo_meta["width"], geo_meta["height"],
        geo_meta["crs"], geo_meta["res"][0],
    )

    return image, geo_meta


def write_depth_geotiff(
    depth_array: np.ndarray,
    geo_meta: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a float32 relative-depth array as a co-registered single-band GeoTIFF.

    The output is spatially aligned with the source image: it carries the same
    CRS and affine transform, so it can be overlaid directly in any GIS tool.

    The values are relative depth (dimensionless model output).  They are NOT
    metric elevation, DSM, or height above ground.

    Parameters
    ----------
    depth_array:
        2-D float32 numpy array of shape (height, width).
    geo_meta:
        Metadata dict produced by ``read_geotiff_rgb()``.
    output_path:
        Destination path for the GeoTIFF (parent dirs created automatically).

    Returns
    -------
    Path
        Resolved path of the written file.

    Raises
    ------
    ValueError
        If depth_array dimensions do not match the stored width/height.
    ImportError
        If rasterio is not installed.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for GeoTIFF output. "
            "Install it with: pip install rasterio"
        ) from exc

    if depth_array.ndim != 2:
        raise ValueError(
            f"depth_array must be 2-D, got shape {depth_array.shape}"
        )

    expected_h = geo_meta["height"]
    expected_w = geo_meta["width"]
    actual_h, actual_w = depth_array.shape

    if actual_h != expected_h or actual_w != expected_w:
        raise ValueError(
            f"depth_array shape ({actual_h}, {actual_w}) does not match "
            f"geo_meta dimensions ({expected_h}, {expected_w}). "
            "The depth array must have the same spatial extent as the source."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver"    : "GTiff",
        "dtype"     : "float32",
        "width"     : expected_w,
        "height"    : expected_h,
        "count"     : 1,
        "crs"       : geo_meta["crs"],
        "transform" : geo_meta["transform"],
        # LZW compression keeps file size reasonable for float32 rasters
        # without any lossy degradation.
        "compress"  : "lzw",
        "predictor" : 3,   # horizontal differencing for float data
        "tiled"     : True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    # Preserve NoData if the source had one (downstream tools may need it).
    if geo_meta.get("nodata") is not None:
        profile["nodata"] = geo_meta["nodata"]

    arr = depth_array.astype(np.float32)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(
            DEPTH_TYPE="relative",
            SOURCE_CRS=str(geo_meta["crs"]),
            NOTE=(
                "Relative depth from Depth Anything V2. "
                "Values are dimensionless model output — NOT metric elevation."
            ),
        )

    logger.info(
        "write_depth_geotiff: wrote %s  (%dx%d float32, CRS=%s)",
        output_path.name, expected_w, expected_h, geo_meta["crs"],
    )

    return output_path.resolve()


def validate_geotiff_alignment(
    source_meta: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Re-open a written GeoTIFF and compare its metadata against the source.

    Returns a dict with one key per checked property whose value is a dict:
        { "source": <value>, "output": <value>, "match": bool }

    All keys are expected to match.  Any mismatch is a bug in write_depth_geotiff.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("rasterio required") from exc

    output_path = Path(output_path)
    if not output_path.is_file():
        raise FileNotFoundError(f"Output file not found: {output_path}")

    with rasterio.open(output_path) as dst:
        out_meta = {
            "crs"      : dst.crs,
            "transform": dst.transform,
            "width"    : dst.width,
            "height"   : dst.height,
            "bounds"   : dst.bounds,
            "res"      : dst.res,
            "dtype"    : dst.dtypes[0],
            "count"    : dst.count,
        }

    checks: dict[str, Any] = {}
    for key in ("crs", "transform", "width", "height", "bounds", "res"):
        src_val = source_meta[key]
        out_val = out_meta[key]
        checks[key] = {
            "source" : src_val,
            "output" : out_val,
            "match"  : _meta_equal(key, src_val, out_val),
        }

    checks["dtype"] = {
        "source" : "float32 (depth output)",
        "output" : out_meta["dtype"],
        "match"  : out_meta["dtype"] == "float32",
    }
    checks["count"] = {
        "source" : "1 (depth band)",
        "output" : out_meta["count"],
        "match"  : out_meta["count"] == 1,
    }
    return checks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Stretch arbitrary numeric raster values into the uint8 range.

    Uses the 1–99th percentile of the combined data so that extreme outliers
    (sensor artefacts, cloud shadows) do not dominate the contrast stretch.
    For data already stored as uint8 (0–255) the array is returned as-is.
    """
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.dtype == np.uint8:
        return image
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _meta_equal(key: str, a: Any, b: Any) -> bool:
    """Compare two metadata values, handling floating-point tolerance."""
    if key == "transform":
        # Affine objects compare element-wise; use near-equality for floats.
        try:
            import numpy as np
            return np.allclose(list(a), list(b), rtol=1e-6, atol=1e-6)
        except Exception:
            return a == b
    if key == "res":
        try:
            return (
                abs(a[0] - b[0]) < 1e-8
                and abs(a[1] - b[1]) < 1e-8
            )
        except Exception:
            return a == b
    if key == "bounds":
        try:
            return all(abs(getattr(a, f) - getattr(b, f)) < 1e-6
                       for f in ("left", "bottom", "right", "top"))
        except Exception:
            return a == b
    return a == b
