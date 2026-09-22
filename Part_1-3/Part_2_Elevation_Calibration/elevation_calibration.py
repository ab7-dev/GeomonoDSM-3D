"""
Elevation Calibration — Part 2 of the GeoMonoDSM-3D pipeline.

Converts a relative depth array (Part 1 output) into a metric Digital Surface
Model (DSM) in metres by fitting a linear calibration:

    DSM(x, y) = a · depth(x, y) + b

The scale (a) and offset (b) are estimated against real SRTM elevations queried
live from https://api.opentopodata.org — no local DEM files needed.

Public entry point
------------------
    run_elevation_calibration(depth_array_path, geotiff_path, output_dir) -> dict

Returned dict keys
------------------
    "a"               : float  — linear scale factor
    "b"               : float  — additive offset (metres)
    "dsm_array"       : np.ndarray  — calibrated elevation array (H×W, float32)
    "dsm_tif_path"    : str    — path to calibrated_dsm.tif
    "preview_png_path": str    — path to calibrated_dsm_preview.png
    "gamus_validation": dict   — {"mae": float, "rmse": float} or {} on skip

Usage
-----
    python elevation_calibration.py \\
        --depth  path/to/depth.npy \\
        --geotiff path/to/image.tif \\
        [--output-dir Elevation_Calibration/outputs]
        [--checkpoint-dir D:/path/to/checkpoints]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elevation_calibration")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SRTM_API      = "https://api.opentopodata.org/v1/srtm30m"
GAMUS_ROWS    = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=earthflow%2FGAMUS&config=default&split=train&offset=0&length=5"
)
GAMUS_FIRST   = (
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=earthflow%2FGAMUS&config=default&split=train"
)
RETRY_DELAYS  = [2, 5, 10]       # seconds between SRTM retries
REQUEST_TIMEOUT = 30             # seconds per HTTP request


# ===========================================================================
# STEP 0 — helpers
# ===========================================================================

def _ensure_dir(path: str | Path) -> Path:
    """Create parent directories and return a Path object."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _pixel_to_latlon(
    row: int,
    col: int,
    transform,          # affine.Affine
    crs,                # rasterio.crs.CRS
) -> tuple[float, float]:
    """Convert pixel (row, col) → (lat, lon) in EPSG:4326.

    The GeoTIFF transform maps pixel (col, row) to (x, y) in its native CRS.
    If the CRS is not geographic (lat/lon), we reproject with pyproj.
    """
    from pyproj import Transformer

    # Affine: (col + 0.5, row + 0.5) → pixel centre in native CRS
    x = transform.c + (col + 0.5) * transform.a + (row + 0.5) * transform.b
    y = transform.f + (col + 0.5) * transform.d + (row + 0.5) * transform.e

    if crs.is_geographic:
        # Already lon/lat
        return float(y), float(x)      # (lat, lon)

    # Projected CRS → reproject to EPSG:4326
    transformer = Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lat), float(lon)


def _build_gcp_grid(height: int, width: int, n: int = 16) -> list[tuple[int, int]]:
    """Return ~n evenly-spaced (row, col) GCP locations inside the image.

    Uses a regular grid of sqrt(n)×sqrt(n) interior points, skipping the very
    edge (10% inset) to avoid NoData borders common in satellite imagery.
    """
    side = max(2, int(np.ceil(np.sqrt(n))))
    margin_r = max(1, int(height * 0.10))
    margin_c = max(1, int(width  * 0.10))

    rows = np.linspace(margin_r, height - margin_r - 1, side, dtype=int)
    cols = np.linspace(margin_c, width  - margin_c - 1, side, dtype=int)

    gcps = [(int(r), int(c)) for r in rows for c in cols]
    return gcps[:n]     # cap at requested n


# ===========================================================================
# STEP 1 — Reference elevation via SRTM API
# ===========================================================================

def _query_srtm(lat_lons: list[tuple[float, float]]) -> list[float | None]:
    """Batch-query OpenTopoData SRTM30m for a list of (lat, lon) pairs.

    Sends one request with pipe-separated locations.
    Returns a list of elevations in metres (None on per-point failure).

    Retries up to len(RETRY_DELAYS) times on transient HTTP errors.
    The free API allows 100 locations per request.
    """
    locations = "|".join(f"{lat},{lon}" for lat, lon in lat_lons)
    payload   = {"locations": locations}

    last_err: Exception | None = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS, start=1):
        if delay:
            logger.info("  SRTM retry %d/%d — waiting %ds …", attempt, len(RETRY_DELAYS) + 1, delay)
            time.sleep(delay)
        try:
            resp = requests.get(SRTM_API, params=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                logger.warning("  SRTM rate-limited (429). Will retry.")
                last_err = RuntimeError("Rate limited")
                continue
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("  SRTM network error (attempt %d): %s", attempt, exc)
            last_err = exc
            continue

        if data.get("status") != "OK":
            logger.warning("  SRTM API status: %s", data.get("status"))
            last_err = RuntimeError(f"API status={data.get('status')}")
            continue

        elevations: list[float | None] = []
        for result in data.get("results", []):
            elev = result.get("elevation")
            elevations.append(float(elev) if elev is not None else None)
        logger.info("  SRTM: received %d elevations (%d valid)",
                    len(elevations), sum(e is not None for e in elevations))
        return elevations

    raise RuntimeError(
        f"SRTM API failed after {len(RETRY_DELAYS) + 1} attempts. "
        f"Last error: {last_err}\n"
        "Check your internet connection or try again later."
    )


def fetch_reference_elevations(
    depth_array: np.ndarray,
    transform,
    crs,
    n_gcps: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick GCP grid, convert to lat/lon, query SRTM, return paired arrays.

    Returns
    -------
    depth_vals  : 1-D float32 array — depth values at valid GCP pixels
    srtm_vals   : 1-D float32 array — matching SRTM elevations in metres

    Both arrays have the same length (invalid/None points dropped).
    """
    height, width = depth_array.shape
    gcps = _build_gcp_grid(height, width, n=n_gcps)

    logger.info("Step 1 — Converting %d GCPs to lat/lon …", len(gcps))
    lat_lons: list[tuple[float, float]] = []
    for row, col in gcps:
        try:
            ll = _pixel_to_latlon(row, col, transform, crs)
            lat_lons.append(ll)
        except Exception as exc:
            logger.warning("  GCP (%d,%d) skipped — CRS error: %s", row, col, exc)

    if not lat_lons:
        raise RuntimeError("No GCPs could be converted to lat/lon. Check the GeoTIFF CRS.")

    logger.info("Step 1 — Querying SRTM for %d points …", len(lat_lons))
    elevations = _query_srtm(lat_lons)

    depth_vals, srtm_vals = [], []
    for (row, col), elev in zip(gcps[:len(lat_lons)], elevations):
        if elev is None:
            continue
        d = float(depth_array[row, col])
        if not np.isfinite(d):
            continue
        depth_vals.append(d)
        srtm_vals.append(elev)

    if len(depth_vals) < 2:
        raise RuntimeError(
            f"Only {len(depth_vals)} valid GCP(s) after SRTM lookup — need ≥ 2 to fit a line. "
            "Try a different image or check API connectivity."
        )

    logger.info("Step 1 — %d valid GCP pairs collected.", len(depth_vals))
    return np.array(depth_vals, dtype=np.float32), np.array(srtm_vals, dtype=np.float32)


# ===========================================================================
# STEP 2 — Robust linear calibration fit
# ===========================================================================

def fit_calibration(
    depth_vals: np.ndarray,
    srtm_vals:  np.ndarray,
) -> tuple[float, float, dict]:
    """Fit  elevation ≈ a·depth + b  robustly and return diagnostics.

    Uses RANSAC when there are ≥ 4 pairs (tolerant of bad SRTM lookups).
    Falls back to numpy least-squares for smaller sets.

    Returns
    -------
    a    : float — scale factor
    b    : float — offset in metres
    diag : dict  — RANSAC/fit diagnostics (n_inliers, residuals, etc.)
    """
    X = depth_vals.reshape(-1, 1).astype(np.float64)
    y = srtm_vals.astype(np.float64)
    n_total = len(X)

    diag: dict = {
        "method":          "polyfit_fallback",
        "n_total":         n_total,
        "n_inliers":       n_total,
        "inlier_ratio":    1.0,
        "residual_mae_m":  None,
        "residual_rmse_m": None,
        "residual_median_m": None,
    }

    if len(X) >= 4:
        try:
            from sklearn.linear_model import RANSACRegressor
            from sklearn.linear_model import LinearRegression

            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=max(4, int(len(X) * 0.5)),
                residual_threshold=50.0,   # 50 m tolerance — SRTM ~30 m RMSE
                max_trials=500,
                random_state=42,
            )
            ransac.fit(X, y)
            a = float(ransac.estimator_.coef_[0])
            b = float(ransac.estimator_.intercept_)
            inlier_mask = ransac.inlier_mask_
            n_inliers = int(inlier_mask.sum())
            inlier_ratio = n_inliers / n_total if n_total > 0 else 0.0

            # Compute residuals over inlier set
            y_pred_inliers = (a * X[inlier_mask, 0]) + b
            residuals      = y[inlier_mask] - y_pred_inliers
            mae    = float(np.mean(np.abs(residuals)))
            rmse   = float(np.sqrt(np.mean(residuals ** 2)))
            median = float(np.median(np.abs(residuals)))

            diag.update({
                "method":            "ransac",
                "n_inliers":         n_inliers,
                "inlier_ratio":      round(inlier_ratio, 4),
                "residual_mae_m":    round(mae, 3),
                "residual_rmse_m":   round(rmse, 3),
                "residual_median_m": round(median, 3),
            })

            logger.info(
                "Step 2 — RANSAC fit: a=%.4f  b=%.2f m  "
                "(inliers %d/%d  MAE=%.2f m  RMSE=%.2f m)",
                a, b, n_inliers, n_total, mae, rmse,
            )
            return a, b, diag
        except Exception as exc:
            logger.warning("Step 2 — RANSAC failed (%s), falling back to polyfit.", exc)

    # Fallback: least-squares via numpy.polyfit (degree 1)
    coeffs = np.polyfit(depth_vals.astype(np.float64), srtm_vals.astype(np.float64), 1)
    a, b = float(coeffs[0]), float(coeffs[1])

    # Compute residuals for polyfit
    y_pred = a * depth_vals.astype(np.float64) + b
    residuals = srtm_vals.astype(np.float64) - y_pred
    mae    = float(np.mean(np.abs(residuals)))
    rmse   = float(np.sqrt(np.mean(residuals ** 2)))
    median = float(np.median(np.abs(residuals)))

    diag.update({
        "residual_mae_m":    round(mae, 3),
        "residual_rmse_m":   round(rmse, 3),
        "residual_median_m": round(median, 3),
    })

    logger.info(
        "Step 2 — polyfit fallback: a=%.4f  b=%.2f m  MAE=%.2f m  RMSE=%.2f m",
        a, b, mae, rmse,
    )
    return a, b, diag


# ===========================================================================
# STEP 2b — Calibration quality diagnostics and validation
# ===========================================================================

# Thresholds for suspicious calibration detection.
_NEAR_ZERO_SLOPE    = 0.001   # |a| below this = near-zero slope
_MIN_DEPTH_STD      = 0.01    # insufficient depth variance
_MIN_SRTM_STD       = 1.0     # insufficient SRTM variance (metres)
_POOR_MAE_M         = 40.0    # residual MAE above this = poor fit
_MIN_INLIER_RATIO   = 0.40    # fewer inliers than this = suspect


def compute_calibration_diagnostics(
    a: float,
    b: float,
    depth_vals: np.ndarray,
    srtm_vals: np.ndarray,
    fit_diag: dict,
    n_gcps_requested: int = 16,
) -> dict:
    """Assemble full calibration diagnostic dict.

    Parameters
    ----------
    a, b        : calibration coefficients
    depth_vals  : valid depth GCP values (after SRTM filtering)
    srtm_vals   : matching SRTM elevations (metres)
    fit_diag    : diagnostics dict returned by fit_calibration()
    n_gcps_requested : how many GCPs were requested

    Returns
    -------
    dict with GCP stats, fit stats, and "quality" sub-dict.
    """
    depth_std  = float(np.std(depth_vals)) if len(depth_vals) > 1 else 0.0
    srtm_std   = float(np.std(srtm_vals))  if len(srtm_vals) > 1 else 0.0
    depth_rng  = [round(float(depth_vals.min()), 4), round(float(depth_vals.max()), 4)]
    srtm_rng   = [round(float(srtm_vals.min()),  2), round(float(srtm_vals.max()),  2)]

    quality = validate_calibration(a, b, depth_std, srtm_std, fit_diag)

    diag = {
        "a":                 round(a, 6),
        "b":                 round(b, 4),
        "n_gcps":            n_gcps_requested,
        "n_valid_gcps":      len(depth_vals),
        "n_inliers":         fit_diag.get("n_inliers"),
        "inlier_ratio":      fit_diag.get("inlier_ratio"),
        "residual_mae_m":    fit_diag.get("residual_mae_m"),
        "residual_rmse_m":   fit_diag.get("residual_rmse_m"),
        "residual_median_m": fit_diag.get("residual_median_m"),
        "depth_std":         round(depth_std, 4),
        "srtm_std":          round(srtm_std, 3),
        "depth_range":       depth_rng,
        "srtm_range":        srtm_rng,
        "fit_method":        fit_diag.get("method", "unknown"),
        "quality":           quality,
    }

    logger.info(
        "Calibration diagnostics: a=%.6f  b=%.3f  depth_std=%.4f  srtm_std=%.3f  "
        "inlier_ratio=%.2f  MAE=%.2f m  status=%s",
        a, b, depth_std, srtm_std,
        fit_diag.get("inlier_ratio", 1.0),
        fit_diag.get("residual_mae_m", 0.0),
        quality["status"],
    )

    if quality["status"] != "good":
        logger.warning(
            "Calibration quality is '%s': %s",
            quality["status"], quality["reason"],
        )

    return diag


def validate_calibration(
    a: float,
    b: float,
    depth_std: float,
    srtm_std: float,
    fit_diag: dict,
) -> dict:
    """Assess calibration reliability and return a quality classification.

    Checks performed (in priority order):
      1. Non-finite slope/intercept
      2. Near-zero slope (depth has no SRTM signal)
      3. Insufficient depth variance (flat depth map)
      4. Insufficient SRTM variance (flat terrain, no signal)
      5. Low inlier ratio (RANSAC found few consistent points)
      6. High residual MAE (poor linear fit)

    Does NOT blindly reject negative slopes — negative slope is valid if
    the depth convention is inverted (larger depth value = closer = higher
    elevation in SOME configurations). The sign is reported, not corrected.

    Returns
    -------
    dict with keys:
        "status"  : "good" | "warning" | "failed"
        "reason"  : human-readable explanation
        "checks"  : dict of individual check results
    """
    checks: dict[str, Any] = {}

    # --- Check 1: non-finite ---
    if not np.isfinite(a) or not np.isfinite(b):
        return {
            "status": "failed",
            "reason": f"Non-finite calibration coefficients: a={a}, b={b}",
            "checks": {"non_finite": True},
        }
    checks["non_finite"] = False

    # --- Check 2: near-zero slope ---
    checks["near_zero_slope"] = abs(a) < _NEAR_ZERO_SLOPE
    if checks["near_zero_slope"]:
        return {
            "status": "failed",
            "reason": (
                f"Near-zero slope a={a:.6f} (|a| < {_NEAR_ZERO_SLOPE}). "
                "Depth values have negligible linear relationship with SRTM elevations. "
                "DSM will be essentially flat (≈ b everywhere). "
                "Possible causes: depth map is constant, wrong image, or SRTM query failure."
            ),
            "checks": checks,
        }

    # --- Check 3: insufficient depth variance ---
    checks["low_depth_variance"] = depth_std < _MIN_DEPTH_STD
    if checks["low_depth_variance"]:
        return {
            "status": "failed",
            "reason": (
                f"Insufficient depth variance (std={depth_std:.4f} < {_MIN_DEPTH_STD}). "
                "The depth array appears nearly constant — calibration cannot be reliable."
            ),
            "checks": checks,
        }

    # --- Check 4: insufficient SRTM variance ---
    checks["low_srtm_variance"] = srtm_std < _MIN_SRTM_STD
    if checks["low_srtm_variance"]:
        # Warning rather than failure — very flat terrain is real
        return {
            "status": "warning",
            "reason": (
                f"Low SRTM elevation variance (std={srtm_std:.2f} m < {_MIN_SRTM_STD} m). "
                "Terrain may be very flat; calibration is less constrained. "
                "DSM may be accurate in flat areas but unreliable for height estimation."
            ),
            "checks": checks,
        }

    # --- Check 5: low inlier ratio ---
    inlier_ratio = fit_diag.get("inlier_ratio", 1.0)
    n_inliers    = fit_diag.get("n_inliers", 0)
    checks["low_inlier_ratio"] = inlier_ratio < _MIN_INLIER_RATIO
    if checks["low_inlier_ratio"]:
        return {
            "status": "warning",
            "reason": (
                f"Low RANSAC inlier ratio ({inlier_ratio:.2f}, {n_inliers} inliers). "
                "Many GCPs were rejected as outliers — calibration may be unreliable. "
                "Possible causes: mixed terrain types, SRTM errors, or domain mismatch."
            ),
            "checks": checks,
        }

    # --- Check 6: poor residual fit ---
    mae = fit_diag.get("residual_mae_m")
    if mae is not None:
        checks["high_residual"] = mae > _POOR_MAE_M
        if checks["high_residual"]:
            return {
                "status": "warning",
                "reason": (
                    f"High calibration residual MAE={mae:.1f} m > {_POOR_MAE_M} m. "
                    "Linear fit is poor — elevation surface may have systematic errors. "
                    "Building height estimates will have correspondingly higher uncertainty."
                ),
                "checks": checks,
            }
    else:
        checks["high_residual"] = False

    # --- Note on negative slope (informational, not a failure) ---
    sign_note = None
    if a < 0:
        sign_note = (
            f"Negative slope (a={a:.6f}): depth and SRTM elevation are inversely related. "
            "This can be valid if Depth Anything V2 assigns LARGER depth values to CLOSER "
            "(higher elevation) pixels at the GCP locations. Verify against known landmarks."
        )
        logger.info("Calibration note: negative slope. %s", sign_note)
        checks["negative_slope"] = True
    else:
        checks["negative_slope"] = False

    result: dict = {
        "status":  "good",
        "reason":  "All calibration quality checks passed.",
        "checks":  checks,
    }
    if sign_note:
        result["slope_sign_note"] = sign_note

    return result


# ===========================================================================
# STEP 3 — GAMUS structural-consistency validation
# ===========================================================================
# NOTE: GAMUS nDSM values are height-above-ground (relative to bare earth),
# NOT absolute geodetic elevation. The SRTM-calibrated DSM produced here
# represents absolute elevation above sea level. Comparing them directly
# gives a rough structural-consistency check only — buildings, trees, and
# terrain gradients should correlate — but the numeric MAE/RMSE should NOT
# be interpreted as absolute accuracy against a georeferenced benchmark.
# This is an indicative signal, not a quantitative accuracy guarantee.

def _fetch_gamus_rows() -> dict | None:
    """Try /rows endpoint, fall back to /first-rows. Returns parsed JSON or None."""
    for url, label in [(GAMUS_ROWS, "/rows"), (GAMUS_FIRST, "/first-rows")]:
        try:
            logger.info("Step 3 — Fetching GAMUS slice from %s …", label)
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("  GAMUS %s returned HTTP %d.", label, resp.status_code)
        except requests.RequestException as exc:
            logger.warning("  GAMUS %s network error: %s", label, exc)
    return None


def _find_column_names(features: dict) -> tuple[str | None, str | None]:
    """Inspect the features dict to find the RGB image and nDSM column names.

    Does NOT hardcode guessed names — walks the schema and looks for
    image-type features and numeric/array features likely to be the height map.
    Returns (rgb_col, ndsm_col) — either may be None if not found.
    """
    rgb_col  = None
    ndsm_col = None

    for col_name, col_info in features.items():
        dtype = col_info.get("dtype", "")
        col_type = col_info.get("_type", "")

        # Image columns: HuggingFace uses {"_type": "Image"} or similar
        is_image = (
            col_type in ("Image", "ImageFile")
            or dtype in ("image", "ImageFile")
            or "image" in col_name.lower()
            or "rgb"   in col_name.lower()
            or "photo" in col_name.lower()
        )
        # nDSM / height columns
        is_height = (
            "ndsm"   in col_name.lower()
            or "dsm"    in col_name.lower()
            or "height" in col_name.lower()
            or "chm"    in col_name.lower()
            or "dem"    in col_name.lower()
        )

        if is_image and rgb_col is None:
            rgb_col = col_name
        if is_height and ndsm_col is None:
            ndsm_col = col_name

    return rgb_col, ndsm_col


def _load_image_from_row(row_cell: dict) -> "np.ndarray | None":
    """Download an image from a GAMUS row cell into a numpy uint8 array.

    The cell value from the datasets-server API is typically:
        {"src": "https://...", "bytes": null, ...}
    or contains a base64 src URI.
    """
    from PIL import Image

    src = row_cell.get("src") or row_cell.get("path") or row_cell.get("url")
    raw_bytes = row_cell.get("bytes")

    if raw_bytes:
        # Sometimes bytes is a base64 string or raw bytes
        if isinstance(raw_bytes, str):
            import base64
            # Strip data URI prefix if present
            if "," in raw_bytes:
                raw_bytes = raw_bytes.split(",", 1)[1]
            raw_bytes = base64.b64decode(raw_bytes)
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return np.array(img, dtype=np.uint8)

    if src and src.startswith("http"):
        try:
            resp = requests.get(src, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return np.array(img, dtype=np.uint8)
        except Exception as exc:
            logger.warning("    Could not download image from %s: %s", src, exc)
            return None

    return None


def _load_ndsm_from_row(row_cell: dict) -> "np.ndarray | None":
    """Load nDSM data from a GAMUS row cell into a float32 numpy array."""
    from PIL import Image

    src = row_cell.get("src") or row_cell.get("path") or row_cell.get("url")
    raw_bytes = row_cell.get("bytes")

    def _bytes_to_array(b: bytes) -> np.ndarray | None:
        try:
            img = Image.open(io.BytesIO(b))
            arr = np.array(img, dtype=np.float32)
            return arr
        except Exception:
            pass
        # Try raw numpy
        try:
            arr = np.load(io.BytesIO(b))
            return arr.astype(np.float32)
        except Exception:
            return None

    if raw_bytes:
        if isinstance(raw_bytes, str):
            import base64
            if "," in raw_bytes:
                raw_bytes = raw_bytes.split(",", 1)[1]
            raw_bytes = base64.b64decode(raw_bytes)
        return _bytes_to_array(raw_bytes)

    if src and src.startswith("http"):
        try:
            resp = requests.get(src, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return _bytes_to_array(resp.content)
        except Exception as exc:
            logger.warning("    Could not download nDSM from %s: %s", src, exc)

    return None


def validate_against_gamus(a: float, b: float, depth_fn) -> dict:
    """Run structural-consistency check against 5 GAMUS samples.

    NOTE: This is an approximate check only. GAMUS nDSM is height-above-ground;
    our DSM is absolute elevation above sea level. The comparison tests whether
    the relative structure (buildings, trees) is preserved after calibration,
    not whether absolute elevations are correct.

    Parameters
    ----------
    a, b     : calibration coefficients (elevation = a·depth + b)
    depth_fn : callable(image_path_or_array) → depth_array
               Wraps Part 1 inference; accepts a numpy uint8 array.

    Returns
    -------
    dict with keys "mae", "rmse", "n_samples", "note" — or empty dict on skip.
    """
    logger.info("Step 3 — Starting GAMUS validation (structural-consistency check) …")

    data = _fetch_gamus_rows()
    if data is None:
        logger.warning("Step 3 — GAMUS API unavailable. Skipping validation.")
        return {}

    # Discover column names from the schema — don't guess
    features = data.get("features", {})
    # datasets-server returns features as a list of {name, type} dicts OR a dict
    if isinstance(features, list):
        features = {f["name"]: f["type"] for f in features if "name" in f}

    rgb_col, ndsm_col = _find_column_names(features)
    rows = data.get("rows", [])

    logger.info(
        "  GAMUS schema: %d columns, rgb_col=%r, ndsm_col=%r, %d rows",
        len(features), rgb_col, ndsm_col, len(rows),
    )

    if not rows:
        logger.warning("Step 3 — No GAMUS rows returned. Skipping.")
        return {}

    maes, rmses = [], []

    for i, row_entry in enumerate(rows):
        row = row_entry.get("row", row_entry)   # handle both wrapping styles

        try:
            # ── Load RGB image ──────────────────────────────────────────────
            rgb_arr = None
            if rgb_col and rgb_col in row:
                rgb_arr = _load_image_from_row(row[rgb_col])
            if rgb_arr is None:
                # Fallback: try any column that looks like an image
                for col, val in row.items():
                    if isinstance(val, dict) and ("src" in val or "bytes" in val):
                        rgb_arr = _load_image_from_row(val)
                        if rgb_arr is not None:
                            logger.debug("    Used fallback image col: %r", col)
                            break

            if rgb_arr is None:
                logger.warning("  Sample %d: could not load RGB — skipping.", i)
                continue

            # ── Load nDSM ground truth ──────────────────────────────────────
            ndsm_arr = None
            if ndsm_col and ndsm_col in row:
                ndsm_arr = _load_ndsm_from_row(row[ndsm_col])
            if ndsm_arr is None:
                for col, val in row.items():
                    if col == rgb_col:
                        continue
                    if isinstance(val, dict) and ("src" in val or "bytes" in val):
                        candidate = _load_ndsm_from_row(val)
                        if candidate is not None:
                            ndsm_arr = candidate
                            logger.debug("    Used fallback nDSM col: %r", col)
                            break

            if ndsm_arr is None:
                logger.warning("  Sample %d: could not load nDSM — skipping.", i)
                continue

            # ── Run Part 1 depth inference on the RGB array ─────────────────
            depth_pred = depth_fn(rgb_arr)       # (H, W) float32

            # ── Apply calibration ───────────────────────────────────────────
            dsm_pred = a * depth_pred + b        # (H, W) predicted elevation

            # ── Resize to match nDSM if needed ──────────────────────────────
            gt = ndsm_arr
            if gt.ndim == 3:
                gt = gt.mean(axis=2)             # multi-channel nDSM → scalar
            gt = gt.astype(np.float32)

            if dsm_pred.shape != gt.shape:
                from PIL import Image as PILImage
                gt_resized = np.array(
                    PILImage.fromarray(gt).resize(
                        (dsm_pred.shape[1], dsm_pred.shape[0]),
                        PILImage.BILINEAR,
                    ),
                    dtype=np.float32,
                )
                gt = gt_resized

            # ── Filter obviously invalid pixels ─────────────────────────────
            valid = np.isfinite(dsm_pred) & np.isfinite(gt)
            if valid.sum() < 100:
                logger.warning("  Sample %d: too few valid pixels (%d) — skipping.", i, valid.sum())
                continue

            mae  = float(np.mean(np.abs(dsm_pred[valid] - gt[valid])))
            rmse = float(np.sqrt(np.mean((dsm_pred[valid] - gt[valid]) ** 2)))
            maes.append(mae)
            rmses.append(rmse)
            logger.info("  Sample %d: MAE=%.2f m  RMSE=%.2f m", i, mae, rmse)

        except Exception as exc:
            logger.warning("  Sample %d: error — %s", i, exc)
            continue

    if not maes:
        logger.warning("Step 3 — No valid GAMUS samples processed. Skipping.")
        return {}

    result = {
        "mae":       float(np.mean(maes)),
        "rmse":      float(np.mean(rmses)),
        "n_samples": len(maes),
        "note": (
            "APPROXIMATE structural-consistency check only. "
            "GAMUS nDSM is height-above-ground (relative), "
            "our DSM is absolute elevation above sea level. "
            "These metrics measure structural correlation, not absolute accuracy."
        ),
    }
    logger.info(
        "Step 3 — GAMUS validation: mean MAE=%.2f m  mean RMSE=%.2f m  (n=%d)",
        result["mae"], result["rmse"], result["n_samples"],
    )
    return result


# ===========================================================================
# Output writers
# ===========================================================================

def save_dsm_geotiff(
    dsm_array: np.ndarray,
    geo_meta: dict,
    output_path: str | Path,
) -> Path:
    """Write calibrated DSM as a single-band float32 GeoTIFF with real CRS."""
    import rasterio

    output_path = _ensure_dir(output_path)

    profile = {
        "driver"    : "GTiff",
        "dtype"     : "float32",
        "width"     : geo_meta["width"],
        "height"    : geo_meta["height"],
        "count"     : 1,
        "crs"       : geo_meta["crs"],
        "transform" : geo_meta["transform"],
        "compress"  : "lzw",
        "predictor" : 3,
        "tiled"     : True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if geo_meta.get("nodata") is not None:
        profile["nodata"] = geo_meta["nodata"]

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dsm_array.astype(np.float32), 1)
        dst.update_tags(
            DSM_TYPE="calibrated_absolute_elevation",
            UNITS="metres_above_sea_level",
            METHOD="linear_srtm_calibration_ransac",
            NOTE="Produced by GeoMonoDSM-3D Part 2 — Elevation Calibration",
        )

    logger.info("Saved GeoTIFF: %s", output_path)
    return Path(output_path).resolve()


def save_dsm_preview(
    dsm_array: np.ndarray,
    output_path: str | Path,
    colormap: str = "terrain",
) -> Path:
    """Render the DSM with a terrain colormap and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(dsm_array, cmap=colormap, interpolation="bilinear")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Elevation (m)", fontsize=11)
    ax.set_title("Calibrated DSM — Elevation (metres)", fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info("Saved preview PNG: %s", output_path)
    return Path(output_path).resolve()


def save_calibration_report(
    a: float,
    b: float,
    dsm_array: np.ndarray,
    gamus_result: dict,
    output_path: str | Path,
    calib_diagnostics: dict | None = None,
) -> Path:
    """Write calibration_report.json with fitted params, diagnostics, and DSM stats."""
    output_path = _ensure_dir(output_path)

    valid = dsm_array[np.isfinite(dsm_array)]
    report: dict[str, Any] = {
        "calibration": {
            "a":                 a,
            "b":                 b,
            "method":            "RANSAC linear regression (depth → SRTM elevation)",
            "reference_dataset": "SRTM30m via api.opentopodata.org",
        },
        "dsm_stats": {
            "min_m":  float(valid.min())  if len(valid) else None,
            "max_m":  float(valid.max())  if len(valid) else None,
            "mean_m": float(valid.mean()) if len(valid) else None,
            "std_m":  float(valid.std())  if len(valid) else None,
            "shape":  list(dsm_array.shape),
        },
        "gamus_validation": gamus_result if gamus_result else "skipped",
    }

    # Embed full calibration quality diagnostics when available
    if calib_diagnostics:
        report["calibration_quality"] = calib_diagnostics.get("quality", {})
        report["calibration_details"] = {
            k: v for k, v in calib_diagnostics.items() if k != "quality"
        }

    output_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved report: %s", output_path)
    return output_path.resolve()


# ===========================================================================
# Main entry point
# ===========================================================================

def run_relative_calibration(
    depth_array_path: str | Path,
    output_dir: str | Path = "Elevation_Calibration/outputs",
) -> dict:
    """MODE A: Produce a relative DSM from a depth array without geographic reference.

    Used when the input image has NO CRS / geotransform (e.g. a plain PNG/JPG).
    SRTM or any external DEM cannot be queried because pixel→lat/lon conversion
    is impossible without a CRS.

    The output is NOT in absolute metres above sea level.  It is a relative
    surface model where larger values correspond to higher elevation relative
    to the minimum depth in the scene.  All values are non-negative.

    The result is clearly labelled as a relative DSM (rDSM) and must NOT be
    compared to absolute elevation figures.

    Parameters
    ----------
    depth_array_path : Path to .npy depth array from Part 1 (float32, H×W).
    output_dir       : Output directory for rDSM files.

    Returns
    -------
    dict with keys:
        "mode"              : "MODE_A_relative"
        "rdsm_array"        : np.ndarray float32 (H×W) — relative surface model
        "rdsm_min"          : float — minimum value (always 0.0 after normalisation)
        "rdsm_max"          : float — maximum value (relative, not metres AMSL)
        "rdsm_png_path"     : str — path to rdsm_preview.png
        "depth_stats"       : dict — raw depth statistics
        "note"              : human-readable disclaimer

    Notes
    -----
    Values are produced by:
        rDSM = depth - percentile(depth, 5)   [shift so minimum ≈ 0]
    Negative depths (rare) are then clamped to 0.
    Larger rDSM values indicate higher surfaces relative to the scene floor.
    The scale is in depth units (arbitrary), not metres.
    """
    depth_array_path = Path(depth_array_path)
    output_dir       = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GeoMonoDSM-3D  Part 2 — Relative Calibration (MODE A)")
    logger.info("=" * 60)
    logger.info("  No CRS available — producing relative DSM (rDSM).")
    logger.info("  Output is NOT absolute elevation in metres AMSL.")

    if not depth_array_path.is_file():
        raise FileNotFoundError(f"Depth array not found: {depth_array_path}")

    depth = np.load(depth_array_path)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2-D depth array, got shape {depth.shape}")

    depth = depth.astype(np.float32)

    # Replace non-finite values with the median
    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        raise ValueError("Depth array contains no finite values.")
    median_val = float(np.median(depth[finite_mask]))
    depth = np.where(finite_mask, depth, median_val)

    # Compute depth statistics
    depth_stats = {
        "shape":     list(depth.shape),
        "min":       float(depth.min()),
        "max":       float(depth.max()),
        "mean":      float(depth.mean()),
        "median":    float(np.median(depth)),
        "std":       float(depth.std()),
        "p5":        float(np.percentile(depth, 5)),
        "p95":       float(np.percentile(depth, 95)),
        "n_finite":  int(finite_mask.sum()),
        "n_total":   int(depth.size),
    }

    # Shift so scene floor ≈ 0, clip negative
    floor = float(np.percentile(depth, 5))
    rdsm  = np.clip(depth - floor, 0, None).astype(np.float32)

    # Save rDSM preview
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rdsm_png = output_dir / "rdsm_preview.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    axes[0].imshow(depth, cmap="viridis")
    axes[0].set_title("Relative Depth (raw)", fontsize=11)
    axes[0].axis("off")
    im = axes[1].imshow(rdsm, cmap="terrain")
    axes[1].set_title("Relative DSM (rDSM)", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label("Relative elevation (depth units)")
    fig.suptitle(
        "MODE A — Relative DSM  |  NOT absolute elevation metres AMSL",
        fontsize=10, color="gray"
    )
    fig.tight_layout()
    fig.savefig(rdsm_png, bbox_inches="tight", dpi=120)
    plt.close(fig)

    note = (
        "MODE A — relative DSM produced without geographic reference. "
        "Values are in depth units, NOT metres above sea level. "
        "To produce a metric DSM, provide a georeferenced GeoTIFF input "
        "with a valid CRS (MODE B)."
    )

    logger.info("  rDSM range: %.3f … %.3f (relative depth units)", rdsm.min(), rdsm.max())
    logger.info("  Saved: %s", rdsm_png)

    print("\n" + "=" * 60)
    print("  Part 2 — Relative Calibration (MODE A)  ✓  COMPLETE")
    print("=" * 60)
    print(f"  NOTE: {note}")
    print(f"  rDSM range : {rdsm.min():.3f} … {rdsm.max():.3f} (relative units)")
    print(f"  Output     : {rdsm_png}")
    print("=" * 60 + "\n")

    return {
        "mode":          "MODE_A_relative",
        "rdsm_array":    rdsm,
        "rdsm_min":      float(rdsm.min()),
        "rdsm_max":      float(rdsm.max()),
        "rdsm_png_path": str(rdsm_png),
        "depth_stats":   depth_stats,
        "note":          note,
    }


# ===========================================================================
# ===========================================================================

def run_elevation_calibration(
    depth_array_path: str | Path,
    geotiff_path: str | Path,
    output_dir: str | Path = "Elevation_Calibration/outputs",
    checkpoint_dir: str | Path | None = None,
    skip_gamus: bool = False,
) -> dict:
    """Convert a relative depth array into a metric DSM calibrated against SRTM.

    Parameters
    ----------
    depth_array_path:
        Path to the .npy file produced by Part 1 (float32, H×W).
    geotiff_path:
        Path to the source GeoTIFF (must match the depth array's spatial extent).
    output_dir:
        Directory where calibrated_dsm.tif, calibrated_dsm_preview.png and
        calibration_report.json are written.  Created automatically.
    checkpoint_dir:
        Optional path to the Depth Anything V2 checkpoints folder.
        Sets DA_CHECKPOINT_DIR env variable for Part 1.
    skip_gamus:
        When True, skips the GAMUS structural-consistency validation step.
        Faster — avoids reloading the depth model.

    Processing mode
    ---------------
    This function operates in MODE B only (georeferenced GeoTIFF input).
    For non-georeferenced PNG/JPG inputs, see
    ``run_relative_calibration()`` which produces a relative DSM without
    SRTM reference.

    Depth orientation
    -----------------
    Depth Anything V2 produces relative depth.  Larger values may correspond
    to either CLOSER or FARTHER pixels depending on the scene.  The calibration
    fit will determine the correct slope sign automatically:
      - a > 0 : larger depth → higher elevation (typical for nadir satellite)
      - a < 0 : larger depth → lower elevation (inverted convention)
    Both are valid; the calibration diagnostics report the slope and a note
    if a < 0.  Do not manually flip depth values.

    Returns
    -------
    dict with keys:
        "a", "b", "dsm_array", "dsm_tif_path", "preview_png_path",
        "gamus_validation", "report_path", "calib_diagnostics", "geo_meta",
        "processing_mode"
    """
    # ── Setup ────────────────────────────────────────────────────────────────
    depth_array_path = Path(depth_array_path)
    geotiff_path     = Path(geotiff_path)
    output_dir       = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir:
        os.environ["DA_CHECKPOINT_DIR"] = str(checkpoint_dir)

    logger.info("=" * 60)
    logger.info("GeoMonoDSM-3D  Part 2 — Elevation Calibration")
    logger.info("=" * 60)
    logger.info("Depth array : %s", depth_array_path)
    logger.info("GeoTIFF     : %s", geotiff_path)
    logger.info("Output dir  : %s", output_dir.resolve())

    # ── Load depth array ─────────────────────────────────────────────────────
    if not depth_array_path.is_file():
        raise FileNotFoundError(f"Depth array not found: {depth_array_path}")
    depth_array = np.load(depth_array_path)
    if depth_array.ndim != 2:
        raise ValueError(f"Expected 2-D depth array, got shape {depth_array.shape}")
    logger.info("Loaded depth array: shape=%s  dtype=%s", depth_array.shape, depth_array.dtype)

    # ── Load GeoTIFF metadata ────────────────────────────────────────────────
    if not geotiff_path.is_file():
        raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")

    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required: pip install rasterio")

    with rasterio.open(geotiff_path) as src:
        geo_meta = {
            "crs"      : src.crs,
            "transform": src.transform,
            "width"    : src.width,
            "height"   : src.height,
            "bounds"   : src.bounds,
            "res"      : src.res,
            "nodata"   : src.nodata,
            "count"    : src.count,
            "dtype"    : src.dtypes[0],
            "driver"   : src.driver,
        }

    logger.info(
        "GeoTIFF: %dx%d  CRS=%s  res=%.4f",
        geo_meta["width"], geo_meta["height"],
        geo_meta["crs"], geo_meta["res"][0],
    )

    # Sanity check — depth array must match GeoTIFF dimensions
    h, w = depth_array.shape
    if h != geo_meta["height"] or w != geo_meta["width"]:
        raise ValueError(
            f"Depth array shape ({h},{w}) does not match GeoTIFF dimensions "
            f"({geo_meta['height']},{geo_meta['width']}). "
            "Ensure the depth array was produced from this exact GeoTIFF."
        )

    # ── Step 1: GCPs → SRTM elevations ──────────────────────────────────────
    depth_vals, srtm_vals = fetch_reference_elevations(
        depth_array,
        geo_meta["transform"],
        geo_meta["crs"],
        n_gcps=16,
    )

    # ── Step 2: Fit calibration (a, b) ──────────────────────────────────────
    a, b, fit_diag = fit_calibration(depth_vals, srtm_vals)
    logger.info("Calibration: a=%.6f  b=%.4f m", a, b)

    # ── Calibration diagnostics ──────────────────────────────────────────────
    calib_diag = compute_calibration_diagnostics(
        a, b, depth_vals, srtm_vals, fit_diag, n_gcps_requested=16
    )
    quality_status = calib_diag["quality"]["status"]
    if quality_status == "failed":
        logger.error(
            "Calibration FAILED quality check: %s",
            calib_diag["quality"]["reason"],
        )
        logger.error(
            "The resulting DSM (a=%.6f b=%.4f) may be unreliable. "
            "Proceeding but marking outputs with quality status 'failed'.",
            a, b,
        )
    elif quality_status == "warning":
        logger.warning(
            "Calibration quality WARNING: %s",
            calib_diag["quality"]["reason"],
        )

    # ── Apply calibration to full depth array ────────────────────────────────
    # Stay in float32 throughout — float64 intermediate would double memory
    # (200 MB at 5000×5000) with no accuracy benefit given SRTM ~30 m precision
    dsm_array = np.add(np.multiply(depth_array, np.float32(a), dtype=np.float32),
                       np.float32(b), dtype=np.float32)
    logger.info(
        "DSM array: min=%.1f m  max=%.1f m  mean=%.1f m",
        float(dsm_array.min()), float(dsm_array.max()), float(dsm_array.mean()),
    )

    # ── DSM sanity check ────────────────────────────────────────────────────
    # Log interpretation note so the user understands absolute elevation output.
    _dsm_min = float(dsm_array.min())
    _dsm_max = float(dsm_array.max())
    _dsm_rng = _dsm_max - _dsm_min
    logger.info(
        "DSM interpretation: values represent estimated ABSOLUTE elevation above the "
        "reference datum used by SRTM30m (approx. EGM96 geoid). "
        "These are NOT relative heights or above-ground distances. "
        "Building-height rasters (separate output) are always relative to local ground."
    )
    if _dsm_rng < 1.0:
        logger.warning(
            "DSM elevation range is only %.2f m — surface appears nearly flat. "
            "This may indicate a calibration issue or a genuinely flat scene.",
            _dsm_rng,
        )
    if abs(a) < 0.01:
        logger.warning(
            "Calibration slope |a|=%.6f is very small. "
            "The DSM is dominated by the constant offset b=%.2f m. "
            "Depth variation has minimal influence — calibration may be unreliable.",
            a, b,
        )

    # ── Write outputs ────────────────────────────────────────────────────────
    tif_path     = save_dsm_geotiff(dsm_array, geo_meta, output_dir / "calibrated_dsm.tif")
    preview_path = save_dsm_preview(dsm_array, output_dir / "calibrated_dsm_preview.png")

    # ── Step 3: GAMUS validation ─────────────────────────────────────────────
    # Set up a lightweight depth function for GAMUS (accepts numpy uint8 array)
    def _depth_from_array(rgb_uint8: np.ndarray) -> np.ndarray:
        """Run Part 1 depth inference on an in-memory uint8 RGB array."""
        # Add Part 1 to sys.path so it can be imported from any working dir
        _part1_dir = Path(__file__).resolve().parent.parent / "Depth_Mapping"
        if str(_part1_dir) not in sys.path:
            sys.path.insert(0, str(_part1_dir))

        try:
            from depth_mapping.depth import _clip_exposure, _single_inference, _tiled_inference
            from depth_mapping.model_loader import load_model
            import torch

            device  = "cuda" if torch.cuda.is_available() else "cpu"
            model   = load_model(encoder="vitl", device=device)

            from PIL import Image as PILImage
            pil_img  = PILImage.fromarray(rgb_uint8, mode="RGB")
            pil_img  = _clip_exposure(pil_img)

            MAX_SINGLE = 650 + 128   # MAX_CORE_SIZE + CONTEXT_MARGIN
            if max(pil_img.size) <= MAX_SINGLE:
                return _single_inference(pil_img, model, device)
            return _tiled_inference(pil_img, model, device)

        except Exception as exc:
            logger.warning("  Part 1 inference failed for GAMUS sample: %s", exc)
            # Return a flat array so the sample is skipped gracefully
            h, w = rgb_uint8.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

    gamus_result = {} if skip_gamus else validate_against_gamus(a, b, _depth_from_array)

    # ── Step 4: Save calibration report ─────────────────────────────────────
    report_path = save_calibration_report(
        a, b, dsm_array, gamus_result, output_dir / "calibration_report.json",
        calib_diagnostics=calib_diag,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Part 2 — Elevation Calibration  ✓  COMPLETE")
    print("=" * 60)
    print(f"  Calibration: a = {a:.6f}   b = {b:.4f} m")
    print(f"  Equation   : DSM = {a:.6f} × depth + {b:.4f}")
    print(f"  DSM range  : [{dsm_array.min():.1f}, {dsm_array.max():.1f}] m  "
          f"(absolute elevation, SRTM datum)")
    print(f"  DSM mean   : {dsm_array.mean():.1f} m  (AMSL estimate)")
    print(f"  Quality    : {calib_diag['quality']['status'].upper()}")
    if calib_diag["quality"]["status"] != "good":
        print(f"  Reason     : {calib_diag['quality']['reason']}")
    print(f"  NOTE: DSM values are ABSOLUTE ELEVATION, not building heights.")
    print(f"        Building height = DSM_building - DSM_local_ground  (separate output).")
    if gamus_result:
        print(f"  GAMUS MAE  : {gamus_result['mae']:.2f} m   RMSE: {gamus_result['rmse']:.2f} m")
        print(f"  GAMUS note : structural check only; GAMUS nDSM ≠ absolute elevation.")
    print(f"\n  Outputs → {output_dir.resolve()}")
    print(f"    calibrated_dsm.tif           (absolute elevation, metres, SRTM reference)")
    print(f"    calibrated_dsm_preview.png")
    print(f"    calibration_report.json")
    print("=" * 60 + "\n")

    return {
        "a"                   : a,
        "b"                   : b,
        "dsm_array"           : dsm_array,
        "dsm_tif_path"        : str(tif_path),
        "preview_png_path"    : str(preview_path),
        "gamus_validation"    : gamus_result,
        "report_path"         : str(report_path),
        "calib_diagnostics"   : calib_diag,
        "geo_meta"            : geo_meta,
        "processing_mode"     : "MODE_B_georeferenced",
    }


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GeoMonoDSM-3D Part 2 — Elevation Calibration\n\n"
            "MODE B (georeferenced): --depth depth.npy --geotiff image.tif\n"
            "MODE A (relative only): --depth depth.npy  (no --geotiff)\n"
        ),
        prog="python elevation_calibration.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--depth", required=True,
        help="Path to .npy depth array from Part 1",
    )
    parser.add_argument(
        "--geotiff", default=None,
        help=(
            "Path to the source GeoTIFF (same spatial extent as depth array). "
            "Required for MODE B (metric DSM with SRTM calibration). "
            "Omit for MODE A (relative DSM, no geographic reference)."
        ),
    )
    parser.add_argument(
        "--output-dir", default="Elevation_Calibration/outputs",
        help="Output directory (default: Elevation_Calibration/outputs)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Path to DA V2 checkpoints folder (sets DA_CHECKPOINT_DIR)",
    )
    parser.add_argument(
        "--no-gamus", action="store_true",
        help=(
            "Skip GAMUS structural-consistency validation (faster — avoids "
            "reloading the depth model for GAMUS samples)."
        ),
    )
    args = parser.parse_args()

    if args.geotiff:
        # MODE B — georeferenced metric DSM
        result = run_elevation_calibration(
            depth_array_path=args.depth,
            geotiff_path=args.geotiff,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            skip_gamus=args.no_gamus,
        )
    else:
        # MODE A — relative DSM, no CRS available
        logger.info("No --geotiff provided → running MODE A (relative DSM).")
        result = run_relative_calibration(
            depth_array_path=args.depth,
            output_dir=args.output_dir,
        )
    return result


if __name__ == "__main__":
    main()
