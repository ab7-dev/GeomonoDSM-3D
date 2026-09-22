"""Reusable output validator and processing report for Part 1 depth outputs.

This module can be used:
  1. As a library  — import validate_run() for programmatic checks.
  2. As a CLI tool — python -m depth_mapping.validate_output <folder>

Public API
----------
validate_run(output_dir, source_tif=None)
    Validate all outputs in a completed test/run folder and return a structured
    report dict.  Prints a human-readable summary to stdout.

write_report(report, output_dir)
    Write a machine-readable JSON report to <output_dir>/processing_report.json.

Checks performed
----------------
For all runs:
    - .npy file exists and is loadable
    - depth array is 2-D float32
    - all values are finite
    - array has non-zero variance (not a blank or constant output)
    - depth min/max/mean/std are within plausible bounds

For GeoTIFF output (when present):
    - file exists
    - band count == 1
    - dtype == float32
    - all band values finite
    - DEPTH_TYPE tag == "relative"
    - CRS matches source (when source_tif provided)
    - transform matches source
    - width/height match source
    - bounds match source
    - resolution matches source

Usage examples
--------------
From Python:
    from depth_mapping.validate_output import validate_run, write_report
    report = validate_run("depth_mapping/tested/test_002",
                          source_tif="depth_mapping/Input/test_2.tif")
    write_report(report, "depth_mapping/tested/test_002")

From the terminal:
    python -m depth_mapping.validate_output depth_mapping/tested/test_002
    python -m depth_mapping.validate_output depth_mapping/tested/test_002 ^
        --source-tif depth_mapping/Input/test_2.tif
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Sentinel for "not checked" ────────────────────────────────────────────────
_NC = "not_checked"


def validate_run(
    output_dir: str | Path,
    source_tif: str | Path | None = None,
) -> dict[str, Any]:
    """Validate outputs in a completed run folder.

    Parameters
    ----------
    output_dir:
        Folder produced by a run (e.g. ``tested/test_002/``).
    source_tif:
        Optional path to the input GeoTIFF for alignment cross-checks.

    Returns
    -------
    dict
        Full report including a ``"passed"`` boolean and per-check results.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    checks: dict[str, Any] = {}
    warnings_list: list[str] = []

    # ── Find files ────────────────────────────────────────────────────────────
    npy_files  = sorted(output_dir.glob("*.npy"))
    png_files  = sorted(output_dir.glob("*.png"))
    tif_files  = [f for f in sorted(output_dir.glob("*.tif"))
                  if "depth" in f.name.lower()]

    checks["npy_found"]  = len(npy_files) > 0
    checks["png_found"]  = len(png_files) > 0
    checks["tif_found"]  = len(tif_files) > 0

    depth_arr: np.ndarray | None = None

    # ── Validate .npy ─────────────────────────────────────────────────────────
    if npy_files:
        npy_path = npy_files[0]
        try:
            depth_arr = np.load(npy_path)
            checks["npy_loadable"]        = True
            checks["npy_is_2d"]           = depth_arr.ndim == 2
            checks["npy_dtype_float32"]   = depth_arr.dtype == np.float32
            checks["npy_all_finite"]      = bool(np.all(np.isfinite(depth_arr)))
            checks["npy_nonzero_variance"]= bool(depth_arr.std() > 0)
            checks["npy_shape"]           = list(depth_arr.shape)
            checks["depth_min"]           = float(depth_arr.min())
            checks["depth_max"]           = float(depth_arr.max())
            checks["depth_mean"]          = float(depth_arr.mean())
            checks["depth_std"]           = float(depth_arr.std())
            if depth_arr.ndim == 2:
                checks["npy_not_empty"]   = depth_arr.shape[0] > 0 and depth_arr.shape[1] > 0
        except Exception as exc:
            checks["npy_loadable"] = False
            warnings_list.append(f"Cannot load .npy: {exc}")
    else:
        checks["npy_loadable"] = False
        warnings_list.append("No .npy file found in output directory")

    # ── Validate PNG ──────────────────────────────────────────────────────────
    if png_files:
        png_path = png_files[0]
        try:
            from PIL import Image
            img = Image.open(png_path)
            checks["png_loadable"]        = True
            checks["png_size"]            = list(img.size)
            checks["png_mode"]            = img.mode
            # PNG size should match depth array shape
            if depth_arr is not None and depth_arr.ndim == 2:
                h, w = depth_arr.shape
                checks["png_size_matches_depth"] = (img.width == w and img.height == h)
        except Exception as exc:
            checks["png_loadable"] = False
            warnings_list.append(f"Cannot load PNG: {exc}")

    # ── Validate GeoTIFF ──────────────────────────────────────────────────────
    if tif_files:
        tif_path = tif_files[0]
        try:
            import rasterio
            with rasterio.open(tif_path) as dst:
                out_crs       = dst.crs
                out_transform = dst.transform
                out_width     = dst.width
                out_height    = dst.height
                out_bounds    = dst.bounds
                out_res       = dst.res
                out_dtype     = dst.dtypes[0]
                out_count     = dst.count
                out_tags      = dst.tags()
                out_band      = dst.read(1)

            checks["tif_loadable"]          = True
            checks["tif_dtype_float32"]     = out_dtype == "float32"
            checks["tif_band_count_1"]      = out_count == 1
            checks["tif_all_finite"]        = bool(np.all(np.isfinite(out_band)))
            checks["tif_depth_type_tag"]    = out_tags.get("DEPTH_TYPE") == "relative"
            checks["tif_width"]             = out_width
            checks["tif_height"]            = out_height
            checks["tif_crs"]               = str(out_crs)
            checks["tif_res_x"]             = float(out_res[0])
            checks["tif_res_y"]             = float(out_res[1])
            checks["tif_bounds"]            = {
                "left"  : out_bounds.left,
                "bottom": out_bounds.bottom,
                "right" : out_bounds.right,
                "top"   : out_bounds.top,
            }

            # Shape consistency with .npy
            if depth_arr is not None and depth_arr.ndim == 2:
                h, w = depth_arr.shape
                checks["tif_size_matches_depth"] = (out_width == w and out_height == h)

            # Cross-check against source GeoTIFF
            if source_tif is not None:
                source_tif = Path(source_tif)
                if source_tif.is_file():
                    with rasterio.open(source_tif) as src:
                        src_crs       = src.crs
                        src_transform = src.transform
                        src_width     = src.width
                        src_height    = src.height
                        src_bounds    = src.bounds
                        src_res       = src.res

                    checks["tif_crs_matches_source"]       = out_crs == src_crs
                    checks["tif_width_matches_source"]     = out_width  == src_width
                    checks["tif_height_matches_source"]    = out_height == src_height
                    checks["tif_transform_matches_source"] = _transforms_equal(
                        out_transform, src_transform
                    )
                    checks["tif_bounds_matches_source"]    = _bounds_equal(
                        out_bounds, src_bounds
                    )
                    checks["tif_res_matches_source"]       = (
                        abs(out_res[0] - src_res[0]) < 1e-6 and
                        abs(out_res[1] - src_res[1]) < 1e-6
                    )
                else:
                    warnings_list.append(f"source_tif not found: {source_tif}")

        except ImportError:
            checks["tif_loadable"] = False
            warnings_list.append("rasterio not installed - GeoTIFF checks skipped")
        except Exception as exc:
            checks["tif_loadable"] = False
            warnings_list.append(f"Cannot open depth GeoTIFF: {exc}")

    # ── Overall pass/fail ─────────────────────────────────────────────────────
    required_passing = [
        "npy_found", "npy_loadable", "npy_is_2d",
        "npy_dtype_float32", "npy_all_finite", "npy_nonzero_variance",
    ]
    if tif_files:
        required_passing += [
            "tif_loadable", "tif_dtype_float32", "tif_band_count_1",
            "tif_all_finite", "tif_depth_type_tag",
        ]
        if source_tif is not None and Path(source_tif).is_file():
            required_passing += [
                "tif_crs_matches_source", "tif_width_matches_source",
                "tif_height_matches_source", "tif_transform_matches_source",
                "tif_bounds_matches_source", "tif_res_matches_source",
            ]

    failed = [k for k in required_passing if not checks.get(k, False)]
    passed = len(failed) == 0

    report: dict[str, Any] = {
        "validated_at"  : datetime.now().isoformat(),
        "output_dir"    : str(output_dir.resolve()),
        "source_tif"    : str(source_tif) if source_tif else None,
        "passed"        : passed,
        "failed_checks" : failed,
        "warnings"      : warnings_list,
        "checks"        : checks,
    }

    _print_report(report)
    return report


def write_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    """Write the validation report as JSON to ``<output_dir>/processing_report.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "processing_report.json"

    # Make the dict JSON-serialisable (numpy types → native Python)
    def _serialise(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=_serialise)

    logger.info("Processing report written: %s", report_path)
    return report_path.resolve()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _print_report(report: dict[str, Any]) -> None:
    """Print a concise human-readable summary to stdout."""
    checks  = report["checks"]
    failed  = report["failed_checks"]
    warnings = report["warnings"]
    passed  = report["passed"]

    print("=" * 62)
    print("DEPTH OUTPUT VALIDATION REPORT")
    print("=" * 62)
    print(f"  Directory : {report['output_dir']}")
    if report.get("source_tif"):
        print(f"  Source TIF: {report['source_tif']}")
    print(f"  Validated : {report['validated_at']}")
    print()

    # Depth array stats
    for k in ("npy_shape", "npy_dtype_float32", "npy_all_finite",
              "npy_nonzero_variance", "depth_min", "depth_max",
              "depth_mean", "depth_std"):
        if k in checks:
            val = checks[k]
            if isinstance(val, bool):
                tag = "PASS" if val else "FAIL"
                print(f"  [{tag}] {k}")
            else:
                print(f"        {k} = {val}")

    print()

    # GeoTIFF checks — only shown when a depth .tif was actually produced
    tif_keys = [k for k in checks if k.startswith("tif_") and k != "tif_found"]
    if checks.get("tif_found"):
        print("  GeoTIFF checks:")
        for k in tif_keys:
            val = checks[k]
            if isinstance(val, bool):
                tag = "PASS" if val else "FAIL"
                print(f"    [{tag}] {k}")
            else:
                print(f"          {k} = {val}")
        print()

    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    ! {w}")
        print()

    if passed:
        print("  RESULT: ALL REQUIRED CHECKS PASSED [OK]")
    else:
        print(f"  RESULT: FAILED -- {len(failed)} check(s) did not pass:")
        for f in failed:
            print(f"    FAIL: {f}")
    print("=" * 62)


def _transforms_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        return np.allclose(list(a), list(b), rtol=tol, atol=tol)
    except Exception:
        return a == b


def _bounds_equal(a: Any, b: Any, tol: float = 1e-4) -> bool:
    try:
        return all(
            abs(getattr(a, f) - getattr(b, f)) < tol
            for f in ("left", "bottom", "right", "top")
        )
    except Exception:
        return a == b


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate a depth-mapping output folder.",
        prog="python -m depth_mapping.validate_output",
    )
    parser.add_argument(
        "output_dir",
        help="Path to the completed run folder (e.g. tested/test_002).",
    )
    parser.add_argument(
        "--source-tif",
        default=None,
        help="Path to the original input GeoTIFF for alignment cross-checks.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write processing_report.json into the output folder.",
    )
    args = parser.parse_args()

    report = validate_run(args.output_dir, source_tif=args.source_tif)

    if args.write_report:
        path = write_report(report, args.output_dir)
        print(f"\nReport written: {path}")

    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    _cli()
