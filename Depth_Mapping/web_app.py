"""DepthWizard — Unified web interface.

Serves two functional modules:
  Role 1 — Single-view relative depth estimation (Depth Anything V2)
  Role 3 — Terrain classification (LoveDA SegFormer)

Run from the project root:
    python depth_mapping/web_app.py
    python depth_mapping/web_app.py --port 5000

Or from inside depth_mapping/:
    python web_app.py --port 5000

Opens at http://127.0.0.1:5000 by default.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Flask import
# ---------------------------------------------------------------------------
try:
    from flask import (
        Flask, jsonify, render_template, request,
        send_file, send_from_directory,
    )
except ImportError as exc:
    raise ImportError("Flask is required: pip install flask") from exc

# ---------------------------------------------------------------------------
# Existing Part 1 pipeline — imported, never duplicated
# ---------------------------------------------------------------------------
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Depth Anything V2 checkpoint directory
# ---------------------------------------------------------------------------
# The checkpoints live outside this sub-project, in the sibling GeomonoDSM
# repository.  Set DA_CHECKPOINT_DIR before importing depth.py so that
# model_loader.default_checkpoint_dir() resolves to the correct location
# regardless of the working directory the server is launched from.
# Only override if the environment variable is not already set by the user,
# so that an explicit DA_CHECKPOINT_DIR=... on the command line still wins.
_DEFAULT_CKPT_DIR = Path(r"D:\Collaborate_Projects\SIH\GeomonoDSM-3D-main\checkpoints")
if "DA_CHECKPOINT_DIR" not in os.environ:
    os.environ["DA_CHECKPOINT_DIR"] = str(_DEFAULT_CKPT_DIR)

from depth_mapping.depth import get_depth_with_meta, save_depth_outputs, _DEVICE
from depth_mapping import geo_utils

# ---------------------------------------------------------------------------
# Terrain Classification (Part 3) — absolute model paths
# ---------------------------------------------------------------------------
_TERRAIN_ROOT  = _PROJECT_ROOT / "Terrain_Classification_Part_3"
_LOVEDA_MODEL  = str(_TERRAIN_ROOT / "models" / "loveda-segformer")
_ADE_MODEL     = str(_TERRAIN_ROOT / "models" / "ade-segformer-b0")

_terrain_import_err_msg = ""   # set here so it's always defined
try:
    # Insert terrain root so the package can be found as
    # Terrain_Classification_Part_3.terrain_classification
    sys.path.insert(0, str(_PROJECT_ROOT))
    from Terrain_Classification_Part_3.terrain_classification import classify_terrain as _classify_terrain
    _TERRAIN_AVAILABLE = True
except ImportError as _terrain_import_err:
    _TERRAIN_AVAILABLE = False
    _terrain_import_err_msg = str(_terrain_import_err)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(_HERE / "templates"),
    static_folder=str(_HERE / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("depthwizard")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UPLOAD_DIR          = _HERE / "uploads"
RESULTS_DIR         = _HERE / "results"
TERRAIN_UPLOAD_DIR  = _HERE / "terrain_uploads"
TERRAIN_RESULTS_DIR = _HERE / "terrain_results"

for _d in (UPLOAD_DIR, RESULTS_DIR, TERRAIN_UPLOAD_DIR, TERRAIN_RESULTS_DIR):
    _d.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# In-memory job stores
# ---------------------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

_terrain_jobs: dict[str, dict[str, Any]] = {}
_terrain_jobs_lock = threading.Lock()


# ── Depth job helpers ──────────────────────────────────────────────────────

def _job_get(job_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(job_id)

def _job_set(job_id: str, data: dict) -> None:
    with _jobs_lock:
        _jobs[job_id] = data

def _job_update(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


# ── Terrain job helpers ────────────────────────────────────────────────────

def _tj_get(job_id: str) -> dict | None:
    with _terrain_jobs_lock:
        return _terrain_jobs.get(job_id)

def _tj_set(job_id: str, data: dict) -> None:
    with _terrain_jobs_lock:
        _terrain_jobs[job_id] = data

def _tj_update(job_id: str, **kwargs: Any) -> None:
    with _terrain_jobs_lock:
        if job_id in _terrain_jobs:
            _terrain_jobs[job_id].update(kwargs)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_stem(filename: str) -> str:
    return Path(Path(filename).name).stem[:80]

def _file_format_label(suffix: str) -> str:
    s = suffix.lower()
    if s in {".tif", ".tiff"}:
        return "GeoTIFF"
    return s.lstrip(".").upper()

def _image_dimensions(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
            with rasterio.open(path) as src:
                return src.width, src.height
        except Exception:
            pass
    from PIL import Image
    with Image.open(path) as img:
        return img.size  # (W, H)

def _make_display_preview(src_path: Path, out_path: Path, max_side: int = 1200) -> None:
    from PIL import Image
    suffix = src_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            image, _ = geo_utils.read_geotiff_rgb(src_path)
        except Exception:
            image = Image.open(src_path).convert("RGB")
    else:
        image = Image.open(src_path).convert("RGB")
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    image.save(str(out_path), format="PNG", optimize=True)

def _make_depth_preview(depth_png: Path, out_path: Path, max_side: int = 1200) -> None:
    from PIL import Image
    img = Image.open(depth_png).convert("RGBA")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(str(out_path), format="PNG", optimize=True)

def _resize_png_preview(src: Path, dest: Path, max_side: int = 1200) -> None:
    """Generic PNG resize for browser display."""
    from PIL import Image
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(str(dest), format="PNG", optimize=True)


# ---------------------------------------------------------------------------
# Role 1 — Depth inference thread
# ---------------------------------------------------------------------------

def _run_inference(job_id: str, upload_path: Path, fast: bool) -> None:
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(exist_ok=True)
    stem = _safe_stem(upload_path.name)

    try:
        _job_update(job_id, stage="Preparing image…", progress=5)
        is_geotiff = upload_path.suffix.lower() in {".tif", ".tiff"}

        preview_in = result_dir / "preview_input.png"
        _job_update(job_id, stage="Building input preview…", progress=10)
        _make_display_preview(upload_path, preview_in)

        w, h = _image_dimensions(upload_path)

        geo_input_info: dict[str, Any] = {}
        if is_geotiff:
            try:
                import rasterio
                with rasterio.open(upload_path) as src:
                    geo_input_info = {
                        "crs"       : str(src.crs),
                        "res_x"     : round(src.res[0], 6),
                        "res_y"     : round(src.res[1], 6),
                        "bounds"    : {
                            "left"  : round(src.bounds.left,   4),
                            "bottom": round(src.bounds.bottom, 4),
                            "right" : round(src.bounds.right,  4),
                            "top"   : round(src.bounds.top,    4),
                        },
                        "band_count": src.count,
                        "dtype"     : src.dtypes[0],
                    }
            except Exception as exc:
                logger.warning("Could not read GeoTIFF metadata: %s", exc)

        _job_update(job_id, stage="Running depth inference (Depth Anything V2 Large)…", progress=20)
        t0 = time.perf_counter()
        depth_array, geo_meta = get_depth_with_meta(str(upload_path), fast=fast)
        inference_time = round(time.perf_counter() - t0, 2)

        _job_update(job_id, stage="Saving depth outputs…", progress=80)

        try:
            from PIL import Image as PILImage
            if is_geotiff:
                rgb_img, _ = geo_utils.read_geotiff_rgb(upload_path)
            else:
                rgb_img = PILImage.open(upload_path).convert("RGB")
        except Exception:
            rgb_img = None

        save_depth_outputs(
            depth_array,
            output_dir=str(result_dir),
            filename=stem,
            geo_meta=geo_meta,
            rgb_image=rgb_img,
            save_visualization=True,
        )

        _job_update(job_id, stage="Generating previews…", progress=88)
        depth_png = result_dir / f"{stem}_depth.png"
        preview_depth = result_dir / "preview_depth.png"
        if depth_png.is_file():
            _make_depth_preview(depth_png, preview_depth)

        vis_png = result_dir / f"{stem}_depth_vis.png"

        geo_output_info: dict[str, Any] = {}
        tif_out = result_dir / f"{stem}_depth.tif"
        if tif_out.is_file() and geo_meta is not None:
            try:
                checks = geo_utils.validate_geotiff_alignment(geo_meta, tif_out)
                geo_output_info = {
                    "crs_preserved"       : checks["crs"]["match"],
                    "transform_preserved" : checks["transform"]["match"],
                    "bounds_preserved"    : checks["bounds"]["match"],
                    "res_preserved"       : checks["res"]["match"],
                    "dimensions_match"    : checks["width"]["match"] and checks["height"]["match"],
                    "dtype_float32"       : checks["dtype"]["match"],
                    "depth_type_tag"      : "relative",
                    "output_crs"          : str(checks["crs"]["output"]),
                    "output_res_x"        : round(float(checks["res"]["output"][0]), 6),
                }
            except Exception as exc:
                logger.warning("GeoTIFF alignment check failed: %s", exc)

        import numpy as np
        outputs: dict[str, str] = {}
        npy_path = result_dir / f"{stem}.npy"
        if npy_path.is_file():   outputs["npy"] = npy_path.name
        if depth_png.is_file():  outputs["png"] = depth_png.name
        if tif_out.is_file():    outputs["tif"] = tif_out.name
        if vis_png.is_file():    outputs["vis"] = vis_png.name

        _job_update(
            job_id,
            stage="Complete", progress=100, status="done",
            inference_time=inference_time, device=_DEVICE, fast=fast,
            input_width=w, input_height=h,
            is_geotiff=is_geotiff,
            geo_input_info=geo_input_info, geo_output_info=geo_output_info,
            depth_min=round(float(depth_array.min()), 4),
            depth_max=round(float(depth_array.max()), 4),
            depth_mean=round(float(depth_array.mean()), 4),
            depth_std=round(float(depth_array.std()), 4),
            output_shape=list(depth_array.shape),
            outputs=outputs,
        )

    except MemoryError:
        _job_update(job_id, status="error",
                    error="Image is too large for the current system configuration.")
    except Exception as exc:
        logger.exception("Depth inference failed for job %s", job_id)
        _job_update(job_id, status="error",
                    error=f"Depth generation failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Role 3 — Terrain inference thread
# ---------------------------------------------------------------------------

def _run_terrain_inference(job_id: str, upload_path: Path) -> None:
    result_dir = TERRAIN_RESULTS_DIR / job_id
    result_dir.mkdir(exist_ok=True)
    stem = _safe_stem(upload_path.name)

    try:
        _tj_update(job_id, stage="Preparing image…", progress=5)
        is_geotiff = upload_path.suffix.lower() in {".tif", ".tiff"}

        # Build browser-sized input preview
        preview_in = result_dir / "preview_input.png"
        _tj_update(job_id, stage="Building input preview…", progress=10)
        _make_display_preview(upload_path, preview_in)

        w, h = _image_dimensions(upload_path)

        # Geospatial input metadata
        geo_input_info: dict[str, Any] = {}
        if is_geotiff:
            try:
                import rasterio
                with rasterio.open(upload_path) as src:
                    geo_input_info = {
                        "crs"       : str(src.crs),
                        "res_x"     : round(src.res[0], 6),
                        "res_y"     : round(src.res[1], 6),
                        "band_count": src.count,
                        "dtype"     : src.dtypes[0],
                    }
            except Exception as exc:
                logger.warning("Could not read terrain GeoTIFF metadata: %s", exc)

        _tj_update(job_id, stage="Loading terrain model…", progress=15)

        # Resolve absolute ADE model path (only pass if it actually exists)
        ade_path = _ADE_MODEL if Path(_ADE_MODEL).is_dir() else None

        _tj_update(job_id, stage="Running terrain classification (LoveDA SegFormer)…", progress=20)
        t0 = time.perf_counter()

        json_path, overlay_path = _classify_terrain(
            image_path=str(upload_path),
            output_dir=str(result_dir),
            model_name=_LOVEDA_MODEL,
            ade_model_path=ade_path,
            device=None,           # auto-detect
            enable_fallbacks=True,
        )

        inference_time = round(time.perf_counter() - t0, 2)

        _tj_update(job_id, stage="Generating previews…", progress=88)

        # Build browser-sized previews
        overlay_png   = Path(overlay_path)
        labelmap_path = result_dir / f"{stem}_labelmap.png"

        preview_overlay  = result_dir / "preview_overlay.png"
        preview_labelmap = result_dir / "preview_labelmap.png"

        if overlay_png.is_file():
            _resize_png_preview(overlay_png, preview_overlay)
        if labelmap_path.is_file():
            _resize_png_preview(labelmap_path, preview_labelmap)

        # Parse JSON for stats
        stats: dict[str, Any] = {}
        terrain_classes: list[str] = []
        building_count = 0
        class_distribution: dict[str, float] = {}

        if json_path and Path(json_path).is_file():
            with open(json_path, encoding="utf-8") as fh:
                payload = json.load(fh)
            terrain_classes   = payload.get("segmentation", {}).get("classes", [])
            detections        = payload.get("detections", [])
            building_count    = sum(1 for d in detections if d.get("class") == "building")

        # Collect available output files
        outputs: dict[str, str] = {}
        if overlay_png.is_file():
            outputs["overlay"] = overlay_png.name
        if labelmap_path.is_file():
            outputs["labelmap"] = labelmap_path.name
        if json_path and Path(json_path).is_file():
            outputs["json"] = Path(json_path).name
        # GeoTIFF terrain labels
        tif_labels = result_dir / f"{stem}_terrain_labels.tif"
        if tif_labels.is_file():
            outputs["tif"] = tif_labels.name

        _tj_update(
            job_id,
            stage="Complete", progress=100, status="done",
            inference_time=inference_time,
            device=_DEVICE,
            input_width=w, input_height=h,
            is_geotiff=is_geotiff,
            geo_input_info=geo_input_info,
            terrain_classes=terrain_classes,
            building_count=building_count,
            outputs=outputs,
        )

    except MemoryError:
        _tj_update(job_id, status="error",
                   error="Image is too large for available memory.")
    except Exception as exc:
        logger.exception("Terrain inference failed for job %s", job_id)
        _tj_update(job_id, status="error",
                   error=f"Terrain classification failed: {type(exc).__name__}: {exc}")


# ===========================================================================
# ROUTES — Role 1 (Depth Estimation)
# ===========================================================================

@app.route("/")
def index():
    return render_template("index.html", device=_DEVICE.upper())


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    suffix = Path(f.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{suffix}'. "
                     "Please upload TIFF, TIF, JPG, JPEG, or PNG."
        }), 415

    job_id    = str(uuid.uuid4())
    safe_name = f"{job_id}{suffix}"
    upload_path = UPLOAD_DIR / safe_name
    f.save(str(upload_path))

    try:
        w, h = _image_dimensions(upload_path)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        return jsonify({"error": f"Unable to read image: {exc}"}), 422

    is_geotiff    = suffix in {".tif", ".tiff"}
    file_size_mb  = round(upload_path.stat().st_size / 1e6, 2)

    _job_set(job_id, {
        "status"      : "uploaded",
        "stage"       : "Ready",
        "progress"    : 0,
        "filename"    : f.filename,
        "suffix"      : suffix,
        "format"      : _file_format_label(suffix),
        "is_geotiff"  : is_geotiff,
        "width"       : w,
        "height"      : h,
        "file_size_mb": file_size_mb,
        "upload_path" : str(upload_path),
        "outputs"     : {},
    })

    return jsonify({
        "job_id"      : job_id,
        "filename"    : f.filename,
        "format"      : _file_format_label(suffix),
        "width"       : w,
        "height"      : h,
        "file_size_mb": file_size_mb,
        "is_geotiff"  : is_geotiff,
    })


@app.route("/infer/<job_id>", methods=["POST"])
def start_inference(job_id: str):
    job = _job_get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job ID."}), 404
    if job["status"] not in {"uploaded", "error"}:
        return jsonify({"error": "Inference already running or complete."}), 409

    fast = request.get_json(silent=True, force=True) or {}
    use_fast = bool(fast.get("fast", False))

    _job_update(job_id, status="running", stage="Starting…", progress=2, fast=use_fast)
    upload_path = Path(job["upload_path"])
    threading.Thread(target=_run_inference, args=(job_id, upload_path, use_fast), daemon=True).start()
    return jsonify({"status": "running", "job_id": job_id})


@app.route("/status/<job_id>")
def job_status(job_id: str):
    job = _job_get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job ID."}), 404

    safe = {k: job[k] for k in (
        "status", "stage", "progress", "filename", "format",
        "is_geotiff", "width", "height", "file_size_mb",
    ) if k in job}

    if job["status"] == "done":
        for key in (
            "inference_time", "device", "fast",
            "input_width", "input_height",
            "is_geotiff", "geo_input_info", "geo_output_info",
            "depth_min", "depth_max", "depth_mean", "depth_std",
            "output_shape", "outputs",
        ):
            if key in job:
                safe[key] = job[key]

    if job["status"] == "error":
        safe["error"] = job.get("error", "Unknown error.")

    return jsonify(safe)


@app.route("/preview/<job_id>/<which>")
def serve_preview(job_id: str, which: str):
    job = _job_get(job_id)
    if job is None:
        return "Not found", 404
    result_dir = RESULTS_DIR / job_id
    allowed = {"input": "preview_input.png", "depth": "preview_depth.png"}
    if which not in allowed:
        return "Not found", 404
    img_path = result_dir / allowed[which]
    if not img_path.is_file():
        return "Preview not ready", 404
    return send_file(str(img_path), mimetype="image/png")


@app.route("/download/<job_id>/<which>")
def download_file(job_id: str, which: str):
    job = _job_get(job_id)
    if job is None or job["status"] != "done":
        return "Not found", 404
    outputs  = job.get("outputs", {})
    if which not in outputs:
        return "File not available", 404
    result_dir = RESULTS_DIR / job_id
    filename   = outputs[which]
    file_path  = result_dir / filename
    if not file_path.is_file():
        return "File not found", 404

    stem = _safe_stem(job["filename"])
    friendly = {
        "npy": f"{stem}_depth.npy",
        "png": f"{stem}_depth_heatmap.png",
        "tif": f"{stem}_depth.tif",
        "vis": f"{stem}_depth_visualization.png",
    }
    return send_file(str(file_path), as_attachment=True,
                     download_name=friendly.get(which, filename))


@app.route("/health")
def health():
    return jsonify({
        "status"            : "ok",
        "device"            : _DEVICE,
        "terrain_available" : _TERRAIN_AVAILABLE,
    })


# ===========================================================================
# ROUTES — Role 3 (Terrain Classification)
# ===========================================================================

@app.route("/terrain/upload", methods=["POST"])
def terrain_upload():
    if not _TERRAIN_AVAILABLE:
        return jsonify({
            "error": "Terrain classification module is not available. "
                     f"Check imports: {_terrain_import_err_msg if not _TERRAIN_AVAILABLE else ''}"
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    suffix = Path(f.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{suffix}'. "
                     "Please upload TIFF, TIF, JPG, JPEG, or PNG."
        }), 415

    job_id      = str(uuid.uuid4())
    safe_name   = f"{job_id}{suffix}"
    upload_path = TERRAIN_UPLOAD_DIR / safe_name
    f.save(str(upload_path))

    try:
        w, h = _image_dimensions(upload_path)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        return jsonify({"error": f"Unable to read image: {exc}"}), 422

    is_geotiff   = suffix in {".tif", ".tiff"}
    file_size_mb = round(upload_path.stat().st_size / 1e6, 2)

    _tj_set(job_id, {
        "status"      : "uploaded",
        "stage"       : "Ready",
        "progress"    : 0,
        "filename"    : f.filename,
        "suffix"      : suffix,
        "format"      : _file_format_label(suffix),
        "is_geotiff"  : is_geotiff,
        "width"       : w,
        "height"      : h,
        "file_size_mb": file_size_mb,
        "upload_path" : str(upload_path),
        "outputs"     : {},
    })

    return jsonify({
        "job_id"      : job_id,
        "filename"    : f.filename,
        "format"      : _file_format_label(suffix),
        "width"       : w,
        "height"      : h,
        "file_size_mb": file_size_mb,
        "is_geotiff"  : is_geotiff,
    })


@app.route("/terrain/classify/<job_id>", methods=["POST"])
def terrain_classify(job_id: str):
    job = _tj_get(job_id)
    if job is None:
        return jsonify({"error": "Unknown terrain job ID."}), 404
    if job["status"] not in {"uploaded", "error"}:
        return jsonify({"error": "Classification already running or complete."}), 409

    _tj_update(job_id, status="running", stage="Starting…", progress=2)
    upload_path = Path(job["upload_path"])
    threading.Thread(
        target=_run_terrain_inference,
        args=(job_id, upload_path),
        daemon=True,
    ).start()
    return jsonify({"status": "running", "job_id": job_id})


@app.route("/terrain/status/<job_id>")
def terrain_status(job_id: str):
    job = _tj_get(job_id)
    if job is None:
        return jsonify({"error": "Unknown terrain job ID."}), 404

    safe = {k: job[k] for k in (
        "status", "stage", "progress", "filename", "format",
        "is_geotiff", "width", "height", "file_size_mb",
    ) if k in job}

    if job["status"] == "done":
        for key in (
            "inference_time", "device",
            "input_width", "input_height",
            "is_geotiff", "geo_input_info",
            "terrain_classes", "building_count",
            "outputs",
        ):
            if key in job:
                safe[key] = job[key]

    if job["status"] == "error":
        safe["error"] = job.get("error", "Unknown error.")

    return jsonify(safe)


@app.route("/terrain/preview/<job_id>/<which>")
def terrain_preview(job_id: str, which: str):
    job = _tj_get(job_id)
    if job is None:
        return "Not found", 404
    result_dir = TERRAIN_RESULTS_DIR / job_id
    allowed = {
        "input"   : "preview_input.png",
        "overlay" : "preview_overlay.png",
        "labelmap": "preview_labelmap.png",
    }
    if which not in allowed:
        return "Not found", 404
    img_path = result_dir / allowed[which]
    if not img_path.is_file():
        return "Preview not ready", 404
    return send_file(str(img_path), mimetype="image/png")


@app.route("/terrain/download/<job_id>/<which>")
def terrain_download(job_id: str, which: str):
    job = _tj_get(job_id)
    if job is None or job["status"] != "done":
        return "Not found", 404
    outputs   = job.get("outputs", {})
    if which not in outputs:
        return "File not available", 404
    result_dir = TERRAIN_RESULTS_DIR / job_id
    filename   = outputs[which]
    file_path  = result_dir / filename
    if not file_path.is_file():
        return "File not found", 404

    stem = _safe_stem(job["filename"])
    ct_map = {
        "overlay" : ("image/png",        f"{stem}_terrain_overlay.png"),
        "labelmap": ("image/png",        f"{stem}_terrain_labelmap.png"),
        "json"    : ("application/json", f"{stem}_terrain_data.json"),
        "tif"     : ("image/tiff",       f"{stem}_terrain_labels.tif"),
    }
    mimetype, dl_name = ct_map.get(which, ("application/octet-stream", filename))
    return send_file(str(file_path), as_attachment=True,
                     mimetype=mimetype, download_name=dl_name)


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the DepthWizard unified web interface.")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  DepthWizard — Unified Interface")
    print(f"  Depth Estimation  |  Terrain Classification")
    print(f"  Open http://{args.host}:{args.port} in your browser.")
    print(f"  Device: {_DEVICE.upper()}")
    print(f"  Terrain module: {'available' if _TERRAIN_AVAILABLE else 'UNAVAILABLE'}")
    print(f"  Press Ctrl+C to stop.\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
