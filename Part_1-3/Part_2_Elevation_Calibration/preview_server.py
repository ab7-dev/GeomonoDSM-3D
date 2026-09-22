"""
GeoMonoDSM-3D  Part 2 — Local Verification Viewer
==================================================

Serves a single HTML page that shows the three pipeline outputs side-by-side:
    [Original RGB]  [Depth preview]  [Calibrated DSM preview]

Plus calibration stats: a, b, DSM min/max/mean, GAMUS MAE/RMSE.

Usage (run from the project root or Elevation_Calibration/):
    python preview_server.py
    python preview_server.py --port 8080
    python preview_server.py --output-dir Elevation_Calibration/outputs
                             --rgb-image   path/to/original.tif
                             --depth-npy   path/to/depth.npy
                             --port        8000

The server reads calibration_report.json automatically from --output-dir.
If the JSON is not present yet, it renders a "waiting for calibration" page.

No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = (Path(__file__).resolve().parent / "outputs").resolve()
DEFAULT_PORT       = 8000


# ===========================================================================
# Image helpers (stdlib only, but PIL is already a Part 1 dep)
# ===========================================================================

def _img_to_b64(path: Path, target_w: int = 600) -> str:
    """Load an image, optionally resize, and return a base64 PNG data URI."""
    try:
        from PIL import Image
        img = Image.open(path)
        if img.width > target_w:
            ratio = target_w / img.width
            img = img.resize(
                (target_w, int(img.height * ratio)),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception as exc:
        return f""  # blank 1-px placeholder on error


def _depth_npy_to_b64(npy_path: Path, target_w: int = 600) -> str:
    """Load a .npy depth array and render as viridis colormap → base64 PNG."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr = np.load(npy_path)
        fig, ax = plt.subplots(figsize=(target_w / 100, target_w * arr.shape[0] / arr.shape[1] / 100))
        ax.imshow(arr, cmap="viridis")
        ax.axis("off")
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _geotiff_rgb_to_b64(tif_path: Path, target_w: int = 600) -> str:
    """Extract RGB from a GeoTIFF → base64 PNG."""
    try:
        import rasterio
        import numpy as np
        from PIL import Image

        with rasterio.open(tif_path) as src:
            n = min(src.count, 3)
            data = src.read(list(range(1, n + 1)))  # (bands, H, W)

        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        elif data.shape[0] == 2:
            data = np.concatenate([data, data[:1]], axis=0)

        data = np.moveaxis(data, 0, -1)  # (H, W, 3)
        # Stretch to uint8
        lo = np.percentile(data, 1)
        hi = np.percentile(data, 99)
        data = np.clip((data - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype("uint8")

        img = Image.fromarray(data, "RGB")
        if img.width > target_w:
            r = target_w / img.width
            img = img.resize((target_w, int(img.height * r)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


# ===========================================================================
# HTML builder
# ===========================================================================

def _build_html(
    output_dir: Path,
    rgb_src: str | None,
    depth_npy: str | None,
) -> str:
    """Generate the full HTML page as a string."""

    # Resolve output_dir to an absolute path so report/preview lookups work
    # regardless of what the current working directory happens to be.
    output_dir = Path(output_dir).resolve()

    # ── Load report ──────────────────────────────────────────────────────────
    report_path = output_dir / "calibration_report.json"
    report: dict = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            pass

    cal   = report.get("calibration", {})
    stats = report.get("dsm_stats", {})
    gamus = report.get("gamus_validation", {})
    a_val = cal.get("a", "—")
    b_val = cal.get("b", "—")

    def _fmt(v, digits=2):
        return f"{v:.{digits}f}" if isinstance(v, (int, float)) else str(v)

    # ── Encode images ────────────────────────────────────────────────────────
    # 1. Original RGB
    rgb_b64 = ""
    if rgb_src:
        p = Path(rgb_src)
        if p.suffix.lower() in (".tif", ".tiff"):
            rgb_b64 = _geotiff_rgb_to_b64(p)
        elif p.is_file():
            rgb_b64 = _img_to_b64(p)

    # 2. Depth preview — prefer _depth_vis.png, fall back to .npy rendering
    depth_b64 = ""
    if depth_npy:
        # Look for an existing viridis PNG alongside the .npy
        npy_p = Path(depth_npy)
        vis_candidates = [
            npy_p.parent / (npy_p.stem + "_depth_vis.png"),
            npy_p.parent / (npy_p.stem + "_depth.png"),
            npy_p.with_suffix(".png"),
        ]
        for c in vis_candidates:
            if c.is_file():
                depth_b64 = _img_to_b64(c)
                break
        if not depth_b64 and npy_p.is_file():
            depth_b64 = _depth_npy_to_b64(npy_p)

    # 3. DSM preview PNG
    dsm_preview = output_dir / "calibrated_dsm_preview.png"
    dsm_b64 = _img_to_b64(dsm_preview) if dsm_preview.is_file() else ""

    def _img_card(title: str, b64: str, label: str) -> str:
        src = b64 if b64 else "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="
        return f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <img src="{src}" alt="{title}" onerror="this.style.opacity=0.2" />
          <div class="card-label">{label}</div>
        </div>"""

    gamus_html = ""
    if isinstance(gamus, dict) and gamus:
        gamus_html = f"""
        <div class="stat-block">
          <h3>GAMUS Structural-Consistency Check
            <span class="badge warning">approximate — see note</span>
          </h3>
          <p class="note">{gamus.get('note','')}</p>
          <table>
            <tr><td>Mean MAE</td><td><b>{_fmt(gamus.get('mae','—'))} m</b></td></tr>
            <tr><td>Mean RMSE</td><td><b>{_fmt(gamus.get('rmse','—'))} m</b></td></tr>
            <tr><td>Samples</td><td>{gamus.get('n_samples','—')}</td></tr>
          </table>
        </div>"""
    elif gamus == "skipped":
        gamus_html = '<div class="stat-block"><h3>GAMUS Validation</h3><p>Skipped.</p></div>'
    else:
        gamus_html = '<div class="stat-block"><h3>GAMUS Validation</h3><p>Not yet run or API unavailable.</p></div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>GeoMonoDSM-3D · Part 2 Verification</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f1117; color: #e0e0e0;
      min-height: 100vh; padding: 24px;
    }}
    h1 {{ font-size: 1.6rem; color: #7ecfff; margin-bottom: 4px; }}
    .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 28px; }}
    .panels {{
      display: flex; gap: 20px; flex-wrap: wrap;
      justify-content: center; margin-bottom: 36px;
    }}
    .card {{
      background: #1a1d27; border: 1px solid #2e3147;
      border-radius: 10px; overflow: hidden; width: 340px;
    }}
    .card-title {{
      background: #252840; padding: 10px 14px;
      font-weight: 600; font-size: 0.85rem; color: #aac8ff;
      text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .card img {{
      width: 100%; height: 240px; object-fit: cover; display: block;
    }}
    .card-label {{
      padding: 8px 14px; font-size: 0.78rem; color: #777; background: #161824;
    }}
    .stats-row {{
      display: flex; gap: 20px; flex-wrap: wrap;
      justify-content: center; margin-bottom: 20px;
    }}
    .stat-block {{
      background: #1a1d27; border: 1px solid #2e3147;
      border-radius: 10px; padding: 18px 22px; min-width: 280px; flex: 1;
    }}
    .stat-block h3 {{
      font-size: 0.95rem; color: #7ecfff; margin-bottom: 12px;
      display: flex; align-items: center; gap: 8px;
    }}
    .stat-block table {{ width: 100%; border-collapse: collapse; }}
    .stat-block td {{ padding: 5px 8px; font-size: 0.88rem; }}
    .stat-block td:first-child {{ color: #999; width: 55%; }}
    .stat-block td b {{ color: #e8f5e9; }}
    .badge {{
      font-size: 0.68rem; padding: 2px 7px; border-radius: 20px;
      font-weight: 500; letter-spacing: 0.03em;
    }}
    .badge.ok {{ background: #1b3a27; color: #80e09f; }}
    .badge.warning {{ background: #3a2e14; color: #f0bf6a; }}
    .note {{
      font-size: 0.78rem; color: #888; margin-bottom: 10px;
      line-height: 1.5;
    }}
    footer {{
      text-align: center; color: #444; font-size: 0.8rem; margin-top: 30px;
    }}
    .highlight {{ color: #f0bf6a; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>GeoMonoDSM-3D &mdash; Part 2: Elevation Calibration</h1>
  <p class="subtitle">Pipeline verification viewer &nbsp;|&nbsp; auto-refreshes every 30 s</p>

  <div class="panels">
    {_img_card("Original RGB", rgb_b64, "Source GeoTIFF — input to Part 1")}
    {_img_card("Relative Depth Map", depth_b64, "Part 1 output — dimensionless, larger = taller/closer")}
    {_img_card("Calibrated DSM", dsm_b64, "Part 2 output — elevation in metres (terrain colormap)")}
  </div>

  <div class="stats-row">
    <div class="stat-block">
      <h3>Calibration Parameters <span class="badge ok">SRTM30m</span></h3>
      <table>
        <tr><td>Scale factor (a)</td><td><b class="highlight">{_fmt(a_val, 6)}</b></td></tr>
        <tr><td>Offset (b)</td><td><b class="highlight">{_fmt(b_val, 4)} m</b></td></tr>
        <tr><td>Method</td><td>{cal.get('method','—')}</td></tr>
        <tr><td>Reference</td><td>{cal.get('reference_dataset','—')}</td></tr>
      </table>
    </div>

    <div class="stat-block">
      <h3>DSM Statistics</h3>
      <table>
        <tr><td>Min elevation</td><td><b>{_fmt(stats.get('min_m','—'))} m</b></td></tr>
        <tr><td>Max elevation</td><td><b>{_fmt(stats.get('max_m','—'))} m</b></td></tr>
        <tr><td>Mean elevation</td><td><b>{_fmt(stats.get('mean_m','—'))} m</b></td></tr>
        <tr><td>Std deviation</td><td><b>{_fmt(stats.get('std_m','—'))} m</b></td></tr>
        <tr><td>Array shape</td><td>{stats.get('shape','—')}</td></tr>
      </table>
    </div>

    {gamus_html}
  </div>

  <footer>
    Output directory: {output_dir.resolve()}<br/>
    Served by GeoMonoDSM-3D preview_server.py &mdash; no external dependencies
  </footer>

  <script>
    // Auto-reload every 30 seconds so the page updates if calibration finishes
    setTimeout(() => location.reload(), 30000);
  </script>
</body>
</html>"""
    return html


# ===========================================================================
# HTTP handler
# ===========================================================================

class PreviewHandler(BaseHTTPRequestHandler):
    """Minimal single-page HTTP handler — serves the viewer at any path."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    rgb_src:    str  = ""
    depth_npy:  str  = ""

    def do_GET(self):
        parsed = urlparse(self.path)

        # ── Serve the HTML page ───────────────────────────────────────────
        html = _build_html(
            self.output_dir,
            self.rgb_src or None,
            self.depth_npy or None,
        )
        body = html.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Suppress per-request noise; just log the initial start message
        pass


# ===========================================================================
# Server start
# ===========================================================================

def _find_free_port(preferred: int) -> int:
    """Return `preferred` if free, otherwise find the next available port."""
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    return preferred  # give up, let the OS error out


def start_server(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    rgb_src:    str = "",
    depth_npy:  str = "",
    port:       int = DEFAULT_PORT,
) -> None:
    """Start the preview server and block until Ctrl-C.

    Parameters
    ----------
    output_dir : directory containing calibrated_dsm_preview.png and
                 calibration_report.json
    rgb_src    : path to the original GeoTIFF (for the RGB panel)
    depth_npy  : path to the .npy depth array (for the depth panel)
    port       : preferred TCP port (auto-incremented if busy)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_port = _find_free_port(port)

    # Attach config to handler via class attributes (simple, no closure needed)
    PreviewHandler.output_dir = output_dir
    PreviewHandler.rgb_src    = rgb_src
    PreviewHandler.depth_npy  = depth_npy

    server = HTTPServer(("", actual_port), PreviewHandler)

    url = f"http://localhost:{actual_port}"
    print("\n" + "=" * 60)
    print("  GeoMonoDSM-3D  Part 2 — Preview Server")
    print("=" * 60)
    print(f"\n  Open this URL in your browser:")
    print(f"\n      {url}\n")
    print(f"  Serving outputs from:  {output_dir.resolve()}")
    print(f"  Press Ctrl-C to stop.\n")
    print("=" * 60 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GeoMonoDSM-3D Part 2 — Local verification viewer",
        prog="python preview_server.py",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Directory containing calibration outputs (default: Elevation_Calibration/outputs)",
    )
    parser.add_argument(
        "--rgb-image", default="",
        help="Path to the original GeoTIFF / RGB image (for the first panel)",
    )
    parser.add_argument(
        "--depth-npy", default="",
        help="Path to the .npy depth array from Part 1 (for the depth panel)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"TCP port to listen on (default: {DEFAULT_PORT})",
    )
    args = parser.parse_args()

    start_server(
        output_dir=str(Path(args.output_dir).resolve()),
        rgb_src=str(Path(args.rgb_image).resolve()) if args.rgb_image else "",
        depth_npy=str(Path(args.depth_npy).resolve()) if args.depth_npy else "",
        port=args.port,
    )


if __name__ == "__main__":
    main()
