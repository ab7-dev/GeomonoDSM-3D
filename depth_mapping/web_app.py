"""Small local web interface for trying the relative-depth estimator.

Run ``python -m depth_mapping.web_app`` and open the printed address in a browser.
The interface stays local to this computer and accepts JPG, PNG, and GeoTIFF
files.  Its first prediction can take longer because Hugging Face downloads the
model checkpoint once and stores it in its normal cache.
"""

from __future__ import annotations

import html
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote

from depth_mapping.depth import get_depth, save_depth_outputs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _page(message: str = "", image_url: str = "") -> bytes:
    """Build the deliberately simple, dependency-free upload page."""
    result = ""
    if message:
        result = f"<p class='result'>{html.escape(message)}</p>"
    if image_url:
        result += f"<img class='depth' src='{html.escape(image_url)}' alt='Depth heatmap'>"
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Depth Anything V2 uploader</title>
<style>body{{font:16px system-ui;max-width:760px;margin:3rem auto;padding:0 1rem}}
button{{padding:.6rem 1rem}} .result{{background:#eef6ee;padding:1rem}}
.depth{{max-width:100%;border:1px solid #ccc;margin-top:1rem}}</style></head>
<body><h1>Relative depth estimator</h1>
<p>Select a JPG, PNG, or GeoTIFF. Higher values represent closer/taller areas.</p>
<form method='post' enctype='multipart/form-data'>
<input name='image' type='file' accept='.jpg,.jpeg,.png,.tif,.tiff' required>
<button type='submit'>Estimate depth</button></form>{result}</body></html>""".encode()


class DepthUploadHandler(BaseHTTPRequestHandler):
    """Serve the upload form and pass submitted imagery to the estimator."""

    def _send(self, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - required method name from stdlib
        """Serve the form or a generated PNG from the outputs directory."""
        if self.path.startswith("/outputs/"):
            # ``quote`` encodes spaces and other safe filename characters in the
            # browser URL, so decode it before looking up the saved heatmap.
            requested = (PROJECT_ROOT / unquote(self.path).lstrip("/")).resolve()
            if requested.parent == OUTPUT_DIR.resolve() and requested.suffix == ".png" and requested.is_file():
                self._send(requested.read_bytes(), "image/png")
                return
        self._send(_page())

    def do_POST(self) -> None:  # noqa: N802 - required method name from stdlib
        """Receive one multipart image upload and return its depth heatmap."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type or "boundary=" not in content_type:
            self._send(_page("Please submit an image using the upload form."))
            return
        boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode()
        payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        # Browsers send each field as a boundary-delimited section.  This local
        # app accepts only the file field and keeps the parser intentionally small.
        file_bytes = None
        submitted_name = ""
        for part in payload.split(b"--" + boundary):
            headers, separator, value = part.partition(b"\r\n\r\n")
            if separator and b'name="image"' in headers and b"filename=" in headers:
                submitted_name = headers.split(b'filename="', 1)[1].split(b'"', 1)[0].decode("utf-8", "replace")
                file_bytes = value.rstrip(b"\r\n")
                break
        filename = Path(submitted_name).name
        if not file_bytes or Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
            self._send(_page("Choose a non-empty JPG, PNG, or GeoTIFF file."))
            return

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        upload_path = UPLOAD_DIR / filename
        upload_path.write_bytes(file_bytes)
        try:
            depth = get_depth(str(upload_path))
            save_depth_outputs(depth, str(OUTPUT_DIR), filename)
            heatmap_name = f"{Path(filename).stem}_depth.png"
            message = f"Finished {filename}. Raw depth and heatmap were saved in outputs/."
            self._send(_page(message, "/outputs/" + quote(heatmap_name)))
        except Exception as error:  # Show inference/download errors without stopping the server.
            self._send(_page(f"Could not estimate depth: {error}"))

    def log_message(self, format: str, *args: object) -> None:
        """Keep routine HTTP request logs out of the terminal."""


def main() -> None:
    """Start a browser-accessible uploader on this computer."""
    parser = argparse.ArgumentParser(description="Run the local depth-image uploader.")
    parser.add_argument("--port", type=int, default=8000, help="Local port for the uploader (default: 8000).")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DepthUploadHandler)
    print(f"Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C here to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
