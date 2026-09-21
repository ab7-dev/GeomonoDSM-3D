"""GAMUS dataset loading utilities.

Supports two access modes:

1. **Local mode** (default)
   Assumes GAMUS data is already on disk at ``config.data_dir``.
   Works entirely offline, no extra packages required.

2. **Streaming mode** (``config.streaming = True``)
   Uses the HuggingFace ``datasets`` library to stream samples without
   downloading the full ~80 GB corpus.
   Requires:  pip install datasets

GAMUS height data is nDSM (above-ground height in metres), NOT absolute
terrain elevation.  See ``config.py`` for the scientific explanation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterator

import numpy as np
from PIL import Image

from .config import GAMUSConfig

logger = logging.getLogger(__name__)


def load_sample(
    rgb_path: str | Path,
    ndsm_path: str | Path,
    crop_size: int | None = None,
    seed: int = 0,
) -> dict:
    """Load one RGB / nDSM pair from disk.

    Parameters
    ----------
    rgb_path  : path to the RGB image (any PIL-readable format or GeoTIFF).
    ndsm_path : path to the nDSM raster (.tif preferred; float32 metres).
    crop_size : if given, take a random square crop of this size.
    seed      : random seed for reproducible cropping.

    Returns
    -------
    dict with keys:
        ``rgb``        – PIL Image (uint8 RGB)
        ``ndsm``       – np.ndarray float32, shape (H, W), metres above ground
        ``rgb_path``   – Path
        ``ndsm_path``  – Path
        ``crop``       – (y0, x0) top-left of crop, or None
    """
    rgb_path  = Path(rgb_path)
    ndsm_path = Path(ndsm_path)

    # Load RGB
    rgb = _load_rgb(rgb_path)

    # Load nDSM
    ndsm = _load_ndsm(ndsm_path)

    # Ensure matching spatial size
    rw, rh = rgb.size
    nh, nw = ndsm.shape
    if (rh, rw) != (nh, nw):
        logger.warning(
            "RGB (%dx%d) and nDSM (%dx%d) sizes differ for %s — "
            "resizing nDSM to match RGB.",
            rw, rh, nw, nh, rgb_path.name,
        )
        from PIL import Image as PILImage
        ndsm_img  = PILImage.fromarray(ndsm).resize((rw, rh), PILImage.BILINEAR)
        ndsm = np.asarray(ndsm_img, dtype=np.float32)

    crop_offset = None
    if crop_size is not None and crop_size < min(rh, rw):
        rng = np.random.default_rng(seed)
        y0  = int(rng.integers(0, rh - crop_size))
        x0  = int(rng.integers(0, rw - crop_size))
        rgb  = rgb.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        ndsm = ndsm[y0:y0 + crop_size, x0:x0 + crop_size]
        crop_offset = (y0, x0)

    return {
        "rgb"      : rgb,
        "ndsm"     : ndsm,
        "rgb_path" : rgb_path,
        "ndsm_path": ndsm_path,
        "crop"     : crop_offset,
    }


class GAMUSDataset:
    """Iterable dataset for local GAMUS data.

    Discovers matched RGB / nDSM pairs by filename stem.

    Parameters
    ----------
    config : GAMUSConfig

    Usage
    -----
    ::

        from depth_mapping.gamus import GAMUSConfig, GAMUSDataset
        cfg = GAMUSConfig(data_dir="gamus_data", max_samples=10)
        ds  = GAMUSDataset(cfg)
        for sample in ds:
            rgb  = sample["rgb"]    # PIL Image
            ndsm = sample["ndsm"]   # np.ndarray float32, metres AGL
    """

    def __init__(self, config: GAMUSConfig) -> None:
        self.config = config
        self._pairs: list[tuple[Path, Path]] = []
        self._discovered = False

    def _discover(self) -> None:
        if self._discovered:
            return
        cfg = self.config
        cfg.validate()

        rgb_files = sorted(cfg.rgb_dir.glob(f"*{cfg.image_extension}"))
        ndsm_dir  = cfg.ndsm_dir

        pairs: list[tuple[Path, Path]] = []
        for rgb_path in rgb_files:
            ndsm_path = ndsm_dir / (rgb_path.stem + cfg.ndsm_extension)
            if ndsm_path.is_file():
                pairs.append((rgb_path, ndsm_path))
            else:
                logger.debug("No matching nDSM for %s — skipping.", rgb_path.name)

        if cfg.max_samples is not None:
            pairs = pairs[: cfg.max_samples]

        self._pairs = pairs
        self._discovered = True
        logger.info("GAMUS: found %d matched RGB/nDSM pairs.", len(pairs))

    def __len__(self) -> int:
        self._discover()
        return len(self._pairs)

    def __iter__(self) -> Iterator[dict]:
        self._discover()
        for rgb_path, ndsm_path in self._pairs:
            try:
                yield load_sample(
                    rgb_path, ndsm_path,
                    crop_size=self.config.crop_size,
                )
            except Exception as exc:
                logger.warning("Skipping %s: %s", rgb_path.name, exc)

    def __getitem__(self, idx: int) -> dict:
        self._discover()
        rgb_path, ndsm_path = self._pairs[idx]
        return load_sample(
            rgb_path, ndsm_path,
            crop_size=self.config.crop_size,
        )


# ---------------------------------------------------------------------------
# HuggingFace streaming loader (optional)
# ---------------------------------------------------------------------------

def stream_gamus(
    config: GAMUSConfig,
    split: str = "train",
) -> Generator[dict, None, None]:
    """Yield GAMUS samples via HuggingFace datasets streaming.

    Does NOT download the full dataset.  Each sample is fetched on demand.

    Requires:  pip install datasets

    Parameters
    ----------
    config : GAMUSConfig   (config.hf_repo_id must be correct)
    split  : HuggingFace split name (e.g. "train", "validation")

    Yields
    ------
    dict with keys ``rgb``, ``ndsm``, ``rgb_path`` (None), ``ndsm_path`` (None).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "HuggingFace `datasets` package is required for streaming.\n"
            "Install it with:  pip install datasets"
        ) from exc

    logger.info("Streaming GAMUS from %s (split=%s) ...", config.hf_repo_id, split)
    ds = load_dataset(config.hf_repo_id, split=split, streaming=True)

    count = 0
    for raw in ds:
        if config.max_samples is not None and count >= config.max_samples:
            break

        try:
            # HuggingFace GAMUS fields: "image" (PIL), "ndsm" (PIL or array)
            rgb_pil = raw.get("image") or raw.get("rgb")
            ndsm_raw = raw.get("ndsm") or raw.get("height")

            if rgb_pil is None or ndsm_raw is None:
                logger.debug("Unexpected GAMUS field names: %s", list(raw.keys()))
                continue

            rgb = rgb_pil.convert("RGB") if hasattr(rgb_pil, "convert") else Image.fromarray(np.array(rgb_pil))

            if hasattr(ndsm_raw, "convert"):
                ndsm = np.asarray(ndsm_raw, dtype=np.float32)
            else:
                ndsm = np.asarray(ndsm_raw, dtype=np.float32)

            if config.crop_size is not None:
                sample = load_sample.__wrapped__(rgb, ndsm, config.crop_size) if False else None
                # Direct crop for streaming path
                H, W = ndsm.shape[:2]
                cs = config.crop_size
                if cs < H and cs < W:
                    rng = np.random.default_rng(count)
                    y0 = int(rng.integers(0, H - cs))
                    x0 = int(rng.integers(0, W - cs))
                    rgb  = rgb.crop((x0, y0, x0 + cs, y0 + cs))
                    ndsm = ndsm[y0:y0+cs, x0:x0+cs]

            yield {
                "rgb"       : rgb,
                "ndsm"      : ndsm.squeeze() if ndsm.ndim == 3 else ndsm,
                "rgb_path"  : None,
                "ndsm_path" : None,
                "crop"      : None,
            }
            count += 1

        except Exception as exc:
            logger.warning("Skipping streaming sample %d: %s", count, exc)
            continue


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_rgb(path: Path) -> Image.Image:
    """Load any image format to uint8 RGB PIL Image."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
            with rasterio.open(path) as src:
                n    = min(src.count, 3)
                data = src.read(list(range(1, n + 1)))
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            hwc = np.moveaxis(data, 0, -1)
            if data.dtype != np.uint8:
                lo, hi = np.percentile(hwc, (1, 99))
                hwc = np.clip((hwc - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
            return Image.fromarray(hwc, "RGB")
        except ImportError:
            pass
    return Image.open(path).convert("RGB")


def _load_ndsm(path: Path) -> np.ndarray:
    """Load an nDSM raster to float32 (H, W) array in metres above ground.

    Handles:
      - Single-band float GeoTIFF (standard GAMUS format)
      - PNG/JPEG fallback (uint8, stored as cm or mm — auto-detected)
    """
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            return data
        except ImportError:
            pass

    # Fallback: load as image, assume uint16 cm → metres
    img = Image.open(path)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    # Heuristic: if max > 1000 assume stored in cm → convert to metres
    if arr.max() > 1000.0:
        arr = arr / 100.0
    return arr
