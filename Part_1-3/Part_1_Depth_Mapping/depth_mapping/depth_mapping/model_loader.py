"""Load local Depth Anything V2 checkpoints (.pth) without Hugging Face.

The original DA V2 architecture (vendored under depth_anything_v2/) is used
directly, so the local .pth files in the checkpoints/ directory can be loaded
without downloading any Hugging Face model.

Checkpoint location resolution order
--------------------------------------
1. ``checkpoint_dir`` argument passed directly to ``load_model()``
2. ``DA_CHECKPOINT_DIR`` environment variable
3. ``<project_root>/checkpoints/``          (project-local copy)
4. ``<project_root>/../checkpoints/``       (sibling of project root — the
                                             layout used in the SIH workspace)
5. ``<project_root>/../../checkpoints/``    (two levels above project root)

"project root" is defined as two levels above this file:
    depth_mapping/model_loader.py → depth_mapping/ → Depth_Mapping_Part_1/

Expected filenames (standard Depth Anything V2 naming):
    depth_anything_v2_vitl.pth  – Large  (production, ~1.3 GB)
    depth_anything_v2_vits.pth  – Small  (optional fast mode, ~95 MB)

Model configurations
--------------------
    vitl : embed_dim=1024, depth=24, features=256, out_channels=[256,512,1024,1024]
    vits : embed_dim=384,  depth=12, features=64,  out_channels=[48, 96, 192, 384]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configurations – must match the .pth weights exactly.
# ---------------------------------------------------------------------------
_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
}

_CHECKPOINT_FILENAMES: dict[str, str] = {
    "vitl": "depth_anything_v2_vitl.pth",
    "vitb": "depth_anything_v2_vitb.pth",
    "vits": "depth_anything_v2_vits.pth",
}

# Module-level cache: keyed by (encoder_name, device_str).
_MODEL_CACHE: dict[tuple[str, str], Any] = {}

# ---------------------------------------------------------------------------
# This file's location within the package layout:
#   Depth_Mapping_Part_1/
#     depth_mapping/          ← _PKG_DIR
#       model_loader.py       ← __file__
#   checkpoints/              ← siblings at SIH workspace level
# ---------------------------------------------------------------------------
_PKG_DIR     = Path(__file__).resolve().parent          # depth_mapping/
_PROJECT_DIR = _PKG_DIR.parent                          # Depth_Mapping_Part_1/


def default_checkpoint_dir() -> Path:
    """Return the best available checkpoint directory.

    Search order
    ------------
    1. ``$DA_CHECKPOINT_DIR`` environment variable
    2. ``<project_root>/checkpoints/``          (project-local)
    3. ``<project_root>/../checkpoints/``       (SIH workspace sibling)
    4. ``<project_root>/../../checkpoints/``    (two levels above project)

    Raises
    ------
    FileNotFoundError
        If no candidate directory contains any recognisable ``.pth`` file.
    """
    # 1. Explicit environment override — highest priority.
    env = os.environ.get("DA_CHECKPOINT_DIR")
    if env:
        return Path(env)

    # 2–4. Automatic discovery: walk up from the project directory.
    candidates: list[Path] = [
        _PROJECT_DIR / "checkpoints",
        _PROJECT_DIR.parent / "checkpoints",
        _PROJECT_DIR.parent.parent / "checkpoints",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            pth_files = list(candidate.glob("*.pth"))
            if pth_files:
                logger.debug("Checkpoint dir resolved: %s  (%d .pth files)", candidate, len(pth_files))
                return candidate

    # Fall back to project-local even if empty — caller gets a clear error.
    return _PROJECT_DIR / "checkpoints"


def load_model(
    encoder: str = "vitl",
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> Any:
    """Instantiate and return a Depth Anything V2 model with weights loaded.

    The model is cached after the first load; subsequent calls with the same
    ``encoder`` and resolved ``device`` return the cached instance immediately.

    Parameters
    ----------
    encoder:
        ``"vitl"`` (default, production) or ``"vits"`` (fast/development).
    device:
        PyTorch device string.  ``None`` → CUDA if available, otherwise CPU.
    checkpoint_dir:
        Directory containing the ``.pth`` file.  ``None`` → resolved via
        ``default_checkpoint_dir()`` / ``$DA_CHECKPOINT_DIR``.

    Returns
    -------
    torch.nn.Module
        Model in ``eval()`` mode, placed on the requested device.
    """
    if encoder not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from: {list(_MODEL_CONFIGS)}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = (encoder, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else default_checkpoint_dir()
    ckpt_path = ckpt_dir / _CHECKPOINT_FILENAMES[encoder]

    if not ckpt_path.is_file():
        searched = [
            str(_PROJECT_DIR / "checkpoints"),
            str(_PROJECT_DIR.parent / "checkpoints"),
            str(_PROJECT_DIR.parent.parent / "checkpoints"),
        ]
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Expected file: {_CHECKPOINT_FILENAMES[encoder]}\n\n"
            f"Automatic search locations tried:\n"
            + "\n".join(f"  {p}" for p in searched)
            + "\n\nTo fix, either:\n"
            f"  1. Set the environment variable:  set DA_CHECKPOINT_DIR=<path>\n"
            f"  2. Pass checkpoint_dir= to load_model()\n"
            f"  3. Place the .pth file in one of the searched locations above."
        )

    # Import here to keep the module-level import light.
    from depth_mapping.depth_anything_v2.dpt import DepthAnythingV2

    cfg = _MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg)

    logger.info("Loading %s checkpoint from %s", encoder, ckpt_path)
    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

    model.to(device).eval()
    logger.info(
        "Model loaded on %s (%.0fM parameters)",
        device, sum(p.numel() for p in model.parameters()) / 1e6,
    )

    _MODEL_CACHE[cache_key] = model
    return model


def unload_model(encoder: str = "vitl", device: str | None = None) -> None:
    """Remove a cached model and free its memory.  Useful between test runs."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (encoder, device)
    if key in _MODEL_CACHE:
        del _MODEL_CACHE[key]
        if device == "cuda":
            torch.cuda.empty_cache()
