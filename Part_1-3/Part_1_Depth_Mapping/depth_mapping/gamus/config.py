"""GAMUS dataset configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GAMUSConfig:
    """Configuration for GAMUS dataset access and evaluation.

    Parameters
    ----------
    data_dir : str | Path
        Root directory where GAMUS data is stored locally.
        Expected layout::

            data_dir/
                rgb/          # or any sub-folder — set rgb_subdir / ndsm_subdir
                ndsm/
                ...

    rgb_subdir : str
        Sub-directory (relative to data_dir) containing RGB images.
    ndsm_subdir : str
        Sub-directory containing nDSM rasters (height above ground, metres).
    image_extension : str
        File extension for RGB images (e.g. ".png", ".tif").
    ndsm_extension : str
        File extension for nDSM rasters (usually ".tif").
    max_samples : int | None
        Limit evaluation to the first N samples (None = all available).
    hf_repo_id : str
        HuggingFace dataset repo ID for streaming/downloading.
    streaming : bool
        When True, use HuggingFace datasets streaming to avoid full download.
        Requires:  pip install datasets
    crop_size : int | None
        If set, randomly crop RGB+nDSM pairs to this size before evaluation.
        Useful for fast experimentation on large tiles.
    device : str
        PyTorch device for inference ("cuda" or "cpu").

    Notes
    -----
    nDSM values represent **height above ground** (normalised DSM), not
    absolute geodetic elevation.  Do not use GAMUS nDSM directly as a
    metric elevation reference for georeferencing.
    """

    data_dir:        Path            = field(default_factory=lambda: Path("gamus_data"))
    rgb_subdir:      str             = "rgb"
    ndsm_subdir:     str             = "ndsm"
    image_extension: str             = ".png"
    ndsm_extension:  str             = ".tif"
    max_samples:     Optional[int]   = None
    hf_repo_id:      str             = "earthflow/GAMUS"
    streaming:       bool            = False
    crop_size:       Optional[int]   = None
    device:          str             = "cuda"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)

    @property
    def rgb_dir(self) -> Path:
        return self.data_dir / self.rgb_subdir

    @property
    def ndsm_dir(self) -> Path:
        return self.data_dir / self.ndsm_subdir

    def validate(self) -> None:
        """Raise FileNotFoundError if configured directories do not exist."""
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"GAMUS data_dir not found: {self.data_dir}\n"
                f"Download a subset from https://huggingface.co/datasets/{self.hf_repo_id}\n"
                f"or set GAMUSConfig(data_dir='path/to/your/gamus/subset')."
            )
