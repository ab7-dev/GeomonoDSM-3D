"""Presentation-quality visualization for Part 1 — Relative Depth Estimation.

Public API
----------
make_depth_figure(depth_array, rgb_image=None, output_path=None, title=None)
    Generate a multi-panel figure with heatmap, greyscale, and optional
    side-by-side RGB ↔ depth comparison.  Returns a matplotlib Figure.

save_depth_figure(depth_array, output_dir, stem, rgb_image=None, title=None)
    Convenience wrapper: calls make_depth_figure and saves to
    ``<output_dir>/<stem>_vis.png``.

IMPORTANT
---------
Visualization only — the depth_array is never modified.
All values shown are relative depth (dimensionless model output).
This is NOT metric elevation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Colourmap used for the primary depth heatmap.
_DEPTH_CMAP = "viridis"


def make_depth_figure(
    depth_array: np.ndarray,
    rgb_image: Any | None = None,
    title: str | None = None,
    input_path: str | None = None,
) -> Any:
    """Build a presentation-quality multi-panel depth figure.

    Parameters
    ----------
    depth_array:
        2-D float32 relative-depth array (H × W).
    rgb_image:
        Optional PIL Image or (H, W, 3) uint8 array.  When provided, a
        side-by-side RGB ↔ depth panel is added.
    title:
        Figure suptitle.  Defaults to a descriptive auto-title.
    input_path:
        Source image path used in the subtitle when no explicit title given.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from PIL import Image as PILImage

    if depth_array.ndim != 2:
        raise ValueError(f"depth_array must be 2-D, got {depth_array.shape}")

    arr = depth_array.astype(np.float32)
    H, W = arr.shape
    dmin, dmax = float(arr.min()), float(arr.max())
    dmean, dstd = float(arr.mean()), float(arr.std())

    has_rgb = rgb_image is not None
    if has_rgb:
        if isinstance(rgb_image, PILImage.Image):
            rgb_arr = np.asarray(rgb_image.convert("RGB"))
        else:
            rgb_arr = np.asarray(rgb_image)
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            has_rgb = False
            logger.warning("rgb_image is not H×W×3 — side-by-side panel skipped")

    # ── Layout ────────────────────────────────────────────────────────────────
    # Row 0: large heatmap | greyscale
    # Row 1 (optional): RGB input | depth heatmap (side-by-side comparison)
    n_rows = 2 if has_rgb else 1
    fig = plt.figure(figsize=(16, 8 * n_rows), dpi=120)

    if title is None:
        src = Path(input_path).name if input_path else "input"
        title = (
            f"Part 1 — Georeferenced Relative Depth Map\n"
            f"{src}   {W}×{H} px   "
            f"range [{dmin:.2f}, {dmax:.2f}]   mean {dmean:.2f}   std {dstd:.2f}"
        )

    fig.suptitle(title, fontsize=11, y=1.01)

    # ── Row 0: heatmap + greyscale ─────────────────────────────────────────
    gs0 = gridspec.GridSpec(
        1, 2, figure=fig,
        top=0.97 if not has_rgb else 0.98,
        bottom=0.55 if has_rgb else 0.04,
        left=0.04, right=0.96, wspace=0.12,
    )

    ax_heat = fig.add_subplot(gs0[0, 0])
    im_heat = ax_heat.imshow(arr, cmap=_DEPTH_CMAP, interpolation="nearest",
                             vmin=dmin, vmax=dmax)
    cbar = fig.colorbar(im_heat, ax=ax_heat, fraction=0.03, pad=0.01)
    cbar.set_label("Relative depth (model units)", fontsize=8)
    ax_heat.set_title("Relative Depth Heatmap (viridis)", fontsize=10, pad=6)
    ax_heat.set_xlabel("Column (px)", fontsize=8)
    ax_heat.set_ylabel("Row (px)", fontsize=8)
    ax_heat.tick_params(labelsize=7)

    # Annotation box
    stats_text = (
        f"min={dmin:.2f}  max={dmax:.2f}\n"
        f"mean={dmean:.2f}  std={dstd:.2f}\n"
        f"size={W}×{H}  dtype=float32\n"
        "Output: RELATIVE DEPTH (not metric elevation)"
    )
    ax_heat.text(
        0.01, 0.01, stats_text,
        transform=ax_heat.transAxes,
        fontsize=6.5, color="white",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.55),
    )

    ax_grey = fig.add_subplot(gs0[0, 1])
    ax_grey.imshow(arr, cmap="gray", interpolation="nearest",
                   vmin=dmin, vmax=dmax)
    ax_grey.set_title("Greyscale Depth (structural detail)", fontsize=10, pad=6)
    ax_grey.set_xlabel("Column (px)", fontsize=8)
    ax_grey.set_ylabel("Row (px)", fontsize=8)
    ax_grey.tick_params(labelsize=7)

    # ── Row 1 (optional): RGB ↔ Depth side-by-side ────────────────────────
    if has_rgb:
        gs1 = gridspec.GridSpec(
            1, 2, figure=fig,
            top=0.48, bottom=0.04,
            left=0.04, right=0.96, wspace=0.06,
        )

        ax_rgb = fig.add_subplot(gs1[0, 0])
        ax_rgb.imshow(rgb_arr, interpolation="nearest")
        ax_rgb.set_title("Input RGB Image", fontsize=10, pad=6)
        ax_rgb.set_xlabel("Column (px)", fontsize=8)
        ax_rgb.set_ylabel("Row (px)", fontsize=8)
        ax_rgb.tick_params(labelsize=7)
        ax_rgb.text(
            0.01, 0.99, "INPUT",
            transform=ax_rgb.transAxes,
            fontsize=9, color="white", fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )

        ax_side = fig.add_subplot(gs1[0, 1])
        im_side = ax_side.imshow(arr, cmap=_DEPTH_CMAP,
                                  interpolation="nearest",
                                  vmin=dmin, vmax=dmax)
        fig.colorbar(im_side, ax=ax_side, fraction=0.03, pad=0.01,
                     label="Relative depth")
        ax_side.set_title(
            "Single-View Relative Depth Estimation\n"
            "(Depth Anything V2 Large — NOT metric elevation)",
            fontsize=9, pad=6,
        )
        ax_side.set_xlabel("Column (px)", fontsize=8)
        ax_side.set_ylabel("Row (px)", fontsize=8)
        ax_side.tick_params(labelsize=7)
        ax_side.text(
            0.01, 0.99, "RELATIVE DEPTH OUTPUT",
            transform=ax_side.transAxes,
            fontsize=9, color="white", fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="darkblue", alpha=0.7),
        )

    return fig


def save_depth_figure(
    depth_array: np.ndarray,
    output_dir: str | Path,
    stem: str,
    rgb_image: Any | None = None,
    title: str | None = None,
    input_path: str | None = None,
    dpi: int = 120,
) -> Path:
    """Generate and save a presentation-quality depth visualization.

    Parameters
    ----------
    depth_array:
        2-D float32 relative-depth array.
    output_dir:
        Directory where the figure is saved (created if absent).
    stem:
        Output filename stem.  The file is saved as ``<stem>_vis.png``.
    rgb_image:
        Optional source RGB image for side-by-side comparison.
    title:
        Override figure suptitle.
    input_path:
        Used in auto-generated title when no explicit title provided.
    dpi:
        Output resolution (default 120).

    Returns
    -------
    Path
        Resolved path of the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_vis.png"

    fig = make_depth_figure(
        depth_array,
        rgb_image=rgb_image,
        title=title,
        input_path=input_path,
    )
    fig.savefig(str(out_path), bbox_inches="tight", dpi=dpi)

    import matplotlib.pyplot as plt
    plt.close(fig)

    logger.info("Saved depth visualization: %s", out_path.name)
    return out_path.resolve()


def load_depth_and_visualize(
    npy_path: str | Path,
    output_dir: str | Path | None = None,
    rgb_path: str | Path | None = None,
    title: str | None = None,
    dpi: int = 120,
) -> Path:
    """Load a saved .npy depth array and generate a visualization.

    Useful for inspecting existing test outputs without rerunning inference.

    Parameters
    ----------
    npy_path:
        Path to the ``.npy`` depth array.
    output_dir:
        Where to save the figure.  Defaults to the same directory as npy_path.
    rgb_path:
        Optional path to the source RGB image for side-by-side comparison.
    title:
        Override figure title.
    dpi:
        Output resolution.

    Returns
    -------
    Path
        Saved figure path.
    """
    npy_path = Path(npy_path)
    depth = np.load(npy_path)

    rgb_image = None
    if rgb_path is not None:
        try:
            from PIL import Image
            rgb_image = Image.open(rgb_path).convert("RGB")
        except Exception as exc:
            logger.warning("Could not load RGB image %s: %s", rgb_path, exc)

    if output_dir is None:
        output_dir = npy_path.parent

    stem = npy_path.stem  # e.g. "test_2_depth"
    return save_depth_figure(
        depth,
        output_dir=output_dir,
        stem=stem,
        rgb_image=rgb_image,
        title=title,
        input_path=str(npy_path),
        dpi=dpi,
    )
