"""Terrain classification stage for the GeoMonoDSM-3D pipeline."""

from .segmentation import (
    TerrainClass,
    apply_brightness_fallback,
    apply_greenness_fallback,
    apply_warmth_fallback,
    classify_terrain,
    save_segmentation_outputs,
    segment,
    visualize_confidence_heatmap,
)

__all__ = [
    "TerrainClass",
    "apply_brightness_fallback",
    "apply_greenness_fallback",
    "apply_warmth_fallback",
    "classify_terrain",
    "segment",
    "save_segmentation_outputs",
    "visualize_confidence_heatmap",
]

