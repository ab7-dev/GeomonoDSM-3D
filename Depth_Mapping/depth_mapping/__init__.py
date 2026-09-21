"""Utilities for estimating relative depth from geospatial imagery.

Primary public API
------------------
get_depth(image_path, fast)
    Return a 2-D float32 relative-depth array.  GeoTIFF metadata is not
    returned; use get_depth_with_meta() when geospatial context is needed.

get_depth_with_meta(image_path, fast)
    Return (depth_array, geo_meta).  geo_meta is a dict for GeoTIFF inputs,
    None for JPG/PNG.

save_depth_outputs(depth_array, output_dir, filename, geo_meta=None,
                   rgb_image=None, save_visualization=True)
    Write .npy, viridis .png, optional co-registered .tif, and a
    presentation-quality multi-panel visualization figure.
"""

from .depth import get_depth, get_depth_with_meta, save_depth_outputs

__all__ = ["get_depth", "get_depth_with_meta", "save_depth_outputs"]
