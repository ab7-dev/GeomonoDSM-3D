# GeomonoDSM-3D

Converts a 2D satellite or aerial image into a relative-depth map that can be
used as an input to a 3D reconstruction workflow.

## Progress

The repository now includes `depth_mapping/`, a CPU-oriented relative-depth
stage powered by Depth Anything V2 Large. It supports JPG, PNG, and GeoTIFF
input; clips exposure extremes; uses all available PyTorch CPU threads; and
creates a raw NumPy depth array plus a viridis heatmap.

For large images, the default path uses adaptive, near-518px tile cores. Cores
divide each image axis evenly, receive 128px of real-image overlap context, and
are linearly blended to avoid partial edge tiles and visual seams. Output depth
is clipped to its 1st--99th percentile to reduce isolated artifacts.

## Setup

```bash
pip install -r depth_mapping/requirements.txt
```

## Run depth estimation

Full-quality CPU inference (Depth Anything V2 Large with adaptive tiling):

```bash
python depth_mapping/depth.py --image path/to/image.jpg
```

Quick iteration with the Small checkpoint and no tiles:

```bash
python depth_mapping/depth.py --image path/to/image.jpg --fast
```

Optionally use cached ONNX Runtime CPU inference after installing the listed
ONNX dependencies:

```bash
python depth_mapping/depth.py --image path/to/image.jpg --use-onnx
```

The command saves `<image>.npy` and `<image>_depth.png` in `outputs/` and logs
the inference time.

## Local uploader

```bash
python -m depth_mapping.web_app
```

Open `http://127.0.0.1:8000` and select a JPG, PNG, or GeoTIFF. The resulting
raw array and heatmap are saved to `outputs/`.

## Validation

Run the included offline interface test:

```bash
python -m pytest depth_mapping/test_depth.py -s
```

`depth_mapping/verify_ranking.py` is a standalone manual ranking verifier.
Add rooftop bounding boxes to its `regions` list, then run:

```bash
python depth_mapping/verify_ranking.py
```

It reads the latest saved `.npy` depth map and reports expected versus computed
relative-height rankings, agreement, and Spearman correlation.

## Terrain classification

The `Terrain_Classificartion/` folder adds a pretrained SegFormer terrain
classification stage for JPG, PNG, and GeoTIFF imagery. It creates terrain
labels and a colour-coded overlay for vegetation, buildings, roads, water, and
other areas.

Install the required packages:

```bash
pip install numpy Pillow rasterio torch transformers scipy opencv-python jsonschema
```

Run terrain classification:

```bash
python Terrain_Classificartion/segmentation.py --image path/to/image.tif --output-dir outputs
```

The stage writes a label map, overlay image, and JSON output. GeoTIFF inputs
also produce a terrain-label GeoTIFF that retains the source georeferencing.
