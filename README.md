# GeomonoDSM-3D

Converts a 2D satellite or aerial image into a relative-depth map that can be
used as an input to a 3D reconstruction workflow.

## Pipeline stages

- `depth_mapping/` estimates relative depth with Depth Anything V2.
- `Terrain_Classificartion/` classifies the source image into terrain classes
  for use by later reconstruction stages.

## Depth mapping

The `depth_mapping/` stage is CPU-oriented and powered by Depth Anything V2
Large. It supports JPG, PNG, and GeoTIFF input; clips exposure extremes; uses
all available PyTorch CPU threads; and creates a raw NumPy depth array plus a
viridis heatmap.

For large images, the default path uses adaptive, near-518px tile cores. Cores
divide each image axis evenly, receive 128px of real-image overlap context, and
are linearly blended to avoid partial edge tiles and visual seams. Output depth
is clipped to its 1st--99th percentile to reduce isolated artifacts.

### Setup

```bash
pip install -r depth_mapping/requirements.txt
```

### Run depth estimation

```bash
python depth_mapping/depth.py --image path/to/image.jpg
```

For faster iteration, use the Small checkpoint and disable tiling:

```bash
python depth_mapping/depth.py --image path/to/image.jpg --fast
```

The command saves `<image>.npy` and `<image>_depth.png` in `outputs/`.

### Local uploader

```bash
python -m depth_mapping.web_app
```

Open `http://127.0.0.1:8000` and select a JPG, PNG, or GeoTIFF. The resulting
raw array and heatmap are saved to `outputs/`.

## Terrain classification

`Terrain_Classificartion/` contains a pretrained SegFormer-based
remote-sensing terrain classifier. It labels each pixel as one of:

| Value | Class |
| ---: | --- |
| 0 | Other |
| 1 | Vegetation |
| 2 | Building |
| 3 | Road |
| 4 | Water |

Install its dependencies:

```bash
pip install "numpy>=1.26" "Pillow>=10.0" "rasterio>=1.3" "torch>=2.2" "transformers>=4.40" "scipy>=1.11" "opencv-python>=4.8" "jsonschema>=4.20"
```

Run classification on a JPG, PNG, or GeoTIFF:

```bash
python Terrain_Classificartion/segmentation.py --image path/to/image.tif --output-dir outputs
```

The command produces a terrain label map, a colour-coded overlay, and a JSON
summary. GeoTIFF input also preserves the source CRS and transform in a
`*_terrain_labels.tif` output.

The default model path is `models/loveda-segformer`; provide `--model` with
a compatible local model directory or Hugging Face checkpoint when it is stored
elsewhere. Road labels are useful semantic context but should not be treated as
precise road mapping for imagery whose resolution or geography differs from the
model's training data.

## Validation

Run the included offline depth-mapping interface test:

```bash
python -m pytest depth_mapping/test_depth.py -s
```
