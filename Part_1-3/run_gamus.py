"""
run_gamus.py — Run GAMUS-aligned pretrained land-cover segmentation on any image.

Downloads nave1616/SegFormer-landcover-FT (~256 MB) on first run, cached thereafter.

Usage:
    python run_gamus.py --input Input\\bellingham1.tif --output-dir output\\bellingham1_gamus
    python run_gamus.py --input Depth_Mapping\\depth_mapping\\sample_data\\sample_scene.png --output-dir output\\sample_gamus
    python run_gamus.py --input path\\to\\image.jpg --output-dir output\\my_gamus --device cpu

Windows PowerShell:
    python run_gamus.py --input Input\bellingham1.tif --output-dir output\bellingham1_gamus
"""
from __future__ import annotations
import argparse, sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

sys.path.insert(0, str(Path(__file__).parent / "Depth_Mapping"))

from depth_mapping.gamus.inference import run_gamus_inference, GAMUS_MODEL_ID


def main():
    parser = argparse.ArgumentParser(
        description="GAMUS-aligned pretrained land-cover segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",      required=True, help="Input RGB image (PNG/JPG/TIFF/GeoTIFF)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device",     default=None,  help="cuda or cpu (auto-detected if omitted)")
    parser.add_argument("--tile-size",  type=int, default=512, help="Tile size for large images (default 512)")
    parser.add_argument("--max-size",   type=int, default=1024,
                        help="Max image dimension for inference; larger images are downsampled (default 1024)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  GAMUS Land-Cover Segmentation")
    print(f"  Model : {GAMUS_MODEL_ID}")
    print(f"  Input : {args.input}")
    print(f"{'='*60}")

    result = run_gamus_inference(
        image_path=args.input,
        output_dir=args.output_dir,
        device=args.device,
        tile_size=args.tile_size,
        max_size=args.max_size,
    )

    print(f"\n  Inference time : {result['inference_time_s']:.1f}s")
    print(f"  Device         : cuda" if "cuda" in str(result.get("model_id","")) else f"  Device         : {args.device or 'auto'}")
    print(f"\n  GAMUS class distribution:")
    for cls, pct in result["gamus_distribution"].items():
        bar = "#" * int(pct / 2)
        print(f"    {cls:<22} {pct:5.1f}%  {bar}")
    print(f"\n  TerrainClass distribution:")
    for cls, pct in result["terrain_distribution"].items():
        bar = "#" * int(pct / 2)
        print(f"    {cls:<12} {pct:5.1f}%  {bar}")
    print(f"\n  Outputs -> {Path(args.output_dir).resolve()}")
    for k, v in result.items():
        if k.endswith("_path") and hasattr(v, "name"):
            print(f"    {v.name}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
