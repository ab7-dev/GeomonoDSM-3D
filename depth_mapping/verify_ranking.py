"""Compare manually expected rooftop-height ordering with a saved depth map.

Edit ``regions`` below with 3--5 rooftop boxes from the source image, then run:
    python depth_mapping/verify_ranking.py

Bounding boxes use image pixels as ``(x1, y1, x2, y2)``.  The first corner is
inclusive and the second is exclusive, matching normal NumPy slicing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Fill this list after inspecting your source image.  Rank 1 means the rooftop
# you expect to be tallest/closest according to your own visual judgement.
regions = [
    # {"name": "Building A", "bbox": (x1, y1, x2, y2), "expected_rank": 1},
    # {"name": "Building B", "bbox": (x1, y1, x2, y2), "expected_rank": 2},
    # {"name": "Building C", "bbox": (x1, y1, x2, y2), "expected_rank": 3},
]


def _latest_depth_file(outputs_dir: Path) -> Path:
    """Find the newest raw depth array anywhere in the existing outputs tree."""
    candidates = list(outputs_dir.rglob("*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No .npy depth output found under {outputs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _descending_ranks(values: list[float]) -> list[float]:
    """Return 1-based ranks, averaging ranks when equal depth means tie."""
    ranks = [0.0] * len(values)
    ordered_indices = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    start = 0
    while start < len(ordered_indices):
        end = start + 1
        while end < len(ordered_indices) and values[ordered_indices[end]] == values[ordered_indices[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for ordered_index in ordered_indices[start:end]:
            ranks[ordered_index] = average_rank
        start = end
    return ranks


def _spearman(expected: list[float], computed: list[float]) -> float:
    """Calculate Spearman correlation using only NumPy, including tied ranks."""
    if len(expected) < 2:
        return float("nan")
    expected_ranks = _descending_ranks(expected)
    computed_ranks = _descending_ranks(computed)
    if np.std(expected_ranks) == 0 or np.std(computed_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(expected_ranks, computed_ranks)[0, 1])


def main() -> None:
    """Load one existing depth map and print the manual-vs-computed ranking."""
    if not regions:
        print("Add 3--5 rooftop regions to the 'regions' list at the top of this script, then run again.")
        return

    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    depth_path = _latest_depth_file(outputs_dir)
    depth = np.load(depth_path, mmap_mode="r")  # Read-only memory map; never changes the saved output.
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth array, found shape {depth.shape} in {depth_path}")

    height, width = depth.shape
    mean_depths: list[float] = []
    expected_ranks: list[float] = []
    for region in regions:
        name = region["name"]
        x1, y1, x2, y2 = region["bbox"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"{name}: bbox {(x1, y1, x2, y2)} is outside depth-map bounds {(width, height)}")
        mean_depths.append(float(np.mean(depth[y1:y2, x1:x2])))
        expected_ranks.append(float(region["expected_rank"]))

    computed_ranks = _descending_ranks(mean_depths)
    matches = [expected == computed for expected, computed in zip(expected_ranks, computed_ranks)]

    print(f"Depth file: {depth_path}")
    print(f"{'Region':<24} {'Expected':>9} {'Computed':>9} {'Match':>7}")
    print("-" * 55)
    for region, expected, computed, matched in zip(regions, expected_ranks, computed_ranks, matches):
        computed_text = f"{computed:g}"
        print(f"{region['name']:<24} {expected:>9g} {computed_text:>9} {'yes' if matched else 'no':>7}")

    agreement = sum(matches) / len(regions)
    correlation = _spearman(expected_ranks, computed_ranks)
    print(f"\nAgreement score: {sum(matches)}/{len(regions)} = {agreement:.1%}")
    print(f"Spearman rank correlation: {correlation:.3f}" if not np.isnan(correlation) else "Spearman rank correlation: undefined (need at least two non-tied ranks)")


if __name__ == "__main__":
    main()
