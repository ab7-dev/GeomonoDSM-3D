"""Fast, offline test for the public depth-estimation interface."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from depth_mapping import depth


class _TestProcessor:
    """Minimal processor substitute: avoids downloading a model during unit tests."""

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((1, 3, 4, 5), dtype=torch.float32)}


class _TestModel:
    """Small deterministic depth model exposing the Transformers output shape."""

    def parameters(self):
        yield torch.nn.Parameter(torch.empty(0))

    def __call__(self, **inputs):
        return type("DepthOutput", (), {"predicted_depth": torch.arange(20, dtype=torch.float32).reshape(1, 4, 5)})()


def test_get_depth_returns_2d_numpy_array(monkeypatch):
    """Run get_depth on the bundled image and validate its user-facing result."""
    monkeypatch.setattr(depth, "_load_model", lambda: (_TestProcessor(), _TestModel()))
    image_path = Path(__file__).parent / "sample_data" / "sample_scene.png"

    result = depth.get_depth(str(image_path))

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    print(f"depth shape={result.shape}, min={result.min()}, max={result.max()}")
