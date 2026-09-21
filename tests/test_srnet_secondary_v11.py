import tempfile
from pathlib import Path

import numpy as np

from rdhlab.srnet_secondary_v11 import (
    build_srnet,
    minimal_detection_error,
    parameter_count,
    score_srnet,
)


def test_srnet_forward_shape_and_parameter_count():
    import torch
    model = build_srnet()
    x = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    y = model(x)
    assert tuple(y.shape) == (1, 2)
    assert parameter_count(model) == 4_779_616


def test_score_srnet_shape():
    model = build_srnet()
    imgs = np.zeros((3, 64, 64), dtype=np.uint8)
    s = score_srnet(model, imgs, device="cpu", batch_size=2)
    assert s.shape == (3,)
    assert np.all(np.isfinite(s))


def test_minimal_detection_error_bounds():
    cover = np.array([0.1, 0.2, 0.3, 0.4])
    stego = np.array([0.6, 0.7, 0.8, 0.9])
    pe = minimal_detection_error(cover, stego)
    assert 0.0 <= pe <= 0.5
