import numpy as np

from rdhlab.srnet_secondary import build_srnet, minimal_detection_error, parameter_count


def test_srnet_forward_shape_and_size():
    import torch
    model = build_srnet()
    x = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    y = model(x)
    assert tuple(y.shape) == (1, 2)
    assert parameter_count(model) > 4_000_000


def test_minimal_detection_error_bounds():
    cover = np.array([0.1, 0.2, 0.3, 0.4])
    stego = np.array([0.6, 0.7, 0.8, 0.9])
    pe = minimal_detection_error(cover, stego)
    assert 0.0 <= pe <= 0.5
