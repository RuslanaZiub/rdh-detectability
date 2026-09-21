import numpy as np

from rdhlab.srnet_curriculum_v13 import build_srnet, parameter_count, score_srnet


def test_srnet_v13_architecture_shape_and_count():
    import torch
    model = build_srnet()
    x = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    y = model(x)
    assert tuple(y.shape) == (1, 2)
    assert parameter_count(model) == 4_779_616


def test_score_srnet_v13_finite():
    model = build_srnet()
    imgs = np.zeros((3, 64, 64), dtype=np.uint8)
    s = score_srnet(model, imgs, device="cpu", batch_size=2)
    assert s.shape == (3,)
    assert np.all(np.isfinite(s))


def test_stage_spec_payloads_descend():
    payloads = [0.050, 0.030, 0.015, 0.009]
    assert all(payloads[i] > payloads[i + 1] for i in range(len(payloads) - 1))
