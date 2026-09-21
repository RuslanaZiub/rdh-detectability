import numpy as np
import pytest

from rdhlab.detector_repair import _highpass_bank, _dihedral_pair, build_enhanced_residual_cnn


def test_highpass_bank_is_zero_sum():
    k = _highpass_bank()
    assert k.shape == (8, 1, 5, 5)
    np.testing.assert_allclose(k.sum(axis=(2, 3)), 0.0)


def test_pair_augmentation_preserves_difference_alignment():
    c = np.arange(64, dtype=np.uint8).reshape(8, 8)
    s = c.copy(); s[2, 3] += 1
    for code in range(8):
        cc, ss = _dihedral_pair(c, s, code)
        assert cc.shape == c.shape
        assert ss.shape == s.shape
        assert np.count_nonzero(cc != ss) == 1


def test_enhanced_model_forward_shape():
    torch = pytest.importorskip("torch")
    model = build_enhanced_residual_cnn()
    x = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
    y = model(x)
    assert tuple(y.shape) == (2,)
