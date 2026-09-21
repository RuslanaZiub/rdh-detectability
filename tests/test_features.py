import numpy as np
from rdhlab.features import predictability_score, block_features, residual_hist_features

def test_features_finite():
    x=np.tile(np.arange(64,dtype=np.uint8),(64,1))
    assert np.isfinite(predictability_score(x))
    assert np.isfinite(block_features(x)).all()
    f=residual_hist_features(x)
    assert f.ndim==1 and np.isfinite(f).all() and len(f)>10
