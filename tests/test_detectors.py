import numpy as np
from rdhlab.detectors import srm_lite_features, fit_rich_detector, detector_metrics, build_residual_cnn


def test_srm_lite_feature_shape_stable():
    x=np.zeros((64,64),dtype=np.uint8)
    y=np.ones((64,64),dtype=np.uint8)
    assert srm_lite_features(x).shape == srm_lite_features(y).shape
    assert srm_lite_features(x).ndim == 1


def test_rich_detector_smoke():
    covers=[]; stegos=[]
    rng=np.random.default_rng(3)
    for _ in range(8):
        x=rng.integers(20,230,size=(64,64),dtype=np.uint8)
        y=x.copy(); y[:,::4]=np.clip(y[:,::4].astype(int)+1,0,255).astype(np.uint8)
        covers.append(x); stegos.append(y)
    det=fit_rich_detector(covers,stegos,seed=1)
    scores=np.array([det.score(z) for z in covers+stegos])
    m=detector_metrics(np.array([0]*8+[1]*8),scores)
    assert 0 <= m['auc'] <= 1


def test_cnn_builds():
    model=build_residual_cnn()
    assert model is not None


def test_paired_detector_bootstrap():
    from rdhlab.detectors import paired_detector_bootstrap
    c=np.linspace(.1,.4,20); s=np.linspace(.6,.9,20)
    r=paired_detector_bootstrap(c,s,n_resamples=50,seed=3)
    assert r['auc_low'] <= 1.0 <= r['auc_high']
    assert r['n_pairs']==20
