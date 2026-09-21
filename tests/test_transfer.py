import numpy as np
from rdhlab.transfer import bootstrap_location_ci, paired_method_bootstrap


def test_bootstrap_location_ci_is_reproducible():
    x=np.array([0.1,0.2,0.3,0.4])
    a=bootstrap_location_ci(x,n_resamples=200,seed=7)
    b=bootstrap_location_ci(x,n_resamples=200,seed=7)
    assert a==b
    assert a['low'] <= a['estimate'] <= a['high']


def test_paired_method_bootstrap_detects_lower_scores():
    c=np.linspace(0.1,0.4,40)
    ref=c+0.30
    method=c+0.10
    r=paired_method_bootstrap(c,method,ref,n_resamples=300,seed=11)
    assert r['delta_mean_diff'] < 0
    assert r['delta_mean_diff_high'] < 0
    assert r['auc_diff'] <= 0
