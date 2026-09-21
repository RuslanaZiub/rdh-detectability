import numpy as np
import pandas as pd
from rdhlab.statistics import bootstrap_ci, predictability_detectability_stats, cluster_bootstrap


def test_bootstrap_ci_contains_center():
    x=np.arange(20,dtype=float)
    r=bootstrap_ci(x,'median',n_resamples=300,seed=1)
    assert r['low'] <= r['estimate'] <= r['high']


def test_predictability_stats_and_cluster_bootstrap():
    df=pd.DataFrame({'image':['a']*4+['b']*4,'predictability':[1,2,3,4,2,3,4,5], 'detectability_risk':[4,3,2,1,4,3,2,1]})
    r=predictability_detectability_stats(df)
    assert r['spearman_rho'] < 0
    ci=cluster_bootstrap(df,'image',lambda z: float(z['predictability'].mean()),n_resamples=50,seed=2)
    assert np.isfinite(ci['estimate'])
