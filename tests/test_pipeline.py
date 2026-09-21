import numpy as np
from rdhlab.pipeline import choose_payload_levels


def test_payload_candidates_kept_if_feasible():
    x=np.full(100,0.4)
    r=choose_payload_levels(x,[0.05,0.1,0.2,0.3],0.95)
    assert r['levels']==[0.05,0.1,0.2,0.3]


def test_payload_fallback_uses_validation_quantile():
    x=np.linspace(0.02,0.08,100)
    r=choose_payload_levels(x,[0.05,0.1,0.2,0.3],0.95)
    assert r['method'].startswith('validation_q')
    assert len(r['levels'])>=3
    assert max(r['levels']) <= np.quantile(x,0.05)+1e-9


def test_precomputed_context_roundtrip():
    import numpy as np
    from rdhlab.risk import fit_block_risk_model
    from rdhlab.pipeline import prepare_image_context, run_frozen_image_precomputed
    rng=np.random.default_rng(123)
    imgs=[]
    for i in range(4):
        yy,xx=np.mgrid[:128,:128]
        arr=100+25*np.sin((xx+i)/13)+18*np.cos((yy+i)/17)+rng.normal(0,3,(128,128))
        imgs.append(np.clip(arr,0,255).astype(np.uint8))
    risk=fit_block_risk_model(imgs,block_size=64,max_blocks=8,seed=4)
    x=imgs[0]
    orders,rows,plans=prepare_image_context(x,'a',risk,.5,64,5)
    # Use a very small fixed net payload so the synthetic example is feasible.
    # If even this is not feasible, the test should explain the accounting issue.
    from rdhlab.blockcodec import max_accounted_net_capacity
    maxnet=max_accounted_net_capacity(x,64)['max_net_payload_bits']
    assert maxnet>0
    bpp=min(0.001,maxnet/x.size)
    r=run_frozen_image_precomputed(x,'a',bpp,'joint',orders,rows,64,5,plans=plans)
    assert r['feasible']
    assert r['exact_image'] and r['exact_message']
