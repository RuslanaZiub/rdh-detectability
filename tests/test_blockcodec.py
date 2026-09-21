import numpy as np

from rdhlab.blockcodec import (
    analyze_blocks, embed_blockwise_net, extract_blockwise, max_accounted_net_capacity,
)


def _image(seed=1, size=256):
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[:size,:size]
    arr=100 + 40*np.sin(xx/25.0) + 30*np.cos(yy/31.0) + rng.normal(0,4,(size,size))
    return np.clip(arr,0,255).astype(np.uint8)


def test_blockwise_roundtrip_accounted_net():
    x=_image()
    plans=analyze_blocks(x,64)
    eligible=[p.block_id for p in plans if p.net_gain_bits>0]
    cap=max_accounted_net_capacity(x,64)
    assert cap['max_net_payload_bits']>0
    target=min(256, cap['max_net_payload_bits'])
    rng=np.random.default_rng(7)
    y,bits,meta,stats=embed_blockwise_net(x,target,eligible,rng,64)
    rec,out,dstats=extract_blockwise(y,meta)
    assert np.array_equal(rec,x)
    assert np.array_equal(out,bits)
    assert meta.gross_payload_bits-meta.accounted_sideinfo_bits==target
    assert stats['accounted_net_payload_bits']==target
    assert dstats['decode_ms']>=0


def test_infeasible_target_rejected():
    x=_image(2)
    plans=analyze_blocks(x,64)
    order=[p.block_id for p in plans]
    cap=max_accounted_net_capacity(x,64)['max_net_payload_bits']
    rng=np.random.default_rng(1)
    try:
        embed_blockwise_net(x,cap+1,order,rng,64)
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError')
