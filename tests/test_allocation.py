import numpy as np
from rdhlab.allocation import rank_blocks, joint_score

def test_rank():
    p=np.array([.9,.5,.1]); d=np.array([.1,.4,.8])
    r=rank_blocks(p,d)
    assert r[0]==0
    assert len(set(r.tolist()))==3
    assert joint_score(p,d).shape==(3,)
