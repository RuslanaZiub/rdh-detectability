import numpy as np
from rdhlab.posthoc_revision_v15 import alpha_order, gradient_complexity_order


def test_alpha_endpoints_and_midpoint():
    rows=[
        {"block_id":0,"predictability":0.1,"detectability_risk":0.9},
        {"block_id":1,"predictability":0.8,"detectability_risk":0.2},
        {"block_id":2,"predictability":0.5,"detectability_risk":0.5},
    ]
    assert alpha_order(rows,1.0).tolist()==[1,2,0]
    assert alpha_order(rows,0.0).tolist()==[1,2,0]
    assert set(alpha_order(rows,0.5).tolist())=={0,1,2}


def test_gradient_order_shape():
    x=np.zeros((128,128),dtype=np.uint8)
    x[:64,:64]=np.tile(np.arange(64,dtype=np.uint8),(64,1))
    order,rows=gradient_complexity_order(x,64)
    assert len(order)==4 and len(rows)==4
    assert set(order.tolist())=={0,1,2,3}
