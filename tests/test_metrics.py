import numpy as np
from rdhlab.metrics import mse, psnr, ber, exact_image_recovery

def test_metrics_identity():
    a=np.arange(256,dtype=np.uint8).reshape(16,16)
    assert mse(a,a)==0
    assert psnr(a,a)==float('inf')
    assert ber(np.array([0,1]),np.array([0,1]))==0
    assert exact_image_recovery(a,a.copy())
