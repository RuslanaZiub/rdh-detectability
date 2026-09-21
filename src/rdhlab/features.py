from __future__ import annotations
import numpy as np
from scipy.ndimage import sobel


def med_predict(image: np.ndarray) -> np.ndarray:
    x=image.astype(np.int16)
    pred=np.zeros_like(x)
    a=x[:, :-1]
    b=x[:-1, :]
    c=x[:-1, :-1]
    # interior: JPEG-LS MED predictor
    aa=x[1:, :-1]; bb=x[:-1, 1:]; cc=x[:-1, :-1]
    mx=np.maximum(aa,bb); mn=np.minimum(aa,bb)
    p=np.where(cc>=mx, mn, np.where(cc<=mn, mx, aa+bb-cc))
    pred[1:,1:]=p
    pred[0,:]=x[0,:]
    pred[:,0]=x[:,0]
    return pred


def predictability_score(block: np.ndarray) -> float:
    p=med_predict(block)
    err=np.abs(block.astype(np.int16)-p)
    return float(1.0/(1.0+np.mean(err)))


def block_features(block: np.ndarray) -> np.ndarray:
    x=block.astype(np.float64)
    gx=sobel(x, axis=1, mode='reflect'); gy=sobel(x, axis=0, mode='reflect')
    pred=med_predict(block).astype(np.float64)
    r=x-pred
    hist=np.bincount(block.ravel(), minlength=256).astype(np.float64)
    prob=hist/hist.sum(); nz=prob[prob>0]
    entropy=float(-(nz*np.log2(nz)).sum())
    return np.array([
        x.mean(), x.std(), entropy,
        np.mean(np.abs(gx))+np.mean(np.abs(gy)),
        np.mean(np.abs(r)), np.std(r),
        np.mean(np.abs(np.diff(x, axis=0))),
        np.mean(np.abs(np.diff(x, axis=1))),
    ], dtype=np.float64)


def residual_hist_features(image: np.ndarray, trunc: int = 4) -> np.ndarray:
    """Compact SRM-like residual co-occurrence proxy for a baseline detector."""
    x=image.astype(np.int16)
    feats=[]
    for d in [np.diff(x,axis=0), np.diff(x,axis=1)]:
        q=np.clip(d, -trunc, trunc)+trunc
        n=2*trunc+1
        # first-order histogram
        h=np.bincount(q.ravel(), minlength=n).astype(float); h/=max(h.sum(),1)
        feats.extend(h.tolist())
        # adjacent co-occurrence along flattened residual stream
        v=q.ravel()
        co=np.zeros((n,n),dtype=float)
        if len(v)>1:
            np.add.at(co,(v[:-1],v[1:]),1)
            co/=co.sum()
        feats.extend(co.ravel().tolist())
    return np.asarray(feats,dtype=np.float64)
