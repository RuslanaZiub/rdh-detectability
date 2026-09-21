from __future__ import annotations
import hashlib
import numpy as np
from scipy.ndimage import sobel
from .allocation import rank_blocks
from .blocks import iter_blocks


def alpha_order(block_rows: list[dict], alpha: float) -> np.ndarray:
    bids=np.asarray([r["block_id"] for r in block_rows],dtype=int)
    p=np.asarray([r["predictability"] for r in block_rows],dtype=float)
    d=np.asarray([r["detectability_risk"] for r in block_rows],dtype=float)
    return bids[rank_blocks(p,d,float(alpha),1.0-float(alpha))]


def gradient_complexity_order(image: np.ndarray, block_size: int=64) -> tuple[np.ndarray,list[dict]]:
    rows=[]
    for bid,y,x,b in iter_blocks(np.asarray(image,dtype=np.uint8),int(block_size)):
        z=b.astype(np.float64)
        gx=sobel(z,axis=1,mode="reflect")
        gy=sobel(z,axis=0,mode="reflect")
        c=float(np.mean(np.abs(gx))+np.mean(np.abs(gy)))
        rows.append({"block_id":int(bid),"complexity_gradient":c})
    bids=np.asarray([r["block_id"] for r in rows],dtype=int)
    c=np.asarray([r["complexity_gradient"] for r in rows],dtype=float)
    return bids[np.argsort(-c,kind="stable")],rows


def stable_id_hash(ids) -> str:
    h=hashlib.sha256()
    for sid in map(str,ids):
        h.update(sid.encode('utf-8')); h.update(b'\n')
    return h.hexdigest()
