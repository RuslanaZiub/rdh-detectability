from __future__ import annotations
from pathlib import Path
from time import perf_counter_ns
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from .codec import embed, extract, sideinfo_bits, location_map_storage_bits
from .io import read_gray, write_gray_png
from .metrics import psnr, ssim, ber
from .features import residual_hist_features


def reversibility_trial(image: np.ndarray, payload_bits: int, rng: np.random.Generator,
                        temp_png: str | Path | None = None) -> dict:
    bits=rng.integers(0,2,size=payload_bits,dtype=np.uint8)
    t0=perf_counter_ns()
    stego,meta=embed(image,bits)
    encode_ms=(perf_counter_ns()-t0)/1e6
    target=stego
    io_ms=0.0
    if temp_png is not None:
        t_io=perf_counter_ns()
        write_gray_png(temp_png,stego)
        target=read_gray(temp_png)
        io_ms=(perf_counter_ns()-t_io)/1e6
    t1=perf_counter_ns()
    recovered,msg=extract(target,meta)
    decode_ms=(perf_counter_ns()-t1)/1e6
    return {
        'payload_bits':payload_bits,
        'encode_ms':float(encode_ms),'decode_ms':float(decode_ms),'file_io_ms':float(io_ms),
        'sideinfo_bits':sideinfo_bits(meta),
        'net_payload_bits':payload_bits-sideinfo_bits(meta),
        'ber':ber(bits,msg),
        'exact_image':bool(np.array_equal(image,recovered)),
        'exact_message':bool(np.array_equal(bits,msg)),
        'psnr':psnr(image,stego),
        'ssim':ssim(image,stego),
        'peak':meta.peak,'zero':meta.zero,'direction':meta.direction,
        'location_map_count':meta.location_map_count,
        'location_map_storage_bits':location_map_storage_bits(meta),
    }


def steganalysis_baseline(covers: list[np.ndarray], stegos: list[np.ndarray], seed: int=20260916) -> dict:
    X=[]; y=[]
    for a,b in zip(covers,stegos):
        X.append(residual_hist_features(a)); y.append(0)
        X.append(residual_hist_features(b)); y.append(1)
    X=np.stack(X); y=np.asarray(y)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=seed,stratify=y)
    clf=LogisticRegression(max_iter=3000,C=1.0,solver='liblinear',random_state=seed)
    clf.fit(Xtr,ytr)
    score=clf.predict_proba(Xte)[:,1]
    return {'auc':float(roc_auc_score(yte,score)), 'n_train':int(len(ytr)), 'n_test':int(len(yte)), 'model':clf}
