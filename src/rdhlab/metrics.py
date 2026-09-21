from __future__ import annotations
import math
import numpy as np
from skimage.metrics import structural_similarity


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d*d))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a,b)
    return float("inf") if m == 0 else 10.0 * math.log10((255.0**2)/m)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, data_range=255))


def ber(expected: np.ndarray, actual: np.ndarray) -> float:
    e=np.asarray(expected,dtype=np.uint8).ravel(); a=np.asarray(actual,dtype=np.uint8).ravel()
    if len(e)!=len(a): raise ValueError("length mismatch")
    return 0.0 if len(e)==0 else float(np.mean(e != a))


def exact_image_recovery(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a,b))
