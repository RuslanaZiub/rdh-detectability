from __future__ import annotations
import hashlib
import numpy as np

from .blocks import iter_blocks
from .features import predictability_score


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    lo = float(np.nanmin(x)); hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def joint_score(predictability: np.ndarray, detectability_risk: np.ndarray,
                w_predictability: float = 0.5, w_detectability: float = 0.5) -> np.ndarray:
    p = minmax(predictability)
    d = minmax(detectability_risk)
    return w_predictability * p - w_detectability * d


def rank_blocks(predictability: np.ndarray, detectability_risk: np.ndarray,
                w_predictability: float = 0.5, w_detectability: float = 0.5) -> np.ndarray:
    s = joint_score(predictability, detectability_risk, w_predictability, w_detectability)
    return np.argsort(-s, kind="stable")


def allocation_orders(image: np.ndarray, block_size: int, risk_model, image_key: str, alpha: float = 0.5, seed: int = 20260917) -> tuple[dict[str, np.ndarray], list[dict]]:
    rows = risk_model.score_image_blocks(image, image_key=image_key)
    bids = np.asarray([r["block_id"] for r in rows], dtype=int)
    p = np.asarray([r["predictability"] for r in rows], dtype=float)
    d = np.asarray([r["detectability_risk"] for r in rows], dtype=float)
    # Stable deterministic random baseline keyed to source id.
    digest = hashlib.sha256(f"{seed}|{image_key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    random_order = bids.copy(); rng.shuffle(random_order)
    orders = {
        "raster": bids.copy(),
        "random": random_order,
        "predictability": bids[np.argsort(-p, kind="stable")],
        "detectability": bids[np.argsort(d, kind="stable")],
        "joint": bids[rank_blocks(p, d, alpha, 1.0-alpha)],
    }
    return orders, rows
