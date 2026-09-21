from __future__ import annotations

import hashlib
import threading
import time
from time import perf_counter_ns
from typing import Callable

import numpy as np
import psutil

from .allocation import rank_blocks
from .blocks import iter_blocks
from .features import predictability_score


def deterministic_rng(seed: int, source_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{source_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def build_strategy_order(image: np.ndarray, source_id: str, strategy: str, risk_model,
                         alpha: float, block_size: int, seed: int):
    """Build only the information required by one allocation strategy.

    Returns (order, block_rows, elapsed_ms).  The implementation deliberately
    avoids invoking the SRM-derived local-risk teacher for raster/random/
    predictability, so strategy-level timing does not charge those baselines
    for a model they do not need.
    """
    image=np.asarray(image,dtype=np.uint8)
    t0=perf_counter_ns()

    if strategy in {"raster", "random", "predictability"}:
        raw=list(iter_blocks(image,block_size))
        bids=np.asarray([int(bid) for bid,_,_,_ in raw],dtype=int)
        if strategy == "raster":
            order=bids.copy()
            rows=[{"block_id":int(bid),"predictability":np.nan,"detectability_risk":np.nan}
                  for bid,_,_,_ in raw]
        elif strategy == "random":
            order=bids.copy(); deterministic_rng(seed,source_id).shuffle(order)
            rows=[{"block_id":int(bid),"predictability":np.nan,"detectability_risk":np.nan}
                  for bid,_,_,_ in raw]
        else:
            p=np.asarray([float(predictability_score(block)) for _,_,_,block in raw],dtype=float)
            order=bids[np.argsort(-p,kind="stable")]
            rows=[{"block_id":int(bid),"predictability":float(pi),"detectability_risk":np.nan}
                  for (bid,_,_,_),pi in zip(raw,p)]
    elif strategy in {"detectability", "joint"}:
        rows=risk_model.score_image_blocks(image,image_key=str(source_id))
        bids=np.asarray([r["block_id"] for r in rows],dtype=int)
        p=np.asarray([r["predictability"] for r in rows],dtype=float)
        d=np.asarray([r["detectability_risk"] for r in rows],dtype=float)
        if strategy == "detectability":
            order=bids[np.argsort(d,kind="stable")]
        else:
            order=bids[rank_blocks(p,d,float(alpha),1.0-float(alpha))]
    else:
        raise KeyError(strategy)

    ms=(perf_counter_ns()-t0)/1e6
    return np.asarray(order,dtype=int),rows,float(ms)


def sideinfo_ratios(gross_bits, sideinfo_bits, net_bits):
    gross=float(gross_bits); side=float(sideinfo_bits); net=float(net_bits)
    return {
        "sideinfo_fraction_gross": side/gross if gross>0 else np.nan,
        "sideinfo_per_net": side/net if net>0 else np.nan,
    }


def summarize_exact_recovery(df) -> dict:
    feasible=df[df["feasible"].astype(bool)].copy()
    io=feasible[feasible.get("file_io_checked",False).astype(bool)] if "file_io_checked" in feasible else feasible.iloc[0:0]
    return {
        "cases":int(len(df)),
        "feasible_cases":int(len(feasible)),
        "exact_image_feasible":int(feasible["exact_image"].eq(True).sum()),
        "exact_message_feasible":int(feasible["exact_message"].eq(True).sum()),
        "zero_ber_feasible":int((feasible["ber"].fillna(1).astype(float)==0.0).sum()),
        "file_io_feasible_cases":int(len(io)),
        "file_io_exact_image":int(io["exact_image"].eq(True).sum()) if len(io) else 0,
        "file_io_exact_message":int(io["exact_message"].eq(True).sum()) if len(io) else 0,
    }


def run_with_peak_rss(func: Callable, poll_s: float = 0.005):
    """Run callable while sampling process RSS. Intended for coarse diagnostics.

    The returned peak is an observed polling peak, not an allocator-level exact
    maximum. Timing benchmarks should be run separately because polling adds
    small overhead.
    """
    proc=psutil.Process()
    baseline=proc.memory_info().rss
    peak=[baseline]
    stop=threading.Event()
    def worker():
        while not stop.is_set():
            try:
                peak[0]=max(peak[0],proc.memory_info().rss)
            except Exception:
                pass
            stop.wait(poll_s)
    th=threading.Thread(target=worker,daemon=True)
    th.start()
    try:
        out=func()
    finally:
        stop.set(); th.join(timeout=max(0.1,poll_s*4))
        try: peak[0]=max(peak[0],proc.memory_info().rss)
        except Exception: pass
    return out, {
        "rss_baseline_mib":baseline/1024**2,
        "rss_observed_peak_mib":peak[0]/1024**2,
        "rss_observed_delta_mib":(peak[0]-baseline)/1024**2,
        "poll_interval_ms":poll_s*1000.0,
    }
