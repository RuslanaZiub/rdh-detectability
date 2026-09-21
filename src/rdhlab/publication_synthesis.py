from __future__ import annotations
import math
import pandas as pd

REQUIRED_STRATEGIES = ["raster","random","predictability","detectability","joint"]
REQUIRED_PAYLOADS = [0.003,0.006,0.009,0.012]

def validate_grid(df: pd.DataFrame, strategy_col="strategy", payload_col="target_net_bpp"):
    got_s = set(df[strategy_col].astype(str))
    miss_s = set(REQUIRED_STRATEGIES) - got_s
    if miss_s:
        raise ValueError(f"Missing strategies: {sorted(miss_s)}")
    got_p = sorted({round(float(x), 6) for x in df[payload_col]})
    req_p = sorted(REQUIRED_PAYLOADS)
    if got_p != req_p:
        raise ValueError(f"Payload grid mismatch: got={got_p}, expected={req_p}")
    return True

def compact_metric_table(rdh, det, bpp=0.009):
    a = rdh[rdh.target_net_bpp.astype(float).round(6)==round(float(bpp),6)].copy()
    b = det[det.target_net_bpp.astype(float).round(6)==round(float(bpp),6)].copy()
    m = a.merge(b[["strategy","auc","tpr_at_5pct_fpr","score_delta_mean"]],
                on="strategy", how="inner", validate="one_to_one")
    cols = ["strategy","n","actual_net_bpp_mean","psnr_mean","ssim_mean",
            "auc","tpr_at_5pct_fpr","score_delta_mean"]
    return m[cols].sort_values("strategy").reset_index(drop=True)

def primary_text(primary: dict) -> str:
    return (
        f"At {float(primary['primary_payload_bpp']):.3f} net bpp, the frozen joint allocator "
        f"(alpha={float(primary['alpha']):.2f}) reduced the paired independent-CNN score change "
        f"relative to predictability-only by {float(primary['primary_difference']):.4f} "
        f"(95% CI {float(primary['primary_ci_low']):.4f} to {float(primary['primary_ci_high']):.4f}; "
        f"n={int(primary['pairs'])})."
    )
