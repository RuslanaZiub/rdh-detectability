from __future__ import annotations

from collections.abc import Callable
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def bootstrap_ci(values, statistic: str = "median", confidence: float = 0.95, n_resamples: int = 5000, seed: int = 20260917):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"estimate": np.nan, "low": np.nan, "high": np.nan, "n": 0}
    fn = np.median if statistic == "median" else np.mean
    estimate = float(fn(x))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        boots[i] = fn(rng.choice(x, size=len(x), replace=True))
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return {"estimate": estimate, "low": float(lo), "high": float(hi), "n": int(len(x))}


def cluster_bootstrap(df: pd.DataFrame, cluster_col: str, stat_fn: Callable[[pd.DataFrame], float], n_resamples: int = 2000, confidence: float = 0.95, seed: int = 20260917):
    clusters = np.asarray(pd.unique(df[cluster_col]))
    if len(clusters) == 0:
        return {"estimate": np.nan, "low": np.nan, "high": np.nan, "n_clusters": 0}
    estimate = float(stat_fn(df))
    grouped = {c: df[df[cluster_col] == c] for c in clusters}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([grouped[c] for c in sampled], ignore_index=True)
        try:
            v = float(stat_fn(boot))
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"estimate": estimate, "low": np.nan, "high": np.nan, "n_clusters": int(len(clusters))}
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1 - alpha])
    return {"estimate": estimate, "low": float(lo), "high": float(hi), "n_clusters": int(len(clusters))}


def predictability_detectability_stats(df: pd.DataFrame, p_col: str = "predictability", d_col: str = "detectability_risk") -> dict:
    x = df[p_col].to_numpy(float)
    y = df[d_col].to_numpy(float)
    s = spearmanr(x, y, nan_policy="omit")
    k = kendalltau(x, y, nan_policy="omit")
    p_q75 = np.nanquantile(x, 0.75)
    d_q75 = np.nanquantile(y, 0.75)
    high_p = x >= p_q75
    conflict = np.mean(y[high_p] >= d_q75) if np.any(high_p) else np.nan
    return {
        "spearman_rho": float(s.statistic),
        "spearman_p": float(s.pvalue),
        "kendall_tau": float(k.statistic),
        "kendall_p": float(k.pvalue),
        "top_predictability_top_risk_fraction": float(conflict),
        "n_blocks": int(np.isfinite(x + y).sum()),
    }
