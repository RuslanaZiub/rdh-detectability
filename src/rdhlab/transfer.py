from __future__ import annotations

import numpy as np

from .detectors import detector_metrics


def bootstrap_location_ci(
    values: np.ndarray,
    *,
    statistic: str = "mean",
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260917,
) -> dict:
    """Paired-image bootstrap CI for a one-sample location statistic.

    ``values`` should contain one value per source image, e.g. stego_score-cover_score.
    Resampling therefore preserves the image as the experimental unit.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"estimate": np.nan, "low": np.nan, "high": np.nan, "n": 0}
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    fn = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_resamples), dtype=float)
    n = len(x)
    for b in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        boots[b] = fn(x[idx])
    a = (1.0 - float(confidence)) / 2.0
    return {
        "estimate": float(fn(x)),
        "low": float(np.quantile(boots, a)),
        "high": float(np.quantile(boots, 1.0 - a)),
        "n": int(n),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "statistic": statistic,
    }


def paired_method_bootstrap(
    cover_scores: np.ndarray,
    method_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    fixed_fpr: float = 0.05,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260917,
) -> dict:
    """Cluster bootstrap comparison of two stego methods on identical covers.

    All three arrays must be aligned by source image. Negative differences favor
    ``method_scores`` over ``reference_scores`` for delta-score, AUC and TPR.
    """
    c = np.asarray(cover_scores, dtype=float)
    a = np.asarray(method_scores, dtype=float)
    b = np.asarray(reference_scores, dtype=float)
    if not (len(c) == len(a) == len(b)):
        raise ValueError("cover/method/reference score length mismatch")
    n = len(c)
    if n == 0:
        return {
            "n_pairs": 0,
            "delta_mean_diff": np.nan,
            "delta_mean_diff_low": np.nan,
            "delta_mean_diff_high": np.nan,
            "auc_diff": np.nan,
            "auc_diff_low": np.nan,
            "auc_diff_high": np.nan,
            "tpr_diff": np.nan,
            "tpr_diff_low": np.nan,
            "tpr_diff_high": np.nan,
        }

    da = a - c
    db = b - c
    y = np.tile([0, 1], n)
    ma = detector_metrics(y, np.column_stack([c, a]).reshape(-1), fixed_fpr)
    mb = detector_metrics(y, np.column_stack([c, b]).reshape(-1), fixed_fpr)

    rng = np.random.default_rng(seed)
    dm = np.empty(int(n_resamples), dtype=float)
    med = np.empty(int(n_resamples), dtype=float)
    aucd = np.empty(int(n_resamples), dtype=float)
    tprd = np.empty(int(n_resamples), dtype=float)
    for k in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        ci, ai, bi = c[idx], a[idx], b[idx]
        dai = ai - ci
        dbi = bi - ci
        dm[k] = np.mean(dai - dbi)
        med[k] = np.median(dai - dbi)
        yy = np.tile([0, 1], n)
        mai = detector_metrics(yy, np.column_stack([ci, ai]).reshape(-1), fixed_fpr)
        mbi = detector_metrics(yy, np.column_stack([ci, bi]).reshape(-1), fixed_fpr)
        aucd[k] = mai["auc"] - mbi["auc"]
        tprd[k] = mai["tpr_at_fpr"] - mbi["tpr_at_fpr"]

    q = (1.0 - float(confidence)) / 2.0
    def ci(arr):
        return float(np.quantile(arr, q)), float(np.quantile(arr, 1.0 - q))

    dm_lo, dm_hi = ci(dm)
    med_lo, med_hi = ci(med)
    auc_lo, auc_hi = ci(aucd)
    tpr_lo, tpr_hi = ci(tprd)
    return {
        "n_pairs": int(n),
        "delta_mean_diff": float(np.mean(da - db)),
        "delta_mean_diff_low": dm_lo,
        "delta_mean_diff_high": dm_hi,
        "delta_median_diff": float(np.median(da - db)),
        "delta_median_diff_low": med_lo,
        "delta_median_diff_high": med_hi,
        "auc_method": float(ma["auc"]),
        "auc_reference": float(mb["auc"]),
        "auc_diff": float(ma["auc"] - mb["auc"]),
        "auc_diff_low": auc_lo,
        "auc_diff_high": auc_hi,
        "tpr_method": float(ma["tpr_at_fpr"]),
        "tpr_reference": float(mb["tpr_at_fpr"]),
        "tpr_diff": float(ma["tpr_at_fpr"] - mb["tpr_at_fpr"]),
        "tpr_diff_low": tpr_lo,
        "tpr_diff_high": tpr_hi,
        "fixed_fpr": float(fixed_fpr),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
    }
