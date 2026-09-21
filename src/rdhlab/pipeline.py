from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import pandas as pd
import joblib

from .allocation import allocation_orders
from .blockcodec import analyze_blocks, embed_blockwise_net, extract_blockwise, max_accounted_net_capacity
from .io import read_gray, write_gray_png
from .metrics import psnr, ssim


def deterministic_seed(base_seed: int, *parts) -> int:
    text = "|".join([str(base_seed), *map(str, parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_payload_freeze(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_payload_freeze(path: str | Path, obj: dict):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def run_frozen_image(image: np.ndarray, source_id: str, target_bpp: float, strategy: str, risk_model, alpha: float, block_size: int = 64, seed: int = 20260917, file_io_check: bool = False, temp_path: str | Path | None = None) -> dict:
    target_net_bits = int(round(float(target_bpp) * image.size))
    orders, block_rows = allocation_orders(image, block_size, risk_model, source_id, alpha=alpha, seed=seed)
    if strategy not in orders:
        raise KeyError(strategy)
    rng = np.random.default_rng(deterministic_seed(seed, source_id, target_bpp, strategy))
    try:
        stego, gross_bits, meta, estats = embed_blockwise_net(image, target_net_bits, orders[strategy], rng, block_size)
    except ValueError as e:
        cap = max_accounted_net_capacity(image, block_size)
        return {
            "source_id": source_id, "strategy": strategy, "target_net_bpp": float(target_bpp),
            "target_net_bits": target_net_bits, "feasible": False, "error": str(e), **cap,
        }

    target = stego
    if file_io_check:
        if temp_path is None:
            raise ValueError("temp_path required when file_io_check=True")
        write_gray_png(temp_path, stego)
        target = read_gray(temp_path)
    recovered, recovered_bits, dstats = extract_blockwise(target, meta)
    changed = np.abs(image.astype(np.int16) - stego.astype(np.int16))
    selected_ids = {u.block_id for u in meta.used_blocks}
    selected = [r for r in block_rows if r["block_id"] in selected_ids]
    return {
        "source_id": source_id,
        "strategy": strategy,
        "target_net_bpp": float(target_bpp),
        "target_net_bits": int(target_net_bits),
        "feasible": True,
        "gross_payload_bits": int(meta.gross_payload_bits),
        "sideinfo_bits": int(meta.accounted_sideinfo_bits),
        "net_payload_bits": int(meta.gross_payload_bits - meta.accounted_sideinfo_bits),
        "actual_net_bpp": float((meta.gross_payload_bits - meta.accounted_sideinfo_bits) / image.size),
        "psnr": float(psnr(image, stego)),
        "ssim": float(ssim(image, stego)),
        "exact_image": bool(np.array_equal(image, recovered)),
        "exact_message": bool(np.array_equal(gross_bits, recovered_bits)),
        "ber": float(np.mean(gross_bits != recovered_bits)) if len(gross_bits) else 0.0,
        "encode_ms": float(estats["encode_ms"]),
        "decode_ms": float(dstats["decode_ms"]),
        "used_blocks": int(estats["used_blocks"]),
        "eligible_blocks": int(estats["eligible_blocks"]),
        "total_blocks": int(estats["total_blocks"]),
        "changed_pixels": int(np.count_nonzero(changed)),
        "mean_abs_change": float(changed.mean()),
        "selected_predictability_mean": float(np.mean([r["predictability"] for r in selected])) if selected else np.nan,
        "selected_detectability_risk_mean": float(np.mean([r["detectability_risk"] for r in selected])) if selected else np.nan,
        "stego": stego,
    }


def append_csv(path: str | Path, row: dict, drop_keys: tuple[str, ...] = ("stego",)):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in row.items() if k not in drop_keys}
    pd.DataFrame([clean]).to_csv(path, mode="a", header=not path.exists(), index=False)


def choose_payload_levels(validation_capacity_bpp: np.ndarray, candidates: list[float], minimum_feasibility: float = 0.95, fallback_levels: int = 4, fallback_quantile: float = 0.05, minimum_level_bpp: float = 0.001) -> dict:
    x = np.asarray(validation_capacity_bpp, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("No finite validation capacities")
    feasibility = {float(c): float(np.mean(x >= float(c))) for c in candidates}
    if all(feasibility[float(c)] >= minimum_feasibility for c in candidates):
        levels = [float(c) for c in candidates]
        method = "predeclared_candidates"
    else:
        q = float(np.quantile(x, fallback_quantile))
        if q < minimum_level_bpp:
            raise RuntimeError(
                f"Validation {fallback_quantile:.3f} capacity quantile is only {q:.6f} bpp; "
                "the current blockwise side-information model does not support a meaningful fixed net-payload grid."
            )
        raw = np.linspace(q / fallback_levels, q * 0.95, fallback_levels)
        levels = sorted(set(max(minimum_level_bpp, math_floor_001(v)) for v in raw))
        method = f"validation_q{fallback_quantile:g}_derived"
    return {"levels": levels, "method": method, "candidate_feasibility": feasibility}


def math_floor_001(x: float) -> float:
    return float(np.floor(float(x) * 1000.0) / 1000.0)


def simple_image_features(image: np.ndarray) -> np.ndarray:
    # Reuse the 8-dimensional image statistics from block_features on the full image.
    from .features import block_features
    return block_features(image)


def fit_aux_image_detector(covers: list[np.ndarray], stegos: list[np.ndarray], seed: int = 20260917):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if len(covers) != len(stegos):
        raise ValueError("covers/stegos length mismatch")
    X=[]; y=[]
    for c,s in zip(covers,stegos):
        X += [simple_image_features(c), simple_image_features(s)]
        y += [0,1]
    model=Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, solver="liblinear", random_state=seed)),
    ])
    model.fit(np.stack(X), np.asarray(y))
    return model


def score_aux_image_detector(model, image: np.ndarray) -> float:
    return float(model.predict_proba(simple_image_features(image).reshape(1,-1))[0,1])


def prepare_image_context(image: np.ndarray, source_id: str, risk_model, alpha: float, block_size: int = 64, seed: int = 20260917):
    orders, block_rows = allocation_orders(image, block_size, risk_model, source_id, alpha=alpha, seed=seed)
    plans = analyze_blocks(image, block_size)
    return orders, block_rows, plans


def prepare_image_orders(image: np.ndarray, source_id: str, risk_model, alpha: float, block_size: int = 64, seed: int = 20260917):
    orders, rows, _ = prepare_image_context(image, source_id, risk_model, alpha, block_size, seed)
    return orders, rows


def run_frozen_image_precomputed(image: np.ndarray, source_id: str, target_bpp: float, strategy: str, orders: dict[str, np.ndarray], block_rows: list[dict], block_size: int = 64, seed: int = 20260917, file_io_check: bool = False, temp_path: str | Path | None = None, plans=None) -> dict:
    target_net_bits = int(round(float(target_bpp) * image.size))
    if strategy not in orders:
        raise KeyError(strategy)
    rng = np.random.default_rng(deterministic_seed(seed, source_id, target_bpp, strategy))
    try:
        stego, gross_bits, meta, estats = embed_blockwise_net(image, target_net_bits, orders[strategy], rng, block_size, plans=plans)
    except ValueError as e:
        cap = max_accounted_net_capacity(image, block_size)
        return {
            "source_id": source_id, "strategy": strategy, "target_net_bpp": float(target_bpp),
            "target_net_bits": target_net_bits, "feasible": False, "error": str(e), **cap,
        }
    target = stego
    if file_io_check:
        if temp_path is None:
            raise ValueError("temp_path required when file_io_check=True")
        write_gray_png(temp_path, stego); target = read_gray(temp_path)
    recovered, recovered_bits, dstats = extract_blockwise(target, meta)
    changed = np.abs(image.astype(np.int16) - stego.astype(np.int16))
    selected_ids = {u.block_id for u in meta.used_blocks}
    selected = [r for r in block_rows if r["block_id"] in selected_ids]
    return {
        "source_id": source_id, "strategy": strategy, "target_net_bpp": float(target_bpp),
        "target_net_bits": int(target_net_bits), "feasible": True,
        "gross_payload_bits": int(meta.gross_payload_bits), "sideinfo_bits": int(meta.accounted_sideinfo_bits),
        "net_payload_bits": int(meta.gross_payload_bits-meta.accounted_sideinfo_bits),
        "actual_net_bpp": float((meta.gross_payload_bits-meta.accounted_sideinfo_bits)/image.size),
        "psnr": float(psnr(image,stego)), "ssim": float(ssim(image,stego)),
        "exact_image": bool(np.array_equal(image,recovered)),
        "exact_message": bool(np.array_equal(gross_bits,recovered_bits)),
        "ber": float(np.mean(gross_bits!=recovered_bits)) if len(gross_bits) else 0.0,
        "encode_ms": float(estats["encode_ms"]), "decode_ms": float(dstats["decode_ms"]),
        "used_blocks": int(estats["used_blocks"]), "eligible_blocks": int(estats["eligible_blocks"]),
        "total_blocks": int(estats["total_blocks"]), "changed_pixels": int(np.count_nonzero(changed)),
        "mean_abs_change": float(changed.mean()),
        "selected_predictability_mean": float(np.mean([r["predictability"] for r in selected])) if selected else np.nan,
        "selected_detectability_risk_mean": float(np.mean([r["detectability_risk"] for r in selected])) if selected else np.nan,
        "stego": stego,
    }
