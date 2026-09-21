from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PayloadSelectionRule:
    reference_bpp: float = 0.012
    min_feasible_fraction: float = 0.90
    require_exact_recovery: bool = True


def deterministic_probe_indices(n_pool: int, n_probe: int, seed: int) -> np.ndarray:
    """Return a deterministic TRAIN-only subsample without replacement."""
    if n_pool <= 0:
        raise ValueError("n_pool must be positive")
    if n_probe <= 0 or n_probe > n_pool:
        raise ValueError("n_probe must satisfy 1 <= n_probe <= n_pool")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(int(n_pool), size=int(n_probe), replace=False)
    return np.sort(idx.astype(int))


def deterministic_random_order(block_ids: Iterable[int], *, seed: int, source_id: str, payload_bpp: float) -> np.ndarray:
    """Deterministic random block order for detector-development payload probing."""
    bids = np.asarray(list(block_ids), dtype=int)
    digest = hashlib.sha256(
        f"{int(seed)}|srnet-v12-payload-probe|{float(payload_bpp):.6f}|{source_id}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    out = bids.copy()
    rng.shuffle(out)
    return out


def summarize_probe(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-case TRAIN-only feasibility results by target payload."""
    required = {"target_net_bpp", "feasible", "exact_image", "exact_message"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    records = []
    for payload, g in rows.groupby("target_net_bpp", sort=True):
        feasible = g[g["feasible"].astype(bool)]
        n = int(len(g))
        nf = int(len(feasible))
        exact = (
            feasible["exact_image"].fillna(False).astype(bool)
            & feasible["exact_message"].fillna(False).astype(bool)
        )
        rec = {
            "target_net_bpp": float(payload),
            "n": n,
            "feasible_n": nf,
            "feasible_fraction": float(nf / n) if n else np.nan,
            "exact_recovery_fraction_feasible": float(exact.mean()) if nf else np.nan,
        }
        for col in [
            "actual_net_bpp", "psnr", "ssim", "sideinfo_bits", "sideinfo_fraction_gross",
            "used_blocks", "changed_pixels", "elapsed_ms",
        ]:
            if col in feasible.columns and nf:
                vals = pd.to_numeric(feasible[col], errors="coerce").to_numpy(float)
                vals = vals[np.isfinite(vals)]
                rec[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
                rec[f"{col}_median"] = float(np.median(vals)) if len(vals) else np.nan
        records.append(rec)
    return pd.DataFrame.from_records(records).sort_values("target_net_bpp").reset_index(drop=True)


def select_strong_payload(summary: pd.DataFrame, rule: PayloadSelectionRule | None = None) -> dict:
    """Select the strongest pre-specified training payload meeting the frozen rule.

    This selects a detector-training curriculum payload only. It does not alter any
    RDH evaluation payload, alpha, allocator, or primary endpoint.
    """
    if rule is None:
        rule = PayloadSelectionRule()
    required = {
        "target_net_bpp", "feasible_n", "feasible_fraction", "exact_recovery_fraction_feasible"
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Missing summary columns: {sorted(missing)}")

    cand = summary[summary["target_net_bpp"].astype(float) > float(rule.reference_bpp)].copy()
    cand = cand[cand["feasible_fraction"].astype(float) >= float(rule.min_feasible_fraction)]
    if rule.require_exact_recovery:
        cand = cand[np.isclose(cand["exact_recovery_fraction_feasible"].astype(float), 1.0)]

    if cand.empty:
        return {
            "status": "NO_STRONG_PAYLOAD_FOUND",
            "selected_payload_bpp": None,
            "reference_payload_bpp": float(rule.reference_bpp),
            "min_feasible_fraction": float(rule.min_feasible_fraction),
            "selection_rule": "highest pre-specified payload > reference with feasibility >= threshold and exact recovery on all feasible cases",
        }

    row = cand.sort_values("target_net_bpp").iloc[-1]
    return {
        "status": "STRONG_PAYLOAD_SELECTED",
        "selected_payload_bpp": float(row["target_net_bpp"]),
        "selected_feasible_n": int(row["feasible_n"]),
        "selected_feasible_fraction": float(row["feasible_fraction"]),
        "selected_exact_recovery_fraction_feasible": float(row["exact_recovery_fraction_feasible"]),
        "reference_payload_bpp": float(rule.reference_bpp),
        "min_feasible_fraction": float(rule.min_feasible_fraction),
        "selection_rule": "highest pre-specified payload > reference with feasibility >= threshold and exact recovery on all feasible cases",
    }


def curriculum_path(selected_payload_bpp: float | None, candidate_payloads: Iterable[float], *, target_bpp: float = 0.009, reference_bpp: float = 0.012) -> list[float]:
    """Create a descending, pre-defined curriculum path from the selected probe payload."""
    if selected_payload_bpp is None:
        return []
    selected = float(selected_payload_bpp)
    mids = sorted(
        {float(x) for x in candidate_payloads if float(target_bpp) < float(x) < selected},
        reverse=True,
    )
    path = [selected] + mids
    if reference_bpp < selected and float(reference_bpp) not in path:
        path.append(float(reference_bpp))
    if float(target_bpp) not in path:
        path.append(float(target_bpp))
    # preserve order while removing accidental duplicates
    out = []
    for x in path:
        if x not in out:
            out.append(x)
    return out
