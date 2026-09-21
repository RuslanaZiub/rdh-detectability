from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .statistics import bootstrap_ci


def aggregate_test_results(per_image_csv: str | Path, out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(per_image_csv)
    ok = df[df["feasible"] == True].copy()  # noqa: E712
    rows = []
    for (strategy, bpp), g in ok.groupby(["strategy", "target_net_bpp"]):
        row = {"strategy": strategy, "target_net_bpp": bpp, "n": len(g)}
        for col in ["psnr", "ssim", "encode_ms", "decode_ms", "sideinfo_bits", "used_blocks", "changed_pixels"]:
            ci = bootstrap_ci(g[col].to_numpy(), "median", n_resamples=5000)
            row[f"{col}_median"] = ci["estimate"]
            row[f"{col}_ci_low"] = ci["low"]
            row[f"{col}_ci_high"] = ci["high"]
        row["exact_recovery_rate"] = float((g["exact_image"] & g["exact_message"]).mean())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["target_net_bpp", "strategy"]).reset_index(drop=True)
    feasibility = df.groupby(["strategy", "target_net_bpp"])["feasible"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "feasible_fraction"})

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "table_test_metrics.csv", index=False)
    feasibility.to_csv(out / "table_feasibility.csv", index=False)

    if not summary.empty:
        for metric, ylabel in [("psnr_median", "PSNR (dB)"), ("ssim_median", "SSIM")]:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            for strategy, g in summary.groupby("strategy"):
                ax.plot(g["target_net_bpp"], g[metric], marker="o", label=strategy)
            ax.set_xlabel("Accounted net payload (bpp)"); ax.set_ylabel(ylabel); ax.grid(True, alpha=.25); ax.legend()
            fig.tight_layout(); fig.savefig(out / f"fig_{metric}.png", dpi=300); plt.close(fig)
    return summary, feasibility
