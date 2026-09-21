import pandas as pd
from rdhlab.publication_synthesis import validate_grid, compact_metric_table, primary_text

def _grid():
    rows=[]
    for s in ["raster","random","predictability","detectability","joint"]:
        for p in [0.003,0.006,0.009,0.012]:
            rows.append({"strategy":s,"target_net_bpp":p,"n":10,"actual_net_bpp_mean":p,
                         "psnr_mean":60,"ssim_mean":.99})
    return pd.DataFrame(rows)

def test_validate_grid():
    assert validate_grid(_grid())

def test_compact():
    r=_grid()
    d=_grid()[["strategy","target_net_bpp"]].copy()
    d["auc"]=0.7; d["tpr_at_5pct_fpr"]=0.2; d["score_delta_mean"]=0.1
    out=compact_metric_table(r,d,0.009)
    assert len(out)==5

def test_primary_text():
    x={"primary_payload_bpp":.009,"alpha":.25,"primary_difference":-.2,
       "primary_ci_low":-.3,"primary_ci_high":-.1,"pairs":100}
    s=primary_text(x)
    assert "0.009" in s and "100" in s
