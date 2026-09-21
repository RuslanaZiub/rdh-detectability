import numpy as np
import pandas as pd

from rdhlab.srnet_payload_probe_v12 import (
    PayloadSelectionRule,
    curriculum_path,
    deterministic_probe_indices,
    deterministic_random_order,
    select_strong_payload,
    summarize_probe,
)


def test_probe_indices_are_deterministic_and_unique():
    a = deterministic_probe_indices(1000, 100, 123)
    b = deterministic_probe_indices(1000, 100, 123)
    assert np.array_equal(a, b)
    assert len(np.unique(a)) == 100
    assert a.min() >= 0 and a.max() < 1000


def test_random_order_is_payload_and_source_deterministic():
    ids = np.arange(64)
    a = deterministic_random_order(ids, seed=7, source_id="x", payload_bpp=0.03)
    b = deterministic_random_order(ids, seed=7, source_id="x", payload_bpp=0.03)
    c = deterministic_random_order(ids, seed=7, source_id="y", payload_bpp=0.03)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert sorted(a.tolist()) == ids.tolist()


def test_summary_and_strong_payload_selection():
    rows = []
    for p, nf in [(0.012, 10), (0.020, 10), (0.030, 9), (0.050, 7)]:
        for i in range(10):
            ok = i < nf
            rows.append({
                "target_net_bpp": p,
                "feasible": ok,
                "exact_image": ok,
                "exact_message": ok,
                "psnr": 60.0 if ok else np.nan,
            })
    s = summarize_probe(pd.DataFrame(rows))
    sel = select_strong_payload(s, PayloadSelectionRule(min_feasible_fraction=0.90))
    assert sel["status"] == "STRONG_PAYLOAD_SELECTED"
    assert np.isclose(sel["selected_payload_bpp"], 0.030)
    assert curriculum_path(sel["selected_payload_bpp"], [0.012, 0.015, 0.020, 0.030, 0.050]) == [0.03, 0.02, 0.015, 0.012, 0.009]
