import math
from pathlib import Path
import os

import numpy as np

from rdhlab.srnet_frozen_test_v14 import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_COMMON_TEST_PAIRS,
    EXPECTED_SANITY_AUC,
    EXPECTED_TARGET_BPP,
    minimal_detection_error,
    validate_locked_v13_inputs,
)


def test_locked_v13_bundle_is_exact():
    locked = Path(os.environ.get("RDH_V14_LOCKED_DIR", "/workspace/results/srnet_frozen_test_v14/locked_inputs"))
    out = validate_locked_v13_inputs(locked)
    assert out["sanity"]["sanity_pass"] is True
    assert math.isclose(float(out["sanity"]["auc"]), EXPECTED_SANITY_AUC, rel_tol=0, abs_tol=1e-12)
    assert out["sanity"]["selected_checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256


def test_metric_direction_sanity():
    c = np.array([0.1, 0.2, 0.3, 0.4])
    easy = np.array([0.9, 0.8, 0.9, 0.8])
    hard = np.array([0.15, 0.25, 0.35, 0.45])
    assert minimal_detection_error(c, easy) < minimal_detection_error(c, hard)


def test_frozen_constants():
    assert EXPECTED_COMMON_TEST_PAIRS == 1968
    assert math.isclose(EXPECTED_TARGET_BPP, 0.009)
