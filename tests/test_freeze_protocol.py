
import json
import pytest
from rdhlab.freeze_protocol import (
    validate_freeze_inputs, build_allocator_freeze, atomic_write_json,
    append_jsonl, load_jsonl, case_key,
)

def good():
    decision = {
        "decision": "TRANSFER_CONFIRMED_PRIMARY_ENDPOINT",
        "detector_sanity_pass": True,
        "candidate_alpha": 0.25,
        "payload_bpp": 0.009,
        "primary_endpoint": "paired endpoint",
        "primary_difference": -0.2,
        "primary_ci_low": -0.25,
        "primary_ci_high": -0.15,
        "detector_dev_auc": 0.83,
        "test_split_used": False,
        "frozen_allocator_modified": False,
        "source_ids_locked_from_05d": True,
    }
    rule = {
        "candidate_alpha": 0.25,
        "teacher_payload_bpp": 0.009,
        "rule": "predeclared",
        "defined_before_cnn_transfer_result": True,
    }
    payload = {"levels": [0.003, 0.006, 0.009, 0.012]}
    return decision, rule, payload

def test_validate_good():
    d, r, p = good()
    assert validate_freeze_inputs(d, r, p) == 0.25

def test_reject_ci_crossing_zero():
    d, r, p = good()
    d["primary_ci_high"] = 0.01
    with pytest.raises(RuntimeError):
        validate_freeze_inputs(d, r, p)

def test_build_freeze():
    d, r, p = good()
    f = build_allocator_freeze(
        d, r, p,
        decision_sha256="a", rule_sha256="b", payload_freeze_sha256="c",
        dataset_manifest_sha256="d", local_risk_model_sha256="e",
        enhanced_cnn_sha256="f",
    )
    assert f["alpha"] == 0.25
    assert f["weights"]["detectability"] == 0.75
    assert f["test_split_used_during_selection"] is False

def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, {"source_id": "1", "strategy": "joint", "target_net_bpp": 0.009, "x": 3})
    rows = load_jsonl(p)
    assert len(rows) == 1
    assert case_key(rows[0]["source_id"], rows[0]["strategy"], rows[0]["target_net_bpp"]) == "1|joint|0.009000000"

def test_atomic_write(tmp_path):
    p = tmp_path / "f.json"
    atomic_write_json(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
