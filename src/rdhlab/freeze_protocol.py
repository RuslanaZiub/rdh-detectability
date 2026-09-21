
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import tempfile

CONFIRMED_DECISION = "TRANSFER_CONFIRMED_PRIMARY_ENDPOINT"


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id_hash(values) -> str:
    return sha256_text("\n".join(map(str, values)))


def validate_freeze_inputs(decision: dict, rule: dict, payload_freeze: dict) -> float:
    if decision.get("decision") != CONFIRMED_DECISION:
        raise RuntimeError(f"Independent transfer was not confirmed: {decision.get('decision')!r}")
    if not bool(decision.get("detector_sanity_pass", False)):
        raise RuntimeError("Independent detector sanity check did not pass.")
    if bool(decision.get("test_split_used", True)):
        raise RuntimeError("05e reports that the test split was already used.")
    if bool(decision.get("frozen_allocator_modified", True)):
        raise RuntimeError("05e reports that the allocator had already been modified.")
    if not bool(decision.get("source_ids_locked_from_05d", False)):
        raise RuntimeError("05e transfer IDs were not locked from 05d.")

    a_dec = float(decision["candidate_alpha"])
    a_rule = float(rule["candidate_alpha"])
    if not math.isclose(a_dec, a_rule, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Candidate alpha mismatch: 05e={a_dec}, 05d rule={a_rule}")
    if not (0.0 < a_dec < 1.0):
        raise RuntimeError(f"Candidate alpha must be strictly between 0 and 1, got {a_dec}")
    if not bool(rule.get("defined_before_cnn_transfer_result", False)):
        raise RuntimeError("05d candidate rule was not declared pre-transfer.")

    bpp_dec = float(decision["payload_bpp"])
    bpp_rule = float(rule["teacher_payload_bpp"])
    if not math.isclose(bpp_dec, bpp_rule, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Teacher payload mismatch: 05e={bpp_dec}, 05d={bpp_rule}")

    levels = [float(x) for x in payload_freeze["levels"]]
    if not any(math.isclose(bpp_dec, x, rel_tol=0.0, abs_tol=1e-12) for x in levels):
        raise RuntimeError(f"Teacher payload {bpp_dec} is not one of the frozen payloads: {levels}")

    lo = float(decision["primary_ci_low"])
    hi = float(decision["primary_ci_high"])
    diff = float(decision["primary_difference"])
    if not (diff < 0.0 and hi < 0.0 and lo < hi):
        raise RuntimeError(
            f"Primary independent-detector endpoint is not conclusively favorable: diff={diff}, CI=[{lo},{hi}]"
        )
    return a_dec


def build_allocator_freeze(
    decision: dict,
    rule: dict,
    payload_freeze: dict,
    *,
    decision_sha256: str,
    rule_sha256: str,
    payload_freeze_sha256: str,
    dataset_manifest_sha256: str,
    local_risk_model_sha256: str,
    enhanced_cnn_sha256: str | None = None,
) -> dict:
    alpha = validate_freeze_inputs(decision, rule, payload_freeze)
    return {
        "schema_version": 2,
        "status": "FROZEN_AFTER_05E_TRANSFER_CONFIRMATION",
        "alpha": float(alpha),
        "weights": {"predictability": float(alpha), "detectability": float(1.0 - alpha)},
        "strategy": "joint",
        "candidate_selection_rule": rule["rule"],
        "candidate_rule_defined_before_cnn_transfer_result": True,
        "teacher_payload_bpp": float(decision["payload_bpp"]),
        "independent_transfer_decision": decision["decision"],
        "independent_detector_primary_endpoint": decision["primary_endpoint"],
        "independent_detector_primary_difference": float(decision["primary_difference"]),
        "independent_detector_primary_ci": [float(decision["primary_ci_low"]), float(decision["primary_ci_high"])],
        "independent_detector_dev_auc": float(decision["detector_dev_auc"]),
        "test_split_used_during_selection": False,
        "payload_levels": [float(x) for x in payload_freeze["levels"]],
        "provenance_sha256": {
            "05e_decision_json": decision_sha256,
            "05d_candidate_alpha_rule_json": rule_sha256,
            "frozen_payloads_json": payload_freeze_sha256,
            "dataset_manifest_csv": dataset_manifest_sha256,
            "srm_teacher_local_risk_joblib": local_risk_model_sha256,
            "enhanced_residual_cnn_05e_pt": enhanced_cnn_sha256,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lock_note": (
            "Method selection is complete. Do not alter alpha, payload levels, risk model, "
            "strategy definitions, or test inclusion rules after this freeze."
        ),
    }


def atomic_write_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, p)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def jsonable_record(row: dict, *, drop_keys=("stego",)) -> dict:
    out = {}
    for k, v in row.items():
        if k in drop_keys:
            continue
        if hasattr(v, "item") and callable(getattr(v, "item")):
            try:
                v = v.item()
            except Exception:
                pass
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        out[str(k)] = v
    return out


def append_jsonl(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = jsonable_record(record)
    line = json.dumps(clean, sort_keys=True, allow_nan=False)
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Malformed JSONL checkpoint at line {lineno}: {e}") from e
    return rows


def case_key(source_id, strategy, target_net_bpp) -> str:
    return f"{source_id}|{strategy}|{float(target_net_bpp):.9f}"
