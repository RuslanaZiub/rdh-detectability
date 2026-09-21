from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import math
import os
import tempfile

import numpy as np

from rdhlab.srnet_secondary_v11 import build_srnet, parameter_count, score_srnet


EXPECTED_LOCKED_HASHES = {
    "srnet_v13_train_history.csv": "d5ffd108dd55d5e317018f188a4f848c5ed37658c5e00efa699015981825fb20",
    "srnet_v13_dev_sanity.json": "afad2ec4bdd31bc222cf39b68a77f31f6b55ce02ec71992edc54a818fd48785c",
    "srnet_v13_protocol_pretest.json": "41aa5f32cc783edf63f2db1c96c889ebd5bd3ea940fc55b54aaa1466a7f15685",
    "srnet_v13_stage_summary.csv": "9cf96638f12be688e8e6732005a2f9001fbdf89bee8e59f04616cb481a019aab",
    "stage_04_target_0p009_best.pt": "299dca8453fdd580d6cb169427cce67e5c98784df1f674c81172e327d791d640",
}
EXPECTED_PROTOCOL_SHA256 = EXPECTED_LOCKED_HASHES["srnet_v13_protocol_pretest.json"]
EXPECTED_CHECKPOINT_SHA256 = EXPECTED_LOCKED_HASHES["stage_04_target_0p009_best.pt"]
EXPECTED_SANITY_AUC = 0.8644054128159223
EXPECTED_SANITY_THRESHOLD = 0.65
EXPECTED_TARGET_BPP = 0.009
EXPECTED_SELECTED_EPOCH = 4
EXPECTED_COMMON_TEST_PAIRS = 1968


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def validate_locked_v13_inputs(locked_dir: str | Path) -> dict:
    locked_dir = Path(locked_dir)
    missing = [name for name in EXPECTED_LOCKED_HASHES if not (locked_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing locked Patch-13 inputs: {missing}")

    actual_hashes = {name: sha256_file(locked_dir / name) for name in EXPECTED_LOCKED_HASHES}
    bad = {
        name: {"expected": EXPECTED_LOCKED_HASHES[name], "actual": actual_hashes[name]}
        for name in EXPECTED_LOCKED_HASHES
        if actual_hashes[name] != EXPECTED_LOCKED_HASHES[name]
    }
    if bad:
        raise RuntimeError(f"Locked Patch-13 hash mismatch: {bad}")

    protocol = json.loads((locked_dir / "srnet_v13_protocol_pretest.json").read_text(encoding="utf-8"))
    sanity = json.loads((locked_dir / "srnet_v13_dev_sanity.json").read_text(encoding="utf-8"))

    if protocol.get("analysis_status") != "POST_HOC_SECONDARY_ROBUSTNESS_DETECTOR_DEVELOPMENT_V13":
        raise RuntimeError("Unexpected Patch-13 analysis status.")
    if not math.isclose(float(protocol["allocator_alpha_frozen"]), 0.25, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Frozen allocator alpha is not 0.25.")
    if not math.isclose(float(protocol["primary_payload_bpp"]), EXPECTED_TARGET_BPP, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Patch-13 target payload mismatch.")
    if bool(protocol.get("test_scoring_in_this_patch", True)):
        raise RuntimeError("Patch 13 unexpectedly reports TEST scoring.")
    if not bool(protocol.get("no_allocator_or_primary_endpoint_retuning_permitted", False)):
        raise RuntimeError("Patch 13 does not enforce the no-retuning lock.")

    if not bool(sanity.get("sanity_pass", False)):
        raise RuntimeError("Patch-13 final DEV sanity did not pass.")
    if bool(sanity.get("test_split_scored", True)):
        raise RuntimeError("Patch-13 sanity reports that TEST was already scored by SRNet.")
    if bool(sanity.get("allocator_retuned", True)):
        raise RuntimeError("Patch-13 sanity reports allocator retuning.")
    if sanity.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Patch-13 sanity protocol hash mismatch.")
    if sanity.get("selected_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Patch-13 sanity checkpoint hash mismatch.")
    if not math.isclose(float(sanity["sanity_auc_threshold"]), EXPECTED_SANITY_THRESHOLD, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Unexpected Patch-13 DEV sanity threshold.")
    if not math.isclose(float(sanity["auc"]), EXPECTED_SANITY_AUC, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Unexpected Patch-13 selected DEV AUC.")

    stages = _read_csv_rows(locked_dir / "srnet_v13_stage_summary.csv")
    final = [r for r in stages if r.get("stage") == "target_0p009"]
    if len(final) != 1:
        raise RuntimeError("Could not identify a unique target_0p009 stage.")
    final = final[0]
    if int(float(final["best_epoch_in_stage"])) != EXPECTED_SELECTED_EPOCH:
        raise RuntimeError("Unexpected selected target epoch.")
    if not math.isclose(float(final["best_dev_auc"]), EXPECTED_SANITY_AUC, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Stage summary DEV AUC disagrees with sanity JSON.")

    # Validate the actual full training checkpoint and its embedded lock state.
    import torch
    checkpoint_path = locked_dir / "stage_04_target_0p009_best.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in payload or "state" not in payload:
        raise RuntimeError("Locked checkpoint is not a Patch-13 full training checkpoint.")
    state = payload["state"]
    if state.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Checkpoint embeds a different Patch-13 protocol hash.")
    if state.get("stage_name") != "target_0p009":
        raise RuntimeError("Checkpoint is not the selected target stage.")
    if int(state.get("selected_epoch_in_stage", -1)) != EXPECTED_SELECTED_EPOCH:
        raise RuntimeError("Checkpoint selected epoch mismatch.")
    if not math.isclose(float(state.get("payload_bpp", -1)), EXPECTED_TARGET_BPP, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Checkpoint target payload mismatch.")
    if not math.isclose(float(state.get("dev_auc", -1)), EXPECTED_SANITY_AUC, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Checkpoint DEV AUC mismatch.")

    return {
        "protocol": protocol,
        "sanity": sanity,
        "stage_final": final,
        "actual_hashes": actual_hashes,
        "checkpoint_state": state,
    }


def validate_pretest_lock(lock_path: str | Path, locked_summary: dict) -> dict:
    p = Path(lock_path)
    if not p.exists():
        raise RuntimeError(f"Patch-13 pretest lock is missing: {p}")
    lock = json.loads(p.read_text(encoding="utf-8"))
    sanity = locked_summary["sanity"]
    if lock.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Pretest lock protocol hash mismatch.")
    if lock.get("selected_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Pretest lock checkpoint hash mismatch.")
    if not bool(lock.get("dev_sanity_pass", False)):
        raise RuntimeError("Pretest lock says DEV sanity failed.")
    if bool(lock.get("test_split_scored", True)):
        raise RuntimeError("Pretest lock says SRNet TEST was already scored.")
    if not bool(lock.get("protocol_locked_before_any_srnet_v13_test_scoring", False)):
        raise RuntimeError("Pretest lock was not marked as created before SRNet TEST scoring.")
    if lock.get("srnet_v13_model_sha256") != sanity.get("model_sha256"):
        raise RuntimeError("Pretest lock model hash disagrees with Patch-13 sanity.")
    return lock


def validate_original_v13_against_locked(original_dir: str | Path, locked_dir: str | Path) -> dict:
    original_dir = Path(original_dir)
    locked_dir = Path(locked_dir)
    mapping = {
        "srnet_v13_train_history.csv": original_dir / "srnet_v13_train_history.csv",
        "srnet_v13_dev_sanity.json": original_dir / "srnet_v13_dev_sanity.json",
        "srnet_v13_protocol_pretest.json": original_dir / "srnet_v13_protocol_pretest.json",
        "srnet_v13_stage_summary.csv": original_dir / "srnet_v13_stage_summary.csv",
        "stage_04_target_0p009_best.pt": original_dir / "checkpoints" / "stage_04_target_0p009_best.pt",
    }
    out = {}
    for name, p in mapping.items():
        if not p.exists():
            raise RuntimeError(f"Original Patch-13 artifact is missing: {p}")
        h = sha256_file(p)
        if h != EXPECTED_LOCKED_HASHES[name]:
            raise RuntimeError(
                f"Original Patch-13 artifact differs from the externally locked copy: {name}"
            )
        if h != sha256_file(locked_dir / name):
            raise RuntimeError(f"Original/locked hash mismatch for {name}")
        out[name] = h
    return out


def load_locked_srnet(checkpoint_path: str | Path, device: str | None = None):
    import torch
    p = Path(checkpoint_path)
    if sha256_file(p) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Refusing to load an unapproved SRNet-v13 checkpoint.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(p, map_location=device, weights_only=False)
    model = build_srnet().to(device)
    if str(device).startswith("cpu"):
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(payload["model_state"])
    model.eval()
    if parameter_count(model) != 4_779_616:
        raise RuntimeError("Unexpected SRNet parameter count.")
    return model, device, payload["state"]


def minimal_detection_error(cover_scores, stego_scores) -> float:
    from sklearn.metrics import roc_curve
    c = np.asarray(cover_scores, dtype=float)
    s = np.asarray(stego_scores, dtype=float)
    if len(c) != len(s) or len(c) == 0:
        raise ValueError("cover/stego scores must be non-empty and aligned")
    y = np.tile([0, 1], len(c))
    scores = np.column_stack([c, s]).reshape(-1)
    fpr, tpr, _ = roc_curve(y, scores)
    return float(np.min(0.5 * (fpr + (1.0 - tpr))))


def paired_pe_bootstrap(
    cover_scores,
    method_scores,
    reference_scores,
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260920,
) -> dict:
    c = np.asarray(cover_scores, dtype=float)
    a = np.asarray(method_scores, dtype=float)
    b = np.asarray(reference_scores, dtype=float)
    if not (len(c) == len(a) == len(b)) or len(c) == 0:
        raise ValueError("score arrays must be non-empty and aligned")
    n = len(c)
    rng = np.random.default_rng(int(seed))
    diff = np.empty(int(n_resamples), dtype=float)
    for k in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        diff[k] = minimal_detection_error(c[idx], a[idx]) - minimal_detection_error(c[idx], b[idx])
    q = (1.0 - float(confidence)) / 2.0
    return {
        "pe_method": minimal_detection_error(c, a),
        "pe_reference": minimal_detection_error(c, b),
        "pe_diff": float(minimal_detection_error(c, a) - minimal_detection_error(c, b)),
        "pe_diff_low": float(np.quantile(diff, q)),
        "pe_diff_high": float(np.quantile(diff, 1.0 - q)),
        "n_pairs": int(n),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
    }


def single_pe_bootstrap(
    cover_scores,
    stego_scores,
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260920,
) -> dict:
    c = np.asarray(cover_scores, dtype=float)
    s = np.asarray(stego_scores, dtype=float)
    if len(c) != len(s) or len(c) == 0:
        raise ValueError("score arrays must be non-empty and aligned")
    n = len(c)
    rng = np.random.default_rng(int(seed))
    pe = np.empty(int(n_resamples), dtype=float)
    for k in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        pe[k] = minimal_detection_error(c[idx], s[idx])
    q = (1.0 - float(confidence)) / 2.0
    return {
        "pe": minimal_detection_error(c, s),
        "pe_ci_low": float(np.quantile(pe, q)),
        "pe_ci_high": float(np.quantile(pe, 1.0 - q)),
        "n_pairs": int(n),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
    }
