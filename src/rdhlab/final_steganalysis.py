from __future__ import annotations
import math

CONFIRMED = "TEST_DETECTABILITY_PRIMARY_CONFIRMED"
NOT_CONFIRMED = "TEST_DETECTABILITY_PRIMARY_NOT_CONFIRMED"


def validate_test_completion(complete: dict, *, expected_cases: int = 40000) -> None:
    if complete.get("status") != "COMPLETE":
        raise RuntimeError(f"Frozen test is not complete: {complete.get('status')!r}")
    if int(complete.get("cases", -1)) != int(expected_cases):
        raise RuntimeError(f"Expected {expected_cases} cases, got {complete.get('cases')}")
    if int(complete.get("expected_cases", -1)) != int(expected_cases):
        raise RuntimeError("Frozen-test expected_cases mismatch.")
    if not bool(complete.get("no_retuning_permitted", False)):
        raise RuntimeError("Frozen-test artifact does not enforce no-retuning.")
    feasible = int(complete.get("feasible_cases", -1))
    exact = int(complete.get("exact_recovery_feasible_cases", -2))
    if feasible < 0 or exact != feasible:
        raise RuntimeError(f"Exact-recovery count mismatch: feasible={feasible}, exact={exact}")


def primary_endpoint_decision(comparison: dict) -> dict:
    diff = float(comparison["delta_mean_diff"])
    lo = float(comparison["delta_mean_diff_low"])
    hi = float(comparison["delta_mean_diff_high"])
    if not (math.isfinite(diff) and math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError("Non-finite or invalid primary endpoint values.")
    confirmed = bool(diff < 0.0 and hi < 0.0)
    return {
        "decision": CONFIRMED if confirmed else NOT_CONFIRMED,
        "primary_confirmed": confirmed,
        "primary_difference": diff,
        "primary_ci_low": lo,
        "primary_ci_high": hi,
    }
