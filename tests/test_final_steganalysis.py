from rdhlab.final_steganalysis import (
    validate_test_completion, primary_endpoint_decision,
    CONFIRMED, NOT_CONFIRMED,
)

def test_validate_complete():
    x = {
        "status":"COMPLETE","cases":40000,"expected_cases":40000,
        "feasible_cases":39405,"exact_recovery_feasible_cases":39405,
        "no_retuning_permitted":True,
    }
    validate_test_completion(x)

def test_primary_confirmed():
    out = primary_endpoint_decision({
        "delta_mean_diff":-0.1,
        "delta_mean_diff_low":-0.2,
        "delta_mean_diff_high":-0.03,
    })
    assert out["decision"] == CONFIRMED
    assert out["primary_confirmed"] is True

def test_primary_not_confirmed_when_ci_crosses_zero():
    out = primary_endpoint_decision({
        "delta_mean_diff":-0.01,
        "delta_mean_diff_low":-0.05,
        "delta_mean_diff_high":0.02,
    })
    assert out["decision"] == NOT_CONFIRMED
    assert out["primary_confirmed"] is False
