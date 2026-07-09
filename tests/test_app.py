import numpy as np

from app import build_drawdown_series, get_risk_profile_config, optimize_weights


def test_best_mix_uses_moderate_weights():
    weights = optimize_weights("Best Mix (Help me choose)")
    moderate_weights = optimize_weights("Moderate")

    np.testing.assert_allclose(weights, moderate_weights)


def test_risk_profile_config_returns_expected_safety_and_drawdown():
    config = get_risk_profile_config("Aggressive")

    assert config["risk_class"] == "aggressive"
    assert config["safety_score"] < 80
    assert config["max_drawdown"] > 0.18


def test_drawdown_series_matches_risk_profile_shape():
    series = build_drawdown_series("Conservative")

    assert len(series) == 12
    assert series["portfolio_value"].iloc[0] == 100.0
    assert series["portfolio_value"].min() > 95.0
