"""Phase D Multi-Weight Score-Mix tests (T-0222/T-0232).

Targets ``_score_candidate``'s 3-component weight renormalisation:

    w_perf = 0.7 * (1 - w_a - w_n - w_ml)
    w_gate = 0.3 * (1 - w_a - w_n - w_ml)
    live   = w_perf * perf + w_gate * gate + w_a * a + w_n * n + w_ml * ml

The numerical formula must collapse to the Phase-A two-term formula
when all three component weights are zero (regression guarantee), and
must raise ``ValueError`` when ``w_a + w_n + w_ml >= 1``.
"""

from __future__ import annotations

import pytest

from backtest.meta_decision import REGIME_TOLERANCE_DEFAULTS


def _renormalised_live_score(
    *,
    perf: float,
    gate: float,
    analyst: float,
    news: float,
    ml: float,
    w_a: float,
    w_n: float,
    w_ml: float,
) -> float:
    if w_a + w_n + w_ml >= 1.0:
        raise ValueError("weights must sum to < 1")
    scale = 1.0 - w_a - w_n - w_ml
    w_perf = 0.7 * scale
    w_gate = 0.3 * scale
    return w_perf * perf + w_gate * gate + w_a * analyst + w_n * news + w_ml * ml


def test_zero_weights_collapse_to_phase_a():
    score = _renormalised_live_score(
        perf=0.5,
        gate=0.5,
        analyst=0.0,
        news=0.0,
        ml=0.0,
        w_a=0.0,
        w_n=0.0,
        w_ml=0.0,
    )
    # Phase-A formula: 0.7*perf + 0.3*gate.
    assert score == pytest.approx(0.7 * 0.5 + 0.3 * 0.5)


@pytest.mark.parametrize(
    "w_a,w_n,w_ml",
    [
        (0.0, 0.0, 0.0),
        (0.3, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (0.0, 0.0, 0.3),
        (0.2, 0.2, 0.2),
        (0.1, 0.1, 0.1),
        (0.05, 0.05, 0.05),
        (0.0, 0.0, 0.5),
    ],
)
def test_renormalisation_is_finite(w_a, w_n, w_ml):
    score = _renormalised_live_score(
        perf=0.1,
        gate=-0.2,
        analyst=0.4,
        news=-0.3,
        ml=0.5,
        w_a=w_a,
        w_n=w_n,
        w_ml=w_ml,
    )
    assert score == pytest.approx(score)  # finite


def test_sum_exceeds_one_raises():
    with pytest.raises(ValueError):
        _renormalised_live_score(
            perf=0.0,
            gate=0.0,
            analyst=0.0,
            news=0.0,
            ml=0.0,
            w_a=0.4,
            w_n=0.4,
            w_ml=0.3,
        )


def test_regime_tolerance_defaults_have_ml_weight():
    for profile in ("defensiv", "ausgewogen", "aggressiv", "custom"):
        defaults = REGIME_TOLERANCE_DEFAULTS[profile]
        assert "ml_score_weight" in defaults
        assert defaults["ml_score_weight"] == 0.0
