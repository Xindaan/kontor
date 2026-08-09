"""Tests for sector_regime_aggregation.aggregate_sector_stress."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.external_features.ml.sector_regime_aggregation import (
    AGGREGATION_MODES,
    DEFAULT_MODE,
    aggregate_sector_stress,
)


def test_default_mode_is_std():
    assert DEFAULT_MODE == "std"


def test_all_modes_listed():
    expected = {"mean", "std", "quantile_75", "quantile_95", "max", "min"}
    assert set(AGGREGATION_MODES) == expected


def test_mean_matches_numpy():
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    out = aggregate_sector_stress(probs, mode="mean")
    assert out == pytest.approx(np.mean(probs))


def test_std_matches_numpy_ddof0():
    probs = [0.0, 0.5, 0.5, 0.0, 0.5]
    out = aggregate_sector_stress(probs, mode="std")
    assert out == pytest.approx(np.std(probs, ddof=0))


def test_quantile_75_matches_numpy():
    probs = list(np.linspace(0.0, 1.0, 21))
    out = aggregate_sector_stress(probs, mode="quantile_75")
    assert out == pytest.approx(np.quantile(probs, 0.75))


def test_quantile_95_max_min():
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.95, 0.99, 0.05]
    assert aggregate_sector_stress(probs, mode="max") == pytest.approx(0.99)
    assert aggregate_sector_stress(probs, mode="min") == pytest.approx(0.05)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        aggregate_sector_stress([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], mode="bogus")


def test_returns_none_when_too_few_sectors():
    out = aggregate_sector_stress([0.1, 0.2], mode="std", min_sectors=5)
    assert out is None


def test_min_sectors_threshold_lower():
    out = aggregate_sector_stress([0.1, 0.2], mode="std", min_sectors=2)
    assert out is not None and out == pytest.approx(np.std([0.1, 0.2], ddof=0))


def test_accepts_dict_input():
    d = {"XLK": 0.2, "XLF": 0.5, "XLY": 0.1, "XLV": 0.4, "XLE": 0.8, "XLI": 0.3}
    out = aggregate_sector_stress(d, mode="mean")
    assert out == pytest.approx(np.mean(list(d.values())))


def test_accepts_series_input():
    s = pd.Series({"XLK": 0.2, "XLF": 0.5, "XLY": 0.1, "XLV": 0.4, "XLE": 0.8, "XLI": 0.3})
    out = aggregate_sector_stress(s, mode="mean")
    assert out == pytest.approx(np.mean(s.values))


def test_clips_out_of_range_values():
    """Values > 1.0 or < 0.0 are clipped (no probability range violation)."""
    probs = [0.5, 1.2, -0.3, 0.4, 0.6, 0.7]
    out = aggregate_sector_stress(probs, mode="max")
    assert out == pytest.approx(1.0)
    out = aggregate_sector_stress(probs, mode="min")
    assert out == pytest.approx(0.0)


def test_ignores_non_finite_values():
    probs = [0.1, 0.2, float("nan"), 0.4, None, 0.5, float("inf"), 0.6]
    out = aggregate_sector_stress(probs, mode="mean")
    # NaN, None, Inf are removed before aggregation.
    expected = float(np.mean([0.1, 0.2, 0.4, 0.5, 0.6]))
    assert out == pytest.approx(expected)


def test_pit_function_is_pure_no_external_state():
    """Function is purist — same input -> always same output, no state."""
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    out1 = aggregate_sector_stress(probs, mode="std")
    out2 = aggregate_sector_stress(probs, mode="std")
    out3 = aggregate_sector_stress(probs, mode="std")
    assert out1 == out2 == out3
    # Order invariance (aggregate over a set, not a sequence):
    out_shuffled = aggregate_sector_stress(list(reversed(probs)), mode="std")
    assert out1 == pytest.approx(out_shuffled)
