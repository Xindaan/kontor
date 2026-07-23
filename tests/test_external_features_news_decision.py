"""Phase C T-0127 numerical tests for multi-weight score-mix.

Verifies that adding the news component never alters the Phase-A
behaviour when both new weights are zero, and that
``analyst_score_weight + news_score_weight >= 1`` is rejected hard.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from backtest.external_features.loader import ExternalFeatureSnapshot
from backtest.meta_decision import _score_candidate

pytestmark = pytest.mark.no_network


@pytest.fixture
def _mock_score_environment(monkeypatch, tmp_path: Path):
    """Patch the heavy IO inside ``_score_candidate`` so we only test the
    score-mixing logic."""

    strategy = SimpleNamespace(assets=["AAA", "BBB"], rebalance_frequency="monthly")

    def _fake_load_strategy_instance(path, params=None):
        return strategy

    def _fake_yahoo(**kwargs):
        dates = pd.date_range("2024-01-01", periods=400, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(100, 140, len(dates)),
                "BBB": np.linspace(100, 130, len(dates)),
            },
            index=dates,
        )
        return SimpleNamespace(prices=prices)

    class _FakeMetrics:
        total_return = 0.4
        sharpe_ratio = 0.5

    class _FakeBacktester:
        def __init__(self, *args, **kwargs):
            self._equity = pd.Series(
                np.linspace(1.0, 1.2, 400),
                index=pd.date_range("2024-01-01", periods=400, freq="B"),
            )

        def run(self):
            return SimpleNamespace(equity_curve=self._equity, metrics=_FakeMetrics())

    monkeypatch.setattr("backtest.meta_decision.load_strategy_instance", _fake_load_strategy_instance)
    monkeypatch.setattr("backtest.meta_decision.DataLoader.yahoo", staticmethod(_fake_yahoo))
    monkeypatch.setattr("backtest.meta_decision.Backtester", _FakeBacktester)
    monkeypatch.setattr(
        "backtest.meta_decision.assess_regime_from_equity_curve",
        lambda *args, **kwargs: SimpleNamespace(
            bucket="normal", reasons=[], metrics={}, percentiles={}, status="ok"
        ),
    )
    return strategy


class _ConstantScoreProvider:
    """Minimal provider that returns a fixed score per ticker."""

    def __init__(self, dataset_id: str, feature_name: str, scores: Dict[str, float]):
        self._dataset_id = dataset_id
        self._feature_name = feature_name
        self._scores = scores

    def _frame(self, tickers):
        rows = []
        for ticker, score in self._scores.items():
            if tickers and ticker not in tickers:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "release_date": pd.Timestamp("2026-05-13"),
                    "snapshot_ts": pd.Timestamp("2026-05-13"),
                    "feature_name": self._feature_name,
                    "feature_value": float(score),
                    "source": "Mock",
                    "dataset": self._dataset_id,
                }
            )
        return pd.DataFrame(rows)

    def snapshot(self, as_of, tickers=None):
        return ExternalFeatureSnapshot(
            as_of=as_of, dataset=self._dataset_id, data=self._frame(tickers)
        )

    def snapshot_dataset(self, dataset_id, as_of, tickers=None):
        if dataset_id != self._dataset_id:
            return ExternalFeatureSnapshot(
                as_of=as_of, dataset=dataset_id, data=pd.DataFrame()
            )
        return self.snapshot(as_of, tickers=tickers)


def _call(_mock_score_environment, **kwargs):
    return _score_candidate(
        strategy_path="dummy.py",
        params={},
        as_of=date(2026, 5, 13),
        skip_failed=True,
        scoring_mode="hybrid",
        performance_windows_days=[63, 126, 252],
        performance_weights=[0.5, 0.3, 0.2],
        regime_profile="ausgewogen",
        **kwargs,
    )


def test_zero_weights_match_phase_a_formula(_mock_score_environment):
    candidate = _call(_mock_score_environment)
    expected = 0.7 * candidate.performance_score + 0.3 * candidate.gate_score
    assert candidate.live_score == pytest.approx(expected)
    assert candidate.analyst_score == 0.0
    assert candidate.news_score == 0.0


def test_analyst_only_applies_analyst_weight(_mock_score_environment):
    analyst_provider = _ConstantScoreProvider("analyst_ds", "analyst_score", {"AAA": 1.0, "BBB": 0.0})
    candidate = _call(
        _mock_score_environment,
        analyst_provider=analyst_provider,
        analyst_dataset="analyst_ds",
        analyst_score_effective_weight=0.3,
        analyst_evidence_status="pass",
    )
    expected = (
        0.7 * (1.0 - 0.3) * candidate.performance_score
        + 0.3 * (1.0 - 0.3) * candidate.gate_score
        + 0.3 * candidate.analyst_score
    )
    assert candidate.live_score == pytest.approx(expected)
    assert candidate.news_score == 0.0


def test_news_only_applies_news_weight(_mock_score_environment):
    news_provider = _ConstantScoreProvider(
        "news_ds", "news_sentiment_score", {"AAA": 0.5, "BBB": -0.5}
    )
    candidate = _call(
        _mock_score_environment,
        news_provider=news_provider,
        news_dataset="news_ds",
        news_score_effective_weight=0.2,
        news_evidence_status="pass",
    )
    expected = (
        0.7 * (1.0 - 0.2) * candidate.performance_score
        + 0.3 * (1.0 - 0.2) * candidate.gate_score
        + 0.2 * candidate.news_score
    )
    assert candidate.live_score == pytest.approx(expected)
    assert candidate.analyst_score == 0.0


def test_both_active_renormalises_perf_and_gate(_mock_score_environment):
    analyst_provider = _ConstantScoreProvider("a_ds", "analyst_score", {"AAA": 0.6})
    news_provider = _ConstantScoreProvider("n_ds", "news_sentiment_score", {"AAA": 0.3})
    candidate = _call(
        _mock_score_environment,
        analyst_provider=analyst_provider,
        analyst_dataset="a_ds",
        analyst_score_effective_weight=0.3,
        news_provider=news_provider,
        news_dataset="n_ds",
        news_score_effective_weight=0.2,
    )
    expected = (
        0.7 * (1.0 - 0.3 - 0.2) * candidate.performance_score
        + 0.3 * (1.0 - 0.3 - 0.2) * candidate.gate_score
        + 0.3 * candidate.analyst_score
        + 0.2 * candidate.news_score
    )
    assert candidate.live_score == pytest.approx(expected)


def test_weights_summing_to_one_or_more_raise(_mock_score_environment):
    with pytest.raises(ValueError, match="must be < 1.0"):
        _call(
            _mock_score_environment,
            analyst_score_effective_weight=0.6,
            news_score_effective_weight=0.5,
        )
