"""Tests for the Phase C sentiment engine layer (T-0102).

VADER + FinBERT must never trigger a runtime download; tests rely on
the deterministic :class:`MockSentimentEngine` and only patch the
import-error path to verify the setup-hint messaging.
"""

from __future__ import annotations

import pytest

from backtest.external_features.sentiment import (
    MockSentimentEngine,
    SentimentEngine,
    get_sentiment_engine,
)

pytestmark = pytest.mark.no_network


def test_mock_engine_returns_deterministic_score():
    engine = MockSentimentEngine()
    assert isinstance(engine, SentimentEngine)
    assert -1.0 <= engine.score("good news for AAPL") <= 1.0
    # Same input -> same score across runs.
    a = engine.score("Apple beats earnings")
    b = engine.score("Apple beats earnings")
    assert a == b


def test_mock_engine_handles_empty_strings():
    engine = MockSentimentEngine()
    assert engine.score("") == 0.0
    assert engine.score(None) == 0.0  # type: ignore[arg-type]


def test_engine_codes_are_stable_strings():
    engine = MockSentimentEngine()
    assert engine.engine_code == "mock@1.0"
    assert isinstance(engine.engine_version, int)


def test_get_sentiment_engine_factory_returns_mock():
    engine = get_sentiment_engine("mock")
    assert isinstance(engine, MockSentimentEngine)


def test_get_sentiment_engine_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_sentiment_engine("does_not_exist")


def test_vader_engine_raises_runtime_error_without_lexicon(monkeypatch):
    """VaderSentimentEngine must raise *before* attempting a runtime
    download when the lexicon is unavailable."""

    nltk = pytest.importorskip("nltk")

    def _missing(*args, **kwargs):
        raise LookupError("vader_lexicon missing")

    monkeypatch.setattr(nltk.data, "find", _missing)
    from backtest.external_features.sentiment import VaderSentimentEngine

    with pytest.raises(RuntimeError, match="vader_lexicon"):
        VaderSentimentEngine()


def test_score_many_loops_over_input():
    engine = MockSentimentEngine()
    out = engine.score_many(["aaa", "bbb", ""])
    assert len(out) == 3
    assert out[-1] == 0.0
