"""Tests for the news-schema helpers (T-0103)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest.external_features.news_schema import (
    REQUIRED_SCORE_FEATURE,
    hash_headlines_ndjson,
    hash_headlines_rows,
    read_headlines_ndjson,
    validate_news_snapshot,
    write_headlines_ndjson,
)

pytestmark = pytest.mark.no_network


def _frame(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "release_date": pd.Timestamp("2026-05-13"),
                "snapshot_ts": pd.Timestamp("2026-05-13T12:00:00Z"),
                "feature_name": REQUIRED_SCORE_FEATURE,
                "feature_value": 0.1,
                "source": "MockNews",
                "dataset": "test",
            }
        ]
    )
    for key, value in overrides.items():
        base[key] = value
    return base


def test_validate_passes_for_well_formed_frame():
    validate_news_snapshot(_frame())


def test_validate_requires_score_row_per_ticker():
    df = pd.DataFrame(
        [
            {
                "ticker": "BBB",
                "release_date": pd.Timestamp("2026-05-13"),
                "snapshot_ts": pd.Timestamp("2026-05-13"),
                "feature_name": "news_article_count",
                "feature_value": 5,
                "source": "MockNews",
                "dataset": "test",
            }
        ]
    )
    with pytest.raises(ValueError, match="news_sentiment_score"):
        validate_news_snapshot(df)


def test_validate_rejects_out_of_range_score():
    df = _frame()
    df.loc[df["feature_name"] == REQUIRED_SCORE_FEATURE, "feature_value"] = 1.5
    with pytest.raises(ValueError, match="\\[-1, 1\\]"):
        validate_news_snapshot(df)


def test_validate_rejects_negative_counts():
    score_row = _frame().iloc[0].to_dict()
    count_row = {
        **score_row,
        "feature_name": "news_article_count",
        "feature_value": -1,
    }
    df = pd.DataFrame([score_row, count_row])
    with pytest.raises(ValueError, match="news_article_count"):
        validate_news_snapshot(df)


def test_sidecar_round_trip(tmp_path: Path):
    rows = [
        {"ticker": "AAA", "headline": "first", "engine_score": 0.1},
        {"ticker": "AAA", "headline": "second", "engine_score": -0.2},
    ]
    path = tmp_path / "2026-05-13.headlines.ndjson"
    write_headlines_ndjson(path, rows)
    assert path.exists()
    parsed = read_headlines_ndjson(path)
    assert parsed == rows
    assert hash_headlines_ndjson(path) == hash_headlines_rows(rows)


def test_sidecar_hash_is_deterministic_across_orderings(tmp_path: Path):
    rows_a = [{"ticker": "AAA", "engine_score": 0.1}, {"ticker": "BBB", "engine_score": 0.2}]
    rows_b = list(rows_a)
    path_a = tmp_path / "a.ndjson"
    path_b = tmp_path / "b.ndjson"
    write_headlines_ndjson(path_a, rows_a)
    write_headlines_ndjson(path_b, rows_b)
    assert hash_headlines_ndjson(path_a) == hash_headlines_ndjson(path_b)
