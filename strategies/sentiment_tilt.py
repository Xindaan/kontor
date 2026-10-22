"""Demo strategy that tilts an equal-weight allocation by news sentiment.

Phase C T-0124. The strategy:

- Reads ``news_sentiment_score`` for its universe from the optional
  ``_external_features_provider`` slot. When the provider is missing or
  the snapshot is empty, the strategy falls back to equal weights.
- Picks the top-N tickers whose score exceeds ``score_cutoff`` (default
  0.1) and equal-weights them. When fewer than ``min_picks`` tickers
  qualify, the strategy falls back to equal-weight on the full universe
  to stay invested.

Defaults are intentionally conservative so the strategy is usable both
as a CLI smoke target and as a meta-decision candidate.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd

from backtest.strategy import Allocation, Strategy


class SentimentTilt(Strategy):
    name = "Sentiment Tilt"
    assets: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    rebalance_frequency = "monthly"

    def __init__(
        self,
        top_n: int = 3,
        score_cutoff: float = 0.10,
        min_picks: int = 2,
        news_dataset: str = "synthetic_news_pit",
    ) -> None:
        self.top_n = int(top_n)
        self.score_cutoff = float(score_cutoff)
        self.min_picks = int(min_picks)
        self.news_dataset = str(news_dataset)
        self.params = {
            "top_n": self.top_n,
            "score_cutoff": self.score_cutoff,
            "min_picks": self.min_picks,
            "news_dataset": self.news_dataset,
        }

    def _read_sentiment(self, current_date: date) -> Dict[str, float]:
        provider = getattr(self, "_external_features_provider", None)
        if provider is None:
            return {}
        try:
            if hasattr(provider, "snapshot_dataset"):
                snap = provider.snapshot_dataset(
                    self.news_dataset,
                    as_of=current_date,
                    tickers=list(self.assets),
                )
            else:
                snap = provider.snapshot(as_of=current_date, tickers=list(self.assets))
        except Exception:
            return {}
        if snap is None or getattr(snap, "data", None) is None:
            return {}
        df = snap.data
        if df.empty or "feature_name" not in df.columns:
            return {}
        rows = df.loc[df["feature_name"] == "news_sentiment_score"]
        scores: Dict[str, float] = {}
        for ticker, group in rows.groupby("ticker"):
            try:
                scores[str(ticker).upper()] = float(group["feature_value"].iloc[-1])
            except (TypeError, ValueError):
                continue
        return scores

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        scores = self._read_sentiment(current_date)
        eligible = [
            (ticker, scores.get(ticker.upper(), 0.0))
            for ticker in self.assets
            if scores.get(ticker.upper(), 0.0) >= self.score_cutoff
        ]
        eligible.sort(key=lambda kv: kv[1], reverse=True)
        picks = [ticker for ticker, _ in eligible[: self.top_n]]
        if len(picks) < self.min_picks:
            picks = list(self.assets)
        if not picks:
            return Allocation({})
        weight = 1.0 / len(picks)
        return Allocation({ticker: weight for ticker in picks})


strategy = SentimentTilt()
