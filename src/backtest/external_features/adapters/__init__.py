"""Adapter registry for external feature pulls.

Concrete adapters register themselves here. Phase A only ships the
deterministic mock adapter; real adapters (Yahoo, Finnhub, ...) arrive
in phase B/C/D.
"""

from __future__ import annotations

from typing import Dict, Iterable

from backtest.external_features.adapters.base import ExternalFeatureAdapter
from backtest.external_features.adapters.finnhub_analyst_actions import (
    FinnhubAnalystActionsAdapter,
)
from backtest.external_features.adapters.finnhub_analyst_current import (
    FinnhubAnalystCurrentAdapter,
)
from backtest.external_features.adapters.finnhub_analyst_trends import (
    FinnhubAnalystTrendsAdapter,
)
from backtest.external_features.adapters.finnhub_news import FinnhubNewsAdapter
from backtest.external_features.adapters.lightgbm_forecast import (
    LightGBMForecastAdapter,
)
from backtest.external_features.adapters.ml_ensemble import (
    EnsembleForecastAdapter,
)
from backtest.external_features.adapters.newsapi_news import NewsAPIAdapter
from backtest.external_features.adapters.synthetic_analyst_pit import (
    SyntheticAnalystPITAdapter,
)
from backtest.external_features.adapters.synthetic_ml_forecast import (
    SyntheticMLForecastAdapter,
)
from backtest.external_features.adapters.synthetic_news_pit import (
    SyntheticNewsPITAdapter,
)
from backtest.external_features.adapters.xgboost_forecast import (
    XGBoostForecastAdapter,
)
from backtest.external_features.adapters.yahoo_analyst_current import (
    YahooAnalystCurrentAdapter,
)
from backtest.external_features.adapters.yahoo_news import YahooNewsAdapter
from backtest.external_features.adapters.mock import MockAnalystAdapter


_REGISTRY: Dict[str, ExternalFeatureAdapter] = {}


def register_adapter(adapter: ExternalFeatureAdapter) -> None:
    _REGISTRY[adapter.dataset_id] = adapter


def get_adapter(dataset_id: str) -> ExternalFeatureAdapter:
    if dataset_id not in _REGISTRY:
        raise KeyError(
            f"no adapter registered for dataset '{dataset_id}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[dataset_id]


def available_datasets() -> Iterable[str]:
    return sorted(_REGISTRY)


def datasets_allowing_empty_tickers() -> Iterable[str]:
    """Datasets where 'features pull' may run without explicit --tickers."""

    return ("mock_analyst",)


register_adapter(MockAnalystAdapter())
# Phase B adapters. Finnhub adapters lazily instantiate their client
# so a missing FINNHUB_API_KEY only fails when the adapter is actually
# used. Registering bare instances keeps `get_adapter` light.
register_adapter(YahooAnalystCurrentAdapter())
register_adapter(FinnhubAnalystCurrentAdapter())
register_adapter(FinnhubAnalystActionsAdapter())
register_adapter(FinnhubAnalystTrendsAdapter())
register_adapter(SyntheticAnalystPITAdapter())
# Phase C news adapters. Constructed lazily so an optional FinBERT install
# is not required just to import the package.
register_adapter(YahooNewsAdapter())
register_adapter(SyntheticNewsPITAdapter())


def _register_optional_finnhub_news() -> None:
    try:
        register_adapter(FinnhubNewsAdapter())
    except Exception:
        # FinnhubClient demands an API key in __init__ — skip registration
        # if env is missing; CLI/tests can instantiate the adapter directly.
        pass


def _register_optional_newsapi() -> None:
    try:
        register_adapter(NewsAPIAdapter())
    except Exception:
        pass


_register_optional_finnhub_news()
_register_optional_newsapi()

# Phase D ML adapters. Lightgbm/XGBoost/Ensemble adapters do NOT require
# the heavy ML extras to be installed at import time. The `with_options`
# factory wires the runtime bundle dir per pull (Codex D5/D20); a
# `pull_snapshot` call that cannot resolve a bundle raises at fetch time
# instead of failing the package import.
register_adapter(LightGBMForecastAdapter())
register_adapter(XGBoostForecastAdapter())
register_adapter(EnsembleForecastAdapter())
register_adapter(SyntheticMLForecastAdapter())


__all__ = [
    "EnsembleForecastAdapter",
    "ExternalFeatureAdapter",
    "FinnhubAnalystActionsAdapter",
    "FinnhubAnalystCurrentAdapter",
    "FinnhubAnalystTrendsAdapter",
    "FinnhubNewsAdapter",
    "LightGBMForecastAdapter",
    "MockAnalystAdapter",
    "NewsAPIAdapter",
    "SyntheticAnalystPITAdapter",
    "SyntheticMLForecastAdapter",
    "SyntheticNewsPITAdapter",
    "XGBoostForecastAdapter",
    "YahooAnalystCurrentAdapter",
    "YahooNewsAdapter",
    "register_adapter",
    "get_adapter",
    "available_datasets",
    "datasets_allowing_empty_tickers",
]
