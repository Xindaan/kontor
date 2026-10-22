"""Phase D + E forecast-model implementations.

Phase E3 (Codex R2.11): factory strings instead of direct imports for
the DL models, so that ``import`` does not fail without `torch`. The
actual class is only loaded on constructor call via
:func:`resolve_factory`.
"""

import importlib
from typing import Any, Type

from backtest.external_features.ml.models.base import (
    ForecastModel,
    MockForecastModel,
)


# Factory strings for the Phase-E DL models (Codex R2.11).
LSTM_FACTORY_PATH = (
    "backtest.external_features.ml.models.lstm_impl:LSTMForecastModel"
)
TRANSFORMER_FACTORY_PATH = (
    "backtest.external_features.ml.models.transformer_impl:TransformerForecastModel"
)


def resolve_factory(path: str) -> Type[Any]:
    """Lazily loads a class via a ``module:Class`` path.

    Phase E3 (Codex R2.11): allows ``import backtest.external_features.
    ml.models`` without `torch` installed; `import torch` only happens
    when `resolve_factory(LSTM_FACTORY_PATH)` is called.
    """

    if ":" not in path:
        raise ValueError(f"factory path must contain ':' separator: {path!r}")
    module_path, class_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


__all__ = [
    "LSTM_FACTORY_PATH",
    "TRANSFORMER_FACTORY_PATH",
    "ForecastModel",
    "MockForecastModel",
    "resolve_factory",
]
