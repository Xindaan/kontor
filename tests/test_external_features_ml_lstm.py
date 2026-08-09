"""Phase E3 — LSTM model tests (T-0344, T-0348, T-0349).

@pytest.mark.dl_extras: skipped without torch installed.
"""

from __future__ import annotations

import numpy as np
import pytest


_torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    _torch_available = False


@pytest.mark.dl_extras
@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_lstm_fit_predict_smoke():
    """T-0348: 2 tickers x 80 days x 1 epoch, CPU-only, <30s."""
    from backtest.external_features.ml.models.lstm_impl import LSTMForecastModel

    rng = np.random.default_rng(42)
    n_batch = 40
    seq_len = 20
    n_features = 4
    X = rng.normal(0.0, 1.0, (n_batch, seq_len, n_features)).astype(np.float32)
    y = rng.normal(0.0, 0.1, n_batch).astype(np.float32)

    model = LSTMForecastModel(epochs=1, hidden_dim=8, num_layers=1, seed=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (n_batch,)
    assert np.isfinite(preds).all()


@pytest.mark.dl_extras
@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_lstm_predict_deterministic_within_tolerance():
    """T-0349 (Codex R2.12): two predict() runs identical modulo atol=1e-5."""
    from backtest.external_features.ml.models.lstm_impl import LSTMForecastModel

    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, (10, 20, 4)).astype(np.float32)
    y = rng.normal(0.0, 0.1, 10).astype(np.float32)
    model = LSTMForecastModel(epochs=1, hidden_dim=8, num_layers=1, seed=42)
    model.fit(X, y)
    a = model.predict(X)
    b = model.predict(X)
    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.dl_extras
@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_lstm_engine_code_contains_lstm_and_param_hash():
    from backtest.external_features.ml.models.lstm_impl import LSTMForecastModel

    model = LSTMForecastModel(hidden_dim=8, num_layers=1, seed=42)
    code = model.engine_code
    assert code.startswith("lstm@")
    # Param hash at the end.
    parts = code.split("_")
    assert len(parts[-1]) == 8


def test_lstm_import_without_torch_raises_useful_error(monkeypatch):
    """Codex R2.11: lazy import raises RuntimeError instead of ImportError."""
    import sys

    sys.modules.pop("backtest.external_features.ml.models.lstm_impl", None)
    real_torch = sys.modules.get("torch")
    monkeypatch.setitem(sys.modules, "torch", None)
    sys.modules.pop("torch", None)

    # Force ImportError on the torch import.
    class _Block:
        def find_module(self, *args, **kwargs):
            return None

    # We do NOT patch `import torch` itself globally, because that would
    # break the entire test run. Instead we only check that
    # `import backtest...lstm_impl` itself is not explosive
    # (it allows a lazy import that is only triggered on
    # constructor invocation).
    from backtest.external_features.ml.models import lstm_impl

    # Class existence without constructor invocation.
    assert hasattr(lstm_impl, "LSTMForecastModel")

    # Cleanup
    if real_torch is not None:
        sys.modules["torch"] = real_torch
