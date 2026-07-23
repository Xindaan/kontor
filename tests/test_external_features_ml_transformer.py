"""Phase E3 — Transformer-Model-Tests (T-0345, Codex R3.13)."""

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
def test_transformer_fit_predict_smoke():
    from backtest.external_features.ml.models.transformer_impl import (
        TransformerForecastModel,
    )

    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, (40, 20, 4)).astype(np.float32)
    y = rng.normal(0.0, 0.1, 40).astype(np.float32)

    model = TransformerForecastModel(
        epochs=1, d_model=16, nhead=2, num_layers=1, seed=42
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (40,)
    assert np.isfinite(preds).all()


@pytest.mark.dl_extras
@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_transformer_predict_deterministic():
    from backtest.external_features.ml.models.transformer_impl import (
        TransformerForecastModel,
    )

    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, (10, 20, 4)).astype(np.float32)
    y = rng.normal(0.0, 0.1, 10).astype(np.float32)
    model = TransformerForecastModel(
        epochs=1, d_model=16, nhead=2, num_layers=1, seed=42
    )
    model.fit(X, y)
    a = model.predict(X)
    b = model.predict(X)
    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.dl_extras
@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_transformer_engine_code():
    from backtest.external_features.ml.models.transformer_impl import (
        TransformerForecastModel,
    )

    model = TransformerForecastModel(d_model=16, nhead=2, num_layers=1, seed=42)
    code = model.engine_code
    assert code.startswith("transformer@")
