"""Phase D XGBoost forecast adapter (T-0215).

Mirrors :class:`LightGBMForecastAdapter`. The only differences are the
manifest discriminator ``model_family="xgboost"`` and the engine label
that flows into ``ml_model_version``. Attribution falls back to
``contributions_unavailable_reason="xgboost_pred_contribs_not_supported"``
when the installed xgboost version cannot emit ``pred_contribs``
(Codex D24); the base sidecar writer already records the reason.
"""

from __future__ import annotations

from backtest.external_features.adapters.lightgbm_forecast import (
    _BaseMLForecastAdapter,
)


class XGBoostForecastAdapter(_BaseMLForecastAdapter):
    """XGBoost inference adapter (Codex D5/D20/D24)."""

    model_family = "xgboost"
    dataset_id_value = "xgboost_forecast"
    source_name_value = "XGBoostForecast"
    engine_label = "xgboost"
    license_note = (
        "internal XGBoost bundle; pickled artefacts, manifest carries lib_versions"
    )


__all__ = ["XGBoostForecastAdapter"]
