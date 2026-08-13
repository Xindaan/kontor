"""Tests for FinnhubClient (Phase B T-0053).

All tests must be netzwerklos (``@pytest.mark.no_network``). Every HTTP
call is intercepted via monkeypatch on ``requests.get``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import requests

from backtest.external_features.adapters.finnhub_client import (
    FINNHUB_API_KEY_ENV,
    FinnhubAPIError,
    FinnhubClient,
)


pytestmark = pytest.mark.no_network


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or "<no body>"

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _install_fake_get(monkeypatch, responses: List[_FakeResponse]):
    calls: List[Dict[str, Any]] = []

    def _fake_get(url, params=None, timeout=None):  # noqa: ANN001
        calls.append({"url": url, "params": params, "timeout": timeout})
        if not responses:
            raise AssertionError("unexpected extra HTTP call")
        return responses.pop(0)

    monkeypatch.setattr(
        "backtest.external_features.adapters.finnhub_client.requests.get",
        _fake_get,
    )
    return calls


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv(FINNHUB_API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError):
        FinnhubClient()


def test_env_api_key_used(monkeypatch):
    monkeypatch.setenv(FINNHUB_API_KEY_ENV, "envkey123")
    calls = _install_fake_get(monkeypatch, [_FakeResponse(payload={"ok": True})])
    client = FinnhubClient(throttle_seconds=0.0)
    client.get("stock/recommendation", {"symbol": "AAPL"})
    assert calls[0]["params"]["token"] == "envkey123"
    assert calls[0]["params"]["symbol"] == "AAPL"


def test_constructor_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv(FINNHUB_API_KEY_ENV, "envkey")
    calls = _install_fake_get(monkeypatch, [_FakeResponse(payload={})])
    client = FinnhubClient(api_key="explicit", throttle_seconds=0.0)
    client.get("stock/price-target", {"symbol": "AAPL"})
    assert calls[0]["params"]["token"] == "explicit"


def test_recommendation_trends_calls_correct_endpoint(monkeypatch):
    calls = _install_fake_get(monkeypatch, [_FakeResponse(payload=[{"period": "2026-04-01"}])])
    client = FinnhubClient(api_key="k", throttle_seconds=0.0)
    out = client.recommendation_trends("MSFT")
    assert out == [{"period": "2026-04-01"}]
    assert calls[0]["url"].endswith("/stock/recommendation")
    assert calls[0]["params"]["symbol"] == "MSFT"


def test_upgrade_downgrade_passes_date_range(monkeypatch):
    calls = _install_fake_get(monkeypatch, [_FakeResponse(payload=[])])
    client = FinnhubClient(api_key="k", throttle_seconds=0.0)
    client.upgrade_downgrade("AAPL", "2024-01-01", "2024-06-01")
    assert calls[0]["params"]["from"] == "2024-01-01"
    assert calls[0]["params"]["to"] == "2024-06-01"


def test_http_error_raises(monkeypatch):
    _install_fake_get(monkeypatch, [_FakeResponse(status_code=403, text="forbidden")])
    client = FinnhubClient(api_key="k", throttle_seconds=0.0, max_retries=1)
    with pytest.raises(FinnhubAPIError):
        client.get("stock/recommendation", {"symbol": "AAPL"})


def test_retries_on_429_then_success(monkeypatch):
    calls = _install_fake_get(
        monkeypatch,
        [
            _FakeResponse(status_code=429, text="rate limit"),
            _FakeResponse(payload={"ok": True}),
        ],
    )
    client = FinnhubClient(api_key="k", throttle_seconds=0.0, max_retries=3)
    # Replace time.sleep so the test does not actually sleep.
    monkeypatch.setattr(
        "backtest.external_features.adapters.finnhub_client.time.sleep",
        lambda *args, **kwargs: None,
    )
    out = client.get("stock/recommendation", {"symbol": "AAPL"})
    assert out == {"ok": True}
    assert len(calls) == 2


def test_retries_on_transport_error_then_success(monkeypatch):
    responses = [_FakeResponse(payload={"ok": True})]
    call_count = {"value": 0}

    def _fake_get(url, params=None, timeout=None):  # noqa: ANN001
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise requests.ConnectionError("boom")
        return responses.pop(0)

    monkeypatch.setattr(
        "backtest.external_features.adapters.finnhub_client.requests.get",
        _fake_get,
    )
    monkeypatch.setattr(
        "backtest.external_features.adapters.finnhub_client.time.sleep",
        lambda *args, **kwargs: None,
    )
    client = FinnhubClient(api_key="k", throttle_seconds=0.0, max_retries=3)
    out = client.get("stock/recommendation", {"symbol": "AAPL"})
    assert out == {"ok": True}
    assert call_count["value"] == 2


def test_retries_exhausted_raises(monkeypatch):
    _install_fake_get(
        monkeypatch,
        [
            _FakeResponse(status_code=503, text="down"),
            _FakeResponse(status_code=503, text="down"),
            _FakeResponse(status_code=503, text="down"),
        ],
    )
    monkeypatch.setattr(
        "backtest.external_features.adapters.finnhub_client.time.sleep",
        lambda *args, **kwargs: None,
    )
    client = FinnhubClient(api_key="k", throttle_seconds=0.0, max_retries=3)
    with pytest.raises(FinnhubAPIError):
        client.get("stock/recommendation", {"symbol": "AAPL"})
