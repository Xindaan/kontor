from datetime import date

import pandas as pd
import pytest

from backtest.data import PriceData
from backtest.signals import Portfolio, SignalGenerator, SignalReport, format_signal_report
from backtest.strategy import Allocation, Strategy


class _TargetWeightsStrategy(Strategy):
    name = "Target Weights"
    assets = ["AAA", "CCC"]
    rebalance_frequency = "monthly"

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        _ = (current_date, data)
        return Allocation({"AAA": 0.5, "CCC": 0.3})


class _A4Strategy(Strategy):
    name = "A4 Strategy"
    assets = ["3SEM.L"]
    rebalance_frequency = "daily"

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        _ = (current_date, data)
        return Allocation({"3SEM.L": 1.0})


def _build_price_data(tickers):
    idx = pd.to_datetime(["2024-01-09", "2024-01-10"])
    base = {
        "AAA": [19.0, 20.0],
        "BBB": [39.0, 40.0],
        "CCC": [9.5, 10.0],
    }
    prices = pd.DataFrame(
        {ticker: base.get(ticker, [10.0, 10.0]) for ticker in tickers},
        index=idx,
    )
    return PriceData(
        prices=prices,
        currency={ticker: "USD" for ticker in tickers},
    )


def _build_exposure_price_data(tickers):
    idx = pd.bdate_range("2026-01-01", periods=35)
    raw = [100.0] * 29 + [100.0, 96.0, 92.0, 88.0, 84.0, 80.0]
    base = {
        "3SEM.L": raw,
        "VVSM.DE": [100.0 + i * 0.2 for i in range(35)],
        "SXR8.DE": [100.0 + i * 0.1 for i in range(35)],
    }
    prices = pd.DataFrame(
        {ticker: base.get(ticker, [100.0] * len(idx)) for ticker in tickers},
        index=idx,
    )
    return PriceData(prices=prices, currency={ticker: "USD" for ticker in tickers})


class _BrokerPositionsStrategy(Strategy):
    name = "Broker Positions"
    assets = ["GLEN.L", "AMD"]
    rebalance_frequency = "daily"

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        _ = (current_date, data)
        return Allocation({"GLEN.L": 0.5, "AMD": 0.5})


def test_generate_signals_builds_order_proposals_and_drift(monkeypatch):
    seen = {}

    def fake_yahoo(
        tickers,
        start,
        end=None,
        currency="EUR",
        align="ffill",
        skip_failed=False,
    ):
        _ = (start, end, currency, align, skip_failed)
        seen["tickers"] = list(tickers)
        return _build_price_data(tickers)

    monkeypatch.setattr("backtest.data.DataLoader.yahoo", fake_yahoo)

    portfolio = Portfolio(positions={"AAA": 10.0, "BBB": 5.0}, cash=100.0)
    report = SignalGenerator(
        strategy=_TargetWeightsStrategy(),
        portfolio=portfolio,
    ).generate(as_of=date(2024, 1, 10), lookback_days=30)

    assert set(seen["tickers"]) == {"AAA", "BBB", "CCC"}
    assert report.portfolio_value == pytest.approx(500.0)
    assert report.current_cash_weight == pytest.approx(0.2)
    assert report.target_cash_weight == pytest.approx(0.2)
    assert report.missing_prices == []

    by_ticker = {signal.ticker: signal for signal in report.signals}
    assert set(by_ticker.keys()) == {"AAA", "BBB", "CCC"}

    aaa = by_ticker["AAA"]
    assert aaa.action == "HOLD"
    assert aaa.order_action == "BUY"
    assert aaa.current_shares == pytest.approx(10.0)
    assert aaa.target_shares == pytest.approx(12.5)
    assert aaa.shares_delta == pytest.approx(2.5)
    assert aaa.value_delta == pytest.approx(50.0)
    assert aaa.drift_bps == pytest.approx(1000.0)
    assert aaa.drift_in_tolerance is False

    bbb = by_ticker["BBB"]
    assert bbb.action == "SELL"
    assert bbb.order_action == "SELL"
    assert bbb.target_shares == pytest.approx(0.0)
    assert bbb.shares_delta == pytest.approx(-5.0)
    assert bbb.value_delta == pytest.approx(-200.0)

    ccc = by_ticker["CCC"]
    assert ccc.action == "BUY"
    assert ccc.order_action == "BUY"
    assert ccc.current_shares == pytest.approx(0.0)
    assert ccc.target_shares == pytest.approx(15.0)
    assert ccc.shares_delta == pytest.approx(15.0)
    assert ccc.value_delta == pytest.approx(150.0)

    payload = report.to_dict()
    assert payload["summary"]["orders"]["buy"] == 2
    assert payload["summary"]["orders"]["sell"] == 1
    assert payload["summary"]["orders"]["actionable"] == 3
    assert payload["summary"]["drift"]["out_of_tolerance"] == 3
    assert len(payload["orders"]) == 3


def test_portfolio_from_json_reads_manual_positionen_format(tmp_path):
    portfolio_path = tmp_path / "portfolio_tr.json"
    portfolio_path.write_text(
        """
{
  "broker": "Trade Republic",
  "stand": "2026-06-06",
  "positionen": [
    {"name": "S&P 500", "wkn": "A0YEDG", "price_ticker": "SXR8.DE", "waehrung": "EUR", "shares": 10.0, "rolle": "safe_sp500"},
    {"name": "Semi 3x", "wkn": "A4ANZ5", "price_ticker": "3SEM.L", "waehrung": "EUR", "shares": 25.0, "rolle": "risk_3x_semi"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    portfolio = Portfolio.from_json(str(portfolio_path))

    assert portfolio.positions == {
        "SXR8.DE": pytest.approx(10.0),
        "3SEM.L": pytest.approx(25.0),
    }
    assert portfolio.cash == pytest.approx(0.0)
    assert portfolio.position_metadata["3SEM.L"]["rolle"] == "risk_3x_semi"


def test_generate_signals_values_manual_positions_in_eur(monkeypatch, tmp_path):
    portfolio_path = tmp_path / "portfolio_db.json"
    portfolio_path.write_text(
        """
{
  "broker": "Deutsche Bank / maxblue",
  "cash": 100.0,
  "positionen": [
    {"name": "Glencore", "price_ticker": "GLEN.L", "waehrung": "GBp", "shares": 100, "rolle": "commod_verkaufen"},
    {"name": "AMD", "price_ticker": "AMD", "waehrung": "USD", "shares": 20, "rolle": "satellite_semi", "ticker_verify": true}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    def fake_yahoo(
        tickers,
        start,
        end=None,
        currency="EUR",
        align="ffill",
        skip_failed=False,
    ):
        _ = (tickers, start, end, currency, align, skip_failed)
        idx = pd.to_datetime(["2026-06-05", "2026-06-08"])
        prices = pd.DataFrame(
            {
                "GLEN.L": [245.0, 250.0],  # GBp -> 2.50 GBP -> EUR
                "AMD": [118.0, 120.0],     # USD -> EUR
            },
            index=idx,
        )
        fx = pd.DataFrame({"GBP": [0.85, 0.85], "USD": [1.10, 1.10]}, index=idx)
        return PriceData(
            prices=prices,
            currency={"GLEN.L": "GBP", "AMD": "USD"},
            fx_rates=fx,
        )

    monkeypatch.setattr("backtest.data.DataLoader.yahoo", fake_yahoo)

    portfolio = Portfolio.from_json(str(portfolio_path))
    report = SignalGenerator(
        strategy=_BrokerPositionsStrategy(),
        portfolio=portfolio,
    ).generate(as_of=date(2026, 6, 8), lookback_days=5)

    by_ticker = {signal.ticker: signal for signal in report.signals}
    assert by_ticker["GLEN.L"].current_value == pytest.approx((250.0 / 100.0 / 0.85) * 100)
    assert by_ticker["AMD"].current_value == pytest.approx((120.0 / 1.10) * 20)
    assert report.portfolio_value == pytest.approx(
        100.0 + by_ticker["GLEN.L"].current_value + by_ticker["AMD"].current_value
    )
    assert report.price_warnings == []


def test_price_scale_and_currency_follow_detected_line_not_l_suffix():
    """Live valuation must use the per-ticker currency line, not ".L"==pence.

    The LSE lists USD/GBP/pence lines of the same ETP. Blindly treating every ".L"
    ticker as pence mispriced the USD-denominated lines by ~74x and inflated
    the weekly drift into a bogus BUY. Scale/currency now follow
    assets.detect_currency, with an explicit pence marker on the row as override.
    """
    portfolio = Portfolio(
        positions={"3SEM.L": 1.0, "QQQ3.L": 1.0, "3LUS.L": 1.0, "GLEN.L": 1.0},
        cash=0.0,
        position_metadata={
            # Real portfolio JSON convention: waehrung="EUR" is aspirational;
            # the USD-denominated LSE ETP line actually prints in USD.
            "3SEM.L": {"waehrung": "EUR"},
            "QQQ3.L": {},
            "3LUS.L": {},
            # detect_currency resolves GLEN.L to GBP via the suffix, so the pence
            # nature must come from the row marker.
            "GLEN.L": {"waehrung": "GBp"},
        },
    )

    # USD lines: no pence scaling, loader currency (USD) stands.
    for usd_line in ("3SEM.L", "QQQ3.L"):
        assert portfolio.price_scale_for(usd_line) == 1.0
        assert portfolio.currency_override_for(usd_line) is None

    # Genuine pence lines: divide to GBP, report GBP for the single FX pass.
    for pence_line in ("3LUS.L", "GLEN.L"):
        assert portfolio.price_scale_for(pence_line) == 0.01
        assert portfolio.currency_override_for(pence_line) == "GBP"


def test_generate_signals_values_3sem_l_at_consensus_eur(monkeypatch, tmp_path):
    """End-to-end: a USD-denominated LSE ETP line must value at ~USD/EURUSD, not /74.

    Before the fix, signals.py forced ".L" -> pence(GBP): raw 165.67 USD became
    1.6567 "GBP" and then /GBP-FX, ~74x too low. The consensus value is USD->EUR.
    """
    portfolio_path = tmp_path / "portfolio_tr.json"
    portfolio_path.write_text(
        """
{
  "broker": "Trade Republic",
  "cash": 0.0,
  "positionen": [
    {"name": "Semi 3x", "wkn": "A4ANZ5", "price_ticker": "3SEM.L", "waehrung": "EUR", "shares": 25.0, "rolle": "risk_3x_semi"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    def fake_yahoo(tickers, start, end=None, currency="EUR", align="ffill", skip_failed=False):
        _ = (tickers, start, end, currency, align, skip_failed)
        idx = pd.to_datetime(["2026-07-10", "2026-07-13"])
        prices = pd.DataFrame({"3SEM.L": [160.0, 165.67]}, index=idx)
        # The real DataLoader labels the USD-denominated line via detect_currency.
        fx = pd.DataFrame({"USD": [1.15, 1.15]}, index=idx)
        return PriceData(prices=prices, currency={"3SEM.L": "USD"}, fx_rates=fx)

    monkeypatch.setattr("backtest.data.DataLoader.yahoo", fake_yahoo)

    portfolio = Portfolio.from_json(str(portfolio_path))
    report = SignalGenerator(
        strategy=_A4Strategy(),
        portfolio=portfolio,
    ).generate(as_of=date(2026, 7, 13), lookback_days=5)

    by_ticker = {signal.ticker: signal for signal in report.signals}
    expected_eur = (165.67 / 1.15) * 25.0  # converted EUR value at the ETP line's raw USD price
    assert by_ticker["3SEM.L"].current_value == pytest.approx(expected_eur)
    # Sanity: the old pence path would have produced ~1/74 of this.
    assert by_ticker["3SEM.L"].current_value > expected_eur * 0.5
    assert report.portfolio_value == pytest.approx(expected_eur)


def test_format_signal_report_includes_orders_and_drift_sections(monkeypatch):
    def fake_yahoo(
        tickers,
        start,
        end=None,
        currency="EUR",
        align="ffill",
        skip_failed=False,
    ):
        _ = (start, end, currency, align, skip_failed)
        return _build_price_data(tickers)

    monkeypatch.setattr("backtest.data.DataLoader.yahoo", fake_yahoo)

    report = SignalGenerator(
        strategy=_TargetWeightsStrategy(),
        portfolio=Portfolio(positions={"AAA": 10.0, "BBB": 5.0}, cash=100.0),
    ).generate(as_of=date(2024, 1, 10), lookback_days=30)
    rendered = format_signal_report(report)

    assert "ORDER PROPOSALS" in rendered
    assert "DRIFT RECONCILIATION" in rendered
    assert "AAA" in rendered
    assert "BBB" in rendered
    assert "CCC" in rendered


def test_generate_without_portfolio_context_leaves_orders_unsized(monkeypatch):
    def fake_yahoo(
        tickers,
        start,
        end=None,
        currency="EUR",
        align="ffill",
        skip_failed=False,
    ):
        _ = (start, end, currency, align, skip_failed)
        return _build_price_data(tickers)

    monkeypatch.setattr("backtest.data.DataLoader.yahoo", fake_yahoo)

    report = SignalGenerator(strategy=_TargetWeightsStrategy()).generate(
        as_of=date(2024, 1, 10),
        lookback_days=30,
    )

    assert report.portfolio_value is None
    assert report.current_cash_weight is None
    assert len(report.actionable_orders) == 0
    assert all(signal.shares_delta is None for signal in report.signals)

    rendered = format_signal_report(report)
    assert "ORDER PROPOSALS" not in rendered


def test_signal_report_serializes_meta_decision_block():
    report = SignalReport(
        as_of=date(2026, 2, 17),
        strategy_name="Demo",
        strategy_params={},
        universe_size=1,
        signals=[],
        meta_decision={
            "current_strategy": "a.py",
            "recommended_target": "b.py",
            "score_margin": 0.2,
            "performance_gap": 0.15,
            "evidence_status": "pass",
            "conditioned_evidence_status": "pass",
            "conditioned_windows": 5,
            "decision_rule": "fragility_driven",
            "current_regime_bucket": "stressed",
            "target_regime_bucket": "normal",
            "current_regime_reasons": ["current weak"],
            "target_regime_reasons": ["target stable"],
            "evidence_artifact_id": "artifact-123",
            "switch_allowed": True,
            "executed_action": "switch_to_target",
            "evidence_reasons": [],
            "live_reasons": [],
        },
    )

    payload = report.to_dict()
    assert payload["meta_decision"]["evidence_status"] == "pass"
    assert payload["meta_decision"]["switch_allowed"] is True
    assert payload["meta_decision"]["decision_rule"] == "fragility_driven"

    rendered = format_signal_report(report)
    assert "META DECISION" in rendered
    assert "artifact-123" in rendered
    assert "fragility_driven" in rendered


def test_generate_signals_uses_exposure_policy_adjusted_target(monkeypatch):
    seen = {}

    def fake_yahoo(
        tickers,
        start,
        end=None,
        currency="EUR",
        align="ffill",
        skip_failed=False,
    ):
        _ = (start, end, currency, align, skip_failed)
        seen["tickers"] = list(tickers)
        return _build_exposure_price_data(tickers)

    monkeypatch.setattr("backtest.data.DataLoader.yahoo", fake_yahoo)

    report = SignalGenerator(
        strategy=_A4Strategy(),
        portfolio=Portfolio(cash=10_000.0),
        exposure_policy={
            "enabled": True,
            "profile": "trade_republic",
            "core_asset": "A0YEDG",
        },
    ).generate(as_of=date(2026, 2, 18), lookback_days=60)

    assert {"3SEM.L", "VVSM.DE", "SXR8.DE"}.issubset(set(seen["tickers"]))
    assert report.exposure_policy is not None
    assert report.exposure_policy["raw_strategy_target"] == {"3SEM.L": 1.0}
    assert report.exposure_policy["policy_adjusted_target"] == {"VVSM.DE": 1.0}
    by_ticker = {signal.ticker: signal for signal in report.signals}
    assert by_ticker["VVSM.DE"].target_weight == pytest.approx(1.0)
    assert "3SEM.L" not in by_ticker

    rendered = format_signal_report(report)
    assert "EXPOSURE POLICY" in rendered
    assert "3SEM.L" in rendered
    assert "VVSM.DE" in rendered


# ---------- freshness of valuation prices ----------

def _stale_price_data(tickers):
    """AAA prints right up to the end, BBB has been frozen for 5 trading days.

    Exactly the constellation of incident 1 (Jun 15): the ffill carried the
    old price forward to the end of the frame, `iloc[-1]` looked like
    today's price.
    """
    idx = pd.bdate_range("2024-01-03", periods=6)      # bis 2024-01-10
    frame = pd.DataFrame({t: [10.0] * 6 for t in tickers}, index=idx)
    return PriceData(
        prices=frame,
        currency={t: "EUR" for t in tickers},
        last_print={t: (idx[0] if t == "BBB" else idx[-1]) for t in tickers},
    )


def _report_mit(monkeypatch, price_data_builder, positions):
    def fake_yahoo(tickers, start=None, end=None, currency="EUR", align="ffill",
                   skip_failed=False):
        _ = (start, end, currency, align, skip_failed)
        return price_data_builder(list(tickers))

    monkeypatch.setattr("backtest.data.DataLoader.yahoo", fake_yahoo)
    return SignalGenerator(
        strategy=_TargetWeightsStrategy(),
        portfolio=Portfolio(positions=positions, cash=100.0),
    ).generate(as_of=date(2024, 1, 10), lookback_days=30)


def test_eingefrorene_position_erzeugt_eine_price_warning(monkeypatch):
    """The warning is the trigger for the CLI's refusal gate:
    no order plan on a price nobody can substantiate."""
    report = _report_mit(monkeypatch, _stale_price_data, {"AAA": 10.0, "BBB": 5.0})

    stale = [w for w in report.price_warnings if w.startswith("BBB:")]
    assert len(stale) == 1
    assert "last real price print 2024-01-03" in stale[0]
    assert not [w for w in report.price_warnings if w.startswith("AAA:")]


def test_frische_positionen_erzeugen_keine_warnung(monkeypatch):
    report = _report_mit(monkeypatch, _stale_price_data, {"AAA": 10.0})

    assert report.price_warnings == []


def test_nicht_gehaltener_stale_ticker_warnt_nicht(monkeypatch):
    """A stale candidate with no holding can't trigger an order -- otherwise
    the gate gets noisy and gets dismissed."""
    report = _report_mit(monkeypatch, _stale_price_data, {"AAA": 10.0})

    assert not [w for w in report.price_warnings if w.startswith("BBB:")]
