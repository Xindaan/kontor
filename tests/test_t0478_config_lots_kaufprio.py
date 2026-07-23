"""Config adapter, lot serialization, proportional buy priority.

Failure class: state/configuration that goes missing SILENTLY. Nothing
throws, nothing warns -- the result is just wrong. This exact class is
expensive in the tax path.
"""
import datetime

import pytest

from backtest.backtester import BacktestConfig
from backtest.config.run_config import RunConfig, TaxConfig, config_to_backtest_config
from backtest.tax.de_tax_model import GermanTaxModel


class TestConfigAdapterVerliertNichts:
    """TaxConfig fields must arrive in BacktestConfig."""

    def test_steuerklassen_maps_kommen_an(self):
        tc = TaxConfig(enabled=True,
                       equity_fund_map={'SAFE': True},
                       tax_treatment_map={'L3_NDX': ('general', False)})
        bc = config_to_backtest_config(RunConfig(tax=tc))
        assert bc.equity_fund_map == {'SAFE': True}
        assert bc.tax_treatment_map == {'L3_NDX': ('general', False)}

    def test_cost_basis_method_war_totes_config_feld(self):
        """AVGCOST used to silently run as FIFO."""
        bc = config_to_backtest_config(RunConfig(tax=TaxConfig(enabled=True,
                                                              cost_basis_method='AVGCOST')))
        assert bc.cost_basis_method == 'AVGCOST'

    def test_kein_taxconfig_feld_faellt_still_unter_den_tisch(self):
        """Class-level check instead of a single case: EVERY TaxConfig field must show up in the adapter."""
        from dataclasses import fields
        import inspect
        from backtest.config import run_config as rc
        quelle = inspect.getsource(rc.config_to_backtest_config)
        fehlend = [f.name for f in fields(TaxConfig) if f.name not in quelle]
        assert not fehlend, 'TaxConfig fields do not reach BacktestConfig: %s' % fehlend


class TestLotSerialisierung:
    """Without lots, the tax dump is incomplete."""

    def _modell(self):
        m = GermanTaxModel(tax_rate=0.26375, partial_exemption=0.30, exemption_amount=1000.0)
        m.record_purchase(ticker='AAA', shares=10.0, price_per_share=100.0,
                          purchase_date=datetime.date(2024, 3, 1))
        m.record_purchase(ticker='BBB', shares=5.0, price_per_share=50.0,
                          purchase_date=datetime.date(2024, 6, 1))
        return m

    def test_to_dict_enthaelt_lots(self):
        d = self._modell().to_dict()
        assert 'lots' in d
        assert set(d['lots']) == {'AAA', 'BBB'}
        assert d['lots']['AAA'][0]['shares'] == 10.0
        assert d['lots']['AAA'][0]['cost_per_share'] == 100.0

    def test_lots_round_trip(self):
        m = self._modell()
        wieder = GermanTaxModel.lots_from_dict(m.to_dict())
        for ticker in ('AAA', 'BBB'):
            orig = list(m._lots[ticker])
            neu = list(wieder[ticker])
            assert len(orig) == len(neu)
            for a, b in zip(orig, neu):
                assert (a.shares, a.cost_per_share, a.purchase_date) == \
                       (b.shares, b.cost_per_share, b.purchase_date)

    def test_dump_ohne_lots_bleibt_lesbar(self):
        """Old dumps without 'lots' must not blow up."""
        assert GermanTaxModel.lots_from_dict({'tax_rate': 0.26375}) == {}


class TestProportionaleKaufprioritaet:
    """Under cash scarcity, orders used to be filled alphabetically instead of proportionally."""

    def _portfolio(self, cash):
        from backtest.backtester import Portfolio
        return Portfolio(cash=cash)

    def test_bei_knappheit_proportional_statt_alphabetisch(self):
        import pandas as pd
        p = self._portfolio(cash=1000.0)
        prices = pd.Series({'AAA': 100.0, 'ZZZ': 100.0})
        # Both want 1000 -> together 2000 against 1000 cash: each should get ~half.
        p.execute_buys(target={'AAA': 0.5, 'ZZZ': 0.5}, prices=prices,
                       costs_pct=0.0, slippage_pct=0.0)
        aaa = p.positions.get('AAA', 0.0)
        zzz = p.positions.get('ZZZ', 0.0)
        assert zzz > 0, 'ZZZ came away empty -> alphabetical preference still active'
        assert abs(aaa - zzz) / max(aaa, zzz) < 0.10, \
            'Split not proportional: AAA=%.4f ZZZ=%.4f' % (aaa, zzz)

    def test_bei_ausreichendem_cash_unveraendert(self):
        """The factor is 1.0 -> behavior is bit-identical to before."""
        import pandas as pd
        p = self._portfolio(cash=100_000.0)
        prices = pd.Series({'AAA': 100.0, 'ZZZ': 100.0})
        p.execute_buys(target={'AAA': 0.5, 'ZZZ': 0.5}, prices=prices,
                       costs_pct=0.0, slippage_pct=0.0)
        assert p.positions['AAA'] == pytest.approx(p.positions['ZZZ'], rel=1e-9)
