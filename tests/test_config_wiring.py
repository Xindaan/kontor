"""An option that exists must also arrive.

`BacktestConfig` carries fields that did not reach the sweep, the batch
optimizer and the CLI -- each tool built its config from a subset. Affected
were among others `rebalance_min_deviation` (no-trade band),
`rebalance_weekday`, `tax_partial_exemption` / `tax_treatment_map`,
`spread_pct`, `tax_rate` and `tax_exemption_amount`. Every robustness number
from `backtest sweep` and every optimization run thus used different trading
mechanics than the measurement it was compared against.

The CLASS test below is the actual guard: it does not enumerate today's known
fields but requires EVERY field of `BacktestConfig` to arrive. A new field is
caught when it is added, not when somebody misses it.
"""
import argparse
import datetime
import inspect
import json
import re
from dataclasses import fields

import pytest

from backtest.backtester import BacktestConfig
from backtest.cli import _load_tax_treatment_map, create_parser
from backtest.sweep import SweepConfig

CORE_COMMANDS = ['run', 'compare', 'sweep', 'optimize', 'batch-optimize']


def _flags_per_command():
    parser = create_parser()
    subs = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                subs[name] = {
                    opt for act in sub._actions for opt in act.option_strings
                    if opt.startswith('--')
                }
    return subs


class TestSweepPassesEverythingThrough:
    def test_every_backtestconfig_field_reaches_the_sweep(self):
        """Class check instead of a single case -- the actual guard."""
        from backtest import sweep as sw
        source = inspect.getsource(sw)
        block = source.split('rebal_freq = getattr(strategy')[1]
        block = block.split('backtester = Backtester')[0]
        arrived = set(re.findall(r'^\s+(\w+)=', block, re.M))
        missing = sorted({f.name for f in fields(BacktestConfig)} - arrived)
        assert not missing, (
            'These BacktestConfig fields do not reach the sweep: %s' % missing)

    def test_sweepconfig_knows_the_trading_mechanics(self):
        names = {f.name for f in fields(SweepConfig)}
        for name in ('rebalance_min_deviation', 'rebalance_weekday',
                     'spread_pct', 'tax_partial_exemption', 'tax_treatment_map'):
            assert name in names, 'SweepConfig does not know %s' % name

    def test_defaults_do_not_change_existing_sweeps(self):
        bt, sc = BacktestConfig(), SweepConfig()
        for name in ('rebalance_min_deviation', 'rebalance_weekday', 'spread_pct',
                     'tax_partial_exemption', 'daily_liquidation_tax',
                     'cost_basis_method', 'currency', 'cash_rate'):
            assert getattr(sc, name) == getattr(bt, name), (
                'SweepConfig.%s differs from the BacktestConfig default' % name)

    def test_value_really_reaches_the_backtester(self, monkeypatch):
        """Behavior probe: the regex above finds a FORGOTTEN field but does not
        prove that the VALUE arrives (``x=None`` would match too)."""
        import pandas as pd

        import backtest.backtester as bt_module
        from backtest import sweep as sw
        from backtest.data import PriceData
        from backtest.strategy import Allocation, Strategy

        class Dummy(Strategy):
            name = "Dummy"
            assets = ["AAA", "BBB"]

            def signal(self, date, data):
                return Allocation({"AAA": 0.5, "BBB": 0.5})

        seen = []
        real_backtester = bt_module.Backtester

        class FakeBacktester(real_backtester):
            def __init__(self, strategy, data, config):
                seen.append(config)
                raise RuntimeError("config captured -- the run itself is not needed")

        monkeypatch.setattr(bt_module, "Backtester", FakeBacktester)

        idx = pd.bdate_range("2024-01-02", periods=300)
        data = PriceData(prices=pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=idx),
                         currency={"AAA": "EUR", "BBB": "EUR"})
        window = sw.Window(start=idx[10], end=idx[-1], years=1.0)
        treatment = {"AAA": ("general", False)}
        cfg = sw.SweepConfig(rebalance_min_deviation=0.17, rebalance_weekday=2,
                             tax_treatment_map=treatment, spread_pct=0.0012,
                             benchmark=None, validate=False)
        sw.run_single_window(Dummy(), "x.py", data, window, cfg)

        assert seen, "the sweep built no backtester -- the fixture does not bite"
        assert seen[-1].rebalance_min_deviation == pytest.approx(0.17)
        assert seen[-1].rebalance_weekday == 2
        assert seen[-1].tax_treatment_map == treatment
        assert seen[-1].spread_pct == pytest.approx(0.0012)


class TestBatchOptimizeTakesABaseConfig:
    def test_all_functions_take_base_config(self):
        from backtest.batch_optimize import (
            optimize_single_strategy,
            run_batch_optimization,
            walk_forward_single_strategy,
        )
        for fn in (optimize_single_strategy, walk_forward_single_strategy,
                   run_batch_optimization):
            assert 'base_config' in inspect.signature(fn).parameters, (
                '%s takes no base config' % fn.__name__)


class TestTradingMechanicsReachableViaCli:
    @pytest.mark.parametrize('flag', ['--rebalance-min-deviation', '--rebalance-weekday',
                                      '--tax-treatment-file'])
    def test_flags_exist_on_all_core_commands(self, flag):
        subs = _flags_per_command()
        missing = [c for c in CORE_COMMANDS if flag not in subs.get(c, set())]
        assert not missing, '%s missing on: %s' % (flag, missing)

    def test_sweep_accepts_run_compare_names(self):
        args = create_parser().parse_args(
            ['sweep', 'strategies/buy_and_hold.py', '--tax-exemption', '2000',
             '--output', 'somewhere'])
        assert args.allowance == 2000.0
        assert args.out == 'somewhere'


class TestTaxTreatmentFile:
    def test_file_is_translated_into_the_map(self, tmp_path):
        path = tmp_path / 'tax.json'
        path.write_text(json.dumps({'TQQQ': ['general', False], 'SPY': ['general', True]}),
                        encoding='utf-8')
        treatment = _load_tax_treatment_map(str(path))
        assert treatment['TQQQ'] == ('general', False)
        assert treatment['SPY'] == ('general', True)

    def test_without_file_the_legacy_default_stays(self):
        assert _load_tax_treatment_map(None) is None

    def test_nonsense_class_is_rejected(self, tmp_path):
        """A typo must not silently become the legacy default."""
        path = tmp_path / 'broken.json'
        path.write_text(json.dumps({'X': ['equityfund', False]}), encoding='utf-8')
        with pytest.raises(SystemExit):
            _load_tax_treatment_map(str(path))

    def test_the_forbidden_combination_is_rejected(self, tmp_path):
        """('equity', True) is not expressible -- not via file either."""
        path = tmp_path / 'forbidden.json'
        path.write_text(json.dumps({'X': ['equity', True]}), encoding='utf-8')
        with pytest.raises(SystemExit):
            _load_tax_treatment_map(str(path))

    def test_the_map_really_changes_the_tax(self):
        from backtest.tax import GermanTaxModel

        def tax(instrument_class, is_equity_fund):
            m = GermanTaxModel(tax_rate=0.26375, partial_exemption=0.30,
                               exemption_amount=0.0)
            m.record_purchase(ticker='ETP', shares=100.0, price_per_share=100.0,
                              purchase_date=datetime.date(2024, 1, 1),
                              instrument_class=instrument_class,
                              is_equity_fund=is_equity_fund)
            r = m.apply_sale(ticker='ETP', shares_sold=100.0, sale_price=200.0,
                             sale_date=datetime.date(2025, 1, 1),
                             instrument_class=instrument_class,
                             is_equity_fund=is_equity_fund)
            return r.tax_due

        as_fund = tax('general', True)       # legacy default
        as_etp = tax('general', False)       # debt ETP
        assert as_etp > as_fund * 1.3
