"""TSMOM bond safe-leg (managed-futures crisis hedge) — review-ready, self-verifying deliverable.

Finding (search loop iter 43-55, dossier docs/tsmom_bond_safeleg_2026-06-19.md): holding the master's
(StickyLeveredVolTargeted) de-risked capital in a SIGN-FLEXIBLE bond trend sleeve instead of plain long
Treasury/cash is the most robust DEFENSIVE finding from the search loop. Mechanism: time-series momentum
on bonds goes LONG in flight-to-quality (yields fall, dotcom/GFC/COVID) and SHORT in a rate shock
(yields rise, 2022) -> positive in BOTH crisis types, where long Treasury loses -14% in 2022. Vetted 6x
(mechanism/deep+real+OOS/multi-episode/duration/both accounts/rolling WF), validated against reality via
DBMF + IEF trend (corr synth~real 0.92).

IMPORTANT — not a master-beater, a DEFENSIVE option: costs modern-era CAGR (real ~-1.7pp @25% dose, like
every non-S&P safe leg), largest melt-up drag, fast-reversal whipsaw (2018Q4). Recommended dose 25-33%.

WHY a module instead of a Strategy subclass: the sleeve SHORTS bonds (2022) -> a long-only safe_asset
slot can't do that directly. This module BUILDS the validated safe-leg NAV (with synthetic short) that
you feed in as the safe_asset column in PriceData. REAL implementation (long-only): bond trend sleeve =
{long IEF in an uptrend, inverse bond ETF (e.g. PST/TBX) OR cash in a downtrend}; OR simply buy DBMF/KMLM
(multi-asset MF ETF).

.venv/bin/python strategies/_tsmom_bond_safeleg.py   # self-verification (reproduces the headline numbers)
(File `_` prefix = helper/builder module, NOT a Strategy -> skipped by strategies discovery,
same as _momentum_utils.py.)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def tsmom_bond_sleeve(
    bond_returns: pd.Series,
    lookback: int = 252,
    vol_lookback: int = 63,
    target_vol: float = 0.08,
    weight_cap: float = 3.0,
) -> pd.Series:
    """Sign-flexible TSMOM bond sleeve NAV (start=1.0), look-ahead free.

    Direction = sign(lookback-day return, as of t-1); size = clip(target_vol / realized_vol_t-1, 0, cap)
    (inverse-vol -> the sleeve is duration-INVARIANT, vol ~target_vol at any bond duration). The position
    decision at t uses only data up to t-1 (.shift(1)). Verified in scripts/tsmom_macro_safeleg_test.py
    (iter 43).
    """
    r = bond_returns.astype(float).fillna(0.0)
    nav = (1.0 + r).cumprod()
    mom = nav / nav.shift(lookback) - 1.0
    direction = np.sign(mom)
    vol = r.rolling(vol_lookback).std() * np.sqrt(252.0)
    weight = (direction * (target_vol / vol).clip(0.0, weight_cap)).shift(1).fillna(0.0)
    sleeve_ret = (weight * r).fillna(0.0)
    out = (1.0 + sleeve_ret).cumprod()
    out.name = "tsmom_bond_sleeve"
    return out


def build_safeleg_nav(
    sp_safe_returns: pd.Series,
    bond_returns: pd.Series,
    dose: float = 0.25,
    **sleeve_kwargs,
) -> pd.Series:
    """Safe-leg NAV = (1-dose) * S&P 1x safe + dose * TSMOM bond sleeve (rebalanced daily).

    Feed in as the safe_asset column in PriceData (the master holds (1-w) here). dose 0.25-0.33
    recommended. Higher dose = more 2022/rate-shock protection, but more modern-era CAGR cost + melt-up
    drag.
    """
    sleeve = tsmom_bond_sleeve(bond_returns, **sleeve_kwargs)
    idx = sp_safe_returns.index
    sl_ret = sleeve.pct_change().reindex(idx).fillna(0.0)
    blended = (1.0 - dose) * sp_safe_returns.reindex(idx).fillna(0.0) + dose * sl_ret
    nav = (1.0 + blended.fillna(0.0)).cumprod()
    nav.name = f"safeleg_tsmom_bond_{int(dose * 100)}"
    return nav


def synth_bond_returns_from_yield(yield_pct: pd.Series, duration: float = 8.0) -> pd.Series:
    """Synthesize Treasury total return from a yield (e.g. ^TNX/DGS10): r = carry(y_t-1/252) - D*dy.

    Faithful to real IEF (~D8): corr(synth trend, real IEF trend) = 0.92 (iter 54). For real backtests,
    use the returns of a real Treasury ETF (IEF/TLT) instead.
    """
    y = (yield_pct / 100.0).astype(float).ffill()
    return ((y.shift(1) / 252.0) - duration * y.diff()).fillna(0.0)


# ============================ Self-verification ============================
def _verify() -> None:
    """Reproduces the headline numbers (deep + real 2022 event) offline from the cache."""
    import sys
    import warnings
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    warnings.filterwarnings("ignore")

    import combined_book_stress as cbs
    from backtest.backtester import BacktestConfig, Backtester
    from backtest.data import PriceData
    from backtest.metrics import MetricsCalculator
    from strategies.sticky_levered_vol_targeted import StickyLeveredVolTargeted

    DATA = Path(__file__).resolve().parent.parent / "data"
    load = lambda t: pd.read_csv(DATA / f"{t}.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()

    def run(risky, safe_nav, cols, tf, idx):
        df = pd.DataFrame({**{c: risky[c].reindex(idx) for c in cols}, "SAFE": safe_nav.reindex(idx)}).dropna()
        D = PriceData(prices=df, currency={c: "USD" for c in df.columns}, fx_rates=pd.Series(1.0, index=df.index))
        eq = {c: False for c in cols}; eq["SAFE"] = tf
        cfg = BacktestConfig(initial_capital=10_000.0, costs_pct=0.001, slippage_pct=0.0005, currency="USD",
                             rebalance_frequency="weekly", benchmark=None, tax_enabled=True,
                             metric_basis="net_liquidation", validate=False, equity_fund_map=eq)
        r = Backtester(StickyLeveredVolTargeted(candidates=cols, safe_asset="SAFE", vol_target=0.40), D, cfg).run()
        c = (r.equity_curve_daily if r.equity_curve_daily is not None else r.equity_curve).dropna()
        c = c / c.iloc[0]
        return float(MetricsCalculator.cagr(c)), cbs.maxdd_window(c, "2022-01-01", "2023-06-30")

    TER_1X = 0.0007
    print("Self-Verifikation TSMOM-bond-Safe-Bein (Master vt0.40, after-tax) — Blend25 vs S&P-safe:")
    # DEEP 1994-2024 (synth 3x + synth bond)
    ndx, sp, sox, irx, tnx = (load(t) for t in ["^NDX", "^SP500TR", "^SOX", "^IRX", "^TNX"])
    idx = ndx.index
    for s in [sp, sox, irx, tnx]:
        idx = idx.intersection(s.index)
    idx = idx[(idx >= "1994-01-01") & (idx <= "2024-12-31")]
    rf = (irx / 100.0 / 252.0).reindex(idx).ffill().fillna(0.0)
    risky = {c: cbs.lev_nav(load(t).reindex(idx).pct_change().dropna(), 3.0, rf, cbs.TER_3X)
             for c, t in [("L3_NDX", "^NDX"), ("L3_SP", "^SP500TR"), ("L3_SOX", "^SOX")]}
    sp1x = cbs.lev_nav(sp.reindex(idx).pct_change().dropna(), 1.0, rf, TER_1X); sp1x_r = sp.reindex(idx).pct_change().fillna(0.0)
    bond_r = synth_bond_returns_from_yield(tnx.reindex(idx), 8.0)
    cols = ["L3_NDX", "L3_SP", "L3_SOX"]
    base_c, base_e = run(risky, sp1x, cols, True, idx)
    safe = build_safeleg_nav(sp1x_r, bond_r, dose=0.25)
    ts_c, ts_e = run(risky, safe, cols, False, idx)
    print(f"  DEEP 1994-2024: dCAGR {(ts_c - base_c) * 100:+.1f}pp  d2022 {(ts_e - base_e) * 100:+.1f}pp"
          f"  (erwartet ~ -0.1 / +3.1pp)")
    # REAL 2012-2024 (real proxies)
    tqqq, upro, soxl = (load(t) for t in ["TQQQ", "UPRO", "SOXL"])
    idx2 = tqqq.index
    for s in [upro, soxl, sp, irx, tnx]:
        idx2 = idx2.intersection(s.index)
    idx2 = idx2[(idx2 >= "2012-01-01") & (idx2 <= "2024-12-31")]
    rf2 = (irx / 100.0 / 252.0).reindex(idx2).ffill().fillna(0.0)
    risky2 = {"TQQQ": tqqq.reindex(idx2), "UPRO": upro.reindex(idx2), "SOXL": soxl.reindex(idx2)}
    sp1x2 = cbs.lev_nav(sp.reindex(idx2).pct_change().dropna(), 1.0, rf2, TER_1X)
    sp1x_r2 = sp.reindex(idx2).pct_change().fillna(0.0)
    bond_r2 = synth_bond_returns_from_yield(tnx.reindex(idx2), 8.0)
    base_c2, base_e2 = run(risky2, sp1x2, ["TQQQ", "UPRO", "SOXL"], True, idx2)
    safe2 = build_safeleg_nav(sp1x_r2, bond_r2, dose=0.25)
    ts_c2, ts_e2 = run(risky2, safe2, ["TQQQ", "UPRO", "SOXL"], False, idx2)
    print(f"  REAL 2012-2024: dCAGR {(ts_c2 - base_c2) * 100:+.1f}pp  d2022 {(ts_e2 - base_e2) * 100:+.1f}pp"
          f"  (erwartet ~ -1.7 / +3.0pp)")


if __name__ == "__main__":
    _verify()
