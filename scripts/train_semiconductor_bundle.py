"""Phase E TR bundle v2: LSE/DE universe + SOXL as semiconductor 3x proxy.

Compared to train_tr_only.py: SOXL added so the ML veto layer can
score the semiconductor pick (real 3SEM.L, SOXL proxy in the backtest).
The old TR-only bundle had no semiconductor in the training universe.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, "src")
from backtest.data import DataLoader
from backtest.external_features.ml.config import MLTrainingConfig
from backtest.external_features.ml.training import run_walk_forward_training
from backtest.provenance import ManualDataProvenanceRegistry

UNI = [
    "QQQ3.L", "3LUS.L", "3USL.L", "SOXL",   # 3x LEV (SOXL = 3SEM.L semiconductor proxy)
    "SXR8.DE", "EQQQ.L", "VUSA.L", "CSPX.L", "IUSA.L",
    "VWRL.L", "EUNL.DE", "IS3N.DE",
    "SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "BMW.DE",
    "BAYN.DE", "DBK.DE", "DTE.DE", "MUV2.DE",
    "HSBA.L", "BP.L", "SHEL.L", "AZN.L", "GSK.L",
    "ULVR.L", "RIO.L", "GLEN.L", "LLOY.L", "BARC.L",
    "TLT", "GLD",
]

print(f"Loading {len(UNI)} TR-only+SOXL tickers 2013-2024...")
pd_obj = DataLoader.yahoo(
    tickers=UNI, start="2013-01-01", end="2024-12-31",
    currency="EUR", align="ffill", skip_failed=True,
)
prices = pd_obj.prices
print(f"  shape: {prices.shape}, tickers: {len(prices.columns)}")
print(f"  SOXL present: {'SOXL' in prices.columns}")

config = MLTrainingConfig(
    horizons=(21, 63, 252), model_families=("lightgbm",),
    outer_train_years=4.0, outer_holdout_months=6,
    inner_train_years=2.0, inner_test_months=6,
    grid_size=1, seed=42, tickers=tuple(prices.columns),
)
out_dir = Path("/tmp/phase_e_tr_soxl/models")
out_dir.mkdir(parents=True, exist_ok=True)
reg = ManualDataProvenanceRegistry(path="/tmp/phase_e_tr_soxl/provenance.json")

t1 = time.time()
result = run_walk_forward_training(prices=prices, config=config, output_dir=out_dir, registry=reg)
print(f"DONE in {time.time()-t1:.1f}s, bundles={len(result.manifest_paths)}")
print(f"First: {result.manifest_paths[0]}")
print(f"Last: {result.manifest_paths[-1]}")
