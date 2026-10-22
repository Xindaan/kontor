import pandas as pd

from backtest.fundamentals import FundamentalsLoader


def _write_fundamentals_csv(path):
    df = pd.DataFrame(
        {
            "date": ["2020-03-31", "2020-06-30"],
            "release_date": ["2020-04-30", "2020-07-31"],
            "pe": [10.0, 20.0],
            "ev_ebitda": [8.0, 9.0],
            "pb": [1.2, 1.4],
            "roe": [0.12, 0.14],
            "gross_margin": [0.40, 0.42],
            "debt_to_equity": [0.8, 0.7],
        }
    )
    df.to_csv(path, index=False)


def test_snapshot_respects_release_date(tmp_path):
    root = tmp_path / "fundamentals"
    root.mkdir(parents=True, exist_ok=True)
    _write_fundamentals_csv(root / "AAA.csv")

    loader = FundamentalsLoader(root=root)
    loader.load()

    snap_may = loader.snapshot(as_of=pd.Timestamp("2020-05-15").date(), tickers=["AAA"])
    snap_july_pre = loader.snapshot(as_of=pd.Timestamp("2020-07-15").date(), tickers=["AAA"])
    snap_aug = loader.snapshot(as_of=pd.Timestamp("2020-08-01").date(), tickers=["AAA"])

    assert snap_may.data.loc["AAA", "pe"] == 10.0
    assert snap_july_pre.data.loc["AAA", "pe"] == 10.0
    assert snap_aug.data.loc["AAA", "pe"] == 20.0


def test_snapshot_falls_back_to_report_date_when_release_date_missing(tmp_path):
    root = tmp_path / "fundamentals"
    root.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "date": ["2020-03-31", "2020-06-30"],
            "pe": [11.0, 21.0],
            "ev_ebitda": [8.0, 9.0],
            "pb": [1.2, 1.4],
            "roe": [0.12, 0.14],
            "gross_margin": [0.40, 0.42],
            "debt_to_equity": [0.8, 0.7],
        }
    )
    df.to_csv(root / "BBB.csv", index=False)

    loader = FundamentalsLoader(root=root)
    loader.load()

    snap_apr = loader.snapshot(as_of=pd.Timestamp("2020-04-01").date(), tickers=["BBB"])
    snap_july = loader.snapshot(as_of=pd.Timestamp("2020-07-01").date(), tickers=["BBB"])

    assert snap_apr.data.loc["BBB", "pe"] == 11.0
    assert snap_july.data.loc["BBB", "pe"] == 21.0
