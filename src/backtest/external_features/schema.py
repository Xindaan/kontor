"""CSV schema and stable serialization for external feature snapshots.

The long-form snapshot schema is the contract between adapters, loader and
the provenance hash. Writing must be deterministic across pandas versions
so provenance checksums stay stable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "release_date",
    "snapshot_ts",
    "feature_name",
    "feature_value",
    "source",
    "dataset",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "confidence",
    "raw_payload_hash",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "feature_value",
    "confidence",
)

SNAPSHOT_DIR = Path("data/external_features/snapshots")


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if df is missing required columns or types are wrong.

    Optional columns are tolerated when present but must be numeric where
    applicable. Adapters call this before write_snapshot_csv.
    """

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "external feature snapshot missing required column(s): "
            + ", ".join(missing)
        )

    for col in ("feature_value",):
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                pd.to_numeric(df[col])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"column '{col}' must be numeric or coercible to numeric"
                ) from exc

    if "confidence" in df.columns:
        try:
            pd.to_numeric(df["confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "column 'confidence' must be numeric or coercible to numeric"
            ) from exc

    if df["ticker"].isna().any():
        raise ValueError("column 'ticker' must not contain NaN")
    if df["feature_name"].isna().any():
        raise ValueError("column 'feature_name' must not contain NaN")
    if df["release_date"].isna().any():
        raise ValueError("column 'release_date' must not contain NaN")
    if df["snapshot_ts"].isna().any():
        raise ValueError("column 'snapshot_ts' must not contain NaN")


def _ordered_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = list(REQUIRED_COLUMNS)
    for opt in OPTIONAL_COLUMNS:
        if opt in df.columns:
            cols.append(opt)
    extras = [c for c in df.columns if c not in cols]
    return cols + extras


def write_snapshot_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a snapshot CSV with fixed formatting for stable hashing."""

    validate_schema(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out = out[_ordered_columns(out)]
    if not pd.api.types.is_numeric_dtype(out["feature_value"]):
        out["feature_value"] = pd.to_numeric(out["feature_value"])
    if "confidence" in out.columns and not pd.api.types.is_numeric_dtype(out["confidence"]):
        out["confidence"] = pd.to_numeric(out["confidence"])
    out["release_date"] = pd.to_datetime(out["release_date"]).dt.strftime("%Y-%m-%d")
    out["snapshot_ts"] = pd.to_datetime(out["snapshot_ts"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    out = out.sort_values(
        ["ticker", "feature_name", "source", "release_date", "snapshot_ts"]
    ).reset_index(drop=True)
    out.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )
    return path


def read_snapshot_csv(path: Path) -> pd.DataFrame:
    """Read snapshot CSV and parse datetime columns."""

    df = pd.read_csv(path)
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    return df


def snapshot_path(dataset: str, as_of, root: Path | str = SNAPSHOT_DIR) -> Path:
    """Canonical path for a dataset/as_of snapshot."""

    root_path = Path(root) if not isinstance(root, Path) else root
    return root_path / dataset / f"{as_of.isoformat()}.csv"


def iter_snapshot_files(dataset: str | None = None, root: Path | str = SNAPSHOT_DIR) -> Iterable[Path]:
    """Yield all snapshot CSV paths, optionally filtered by dataset."""

    root_path = Path(root)
    if not root_path.exists():
        return []
    if dataset is not None:
        sub = root_path / dataset
        if not sub.exists():
            return []
        return sorted(sub.glob("*.csv"))
    paths: list[Path] = []
    for sub in sorted(root_path.iterdir()):
        if sub.is_dir():
            paths.extend(sorted(sub.glob("*.csv")))
    return paths
