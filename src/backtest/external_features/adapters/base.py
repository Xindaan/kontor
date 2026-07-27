"""Adapter base class for external feature snapshot pulls.

Adapters implement only ``fetch_remote`` plus a few metadata properties.
The base class handles cache, stable CSV serialization and idempotent
provenance registration — this is the place that guarantees a second
pull of the same snapshot does NOT create a duplicate registry entry.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

import pandas as pd

from backtest.external_features.schema import (
    SNAPSHOT_DIR,
    snapshot_path,
    write_snapshot_csv,
)


@dataclass
class SidecarBlob:
    """Auxiliary payload that lives next to the snapshot CSV (T-0103b).

    Phase C ships exactly one sidecar per news snapshot, but the type is
    generic so phase D adds raw model attributions. ``relative_name``
    is relative to the snapshot's parent directory. Writers register
    via :func:`register_sidecar_writer` (Phase D / Codex D8).
    """

    relative_name: str
    rows: List[Mapping[str, Any]] = field(default_factory=list)
    kind: str = "headlines_ndjson"
    sha256: Optional[str] = None  # filled by base after write


_SIDECAR_WRITERS: dict[str, Any] = {}


def register_sidecar_writer(kind: str, writer) -> None:
    """Register a callable that knows how to persist a sidecar `kind`.

    The callable signature is ``writer(path: Path, rows: list[dict]) -> Path``
    and the matching hash helper resolves via
    :func:`get_sidecar_hash_helper`. Phase A/B/C use ``headlines_ndjson``;
    Phase D adds ``ml_attribution_ndjson``.
    """

    _SIDECAR_WRITERS[str(kind)] = writer


def get_sidecar_writer(kind: str):
    return _SIDECAR_WRITERS.get(str(kind))


_SIDECAR_HASHERS: dict[str, Any] = {}


def register_sidecar_hasher(kind: str, hasher) -> None:
    """Companion to :func:`register_sidecar_writer` — returns SHA256 of
    the canonical rows BEFORE the file is written (so the CSV's
    ``raw_payload_hash`` can be patched in place).
    """

    _SIDECAR_HASHERS[str(kind)] = hasher


def get_sidecar_hasher(kind: str):
    return _SIDECAR_HASHERS.get(str(kind))


@dataclass
class ExternalFeatureFetchResult:
    """Adapter return value when sidecars are needed (T-0103b).

    Phase A/B adapters that return a bare DataFrame continue to work —
    :meth:`ExternalFeatureAdapter.pull_snapshot` normalises both shapes.
    """

    frame: pd.DataFrame
    sidecars: List[SidecarBlob] = field(default_factory=list)


def stable_entry_id(dataset: str, source: str, as_of_iso: str, sha256: str) -> str:
    """Deterministic provenance entry id for a snapshot identity."""

    digest = hashlib.sha256(
        f"{dataset}|{source}|{as_of_iso}|{sha256}".encode("utf-8")
    ).hexdigest()
    return f"{dataset}-{source}-{as_of_iso}-{digest[:12]}".lower()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExternalFeatureAdapter(ABC):
    """Template-method base for all external feature adapters.

    Concrete adapters override only ``fetch_remote`` and the metadata
    properties. The base implements ``pull_snapshot`` so cache and
    provenance behave identically across adapters.
    """

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Stable dataset identifier, e.g. 'mock_analyst' or 'analyst_ratings'."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable source name, e.g. 'YahooFinance' or 'Finnhub'."""

    @property
    @abstractmethod
    def quality_tag(self) -> str:
        """One of: official, proxy, community, manual."""

    @property
    @abstractmethod
    def license_tos_note(self) -> str:
        """Short note about license / terms of service, used in provenance."""

    @property
    def source_url(self) -> Optional[str]:
        """Optional source URL for provenance."""

        return None

    def with_options(self, **kwargs):
        """Phase D adapter factory (Codex D5/D20).

        ML adapters accept runtime overrides (bundle dir, model family,
        sentiment engine) without mutating the registered singleton.
        Phase A/B/C adapters keep the no-op default — they return
        ``self`` because they have no per-pull options. Subclasses that
        accept options MUST return a new instance.
        """

        if not kwargs:
            return self
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(
            f"{type(self).__name__}.with_options does not accept: {unsupported}"
        )

    @abstractmethod
    def fetch_remote(self, tickers: List[str], as_of: date) -> pd.DataFrame:
        """Fetch one snapshot frame from the upstream source.

        Must return a long-form DataFrame conforming to schema.REQUIRED_COLUMNS.
        Must NOT write to disk or register provenance — the base handles that.
        """

    def snapshot_cache_path(self, as_of: date, root: Path | str = SNAPSHOT_DIR) -> Path:
        return snapshot_path(self.dataset_id, as_of, root=root)

    def _throttle(self, seconds: float) -> None:
        """Default throttle. Subclasses can override with token-bucket etc."""

        if seconds > 0:
            time.sleep(seconds)

    # Optional metadata. Adapters override when the upstream service has
    # a freshness / retention / intraday-cutoff property worth recording
    # in provenance (T-0103c).
    @property
    def plan_policy(self) -> Optional[str]:
        return None

    @property
    def max_age_hours(self) -> Optional[int]:
        return None

    def compute_fresh_until(self, snapshot_ts: pd.Timestamp) -> Optional[str]:
        """Return an ISO-timestamp until which the snapshot is considered
        fresh in the **live** path. Default: derived from
        :attr:`max_age_hours`. Adapters can override for finer control."""

        if self.max_age_hours is None:
            return None
        ts = snapshot_ts + pd.Timedelta(hours=int(self.max_age_hours))
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def pull_snapshot(
        self,
        tickers: Iterable[str],
        as_of: date,
        *,
        registry,
        root: Path | str = SNAPSHOT_DIR,
        force: bool = False,
        cutoff_ts_utc: Optional[str] = None,
    ) -> Path:
        """Cache + fetch + persist + idempotent provenance registration.

        Phase C T-0103b: ``fetch_remote`` may return an
        :class:`ExternalFeatureFetchResult` instead of a bare DataFrame.
        When sidecars are present the base writes them deterministically,
        hashes them, and patches ``raw_payload_hash`` of every CSV row
        to the SAME sidecar hash (one sidecar per snapshot — Codex C15).
        """

        ticker_list = [str(t).upper() for t in tickers]
        path = self.snapshot_cache_path(as_of, root=root)
        if path.exists() and not force:
            self._ensure_provenance_idempotent(
                path, as_of, registry, cutoff_ts_utc=cutoff_ts_utc
            )
            return path
        result = self.fetch_remote(ticker_list, as_of)
        if isinstance(result, ExternalFeatureFetchResult):
            df = result.frame
            sidecars = list(result.sidecars or [])
        else:
            df = result
            sidecars = []
        # Sidecars are written first so we can hash them and patch the
        # CSV's raw_payload_hash before serialising the frame. Phase D
        # generalises the dispatcher (Codex D8) so multiple sidecar
        # kinds are supported.
        if sidecars:
            self._ensure_default_sidecar_writers()
            for sidecar in sidecars:
                writer = get_sidecar_writer(sidecar.kind)
                hasher = get_sidecar_hasher(sidecar.kind)
                if writer is None or hasher is None:
                    raise NotImplementedError(
                        f"sidecar kind {sidecar.kind!r} not registered. "
                        "Use register_sidecar_writer/register_sidecar_hasher."
                    )
                sidecar_path = path.parent / sidecar.relative_name
                sidecar.sha256 = hasher(sidecar.rows)
                writer(sidecar_path, sidecar.rows)
            if df is not None and not df.empty:
                shared_hash = sidecars[0].sha256
                if "raw_payload_hash" not in df.columns:
                    df = df.assign(raw_payload_hash=shared_hash)
                else:
                    df = df.copy()
                    df["raw_payload_hash"] = shared_hash
        write_snapshot_csv(df, path)
        self._ensure_provenance_idempotent(
            path, as_of, registry, cutoff_ts_utc=cutoff_ts_utc
        )
        return path

    @staticmethod
    def _ensure_default_sidecar_writers() -> None:
        """Lazy-register the bundled writers (headlines + ml attributions)."""

        if get_sidecar_writer("headlines_ndjson") is None:
            from backtest.external_features.news_schema import (
                hash_headlines_rows,
                write_headlines_ndjson,
            )

            register_sidecar_writer("headlines_ndjson", write_headlines_ndjson)
            register_sidecar_hasher("headlines_ndjson", hash_headlines_rows)
        if get_sidecar_writer("ml_attribution_ndjson") is None:
            try:
                from backtest.external_features.ml.attribution import (
                    hash_ml_attribution_rows,
                    write_ml_attribution_ndjson,
                )
            except ImportError:
                return
            register_sidecar_writer(
                "ml_attribution_ndjson", write_ml_attribution_ndjson
            )
            register_sidecar_hasher(
                "ml_attribution_ndjson", hash_ml_attribution_rows
            )

    def _ensure_provenance_idempotent(
        self,
        path: Path,
        as_of: date,
        registry,
        *,
        cutoff_ts_utc: Optional[str] = None,
    ) -> None:
        sha = sha256_of(path)
        existing = registry.find_entry_by_dedup_key(
            dataset=self.dataset_id,
            source=self.source_name,
            as_of_date=as_of.isoformat(),
            checksum_sha256=sha,
        )
        if existing is not None:
            return
        snapshot_ts = pd.Timestamp.utcnow().tz_localize(None)
        fresh_until = self.compute_fresh_until(snapshot_ts)
        register_kwargs = dict(
            file_path=path,
            dataset=self.dataset_id,
            source=self.source_name,
            quality_tag=self.quality_tag,
            as_of_date=as_of.isoformat(),
            import_method="api_pull",
            license_tos_note=self.license_tos_note,
            source_url=self.source_url,
            entry_id=stable_entry_id(
                self.dataset_id, self.source_name, as_of.isoformat(), sha
            ),
        )
        # Phase C: pass optional metadata only when the registry signature
        # supports it. This keeps the helper backwards compatible against
        # an older provenance schema.
        import inspect as _inspect

        signature = _inspect.signature(registry.register_entry)
        if "plan_policy" in signature.parameters:
            register_kwargs["plan_policy"] = self.plan_policy
        if "fresh_until" in signature.parameters:
            register_kwargs["fresh_until"] = fresh_until
        if "max_age_hours" in signature.parameters:
            register_kwargs["max_age_hours"] = self.max_age_hours
        if "cutoff_ts_utc" in signature.parameters:
            register_kwargs["cutoff_ts_utc"] = cutoff_ts_utc
        registry.register_entry(**register_kwargs)
