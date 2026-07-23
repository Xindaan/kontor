"""Tests for the `backtest features` CLI subcommands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run(args, env=None, cwd=None):
    cmd = [sys.executable, "-m", "backtest", *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)


@pytest.mark.no_network
def test_features_pull_mock_creates_snapshot(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = tmp_path / "prov.json"
    result = _run(
        [
            "features",
            "pull",
            "--dataset",
            "mock_analyst",
            "--as-of",
            "2026-05-01",
            "--registry",
            str(registry),
        ]
    )
    assert result.returncode == 0, result.stderr
    snap = tmp_path / "data" / "external_features" / "snapshots" / "mock_analyst" / "2026-05-01.csv"
    assert snap.exists()
    payload = json.loads(registry.read_text())
    assert len(payload["entries"]) == 1


@pytest.mark.no_network
def test_features_pull_requires_tickers_for_non_mock(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run(
        [
            "features",
            "pull",
            "--dataset",
            "does_not_exist",
            "--as-of",
            "2026-05-01",
        ]
    )
    # Unknown adapter → exits non-zero
    assert result.returncode != 0


@pytest.mark.no_network
def test_features_list_json_lists_snapshots(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = tmp_path / "prov.json"
    _run(
        [
            "features",
            "pull",
            "--dataset",
            "mock_analyst",
            "--as-of",
            "2026-05-01",
            "--registry",
            str(registry),
        ]
    )
    result = _run(
        [
            "features",
            "list",
            "--registry",
            str(registry),
            "-f",
            "json",
        ]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["snapshots"]
    assert payload["snapshots"][0]["registered"] is True


@pytest.mark.no_network
def test_features_verify_ok(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = tmp_path / "prov.json"
    _run(
        [
            "features",
            "pull",
            "--dataset",
            "mock_analyst",
            "--as-of",
            "2026-05-01",
            "--registry",
            str(registry),
        ]
    )
    result = _run(
        [
            "features",
            "verify",
            "--registry",
            str(registry),
            "-f",
            "json",
        ]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] == 1
    assert payload["issue_count"] == 0


@pytest.mark.no_network
def test_run_help_shows_external_features_flags():
    result = _run(["run", "--help"])
    assert result.returncode == 0
    assert "--external-features-enable" in result.stdout
    assert "--external-features-dataset" in result.stdout
    assert "--external-features-provenance-mode" in result.stdout
