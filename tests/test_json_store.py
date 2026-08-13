"""Tests for the atomic money-record write with generations.

`data/manual/portfolio_*.json` is gitignored -- this protective layer is the
ONLY safety net under share counts and trailing highs.
"""

import datetime
import fnmatch
import json
import os

import pytest

from backtest.json_store import save_json_mit_backup


def _now(i: int) -> datetime.datetime:
    return datetime.datetime(2026, 7, 17, 7, 0, i)


def _read(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class TestSaveJsonMitBackup:
    def test_erster_write_legt_datei_ohne_backup_an(self, tmp_path):
        target = tmp_path / "portfolio_db.json"
        bak = save_json_mit_backup(str(target), {"positionen": [1]}, now=_now(0))
        assert bak is None
        assert _read(target) == {"positionen": [1]}
        assert target.read_text(encoding="utf-8").endswith("}\n")
        assert not os.path.exists(str(target) + ".tmp")

    def test_backup_vor_jeder_aenderung_ist_der_alte_stand(self, tmp_path):
        target = tmp_path / "portfolio_db.json"
        save_json_mit_backup(str(target), {"shares": 45}, now=_now(0))
        bak = save_json_mit_backup(str(target), {"shares": 0}, now=_now(1))
        assert bak is not None
        assert _read(bak) == {"shares": 45}     # undo state
        assert _read(target) == {"shares": 0}   # new state

    def test_rotation_haelt_nur_die_juengsten_generationen(self, tmp_path):
        target = tmp_path / "portfolio_db.json"
        for i in range(5):
            save_json_mit_backup(str(target), {"stand": i}, keep=3, now=_now(i))
        backups = sorted(tmp_path.glob("portfolio_db.bak-*.json"))
        assert len(backups) == 3
        # The newest generation is the state BEFORE the last write.
        assert _read(backups[-1]) == {"stand": 3}
        assert _read(backups[0]) == {"stand": 1}

    def test_fehlerhafte_daten_lassen_den_alten_stand_unberuehrt(self, tmp_path):
        """The core case: a bug in the caller must not cost the record."""
        target = tmp_path / "portfolio_db.json"
        save_json_mit_backup(str(target), {"shares": 45}, now=_now(0))
        with pytest.raises(TypeError):
            save_json_mit_backup(str(target), {"kaputt": {1, 2}}, now=_now(1))
        assert _read(target) == {"shares": 45}                      # untouched
        assert list(tmp_path.glob("*.bak-*.json")) == []            # no garbage backup
        assert not os.path.exists(str(target) + ".tmp")

    def test_zwei_writes_in_derselben_sekunde_kollidieren_nicht(self, tmp_path):
        target = tmp_path / "portfolio_db.json"
        for i in range(3):
            save_json_mit_backup(str(target), {"stand": i}, now=_now(0))
        assert len(list(tmp_path.glob("portfolio_db.bak-*.json"))) == 2

    def test_backup_name_faellt_unter_das_portfolio_gitignore(self, tmp_path):
        """`.gitignore` covers `data/manual/portfolio_*.json` -- the generations
        must fall under the SAME entry, otherwise real positions would
        suddenly become trackable (public-repo failure class)."""
        target = tmp_path / "portfolio_db.json"
        save_json_mit_backup(str(target), {"a": 1}, now=_now(0))
        bak = save_json_mit_backup(str(target), {"a": 2}, now=_now(1))
        assert fnmatch.fnmatch(os.path.basename(bak), "portfolio_*.json")
