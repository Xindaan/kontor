"""The web API only loads strategies from the repository.

Endpoints take a strategy path from the request and load it via `importlib` --
which EXECUTES the file. Without a boundary every readable path on the machine
is a code path; with `backtest web --host 0.0.0.0` that is code execution for
anyone on the network.
"""
from pathlib import Path

import pytest

from backtest.web.services.strategies import validate_strategy_path, validate_strategy_paths

REPO = Path(__file__).resolve().parent.parent


class TestPathGuard:
    def test_strategy_from_the_repo_is_allowed(self):
        assert validate_strategy_path(REPO / 'strategies' / 'buy_and_hold.py').name == 'buy_and_hold.py'

    def test_subdirectory_of_strategies_is_allowed(self):
        # No existence check: a fresh clone has no strategy subdirectories, and
        # the guard compares resolved paths, not files on disk.
        target = REPO / 'strategies' / 'subdir'
        assert validate_strategy_path(target / 'anything.py').parent == target.resolve()

    def test_script_is_allowed(self):
        assert validate_strategy_path(REPO / 'scripts' / 'download_pit_data.py').name == \
            'download_pit_data.py'

    def test_foreign_absolute_path_is_rejected(self):
        with pytest.raises(ValueError, match='outside the allowed'):
            validate_strategy_path(Path('/etc/passwd'))

    def test_traversal_is_rejected(self):
        """The prefix alone is not enough -- resolve first, then compare."""
        with pytest.raises(ValueError, match='outside the allowed'):
            validate_strategy_path(REPO / 'strategies' / '..' / '..' / 'evil.py')

    def test_repo_root_itself_is_not_allowed(self):
        with pytest.raises(ValueError, match='outside the allowed'):
            validate_strategy_path(REPO / 'conftest.py')

    def test_list_checks_every_entry(self):
        good = str(REPO / 'strategies' / 'buy_and_hold.py')
        assert validate_strategy_paths([good])
        with pytest.raises(ValueError, match='outside the allowed'):
            validate_strategy_paths([good, '/etc/passwd'])

    def test_empty_list_is_empty(self):
        assert validate_strategy_paths([]) == []
        assert validate_strategy_paths(None) == []


def test_loader_is_guarded_not_only_the_helper(tmp_path):
    """Wiring: the service loader itself refuses a path outside the repo, even
    when the file exists and is a valid strategy module."""
    from backtest.web.services.strategies import load_strategy_instance

    outside = tmp_path / 'outside.py'
    outside.write_text('raise SystemExit("must never execute")\n', encoding='utf-8')
    with pytest.raises(ValueError, match='outside the allowed'):
        load_strategy_instance(outside)


def test_param_schema_is_guarded(tmp_path):
    """The run page passes a client-supplied `?strategy=` path to
    `get_param_schema`, which executes the module to build the schema."""
    from backtest.web.services.strategies import get_param_schema

    outside = tmp_path / 'outside.py'
    outside.write_text('raise SystemExit("must never execute")\n', encoding='utf-8')
    with pytest.raises(ValueError, match='outside the allowed'):
        get_param_schema(outside)
    assert get_param_schema(REPO / 'strategies' / 'buy_and_hold.py')
