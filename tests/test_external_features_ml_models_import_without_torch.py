"""Phase E3 (T-0342, Codex R2.11) — models import without torch.

Ensures that ``import backtest.external_features.ml.models``
does not crash when `torch` is not installed. We simulate this
in a subprocess sandbox so the test run doesn't leave behind any
real `sys.modules` mutations.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap


def test_models_init_does_not_import_torch_at_top_level():
    """Codex R2.11: `import backtest.external_features.ml.models` must
    succeed even without `torch` installed.

    We start a fresh Python subprocess and block `import torch` via a
    meta-path finder that raises `ImportError`. If the models init
    module keeps torch lazy, the import completes cleanly.
    """

    script = textwrap.dedent(
        '''
        import sys
        import importlib.abc
        import importlib.machinery


        class _BlockTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ImportError(f"torch is blocked for this test ({fullname})")
                return None


        # Install the meta-path hook BEFORE the models import.
        sys.meta_path.insert(0, _BlockTorch())

        # If torch is already in sys.modules, remove it, otherwise
        # the block has no effect.
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                del sys.modules[name]

        import backtest.external_features.ml.models as models
        assert hasattr(models, "LSTM_FACTORY_PATH")
        assert hasattr(models, "TRANSFORMER_FACTORY_PATH")
        assert hasattr(models, "resolve_factory")
        print("OK")
        '''
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **{k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}},
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_models_init_exposes_factory_paths():
    """Factory strings must be exported (without loading torch)."""

    mod = importlib.import_module("backtest.external_features.ml.models")
    assert hasattr(mod, "LSTM_FACTORY_PATH")
    assert hasattr(mod, "TRANSFORMER_FACTORY_PATH")
    assert hasattr(mod, "resolve_factory")
    assert mod.LSTM_FACTORY_PATH.endswith(":LSTMForecastModel")
    assert mod.TRANSFORMER_FACTORY_PATH.endswith(":TransformerForecastModel")
