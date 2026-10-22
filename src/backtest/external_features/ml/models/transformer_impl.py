"""Phase E3 — Transformer encoder ForecastModel (T-0345, Codex R3.13).

Lazy torch import (Codex R2.11). Architecture:
- 4 encoder layers, ``d_model=64``, ``nhead=4``, ``dropout=0.1``.
- **No causal mask** (Codex R3.13): the sequence construction in
  ``_sequence.py`` contains no future data, so no autoregressive
  masking is needed.
- Linear output head (single-step prediction).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from backtest.external_features.ml.models.base import (
    ForecastModel,
    _stable_param_hash,
)
from backtest.external_features.ml.models.lstm_impl import _lazy_torch


class TransformerForecastModel(ForecastModel):
    """Transformer encoder Stage-1 alternative."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        self._torch = _lazy_torch()
        self.params: Dict[str, Any] = {
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "seed": int(seed),
        }
        self._net = None
        self._input_dim: Optional[int] = None
        self._seq_len: Optional[int] = None

    @property
    def engine_code(self) -> str:
        torch_version = getattr(self._torch, "__version__", "unknown")
        return f"transformer@{torch_version}_{_stable_param_hash(self.params)}"

    def _build_net(self, input_dim: int):
        torch = self._torch
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self, n_features: int, d_model: int, nhead: int,
                         num_layers: int, dropout: float):
                super().__init__()
                self.input_proj = nn.Linear(n_features, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=num_layers
                )
                self.head = nn.Linear(d_model, 1)

            def forward(self, x):
                # Codex R3.13: NO causal mask, the sequence contains no
                # future data.
                h = self.input_proj(x)
                h = self.encoder(h)
                last = h[:, -1, :]
                return self.head(last).squeeze(-1)

        torch.manual_seed(int(self.params["seed"]))
        return Net(
            n_features=input_dim,
            d_model=int(self.params["d_model"]),
            nhead=int(self.params["nhead"]),
            num_layers=int(self.params["num_layers"]),
            dropout=float(self.params["dropout"]),
        )

    def _ensure_tensor(self, X) -> Any:
        torch = self._torch
        if isinstance(X, np.ndarray):
            if X.ndim != 3:
                raise ValueError(
                    f"TransformerForecastModel erwartet 3D-Input "
                    f"(Batch, Seq, Features), bekam shape={X.shape}"
                )
            return torch.tensor(X, dtype=torch.float32)
        if hasattr(X, "to_numpy") and X.ndim == 2:
            arr = X.to_numpy(dtype=float)
            return torch.tensor(arr[:, None, :], dtype=torch.float32)
        return torch.as_tensor(X, dtype=torch.float32)

    def fit(self, X, y, sample_weight=None) -> "TransformerForecastModel":
        torch = self._torch
        nn = torch.nn

        tensor_X = self._ensure_tensor(X)
        if tensor_X.ndim != 3:
            raise ValueError("fit erwartet 3D-Input (Batch, Seq, Features)")
        tensor_y = torch.tensor(np.asarray(y, dtype=float), dtype=torch.float32)
        self._input_dim = int(tensor_X.shape[-1])
        self._seq_len = int(tensor_X.shape[1])

        self._net = self._build_net(self._input_dim)
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=float(self.params["learning_rate"])
        )
        loss_fn = nn.MSELoss()
        batch_size = int(self.params["batch_size"])
        n = tensor_X.shape[0]

        self._net.train()
        for _ in range(int(self.params["epochs"])):
            permutation = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = permutation[start : start + batch_size]
                pred = self._net(tensor_X[idx])
                loss = loss_fn(pred, tensor_y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self._net.eval()
        return self

    def predict(self, X) -> np.ndarray:
        torch = self._torch
        if self._net is None:
            raise RuntimeError("TransformerForecastModel.predict before fit()")
        tensor_X = self._ensure_tensor(X)
        with torch.no_grad():
            out = self._net(tensor_X)
        return out.detach().cpu().numpy().astype(float)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": dict(self.params),
            "input_dim": self._input_dim,
            "seq_len": self._seq_len,
            "state_dict": self._net.state_dict() if self._net is not None else None,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path) -> "TransformerForecastModel":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        instance = cls(**payload.get("params", {}))
        instance._input_dim = payload.get("input_dim")
        instance._seq_len = payload.get("seq_len")
        if payload.get("state_dict") is not None and instance._input_dim is not None:
            instance._net = instance._build_net(instance._input_dim)
            instance._net.load_state_dict(payload["state_dict"])
            instance._net.eval()
        return instance


__all__ = ["TransformerForecastModel"]
