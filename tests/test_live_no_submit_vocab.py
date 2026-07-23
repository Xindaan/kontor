"""Phase E2 — scoped submit-vocabulary grep (T-0378, Codex R2.1+R3.1).

Checks ONLY runtime paths under ``src/backtest/live/`` + live
subcommands in ``src/backtest/cli.py``. Test files / docs / plan file
are explicitly excluded.

Regex-Patterns (Codex R3.1):
- ``\.placeOrder\(``
- ``\.submit_order\(``
- ``\.cancel_order\(``
- ``add_parser\("execute"``
- ``--live-execute``
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_RUNTIME_DIR = REPO_ROOT / "src" / "backtest" / "live"
CLI_FILE = REPO_ROOT / "src" / "backtest" / "cli.py"


FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"\.placeOrder\(",
    r"\.submit_order\(",
    r"\.cancel_order\(",
    r'add_parser\("execute"',
    r"--live-execute",
)


def _runtime_files() -> list[Path]:
    files: list[Path] = []
    if LIVE_RUNTIME_DIR.exists():
        files.extend(p for p in LIVE_RUNTIME_DIR.rglob("*.py"))
    if CLI_FILE.exists():
        files.append(CLI_FILE)
    return files


def test_no_submit_vocab_in_runtime():
    """Codex R3.1: scoped grep over `src/backtest/live/` + cli.py."""

    findings: list[tuple[Path, str, int, str]] = []
    for path in _runtime_files():
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            for match in re.finditer(pattern, text):
                line_no = text.count("\n", 0, match.start()) + 1
                line = text.splitlines()[line_no - 1].strip()
                findings.append((path, pattern, line_no, line))

    # The CLI is allowed to contain the `--live-execute` string in a
    # rejection error message — we check that there is no
    # `add_argument(..., "--live-execute", ...)` call present.
    real_findings = [
        (p, pat, line_no, line)
        for (p, pat, line_no, line) in findings
        if pat != r"--live-execute"
        or "add_argument" in line
    ]
    assert not real_findings, (
        "Forbidden submit-vocab patterns found in Phase-E runtime:\n"
        + "\n".join(
            f"  {p.name}:{n}: [{pat}] {line}"
            for (p, pat, n, line) in real_findings
        )
    )
