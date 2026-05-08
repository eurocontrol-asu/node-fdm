from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.e2e


def test_fdm_train_help_documents_use_mode_weights_flag() -> None:
    result = subprocess.run(
        ["uv", "run", "fdm", "train", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout + result.stderr
    assert "--use-mode-weights" in out
    assert "--no-use-mode-weights" in out
