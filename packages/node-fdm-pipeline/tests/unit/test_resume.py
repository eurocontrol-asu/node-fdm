"""Unit tests for the resume command (no real I/O)."""

from __future__ import annotations

import pytest


class TestResumeCLI:
    def test_resume_cli_requires_model(self) -> None:
        """Call `fdm resume` without --model → CLI error."""
        from cyclopts.exceptions import MissingArgumentError

        from node_fdm_pipeline.cli import app

        with pytest.raises(MissingArgumentError):
            app(["resume"], exit_on_error=False)
