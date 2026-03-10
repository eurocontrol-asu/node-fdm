# %%
"""09 — Compute prediction error metrics per flight phase.

CLI equivalent::

    fdm evaluate --arch opensky --config config.yaml
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.evaluate import run_evaluate

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    run_evaluate(arch="opensky", config=CONFIG)


if __name__ == "__main__":
    main()
# %%
