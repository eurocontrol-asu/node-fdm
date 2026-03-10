# %%
"""12 — Generate Altair performance comparison charts.

CLI equivalent::

    fdm plot-performance --config config.yaml

Requires ``pip install node-fdm-pipeline[viz]``.
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.visualize import run_plot_performance

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    run_plot_performance(config=CONFIG)


if __name__ == "__main__":
    main()
# %%
