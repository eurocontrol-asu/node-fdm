# %%
"""13 — Generate Altair example flight trajectory chart.

CLI equivalent::

    fdm plot-example --config config.yaml

Requires ``pip install node-fdm-pipeline[viz]``.
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.visualize import run_plot_example

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    run_plot_example(config=CONFIG)


if __name__ == "__main__":
    main()
# %%
