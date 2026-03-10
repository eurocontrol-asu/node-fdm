# %%
"""08 — Visualize prediction comparisons (3-panel: altitude, TAS, gamma).

CLI equivalent::

    fdm visualize --arch opensky --config config.yaml
    fdm visualize --arch opensky --config config.yaml --typecode A320 --flight flight001
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.visualize import run_visualize

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    run_visualize(arch="opensky", config=CONFIG)


if __name__ == "__main__":
    main()
# %%
