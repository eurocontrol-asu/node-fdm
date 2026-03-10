# %%
"""04 — Attach weather data, compute flight parameters, and create splits.

CLI equivalent::

    fdm process --arch opensky --config config.yaml
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.data import process

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    process(arch="opensky", config=CONFIG, dry_run=False)


if __name__ == "__main__":
    main()
# %%
