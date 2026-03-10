# %%
"""01 — Build aircraft database from OpenSky Network.

CLI equivalent::

    fdm aircraft-list --config config.yaml --sample-size 100 --query-date 2025-10-01
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.data import aircraft_list

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    aircraft_list(
        config=CONFIG,
        sample_size=100,
        query_date="2025-10-01",
        dry_run=False,
    )


if __name__ == "__main__":
    main()
# %%
