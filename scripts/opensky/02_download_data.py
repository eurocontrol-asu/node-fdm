# %%
"""02 — Download ADS-B history data from OpenSky Network.

CLI equivalent::

    fdm download --config config.yaml \\
        --start-date 2024-10-01 --end-date 2025-10-15 --step-hours 480
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.data import download

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    download(
        config=CONFIG,
        start_date="2024-10-01",
        end_date="2025-10-15",
        step_hours=480,
        dry_run=False,
    )


if __name__ == "__main__":
    main()
# %%
