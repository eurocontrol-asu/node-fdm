# %%
"""10 — Compute dataset statistics (flight counts, hours per split).

CLI equivalent::

    fdm dataset-stats --arch opensky --config config.yaml
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.stats import run_dataset_stats

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    run_dataset_stats(arch="opensky", config=CONFIG)


if __name__ == "__main__":
    main()
# %%
