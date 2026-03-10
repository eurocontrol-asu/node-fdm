# %%
"""06 — Predict flight trajectories using trained Neural ODE models.

CLI equivalent::

    fdm predict --arch opensky --config config.yaml
    fdm predict --arch opensky --config config.yaml --typecode A320 --device cuda
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.predict import run_predict

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    run_predict(arch="opensky", config=CONFIG)


if __name__ == "__main__":
    main()
# %%
