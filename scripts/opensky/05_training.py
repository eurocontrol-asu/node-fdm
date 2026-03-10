# %%
"""05 — Train Neural ODE models for all typecodes.

CLI equivalent::

    fdm train --arch opensky --config config.yaml
    fdm train --arch opensky --config config.yaml --typecode A320 --epochs 100 --lr 0.001
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.train import run_training

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    run_training(arch="opensky", config=CONFIG)


if __name__ == "__main__":
    main()
# %%
