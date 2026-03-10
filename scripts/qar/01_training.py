# %%
"""01 — Train Neural ODE model on QAR data.

CLI equivalent::

    fdm train --arch qar --config config.yaml --typecode A320
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.train import run_training

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    run_training(arch="qar", config=CONFIG, typecode="A320")


if __name__ == "__main__":
    main()
# %%
