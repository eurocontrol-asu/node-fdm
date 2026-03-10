# %%
"""02 — Predict flight trajectories using trained QAR model.

CLI equivalent::

    fdm predict --arch qar --config config.yaml --typecode A320
    fdm predict --arch qar --config config.yaml --typecode A320 --device cuda
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.predict import run_predict

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    run_predict(arch="qar", config=CONFIG, typecode="A320")


if __name__ == "__main__":
    main()
# %%
