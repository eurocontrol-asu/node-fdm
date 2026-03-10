# %%
"""07 — Run BADA 4.2 baseline predictions.

CLI equivalent::

    fdm predict-bada --config config.yaml
    fdm predict-bada --config config.yaml --typecode A320 --jobs 4
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.predict import run_predict_bada

CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    run_predict_bada(config=CONFIG)


if __name__ == "__main__":
    main()
# %%
