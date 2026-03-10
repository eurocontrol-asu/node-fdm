# %%
"""03 — Preprocess raw ADS-B data (EHS decoding, filtering, resampling).

CLI equivalent (per file)::

    fdm preprocess --config config.yaml \
        --history-file /path/to/history_20241001.parquet --workers 4

For batch processing with GNU parallel::

    ls download/history_*.parquet \
        | parallel -j 20 fdm preprocess --config config.yaml --history-file {}
"""

from __future__ import annotations

from pathlib import Path

from node_fdm_pipeline.commands.data import preprocess

CONFIG = Path(__file__).parent / "config.yaml"


def main(history_file: Path, *, workers: int = 1) -> None:
    preprocess(
        config=CONFIG,
        history_file=history_file,
        workers=workers,
        dry_run=False,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 03_preprocess_data.py <history_file> [--workers N]")
        raise SystemExit(1)

    w = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--workers" else 1
    main(Path(sys.argv[1]), workers=w)
# %%
