"""For up to 500 icao24, query OpenSky and report flights/icao seen per year.

Usage:
    uv run python scripts/debug/check_icao_yearly.py --icaos 3c6589,4b1815
    uv run python scripts/debug/check_icao_yearly.py --model "717-200"
    uv run python scripts/debug/check_icao_yearly.py --typecode A320 --years 2018-2025

Output: a table (year, n_flights, n_icao_seen, ratio) saved to CSV.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import cyclopts
import polars as pl
from cyclopts import Parameter
from traffic.data import aircraft, opensky

app = cyclopts.App()

MAX_ICAOS = 500


def _resolve_icaos(
    icaos: str | None,
    model: str | None,
    typecode: str | None,
) -> list[str]:
    if icaos:
        out = [s.strip().lower() for s in icaos.split(",") if s.strip()]
    else:
        df = pl.from_pandas(
            aircraft.data[["icao24", "model", "typecode", "operator", "operatoricao"]]
        )
        commercial = df.filter(
            (pl.col("operator").is_not_null() & (pl.col("operator") != ""))
            | (pl.col("operatoricao").is_not_null() & (pl.col("operatoricao") != ""))
        )
        if model:
            sub = commercial.filter(pl.col("model") == model)
        elif typecode:
            sub = commercial.filter(pl.col("typecode") == typecode)
        else:
            raise SystemExit("Provide --icaos OR --model OR --typecode")
        out = sub.select("icao24").unique().to_series().to_list()
    if len(out) > MAX_ICAOS:
        raise SystemExit(f"Too many icao24: {len(out)} > {MAX_ICAOS}")
    return out


@app.default
def main(
    *,
    icaos: Annotated[str | None, Parameter(help="comma-separated icao24 list")] = None,
    model: Annotated[str | None, Parameter(help="aircraft.data model")] = None,
    typecode: Annotated[str | None, Parameter(help="aircraft.data typecode")] = None,
    years: Annotated[str, Parameter(help="year range YYYY-YYYY (inclusive)")] = "2018-2025",
    output: Annotated[Path, Parameter(help="output CSV path")] = Path(
        "data/icao_yearly_coverage.csv"
    ),
) -> None:
    icao_list = _resolve_icaos(icaos, model, typecode)
    y_start, y_end = (int(s) for s in years.split("-"))
    label = icaos or model or typecode or "?"
    print(f"label: {label!r}  -> {len(icao_list)} icao24, years {y_start}-{y_end}")

    opensky.trino_client.connect()
    rows = []
    total_t0 = time.time()
    for year in range(y_start, y_end + 1):
        start = f"{year}-01-01"
        stop = f"{year + 1}-01-01"
        t0 = time.time()
        fl = opensky.flightlist(start, stop, icao24=icao_list)
        dt = time.time() - t0
        if fl is None or len(fl) == 0:
            n_flights = 0
            n_icao_seen = 0
        else:
            fl_pl = pl.from_pandas(fl)
            n_flights = len(fl_pl)
            n_icao_seen = fl_pl["icao24"].n_unique()
        ratio = n_icao_seen / len(icao_list) if icao_list else 0.0
        print(
            f"  {year}: {n_flights:>7} flights, "
            f"{n_icao_seen:>4}/{len(icao_list)} icao seen ({ratio:.0%})  [{dt:.1f}s]"
        )
        rows.append(
            {
                "label": label,
                "year": year,
                "n_icao_in": len(icao_list),
                "n_icao_seen": n_icao_seen,
                "n_flights": n_flights,
                "coverage_ratio": round(ratio, 3),
                "query_seconds": round(dt, 1),
            }
        )
    print(f"total: {time.time() - total_t0:.1f}s")

    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    app()
