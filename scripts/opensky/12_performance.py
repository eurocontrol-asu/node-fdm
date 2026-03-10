# %%
"""12 — Generate performance comparison charts (altair)."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl
import yaml


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    figure_dir = data_dir / cfg["paths"]["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(data_dir / "performance.parquet")

    # Rename for display
    rename_map = {
        "Altitude [m]": "altitude",
        "Flight path angle [deg]": "flight path angle",
        "True airspeed [m/s]": "true airspeed",
        "PRED": "prediction",
    }
    for old, new in rename_map.items():
        df = df.with_columns(
            pl.when(pl.col("Variable") == old)
            .then(pl.lit(new))
            .otherwise(pl.col("Variable"))
            .alias("Variable"),
            pl.when(pl.col("Model") == old)
            .then(pl.lit(new))
            .otherwise(pl.col("Model"))
            .alias("Model"),
        )

    # Unit conversions for display
    df = df.with_columns(
        pl.when(pl.col("Variable") == "altitude")
        .then(pl.col("MAE") * 3.28084)  # m → ft
        .when(pl.col("Variable") == "true airspeed")
        .then(pl.col("MAE") * 1.94384)  # m/s → kts
        .otherwise(pl.col("MAE"))
        .alias("MAE")
    )

    # Altair accepts Polars DataFrames natively
    base = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X("MAE"),
            alt.Y("Model").title(None).axis(labelFontSize=0),
            alt.Row("Phase")
            .title(None)
            .header(
                labelOrient="top",
                labelFontSize=13,
                labelFont="Roboto Condensed",
                labelAnchor="end",
                labelAlign="right",
                labelPadding=-18,
            ),
            alt.Color("Model")
            .title(None)
            .legend(
                orient="bottom",
                labelFont="Roboto Condensed",
                labelFontSize=16,
            ),
        )
        .properties(height=20, width=200)
    )

    def make_chart(typecode: str) -> alt.HConcatChart:
        chart = (
            alt.hconcat(
                base.transform_filter(
                    f"datum.Aircraft == '{typecode}' & datum.Variable == 'altitude'"
                ).encode(alt.X("MAE").title("altitude (in ft)")),
                base.transform_filter(
                    f"datum.Aircraft == '{typecode}' & datum.Variable == 'flight path angle'"
                ).encode(alt.X("MAE").title("flight path angle (in deg)")),
                base.transform_filter(
                    f"datum.Aircraft == '{typecode}' & datum.Variable == 'true airspeed'"
                ).encode(alt.X("MAE").title("true airspeed (in kts)")),
            )
            .properties(title=f"{typecode} performance model comparison")
            .configure_title(font="Roboto Condensed", fontSize=18, anchor="start", dy=-10)
            .configure_axisX(
                titleAnchor="start",
                titleFont="Roboto Condensed",
                titleFontSize=14,
                titleFontWeight="normal",
                labelFont="Roboto Condensed",
                labelFontSize=14,
                titlePadding=10,
            )
            .configure_facet(spacing=1)
        )
        chart.save(figure_dir / f"performance_{typecode}.pdf")
        return chart

    for aircraft in df["Aircraft"].unique().to_list():
        make_chart(aircraft)
        print(f"✅ Saved performance_{aircraft}.pdf")


if __name__ == "__main__":
    main()
# %%
