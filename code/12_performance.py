# %%
import sys
import cairosvg
from pathlib import Path

# Détermine le chemin vers la racine du projet
root_path = Path.cwd().parents[0]  # si ton notebook est dans notebooks/
sys.path.append(str(root_path))

from config import DATA_DIR, FIGURE_DIR
import pandas as pd
import os
import altair as alt

df = pd.read_parquet(DATA_DIR / "performance.parquet")
df = (
    df.replace("Altitude [m]", "altitude")
    .replace("Flight path angle [deg]", "flight path angle")
    .replace("True airspeed [m/s]", "true airspeed")
    .replace("PRED", "prediction")
)
df.loc[df.Variable == "altitude", "MAE"] = (
    df.loc[df.Variable == "altitude", "MAE"] * 3.28084
)  # meters to feet
df.loc[df.Variable == "true airspeed", "MAE"] = (
    df.loc[df.Variable == "true airspeed", "MAE"] * 1.94384
)  # m/s to knots
df["BADA_gt_PRED"] = (
    df.groupby(["Aircraft", "Phase", "Variable"])
    .apply(
        lambda g: g.set_index("Model").loc["BADA", "MAE"]
        < g.set_index("Model").loc["prediction", "MAE"]
    )
    .reindex(df.set_index(["Aircraft", "Phase", "Variable"]).index)
    .values
)

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
        alt.Opacity("BADA_gt_PRED:N")
        .title(None)
        .legend(None)
        .scale(range=(0.4, 1)),
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


def chart(typecode: str) -> alt.HConcatChart:
    chart = (
        alt.hconcat(
            base.transform_filter(
                f"datum.Aircraft == '{typecode}' & datum.Variable == 'altitude'"
            ).encode(alt.X("MAE").title("altitude (in ft)")),
            base.transform_filter(
                f"datum.Aircraft == '{typecode}' & datum.Variable == 'flight path angle'"
            ).encode(
                alt.X("MAE").title("flight path angle (in deg)"),
            ),
            base.transform_filter(
                f"datum.Aircraft == '{typecode}' & datum.Variable == 'true airspeed'"
            ).encode(
                alt.X("MAE").title("true airspeed (in kts)"),
            ),
        )
        .properties(title=f"{typecode} performance model comparison")
        .configure_title(
            font="Roboto Condensed", fontSize=18, anchor="start", dy=-10
        )
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
    
    pdf_path = FIGURE_DIR / f"performance_{typecode}.pdf"

    chart.save(pdf_path)
    return chart


for aircraft in df.Aircraft.unique():
    display(chart(aircraft))  # noqa: F821

