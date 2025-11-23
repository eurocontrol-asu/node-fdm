# %%
import sys
from pathlib import Path

root_path = Path.cwd().parents[1]
sys.path.append(str(root_path))

from config import DATA_DIR, FIGURE_DIR

import altair as alt  # type: ignore
import pandas as pd  # type: ignore

dfi = pd.read_parquet(DATA_DIR / "example2.parquet")


# Create a mapping for renaming the values
rename_mapping = {
    "alt_std_m": "observed",
    "bada_alt_std_m": "BADA",
    "pred_alt_std_m": "predicted",
    "alt_sel_m": "SELECTED",
}

# Apply the renaming using transform_calculate
base = (
    alt.Chart(dfi)
    .mark_line()
    .encode(
        x=alt.X("timestamp").title(None).axis(titleAnchor="end", grid=False),
        color=alt.Color("renamed_source:N", title=None)
        .scale(
            domain=["selected", "predicted", "observed", "BADA"],
            range=["#79706e", "#4c78a8", "#f58518", "#54a24b"],
        )
        .legend(
            symbolStrokeWidth=8,
            orient="bottom",
            labelFont="Roboto Condensed",
            labelFontSize=16,
        ),
        strokeDash=alt.StrokeDash(
            "renamed_source:N",
            scale=alt.Scale(
                domain=["selected", "predicted", "observed", "BADA"],
                range=[[6, 3], [1, 0], [1, 0], [1, 0]],
            ),
            legend=None,
        ),
    )
    .properties(width=400, height=200)
)
chart = alt.vconcat(
    # alt.hconcat(
    base.transform_fold(
        ["alt_std_m", "bada_alt_std_m", "pred_alt_std_m", "alt_sel_m"],
        as_=["source", "altitude"],
    )
    .transform_calculate(
        renamed_source='datum.source == "alt_std_m" ? "observed" : datum.source == "bada_alt_std_m" ? "BADA" : datum.source == "alt_sel_m"? "selected": "predicted"'
    )
    .transform_calculate(altitude="datum.altitude / 0.3048")
    .encode(
        y=alt.Y("altitude:Q")
        .title("altitude (in ft)")
        .axis(titleAnchor="end", titleAngle=0, titleAlign="left", titleY=-10),
    ),
    base.transform_fold(
        ["cas_ms", "bada_cas_ms", "pred_cas_ms", "cas_sel_ms"],
        as_=["source", "cas"],
    )
    .transform_calculate(
        renamed_source='datum.source == "cas_ms" ? "observed" : datum.source == "bada_cas_ms" ? "BADA" : datum.source == "cas_sel_ms"? "selected": "predicted"'
    )
    .transform_calculate(cas="datum.cas / 0.514444")
    .encode(
        y=alt.Y("cas:Q")
        .title("CAS (in kts)")
        .axis(titleAnchor="end", titleAngle=0, titleAlign="left", titleY=-10),
    ),
    # ),
    base.transform_fold(
        ["vz_ms", "bada_vz_ms", "pred_vz_ms", "vz_sel_ms"],
        as_=["source", "gamma"],
    )
    .transform_calculate(
        renamed_source='datum.source == "vz_ms" ? "observed" : datum.source == "bada_vz_ms" ? "BADA" : datum.source == "vz_sel_ms"? "selected": "predicted"'
    )
    .transform_calculate(gamma="datum.gamma * 196.850394")  # m/s to ft/min
    .encode(
        y=alt.Y("gamma:Q")
        .title("vertical speed (in ft/min)")
        .axis(titleAnchor="end", titleAngle=0, titleAlign="left", titleY=-10),
    ),
    # base.transform_fold(
    #     ["mass_kg", "bada_mass_kg", "pred_mass_kg"],
    #     as_=["source", "mass"],
    # )
    # .transform_calculate(
    #     renamed_source='datum.source == "mass_kg" ? "observed" : datum.source == "bada_mass_kg" ? "BADA" : datum.source == "alt_sel_m"? "selected": "predicted"'
    # )
    # .encode(
    #     y=alt.Y("mass:Q")
    #     .title("mass (in kg)")
    #     .scale(zero=False)
    #     .axis(titleAnchor="end", titleAngle=0, titleAlign="left", titleY=-10),
    # ),
).configure_axis(
    labelFont="Roboto Condensed",
    labelFontSize=14,
    titleFont="Roboto Condensed",
    titleFontSize=18,
)
chart.save(FIGURE_DIR / "traj_example2.pdf")
chart
