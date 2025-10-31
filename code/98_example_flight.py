# %%

import matplotlib.pyplot as plt

import pandas as pd

dfi = pd.read_parquet("example.parquet")


# Variables
cols = ["alt_std_m", "tas_ms", "gamma_rad", "mass_kg"]
cols_bada = ["bada_" + col for col in cols]
cols_pred = ["pred_" + col for col in cols]

time = dfi["timestamp"].values


# %%
import altair as alt

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
            range=["#e45756", "#4c78a8", "#f58518", "#54a24b"],
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
chart.save("traj_example.pdf")
chart
# %%


# %%
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# --- 1. Altitude ---
axs[0].plot(dfi.alt_std_m, color="r", label="Observed")
axs[0].plot(dfi.pred_alt_std_m, color="b", label="Predicted")
axs[0].plot(dfi.bada_alt_std_m, color="g", label="BADA")
axs[0].plot(dfi.alt_sel_m, "--", color="k", lw=0.5, label="Selected")
axs[0].set_title("Altitude [m]")
axs[0].set_xlabel("Time step")
axs[0].set_ylabel("Altitude [m]")
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.5)

# --- 2. True Airspeed ---
axs[1].plot(dfi.cas_ms, color="r", label="Observed")
axs[1].plot(dfi.pred_cas_ms, color="b", label="Predicted")
axs[1].plot(dfi.bada_cas_ms, color="g", label="BADA")
axs[1].plot(dfi.cas_sel_ms, "--", color="k", lw=0.5, label="Selected")
axs[1].set_title("Calibrated Airspeed [m/s]")
axs[1].set_xlabel("Time step")
axs[1].set_ylabel("CAS [m/s]")
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.5)

# --- 3. Flight Path Angle ---
axs[2].plot(dfi.vz_ms, color="r", label="Observed")
axs[2].plot(dfi.pred_vz_ms, color="b", label="Predicted")
axs[2].plot(dfi.bada_vz_ms, color="g", label="BADA")
axs[2].plot(dfi.vz_sel_ms, "--", color="k", lw=0.5, label="Selected")
axs[2].set_title("Vertical Speed [m/s]")
axs[2].set_xlabel("Time step")
axs[2].set_ylabel("Vertical Speed [m/s]")
axs[2].legend()
axs[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# %%
import altair as alt

import pandas as pd

# Sample data
data = pd.DataFrame(
    {"x": [1, 2, 3, 4], "y": [10, 15, 13, 17], "category": ["A", "A", "A", "A"]}
)

# Basic line chart
chart = (
    alt.Chart(data)
    .mark_line()
    .encode(
        x="x",
        y="y",
        color=alt.Color(
            "category:N",
            legend=alt.Legend(
                symbolStrokeWidth=8,
                symbolSize=100,
            ),
        ),
    )
    .properties(width=400, height=300)
)

chart
