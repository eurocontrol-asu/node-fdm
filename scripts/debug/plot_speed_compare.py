"""Plot Mach/IAS/TAS: clean vs BDS raw vs ERA5, for N flights."""
from __future__ import annotations

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("data/figures/speed_compare")
OUT.mkdir(parents=True, exist_ok=True)

FLIGHTS = [
    "738284_ISR826_s0",
    "40666a_EZY78QT_s0",
    "4a0443_BTI7FR_s0",
    "a88774_JBU493_s0",
    "44006e_AUA445_s0",
]

df = pl.read_delta("data/flights.delta").filter(pl.col("meta_flight_id").is_in(FLIGHTS))

for fid in FLIGHTS:
    f = df.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
    if f.height == 0:
        print(f"skip {fid}: empty")
        continue
    t = f["raw_timestamp"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Mach
    ax = axes[0]
    ax.plot(t, f["bds_mach"], ".", ms=2, alpha=0.4, color="tab:gray", label="bds_mach (raw)")
    ax.plot(t, f["era_mach"], "-", lw=1, alpha=0.7, color="tab:orange", label="era_mach")
    ax.plot(t, f["bds_mach_clean"], ".", ms=2, color="tab:blue", label="bds_mach_clean")
    ax.set_ylabel("Mach")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # IAS / CAS
    ax = axes[1]
    ax.plot(t, f["bds_ias_kt"], ".", ms=2, alpha=0.4, color="tab:gray", label="bds_ias_kt (raw)")
    ax.plot(t, f["era_cas_kt"], "-", lw=1, alpha=0.7, color="tab:orange", label="era_cas_kt")
    ax.plot(t, f["bds_ias_kt_clean"], ".", ms=2, color="tab:blue", label="bds_ias_kt_clean")
    ax.set_ylabel("IAS / CAS [kt]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # TAS
    ax = axes[2]
    ax.plot(t, f["bds_tas_kt"], ".", ms=2, alpha=0.4, color="tab:gray", label="bds_tas_kt (raw)")
    ax.plot(t, f["era_tas_kt"], "-", lw=1, alpha=0.7, color="tab:orange", label="era_tas_kt")
    ax.plot(t, f["bds_tas_kt_clean"], ".", ms=2, color="tab:blue", label="bds_tas_kt_clean")
    if "fdm_tas_from_cas_kt" in f.columns:
        ax.plot(t, f["fdm_tas_from_cas_kt"], "-", lw=1, alpha=0.85, color="tab:green",
                label="fdm_tas_from_cas_kt (CAS→TAS via ERA T)")
    ax.set_ylabel("TAS [kt]")
    ax.set_xlabel("timestamp")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Speed comparison — {fid}", fontsize=12)
    fig.tight_layout()
    out = OUT / f"{fid}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")
