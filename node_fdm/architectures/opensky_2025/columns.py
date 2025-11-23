#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Column definitions and unit mappings for the OpenSky 2025 architecture."""

from utils.data.column import Column

from utils.physics.units import (
    nautical_miles,
    knot,
    feet,
    meter,
    dimensionless,
    degree,
    rad,
    feet_per_minute,
    kelvin,
)

col_dist = Column("distance", "dist", "distance_along_track_m", meter)
col_adep_dist = Column("distance ADEP", "adep_dist", "adep_dist", nautical_miles)
col_ades_dist = Column("distance ADES", "ades_dist", "ades_dist", nautical_miles)

col_long = Column("longitude", "long", "longitude", degree)
col_lat = Column("latitude", "lat", "latitude", degree)
col_alt = Column("altitude standard", "alt_std", "altitude", feet)
col_tas = Column("true air speed", "tas", "TAS", knot)
col_cas = Column("calibrated air speed", "cas", "CAS", knot)
col_gs = Column("ground speed", "gs", "groundspeed", knot)
col_mach = Column("mach number", "mach", "Mach", dimensionless)  #, normalize_mode=None
col_vz = Column("vertical speed", "vz", "vertical_rate", feet_per_minute)
col_gamma = Column("flight path angle", "gamma", "gamma_air", rad)

col_alt_sel = Column("selected altitude", "alt_sel", "selected_mcp", feet)
col_vz_sel = Column("selected vertical speed", "vz_sel", "vz_sel", feet_per_minute)
col_mach_sel = Column("selected mach", "mach_sel", "mach_sel", dimensionless)  #, normalize_mode=None
col_cas_sel = Column("selected cas", "cas_sel", "cas_sel", knot)
col_temp = Column("temperature", "temp", "temperature", kelvin)
col_long_wind_spd = Column("longitudinal wind speed", "long_wind", "long_wind", knot)
col_alt_diff = Column("altitude difference from selected", "alt_diff_sel", None, feet)
