"""Vectorized WMM declination — bit-exact port of pygeomag.GeoMag.calculate."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
from pygeomag import GeoMag  # type: ignore[import-untyped]

__all__ = ["declination_vec"]


def declination_vec(  # noqa: PLR0915 — single numerical kernel, splitting hurts clarity
    geo: GeoMag,
    glat: npt.NDArray[np.float64],
    glon: npt.NDArray[np.float64],
    alt_km: npt.NDArray[np.float64],
    decimal_year: float,
) -> npt.NDArray[np.float64]:
    """Magnetic declination in degrees, vectorized over N samples.

    Reuses *geo*'s loaded coefficients (``_c``, ``_cd``, ``_k``, ``_fn``,
    ``_fm``, ``_p``) so the numerical result matches
    :meth:`pygeomag.GeoMag.calculate` bit-for-bit.  Only the per-sample
    Python loop is replaced with NumPy ops over arrays of shape ``(N,)``.
    """
    geo._load_coefficients()
    maxord = geo._maxord
    size = geo._size
    c = np.array(geo._c, dtype=np.float64)
    cd = np.array(geo._cd, dtype=np.float64)
    k = np.array(geo._k, dtype=np.float64)
    fn = np.array(geo._fn, dtype=np.float64)
    fm = np.array(geo._fm, dtype=np.float64)
    snorm = np.array(geo._p, dtype=np.float64)

    glat_arr = np.asarray(glat, dtype=np.float64)
    glon_arr = np.asarray(glon, dtype=np.float64)
    alt = np.asarray(alt_km, dtype=np.float64)
    n_samples = glat_arr.size

    a = 6378.137
    b = 6356.7523142
    re = 6371.2
    a2 = a * a
    b2 = b * b
    c2_e = a2 - b2
    a4 = a2 * a2
    b4 = b2 * b2
    c4 = a4 - b4

    dt = decimal_year - geo._epoch

    rlon = np.radians(glon_arr)
    rlat = np.radians(glat_arr)
    srlon = np.sin(rlon)
    crlon = np.cos(rlon)
    srlat = np.sin(rlat)
    crlat = np.cos(rlat)
    srlat2 = srlat * srlat
    crlat2 = crlat * crlat

    q = np.sqrt(a2 - c2_e * srlat2)
    q1 = alt * q
    q2 = ((q1 + a2) / (q1 + b2)) ** 2
    ct = srlat / np.sqrt(q2 * crlat2 + srlat2)
    st = np.sqrt(1.0 - ct * ct)
    r2 = (alt * alt) + 2.0 * q1 + (a4 - c4 * srlat2) / (q * q)
    r = np.sqrt(r2)
    d_geo = np.sqrt(a2 * crlat2 + b2 * srlat2)
    ca = (alt + d_geo) / r
    sa = c2_e * crlat * srlat / (r * d_geo)

    sp = np.zeros((size, n_samples), dtype=np.float64)
    cp = np.zeros((size, n_samples), dtype=np.float64)
    sp[0] = 0.0
    cp[0] = 1.0
    sp[1] = srlon
    cp[1] = crlon
    for m in range(2, maxord + 1):
        sp[m] = sp[1] * cp[m - 1] + cp[1] * sp[m - 1]
        cp[m] = cp[1] * cp[m - 1] - sp[1] * sp[m - 1]

    p_flat = np.zeros((size * size, n_samples), dtype=np.float64)
    dp = np.zeros((size, size, n_samples), dtype=np.float64)
    p_flat[0] = snorm[0]
    pp = np.zeros((size, n_samples), dtype=np.float64)
    pp[0] = 1.0

    aor = re / r
    ar = aor * aor
    bt = np.zeros(n_samples)
    bp = np.zeros(n_samples)
    br = np.zeros(n_samples)
    bpp = np.zeros(n_samples)

    pole = st == 0.0

    for n in range(1, maxord + 1):
        ar = ar * aor
        for m in range(0, n + 1):
            if n == m:
                p_flat[n + m * size] = st * p_flat[(n - 1) + (m - 1) * size]
                dp[m, n] = st * dp[m - 1, n - 1] + ct * p_flat[(n - 1) + (m - 1) * size]
            elif n == 1 and m == 0:
                p_flat[n + m * size] = ct * p_flat[(n - 1) + m * size]
                dp[m, n] = ct * dp[m, n - 1] - st * p_flat[(n - 1) + m * size]
            elif n > 1 and n != m:
                p_flat[n + m * size] = (
                    ct * p_flat[(n - 1) + m * size] - k[m, n] * p_flat[(n - 2) + m * size]
                )
                dp[m, n] = (
                    ct * dp[m, n - 1] - st * p_flat[(n - 1) + m * size] - k[m, n] * dp[m, n - 2]
                )

            tc_mn = c[m, n] + dt * cd[m, n]
            if m != 0:
                tc_nm1 = c[n, m - 1] + dt * cd[n, m - 1]

            par = ar * p_flat[n + m * size]
            if m == 0:
                temp1 = tc_mn * cp[m]
                temp2 = tc_mn * sp[m]
            else:
                temp1 = tc_mn * cp[m] + tc_nm1 * sp[m]
                temp2 = tc_mn * sp[m] - tc_nm1 * cp[m]
            bt = bt - ar * temp1 * dp[m, n]
            bp = bp + fm[m] * temp2 * par
            br = br + fn[n] * temp1 * par

            if m == 1:
                if n == 1:
                    pp[n] = pp[n - 1]
                else:
                    pp[n] = ct * pp[n - 1] - k[m, n] * pp[n - 2]
                parp = ar * pp[n]
                bpp = bpp + fm[m] * temp2 * parp

    bp_final = np.where(
        pole,
        bpp,
        np.divide(bp, st, where=~pole, out=np.zeros_like(bp)),
    )
    bx = -bt * ca - br * sa
    by = bp_final
    return cast("npt.NDArray[np.float64]", np.degrees(np.arctan2(by, bx)))
