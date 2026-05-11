"""Cluster aircraft.data 'model' strings for one typecode into:

- canonical: semantic identifier (e.g. A320-214, A320-200 if generic)
- variant:   "" or "W" (winglets/sharklets)
- naming:    list of all raw 'model' strings that map to (canonical, variant)
- n:         icao24 count

Heuristics for A320 / B738 family. Caller picks the typecode.
"""

from __future__ import annotations

import re
from collections import defaultdict

import polars as pl
from traffic.data import aircraft

WINGLET_RE = re.compile(r"\b\(?(?:W|WL|SL)\)?|/W\b", re.IGNORECASE)
TRIM_PREFIX_RE = re.compile(r"^\s*(?:BOEING|AIRBUS|B|A)\s+", re.IGNORECASE)
PARENS_BOEING_RE = re.compile(r"\(\s*(?:Boeing|Airbus)\s*\)", re.IGNORECASE)


def _has_winglet(model: str) -> bool:
    return bool(WINGLET_RE.search(model)) or "SL" in model.upper().replace("AIRSL", "")


def _strip(model: str) -> str:
    s = model.strip()
    s = PARENS_BOEING_RE.sub("", s)
    s = WINGLET_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s


def canonicalize_a320(model: str) -> str:
    """Return canonical form for an A320-family model string."""
    s = _strip(model).upper()
    s = TRIM_PREFIX_RE.sub("", s)  # leading "Airbus "
    # Match A320-XXX or A320 XXX
    m = re.search(r"A?-?320[\s-]*([0-9]{3})", s)
    if m:
        suffix = m.group(1)
        # ACJ corporate: keep label
        if "ACJ" in model.upper():
            return f"ACJ320-{suffix}"
        return f"A320-{suffix}"
    if "ACJ" in model.upper():
        return "ACJ320-200"
    return "A320-200"  # generic CEO


def canonicalize_b738(model: str) -> str:
    """Return canonical form for a 737-800 family model string."""
    s = model.upper()
    if "BBJ" in s:
        return "737-800BBJ"
    if "BCF" in s or re.search(r"\bSF\b|\(SF\)", s):
        return "737-800F"
    return "737-800"


def cluster(typecode: str) -> pl.DataFrame:
    canonicalize = {"A320": canonicalize_a320, "B738": canonicalize_b738}.get(typecode)
    if canonicalize is None:
        raise ValueError(f"No canonicalizer for {typecode}")

    df = pl.from_pandas(aircraft.data[["icao24", "model", "typecode", "operator"]])
    sub = df.filter(
        (pl.col("typecode") == typecode)
        & pl.col("operator").is_not_null()
        & (pl.col("operator") != "")
    )
    rows = sub.select("icao24", "model").to_dicts()
    buckets: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"icaos": set(), "namings": set()}
    )
    for row in rows:
        m = row["model"] or ""
        key = (canonicalize(m), "W" if _has_winglet(m) else "")
        buckets[key]["icaos"].add(row["icao24"])
        buckets[key]["namings"].add(m)

    out = [
        {
            "canonical": canon,
            "variant": var,
            "naming": sorted(b["namings"]),
            "n": len(b["icaos"]),
        }
        for (canon, var), b in buckets.items()
    ]
    return pl.DataFrame(out).sort(["canonical", "variant"])


if __name__ == "__main__":
    import sys

    tc = sys.argv[1] if len(sys.argv) > 1 else "A320"
    res = cluster(tc)
    with pl.Config(tbl_rows=50, fmt_str_lengths=200):
        print(f"=== {tc} ===")
        print(res)
    print(f"\ntotal icao24: {res['n'].sum()}, distinct canonical: {res['canonical'].n_unique()}")
