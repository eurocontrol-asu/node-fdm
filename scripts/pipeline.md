# Pipeline v3 — Commands

> Test: 1 A320 aircraft, 1 day (2025-09-01)

All commands run from the project root:

```bash
cd /Users/gabriel/Documents/Code/python/node-fdm-v2
```

---

## 1. aircraft-list

Creates `data/aircraft_db.csv` — the list of icao24 to process.

```bash
uv run fdm aircraft-list --config config.yaml --sample-size 1 --query-date 2025-09-01
```

---

## 2. download

Downloads ADS-B (history + flightlist + EHS) from OpenSky into the **system-owned** parquet cache at `data/raw/{kind}/date=*/icao24=*/data.parquet` (gitignored, regenerable, safe to `rm -rf`). Diff-based: only the `(date, icao24)` entries missing from the cache are fetched. By default, `decode` is auto-chained to produce the Delta Table — pass `--no-decode` to populate the cache only.

```bash
uv run fdm download --config config.yaml --start-date 2025-09-01 --end-date 2025-09-02
```

Additional flags:

- `--no-decode`: only populate the `data/raw/` cache, skip `decode` (no Delta produced).
- `--force-refresh`: ignore the cache and re-fetch every `(date, icao24)`.

---

## 2b. decode

Rebuilds `data/flights.delta` from the existing `data/raw/` parquet cache. Makes **zero** OpenSky calls — useful for re-running the schema/typecode pipeline after a code change without re-downloading.

`decode` is auto-chained at the end of `download` by default; invoke it standalone (or via `make decode`) only when re-running the decoding pass against an already-warm raw cache (e.g. after a `_RawEHSDecoder` change, a rename-map update, or a typecode refresh in `aircraft_db.csv`).

```bash
uv run fdm decode --config config.yaml --start-date 2025-09-01 --end-date 2025-09-02
```

Additional flags:

- `--icao24-filter PATH`: file with one icao24 per line; intersected with `aircraft_db.csv` to restrict the demanded set.
- `--dry-run`: validate config without writing the Delta.

Missing `(date, icao24)` cache entries are silently skipped; a single `decode_skipped_uncached` warning is logged at the end with the total count.

---

## 3. identify

Segments flights (gap-based) and assigns `meta_flight_id`.

```bash
uv run fdm identify --config config.yaml
```

---

## 4. preprocess

Resamples and smooths flights: sub-segment detection per column group (position, altitude, BDS), linear interpolation per sub-segment, Savitzky-Golay smoothing of position, and resampling onto a regular 4s grid.

Adds 3 gap flags: `fdm_flag_gap_position`, `fdm_flag_gap_altitude`, `fdm_flag_gap_bds`.

> **Note**: this step overwrites the Delta Table (row count changes). Every downstream step (flag → split) must be rerun.

```bash
uv run fdm preprocess --config config.yaml
```

---

## 5. flag

Adds the validity columns (`fdm_flag_*`) on each flight.

```bash
uv run fdm flag --config config.yaml
```

---

## 6. enrich

Enriches the Delta Table with ERA5 weather data.

```bash
uv run fdm enrich --config config.yaml
```

---

## 7. clean-speeds

Cleans the BDS speeds (`bds_mach`, `bds_ias_kt`, `bds_tas_kt`) with:
frozen-run filter, Hampel (window=50, 3 passes), V-shape detector, zigzag-region
detector, ERA fill for long Mode-S gaps, post-fill Hampel, short-gap
interpolation, on-ground mask. Produces `bds_mach_clean`, `bds_ias_kt_clean`,
`bds_tas_kt_clean` and the derived column `fdm_tas_from_cas_kt` (CAS_clean → TAS
via ERA T).

> Requires ERA5 (step 6 enrich) for `era_temp_K` and the ERA fill.
> Feeds `segments`, which consumes the `*_clean` columns directly.

```bash
uv run fdm clean-speeds --config config.yaml
```

---

## 8. derive

Computes the derived physical columns (`fdm_gamma_rad`, `fdm_long_wind_ms`, airport distances, etc.).

```bash
uv run fdm derive --config config.yaml
```

---

## 9. segments

Detects constant segments on `bds_mach_clean`, `bds_ias_kt_clean`,
`fdm_tas_from_cas_kt` and builds the `fdm_*_sel` columns (mach, cas, vz, alt, gamma)
along with `fdm_tas_target_kt` (unified TAS target Mach→TAS / CAS→TAS / TAS_sel).

```bash
uv run fdm segments --config config.yaml
```

---

## 10. convert

Converts to SI units and computes the time derivatives (`fdm_d_*`).

```bash
uv run fdm convert --config config.yaml
```

---

## 11. split

Assigns the train/val/test split via icao24 hashing.

```bash
uv run fdm split --config config.yaml
```

---
