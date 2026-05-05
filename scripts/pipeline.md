# Pipeline v3 — Commandes

> Test: 1 avion A320, 1 journee (2025-09-01)

Toutes les commandes depuis la racine du projet :

```bash
cd /Users/gabriel/Documents/Code/python/node-fdm-v2
```

---

## 1. aircraft-list

Cree `data/aircraft_db.csv` — liste des icao24 a traiter.

```bash
uv run fdm aircraft-list --config config.yaml --sample-size 1 --query-date 2025-09-01
```

---

## 2. download

Telecharge ADS-B (history + flightlist + EHS) depuis OpenSky → Delta Table.

```bash
uv run fdm download --config config.yaml --start-date 2025-09-01 --end-date 2025-09-02
```

---

## 3. identify

Segmente les vols (gap-based) et assigne `meta_flight_id`.

```bash
uv run fdm identify --config config.yaml
```

---

## 4. preprocess

Resample et lisse les vols : detection des sous-segments par groupe de colonnes (position, altitude, BDS), interpolation lineaire par sous-segment, lissage Savitzky-Golay de la position, et resampling a grille reguliere 4s.

Ajoute 3 flags de gap : `fdm_flag_gap_position`, `fdm_flag_gap_altitude`, `fdm_flag_gap_bds`.

> **Note** : cette etape fait un overwrite de la Delta Table (le nombre de rows change). Toutes les etapes suivantes (flag → split) doivent etre relancees.

```bash
uv run fdm preprocess --config config.yaml
```

---

## 5. flag

Ajoute les colonnes de validite (`fdm_flag_*`) sur chaque vol.

```bash
uv run fdm flag --config config.yaml
```

---

## 6. enrich

Enrichit la Delta Table avec les donnees meteo ERA5.

```bash
uv run fdm enrich --config config.yaml
```

---

## 7. clean-speeds

Nettoie les vitesses BDS (`bds_mach`, `bds_ias_kt`, `bds_tas_kt`) avec :
frozen-run filter, Hampel (window=50, 3 passes), V-shape detector, zigzag-region
detector, ERA fill des longs trous Mode-S, post-fill Hampel, interpolation des
gaps courts, on-ground mask. Produit `bds_mach_clean`, `bds_ias_kt_clean`,
`bds_tas_kt_clean` et la colonne derivee `fdm_tas_from_cas_kt` (CAS_clean → TAS
via ERA T).

> Necessite ERA5 (etape 6 enrich) pour `era_temp_K` et le ERA fill.
> Alimente `segments` qui consomme directement les colonnes `*_clean`.

```bash
uv run fdm clean-speeds --config config.yaml
```

---

## 8. derive

Calcule les colonnes physiques derivees (`fdm_gamma_rad`, `fdm_long_wind_ms`, distances aeroports, etc.).

```bash
uv run fdm derive --config config.yaml
```

---

## 9. segments

Detecte les segments constants sur `bds_mach_clean`, `bds_ias_kt_clean`,
`fdm_tas_from_cas_kt` et construit les colonnes `fdm_*_sel` (mach, cas, vz, alt, gamma)
ainsi que `fdm_tas_target_kt` (cible TAS unifiee Mach→TAS / CAS→TAS / TAS_sel).

```bash
uv run fdm segments --config config.yaml
```

---

## 10. convert

Convertit en unites SI et calcule les derivees temporelles (`fdm_d_*`).

```bash
uv run fdm convert --config config.yaml
```

---

## 11. split

Assigne le split train/val/test par hash icao24.

```bash
uv run fdm split --config config.yaml
```

---
