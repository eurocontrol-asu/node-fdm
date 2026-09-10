#!/usr/bin/env python
"""Point d'étape de la campagne d'acquisition ADS-B.

Lecture seule : n'ouvre aucune connexion Trino et ne touche pas au cache.
Peut donc tourner à tout moment pendant la campagne — contrairement à une
requête Trino, qui partagerait la queue du compte et volerait le créneau
d'acquisition (la file est par compte, pas par machine).

Usage :
    ./.venv/bin/python campaign_status.py

Le pourcentage d'avancement est calculé en **aircraft-days**, pas en dates.
Les dates ne sont pas des unités de coût égales : elles vont de 15 à 3 964
aéronefs. Le débit Trino étant plat (~8 000 lignes/s), le temps suit le
volume de lignes, donc les aircraft-days. Compter en dates surestimait la
durée restante d'un facteur ~2,4.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

CAMPAIGN = Path("/data/common/opensky/campaign")
RAW = Path("/data/common/opensky/raw")
JOURNAL = CAMPAIGN / "acquisition.journal.jsonl"
SELECTION = CAMPAIGN / "selection.csv"
LEASE = Path("/data/common/opensky/_campaign/trino.lease")


def _read_journal() -> list[dict]:
    rows = []
    with JOURNAL.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # ligne tronquée = écriture en cours, pas une erreur
    return rows


def main() -> None:
    import polars as pl

    rows = _read_journal()
    finished = [d for d in rows if d.get("event") == "date_finished"]
    ok = [d for d in finished if d.get("kind") == "success"]
    failed = [d for d in finished if d.get("kind") == "error"]
    done_days = {d["date"] for d in ok}

    sel = pl.read_csv(SELECTION, infer_schema=False).with_columns(
        pl.from_epoch(pl.col("firstseen").cast(pl.Int64), time_unit="s")
        .dt.strftime("%Y%m%d")
        .alias("day")
    )
    per_day = sel.group_by("day").agg(
        pl.len().alias("flights"),
        pl.col("icao24").n_unique().alias("ac"),
    )
    done = per_day.filter(pl.col("day").is_in(list(done_days)))
    rest = per_day.filter(~pl.col("day").is_in(list(done_days)))

    total_ad = int(per_day["ac"].sum())
    done_ad = int(done["ac"].sum())
    pct = done_ad / total_ad * 100 if total_ad else 0.0

    stamps = [
        datetime.fromisoformat(d["timestamp"]) for d in finished if d.get("timestamp")
    ]
    elapsed_h = (max(stamps) - min(stamps)).total_seconds() / 3600 if len(stamps) > 1 else 0.0

    print("=" * 62)
    print("  CAMPAGNE ADS-B — POINT D'ÉTAPE")
    print("=" * 62)

    # --- process ---
    alive = [
        p
        for p in Path("/proc").iterdir()
        if p.name.isdigit()
        and "fleet-campaign" in (p / "cmdline").read_bytes().decode("utf8", "replace")
        if (p / "cmdline").exists()
    ]
    if alive:
        pid = alive[0].name
        rss_kb = 0
        for line in (alive[0] / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
        print(f"  process    : VIVANT  pid={pid}  RSS={rss_kb / 1024 / 1024:.1f} Go")
    else:
        print("  process    : /!\\ ARRÊTÉ — plus aucun fleet-campaign")

    if LEASE.exists():
        lease = json.loads(LEASE.read_text())
        age = datetime.now().timestamp() - lease.get("heartbeat_at", 0)
        state = "frais" if age < 900 else f"/!\\ VIEUX ({age / 60:.0f} min)"
        print(f"  lease      : détenu, heartbeat {state}")

    # --- avancement ---
    print("-" * 62)
    print(f"  dates      : {len(ok)} ok + {len(failed)} err  /  {per_day.height}")
    print(f"  vols       : {int(done['flights'].sum()):,} / {int(per_day['flights'].sum()):,}")
    print(f"  ac-days    : {done_ad:,} / {total_ad:,}")
    print(f"  AVANCEMENT : {pct:.2f} %   (base ac-days)")

    if done_days:
        days = sorted(done_days)
        print(f"  plage      : {days[0]} → {days[-1]}")

    # --- projection ---
    if pct > 0 and elapsed_h > 0:
        total_h = elapsed_h / (pct / 100)
        remaining_h = total_h - elapsed_h
        print("-" * 62)
        print(f"  écoulé     : {elapsed_h:.1f} h")
        print(f"  RESTANT    : ~{remaining_h / 24:.1f} jours  (fin totale ~{total_h / 24:.1f} j)")
        if rest.height:
            ratio = rest["ac"].mean() / done["ac"].mean()
            note = "plus légères" if ratio < 1 else "plus lourdes"
            print(f"  charge     : dates restantes {ratio:.1f}x = {note}")

    # --- disque ---
    print("-" * 62)
    for kind in ("history", "extended", "flightlist"):
        root = RAW / kind
        if not root.exists():
            continue
        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        ndates = len([p for p in root.iterdir() if p.name.startswith("date=")])
        print(f"  {kind:11}: {size / 1024**3:7.2f} Go   {ndates} dates")

    # --- alertes ---
    empty = [k for d in ok for k in set(d.get("empty_kinds") or [])]
    if empty:
        from collections import Counter

        print("-" * 62)
        for kind, n in Counter(empty).items():
            print(f"  /!\\ '{kind}' vide sur {n}/{len(ok)} dates")
            if kind == "extended":
                print("      → pas de BDS, donc pas de bloc longitudinal dans derive")
                print("      → attendu avant ~2020 : rollcall_replies_data4 est vide")

    if failed:
        print("-" * 62)
        print(f"  /!\\ {len(failed)} dates en erreur (rejouables, written=0) :")
        for d in failed[-5:]:
            err = str(d.get("error") or "")[:60].replace("\n", " ")
            print(f"      {d['date']}  {err}")

    print("=" * 62)


if __name__ == "__main__":
    main()
