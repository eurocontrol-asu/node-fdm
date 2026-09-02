# node-fdm-pipeline

CLI pipeline for the **node-fdm** Neural ODE framework — typed commands for flight dynamics data processing, training, and evaluation.

## Features

- Typed CLI (cyclopts) for the full v3 data pipeline: download → identify → preprocess → flag → enrich → derive → segments → convert → split
- Pydantic v2 `PipelineConfig` loaded from a single YAML
- Architecture registry (currently `adsb`) wiring preprocessing/segmentation per spec
- Training, resume, predict, evaluate, and visualization commands for Neural ODE models
- BADA 4.2 baseline predictions (`fdm predict-bada`)

## Installation

```bash
# From workspace root
uv sync

# With visualization support
pip install node-fdm-pipeline[viz]
```

## Usage

```bash
fdm --help
fdm version
fdm preprocess --config config.yaml
fdm identify --config config.yaml
fdm train --arch opensky --config config.yaml --typecode A320 --method rk4 --seq-len 200
fdm resume --model models/node_adsb_v1_A320 --config config.yaml --epochs 200 --lr 1e-4
fdm predict --arch opensky --config config.yaml --device cuda:0
fdm evaluate --arch opensky --config config.yaml
```

## Fleet downloads

A historical fleet dry-run needs no campaign option. It validates the discovered cohorts
and prints the planned dates without contacting OpenSky:

```bash
fdm download-fleet --fleet-dir fleet --dry-run
```

Campaign mode is selected by supplying the complete coordinated-run input set:

```bash
fdm download-fleet \
  --fleet-dir fleet \
  --selection run/selection.digest \
  --resolved-config run/resolved-config.json \
  --profile run/profile.json \
  --lease-path /shared/node-fdm/download-fleet.lease \
  --lease-ttl-s 120 \
  --disk-min-gib 8.5
```

The six campaign options are atomic: a partial set exits non-zero and names every missing
input before creating a lease, constructing the provider, or writing payloads. A complete
set validates the recorded digests and acquires the shared lease before remote acquisition;
if another owner holds that lease, the command exits non-zero without publishing payloads.

A successful zero-row response is persisted as an explicit absence receipt for the exact
(day, acquisition kind, aircraft set) combination. Later consumers treat every aircraft
listed by that receipt as cached, so retries do not repeat a live request merely to
rediscover the same absence. See [Raw-cache absence receipts](docs/reference/raw-cache.md)
for the persisted contract.

### Fleet decoding

The historical decoder remains available without campaign options:

```bash
fdm decode-fleet --fleet-dir fleet
```

For a coordinated campaign, pass the same six inputs as one atomic set:

```bash
fdm decode-fleet \
  --fleet-dir fleet \
  --selection run/selection.digest \
  --resolved-config run/resolved-config.json \
  --profile run/profile.json \
  --lease-path /shared/node-fdm/decode-fleet.lease \
  --lease-ttl-s 120 \
  --disk-min-gib 8.5
```

A partial set exits non-zero and reports every missing input before creating a shared
lease or local fallback lock. A complete set validates the resume identity, acquires
the supplied shared lease, then starts decoding. The command records that identity in
`fleet/decode-fleet.resume.json`; a later run whose selection, resolved configuration,
or profile differs exits before producing another decoded artefact.

### Fleet enrichment

Historical enrichment remains available without campaign options:

```bash
fdm enrich-fleet --fleet-dir fleet
```

A coordinated enrichment campaign uses the same atomic six-input contract:

```bash
fdm enrich-fleet \
  --fleet-dir fleet \
  --selection run/selection.digest \
  --resolved-config run/resolved-config.json \
  --profile run/profile.json \
  --lease-path /shared/node-fdm/enrich-fleet.lease \
  --lease-ttl-s 120 \
  --disk-min-gib 8.5
```

A partial set exits non-zero and reports every missing input before creating a lease
or fallback lock. With a complete set, the shared lease path and the authoritative
campaign identity are validated before the enrichment plan is built. The identity is
recorded under `--data-root` (or `--fleet-dir` when no data root is supplied); a later
run whose selection, resolved configuration, or profile differs exits non-zero before
lease acquisition, weather-provider construction, or artefact publication. The lease is
then acquired before the weather provider is constructed, so a rejected or unavailable
campaign leaves the enriched-output tree unchanged and creates no lease or fallback lock.

Commands that load coordinated settings from pipeline YAML use the equivalent
deployment-owned model:

```yaml
fleet_run:
  lease_path: ~/shared/trino.lease
  lease_ttl_s: 120
  disk_min_gib: 8.5
```

When present, all three `fleet_run` values are required, `lease_ttl_s` and
`disk_min_gib` must be strictly positive, and `lease_path` is expanded and resolved to
an absolute path.

## Commands

| Command | Description | Status |
|---|---|---|
| `fdm download-fleet` | Plan historical fleet downloads or run a coordinated campaign behind a shared lease | ✅ Implemented |
| `fdm decode-fleet` | Decode historical fleets or resume a digest-checked campaign behind a shared lease | ✅ Implemented |
| `fdm enrich-fleet` | Enrich historical fleets or run a coordinated campaign behind a shared lease | ✅ Implemented |
| `fdm preprocess` | Resample flights: subsegment detection, position smoothing, fixed-rate resampling | ✅ Implemented |
| `fdm identify` | Segment at gaps, assign flight IDs, join flightlist metadata | ✅ Implemented |
| `fdm derive` | Compute derived physics columns (gamma, wind, distance) — étape 4 | ✅ Implemented |
| `fdm train` | Train Neural ODE models (`--method euler\|rk4`) | Placeholder (AXM-363) |
| `fdm resume` | Resume training from checkpoint (`--model`, `--overwrite`) | ✅ Implemented |
| `fdm predict` | Predict with trained models | Placeholder (AXM-363) |
| `fdm predict-bada` | BADA 4.2 baseline predictions | Placeholder (AXM-363) |
| `fdm evaluate` | Compute error metrics by phase | Placeholder (AXM-363) |
| `fdm process` | Process flight data + split | Placeholder (AXM-362) |
| `fdm split` | Assign train/val/test split column (`meta_split`) by ICAO group | ✅ Implemented |
| `fdm dataset-stats` | Dataset split statistics | Placeholder (AXM-364) |
| `fdm visualize` | Prediction comparison plots | Placeholder (AXM-364) |
| `fdm version` | Print version | ✅ Implemented |

## Development

<!-- 170 tests -->
```bash
uv run pytest packages/node-fdm-pipeline/ -q
uv run ruff check packages/node-fdm-pipeline/
uv run mypy packages/node-fdm-pipeline/src/
```

## License

Licensed under the European Union Public Licence v1.2 (EUPL-1.2).
