# SYNTHETIC scenario dataset — not real observations

| File | Contents |
| --- | --- |
| `scenarios.jsonl` / `scenarios.parquet` | 1,441 rows: one row per 10 Hz step of 24 constructed scenario instances (12 types × 2 variants, seed 42) |
| `manifest.json` | Seed, counts, parent real-event ids used as value references, provenance policy, expected-vs-actual validation |

- **Labels:** every row has `data_source = observation_type = timestamp_source = gps_source = "SYNTHETIC"` and `location_source = "SYNTHETIC_REFERENCE"`.
- **Coordinates:** the GPS coordinates are synthetic, around a generic reference point. They are not collection locations.
- **Purpose:** controlled stress tests of the rule-based risk engine. The rows are **not** evidence of real-world performance and must never be merged with `data/sample/`.
- **Regenerate:** `python scripts/generate_synthetic_scenarios.py`. The script writes to `data/simulated/` locally.
- **Schema:** [`../schema/synthetic_schema.json`](../schema/synthetic_schema.json).
