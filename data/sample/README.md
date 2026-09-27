# Real event sample (LOCAL_REAL)

| File | Contents |
| --- | --- |
| `real_events.jsonl` | All **66 real events** from run `local_real_run_01`, one JSON object per line, 60 fields |
| `real_events.parquet` | The same 66 rows as typed Parquet |
| `real_events_sample.json` | 10 representative events, pretty-printed |
| `evidence/` | Unaltered evidence (frame + `metadata.json`) for 3 events: `event_000001`, `event_000005` (dog at night), `event_000012` (pedestrian on a crossing) |

- **Labels:** every row has `data_source = "LOCAL_REAL"`, and GPS is `UNAVAILABLE`.
- **Evidence paths:** `evidence_path` / `evidence_files` in the rows are relative to `evidence/`.
- **Withheld evidence:** frames for the other 45 events with evidence are withheld for privacy (driver face, number plates, bystanders).
- **Schema:** [`../schema/real_event_schema.json`](../schema/real_event_schema.json) · explained in [DATA_SCHEMA.md](../../DATA_SCHEMA.md).
