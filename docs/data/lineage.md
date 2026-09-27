# Data lineage

## Real

```mermaid
flowchart TD
    A[REAL PHYSICAL RECORDINGS<br/>240 files · phone cameras · 2026-09-26<br/>LOCAL_REAL · not published] --> B[FRAME-LEVEL PERCEPTION<br/>detector boxes/labels · face landmarks · hand state<br/>OBSERVED]
    B --> C[TEMPORAL FEATURES<br/>tracks · persistence · ln-area growth · eye-closure window · durations<br/>INFERRED]
    C --> D[RISK ASSESSMENT<br/>score · level · hazard type · reason · alarm recommendation<br/>INFERRED]
    D --> E[REAL EVENTS<br/>66 rows · run local_real_run_01]
    E --> F[JSONL / PARQUET / EVIDENCE<br/>data/sample/ · evidence frames + metadata.json]
```

## Synthetic (separate)

```mermaid
flowchart TD
    R[REAL TRACKED OBSERVATIONS<br/>value ranges of real events, read-only] -.parent ids only.-> V
    V[CONTROLLED FEATURE VARIATIONS<br/>constructed boxes + scripted driver signals<br/>SYNTHETIC]
    V --> S[SYNTHETIC SCENARIOS<br/>1,441 rows · 24 instances · seed 42<br/>data/synthetic/]
    S --> T[RISK ENGINE VALIDATION<br/>23/24 matched the expectation]
    style V fill:#f1f0ec,stroke:#8a8f96,stroke-dasharray: 5 5
    style S fill:#f1f0ec,stroke:#8a8f96,stroke-dasharray: 5 5
    style T fill:#f1f0ec,stroke:#8a8f96,stroke-dasharray: 5 5
```

## Labels at each step

| Step | Label |
| --- | --- |
| Raw recording | `data_source = LOCAL_REAL` on every derived row |
| Per-field | `METADATA` / `OBSERVED` / `INFERRED` (real schema); `METADATA` / `SYNTHETIC_INPUT` / `ENGINE_ON_SYNTHETIC` / `VALIDATION` (synthetic schema) |
| Per-record | `observation_type`: `INFERRED` for risk/hazard events, `OBSERVED` for hand-state changes, `SYNTHETIC` for synthetic rows |
| Location | real: `gps_source = UNAVAILABLE`, coordinates null; synthetic: `gps_source = SYNTHETIC`, `location_source = SYNTHETIC_REFERENCE` |
| Storage | real and synthetic datasets are separate roots; each writer refuses the other kind |

A third, separately labelled dataset, `SYNTHETIC_COMBINATION`, was produced as a software test. It pairs a real driver clip with an unrelated real front clip, and it is not a drive. It is not included in this repository.
