# Generated reports

These are copies of the reports the scripts wrote during the real runs. Paths inside them refer to the working project's `outputs/` layout; the corresponding public artifacts are in [`../results/`](../results/) and [`../data/`](../data/). Some images they reference are not published, for privacy.

| Report | Produced by |
| --- | --- |
| `dataset_audit.md` | `scripts/audit_dataset.py` — raw dataset audit (240 files) |
| `hand_dataset_split.md`, `hand_test_report.md`, `hand_test_metrics.json` | `scripts/prepare_hand_dataset.py`, `scripts/evaluate_hand_model.py` |
| `driver_camera_domain_comparison.md` | `scripts/compare_driver_domain.py` — training images vs the real driver video |
| `road_perception_real_data.md` | `scripts/test_road_perception.py` |
| `tracking_real_data.md` | `scripts/test_road_tracking.py` |
| `risk_engine_real_data.md` | `scripts/test_risk_engine.py` |
| `event_dataset.md` | `scripts/record_events.py` — the 66-event real dataset |
| `synthetic_scenario_report.md` | `scripts/generate_synthetic_scenarios.py` (SYNTHETIC) |
| `final_demo_report.md` | `scripts/run_demo.py` |
| `safety_summary.md`, `event_statistics.csv`, `final_system_status.md` | `scripts/generate_safety_report.py` |
