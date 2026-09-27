"""Combined final demo report (markdown) from one or more demo_summary.json files."""

from __future__ import annotations

from app.alerts.console import fmt_time
from app.demo.runner import NO_SYNC_STATEMENT


def _t(rows, headers) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join("" if c is None else str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return out


def _counts(d: dict | None) -> str:
    return ", ".join(f"{k} {v}" for k, v in (d or {}).items()) or "none"


def render_final_report(summaries: list[tuple[str, dict]]) -> str:
    """summaries: [(output directory as shown in the report, demo_summary dict), ...]."""
    L = ["# Final integrated demo: real video to safety alert", ""]
    L.append(f"> **{NO_SYNC_STATEMENT}**")
    L.append(">")
    L.append("> **Inputs and outputs.**")
    L.append("> - Every demo input below is a real local recording (`LOCAL_REAL`).")
    L.append("> - Every derived value (tracks, risk, events, alerts) is `INFERRED` by the existing prototype components.")
    L.append("> - No synthetic rows are included: the synthetic scenario dataset (`data/simulated/`) and the SYNTHETIC_COMBINATION test (`data/events_synthetic/`) are separate and unused here.")
    L.append("")
    L.append("## Pipelines")
    L.append("")
    L.append("- **Front camera:** video → YOLO26n detections → IoU tracker → risk engine (road-only) → event recorder → alert layer → evidence, timeline and plot.")
    L.append("- **Driver camera:** video → driver perception (face landmarks, eye closure, drowsiness, hand-state classifier) → risk engine (driver-only) → event recorder → alert layer.")
    L.append("- **Existing components:** all are reused unmodified, with no second risk engine.")
    L.append("- **Alert rules:** the alert layer (`app/alerts`) only maps the engine's level changes to alerts:")
    L.append("  - CAUTION: visual/log notice.")
    L.append("  - HIGH: *WARNING: ROAD HAZARD* (or *WARNING: DRIVER STATE* for a driver-only hazard), with an alarm.")
    L.append("  - CRITICAL: *CRITICAL: IMMEDIATE ATTENTION REQUIRED*, with a strong alarm.")
    L.append("- **Explanations:** each alert is explained by the engine's own `risk_reason`. Alerts are decision support, not collision predictions, and do not guarantee safety.")
    L.append("")
    for out_dir, s in summaries:
        inp, pr, pe, risk, ev, al, evi, prov = (s["input"], s["processing"], s["perception"], s["risk"], s["events"],
                                                s["alerts"], s["evidence"], s["provenance"])
        L.append(f"## {s['camera'].capitalize()}-camera demo: `{inp['file_name']}`")
        L.append("")
        where = f"`<DATASET_ROOT>/{inp['dataset_relative_path']}`" if inp.get("dataset_relative_path") else f"`{inp['video']}`"
        L.append(f"- **Exact real input video:** {where}")
        L.append(f"  - sha256 `{inp['sha256'][:16]}…`")
        L.append(f"  - {inp['width']}×{inp['height']}, container {inp['container_fps'] and round(inp['container_fps'], 2)} fps, {inp['container_frame_count']} frames")
        L.append(f"- **Duration processed:** {inp['duration_seconds_processed']} s of video (container duration {inp['container_duration_seconds'] and round(inp['container_duration_seconds'], 2)} s).")
        L.append(f"- **Frames:** {pr['frames_read']} decoded; {pr['frames_processed']} processed at about {pr['sample_fps']} fps. Wall time {pr['wall_seconds']} s.")
        if s["camera"] == "front":
            L.append(f"- **Detections:** {pe['detections']} ({_counts(pe.get('detections_by_class'))}). Detector {pr.get('detector')} on {pr.get('device')}.")
            L.append(f"- **Tracks:** {pe['tracks']} ({_counts(pe.get('tracks_by_class'))}).")
        else:
            L.append(f"- **Driver observations:** {pe.get('driver_observations')}.")
            L.append(f"  - Drowsiness levels: {_counts(pe.get('drowsiness_levels'))}.")
            L.append(f"  - Hand states: {_counts(pe.get('hand_states'))}.")
        L.append(f"- **Risk:** frames per level {_counts(risk['frames_by_level'])}. Peak {risk['peak_level']} (score {risk['peak_score']}).")
        L.append(f"- **Alarm-recommended frames:** {risk['frames_alarm_recommended']}.")
        L.append("")
        L.append("**Risk transitions**")
        L.append("")
        L += _t([[fmt_time(x["timestamp"]), x["transition"], x["hazard_type"], x["event_id"]] for x in risk["transitions"]],
                ["time", "transition", "hazard type", "event"]) if risk["transitions"] else ["None: the level stayed SAFE."]
        L.append("")
        L.append(f"**Events generated:** {ev['total']} ({_counts(ev['by_type'])}). They are in `{out_dir}/events.jsonl` and `events.parquet`, with run id `{ev['run_id']}`.")
        L.append("")
        L.append(f"**Alerts recommended:** {al['escalations']} escalation alerts ({_counts({k: v for k, v in al['by_severity'].items() if v})}) and {al['deescalations']} de-escalation notices. **{al['alarms']} alarm(s) raised.**")
        L.append("")
        rows = [[fmt_time(a["timestamp"]), a["risk_level"], a["headline"] or a["phrase"], "yes" if a["alarm"] else "",
                 (a["risk_reason"] or "")[:140], ", ".join(a["event_ids"])] for a in al["list"] if a["kind"] == "ESCALATION"]
        if rows:
            L += _t(rows, ["time", "level", "alert", "alarm", "why (engine risk_reason)", "event"])
            L.append("")
        L.append(f"**Evidence files:** {evi['files']} files for {evi['events_with_evidence']} events, under `{out_dir}/evidence/`. Missing: {len(evi['missing_files'])}.")
        L.append("")
        L.append(f"**Other outputs:** `timeline.csv`, `risk_timeline.png`, `alerts.jsonl`, `demo_summary.json`; annotated video generated: **{'yes' if s['outputs']['annotated_demo.mp4'] else 'no'}**.")
        L.append("")
        L.append(f"**Provenance:** events data_source {_counts(prov['event_data_sources'])}; observation_type {_counts(prov['event_observation_types'])}.")
        L.append(f"- Synthetic rows included: {prov['synthetic_rows_included']}.")
        L.append(f"- Driver input: {prov['driver_input']}.")
        L.append(f"- Synchronised driver + road: {prov['synchronized_driver_road']}.")
        L.append(f"- GPS: {prov['gps']}.")
        L.append("")
    L.append("## Limitations")
    L.append("")
    L.append("- **No synchronized real driver+road recording exists.** Front and driver demos are separate runs on separate videos, so no real CRITICAL \"driver + road\" alert can occur. The engine gates CRITICAL on that combination, or on a very strong road hazard. The combination path is exercised only with synthetic test input (tests) and the labelled SYNTHETIC_COMBINATION / synthetic scenario datasets.")
    L.append("- **Image-space road signals only.** There is no calibration, distance, physical speed or TTC. Motion of the camera itself shows up as object motion, so a vehicle can look like it is \"approaching\" because the ego car moves. This was checked on the real clips earlier, and it is a known source of HIGH levels.")
    L.append("- **Prototype score and thresholds.** The risk score is rule-based and not a probability. Levels are decision-support states, and alerts are not collision predictions.")
    L.append("- **Perception errors propagate.** Errors from the pretrained detector (e.g. label flicker) and the tracker (ID switches) flow into risk and events. The hand-state classifier generalises poorly to the demo driver setup (see `driver_camera_domain_comparison.md`).")
    L.append("- **Offline alarm output.** The alarm is a console / terminal-bell output in an offline replay (`--beep` enables the bell), not an in-vehicle system. Timing is video time, not measured latency.")
    L.append("- **No GPS.** No GPS log exists, so coordinates stay null / UNAVAILABLE.")
    L.append("- **`alarm_triggered` stays false.** Event schema 1.0 requires `alarm_triggered = false`, and the schema was not changed. Raised alarms are recorded in `alerts.jsonl` and linked to event ids.")
    L.append("- **Short clips.** The front clips are about 1–11 s long, each starting from SAFE.")
    for out_dir, s in summaries:
        if s["camera"] == "front" and s["perception"].get("detections"):
            pe, pr = s["perception"], s["processing"]
            L.append(f"- **Track fragmentation in `{s['input']['file_name']}`:** {pe['tracks']} tracks from {pe['detections']} "
                     f"detections over {pr['frames_processed']} processed frames. Tracks break on label flicker and busy "
                     "scenes, and a newly started track's growth estimate can be extreme (see the `why` column above).")
    return "\n".join(L) + "\n"
