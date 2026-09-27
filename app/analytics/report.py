"""Post-event safety intelligence: deterministic recommendations, per-event entries, report rendering.

Runs AFTER events are recorded; it is not part of the real-time loop (perception -> tracking -> risk -> alert).
Recommendations are fixed rules over recorded event fields: safety-oriented review suggestions, not medical
diagnoses and not claims of fault. Missing values are shown as "not recorded", never filled in.

generate_post_event_summary(events) is the (optional) LLM hook: deterministic and offline by default.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from app.analytics.event_summary import LEVELS, summarize_events
from app.analytics.hotspot import GPS_UNAVAILABLE, hotspot_analysis, synthetic_gps_summary
from app.analytics.patterns import PATTERN_DISCLAIMER, describe_patterns, find_patterns

NOT_RECORDED = "not recorded"
RECOMMENDATION_NOTE = "Safety-oriented review suggestions derived from recorded events; not medical diagnoses and not claims of fault."
SCOPE_NOTE = ("Prototype Physical-AI safety-observation system. Risk levels are rule-based decision-support states "
              "(score not a probability); road signals are image-space only. No collision prediction, no real-world "
              "safety validation, no distance/speed measurement, no medical drowsiness assessment, no geographic hotspots, "
              "no synchronized driver+road recording.")

REC = {
    "pedestrian": "Review pedestrian interaction and maintain additional visual attention in similar situations.",
    "vehicle": "Review following/approach context and maintain additional observation of closing traffic.",
    "animal": "Review the animal's position relative to the road and maintain additional attention and readiness to slow in similar situations.",
    "drowsiness": "Treat prolonged eye closure as a driver-attention warning; review rest/alertness before continued driving.",
    "hands_off": "Review the driving segment and maintain both hands on the steering wheel when appropriate.",
    "head_away": "Review the segment for sustained looking away from the road and keep attention on the road ahead.",
    "combined": "Review both the driver state and the road situation in this segment; the combination was rated higher than either alone.",
    "road_user_present": "Informational: a road user was present without approach/conflict evidence; no specific action beyond normal observation.",
    "hand_info": "Informational: hand-position change recorded; no action implied on its own.",
    "drowsiness_info": "Informational: drowsiness level decreased to a non-elevated state; continue normal monitoring.",
    "deescalation": "Informational: risk level decreased; no additional action.",
    "none": "No specific action: no hazard recorded for this event.",
}
INFORMATIONAL = {"road_user_present", "hand_info", "drowsiness_info", "deescalation", "none"}
HAZARD_TO_REC = {"PEDESTRIAN_CONFLICT": "pedestrian", "APPROACHING_VEHICLE": "vehicle", "ANIMAL_HAZARD": "animal",
                 "DRIVER_DROWSINESS": "drowsiness", "HANDS_OFF_WHEEL": "hands_off", "DRIVER_HEAD_AWAY": "head_away",
                 "COMBINED_DRIVER_HAZARD": "combined", "ROAD_USER_PRESENT": "road_user_present"}


def recommend(event: dict) -> tuple[str, str]:
    """(key, text): deterministic recommendation for one recorded event."""
    etype = event.get("event_type")
    if etype == "DRIVER_STATE_CHANGE":
        kind = event.get("change_kind")
        to = str(event.get("transition") or "").split("->")[-1]
        if kind == "drowsiness_level":
            key = "drowsiness" if to in ("HIGH", "CRITICAL") else "drowsiness_info"
        elif kind == "hand_state":
            key = "hands_off" if to == "NO_HANDS" else "hand_info"
        elif kind == "driver_activity":
            key = "hands_off" if to == "HANDS_OFF_WHEEL" else "hand_info"
        else:
            key = "none"
        return key, REC[key]
    if etype == "RISK_DEESCALATED":
        return "deescalation", REC["deescalation"]
    key = HAZARD_TO_REC.get(event.get("hazard_type") or "", "none")
    return key, REC[key]


def is_significant(event: dict) -> bool:
    return (event.get("event_type") == "PERSISTENT_HAZARD"
            or event.get("event_type") == "DRIVER_STATE_CHANGE"
            or (event.get("event_type") == "RISK_ESCALATED" and event.get("risk_level") in ("HIGH", "CRITICAL")))


def _v(x, fmt=None):
    if x is None or x == "":
        return NOT_RECORDED
    return fmt.format(x) if fmt else x


def event_entry(e: dict, evidence_prefix: str = "") -> dict:
    key, text = recommend(e)
    if e.get("event_type") == "DRIVER_STATE_CHANGE":
        hazard = f"{e.get('change_kind') or NOT_RECORDED}: {e.get('transition') or NOT_RECORDED}"
        reason = e.get("risk_reason") or f"{e.get('change_kind')} changed {e.get('transition')} (held for the configured observations)"
        risk = NOT_RECORDED if e.get("risk_level") is None else f"{e['risk_level']}"
    else:
        cls = e.get("hazard_class")
        hazard = f"{_v(e.get('hazard_type'))}" + (f" ({cls})" if cls else "")
        reason = _v(e.get("risk_reason"))
        score = e.get("smoothed_risk_score")
        risk = (f"{_v(e.get('risk_level'))}" + (f" (score {score:.1f})" if isinstance(score, (int, float)) else "")
                + (f", {e.get('transition')}" if e.get("transition") else ""))
    ev_path = e.get("evidence_path")
    t = e.get("timestamp")
    return OrderedDict([
        ("event_id", e.get("event_id")), ("run_id", e.get("run_id")), ("event_type", e.get("event_type")),
        ("time", f"{t:.2f} s into {e.get('source_file') or e.get('session_id') or NOT_RECORDED}" if isinstance(t, (int, float)) else NOT_RECORDED),
        ("camera", _v(e.get("camera"))), ("risk", risk), ("hazard", hazard), ("reason", reason),
        ("evidence_path", f"{evidence_prefix}{ev_path}" if ev_path else "none recorded"),
        ("recommendation_key", key), ("recommendation", text), ("observation_type", _v(e.get("observation_type"))),
        ("data_source", _v(e.get("data_source"))),
    ])


def aggregate_recommendations(events: list[dict]) -> list[dict]:
    groups: dict[str, list] = {}
    for e in events:
        key, _ = recommend(e)
        groups.setdefault(key, []).append(e.get("event_id"))
    out = [{"key": k, "recommendation": REC[k], "events": len(ids), "event_ids": ids, "informational": k in INFORMATIONAL}
           for k, ids in groups.items()]
    out.sort(key=lambda r: (r["informational"], -r["events"], r["key"]))
    return out


# ------------------------------------------------------------------------------------ summary hook


def deterministic_summary(events: list[dict]) -> str:
    if not events:
        return "No LOCAL_REAL events were recorded, so there is nothing to summarise."
    s = summarize_events(events)
    pats = find_patterns(events)
    lv = s["events_by_risk_level"]
    parts = [f"{s['total_events']} real events were recorded across {s['sessions']} session(s) "
             f"({', '.join(f'{k} {v}' for k, v in s['events_by_camera'].items())} camera)."]
    parts.append("Risk levels on level-carrying events: " + ", ".join(f"{k} {lv[k]}" for k in LEVELS) + ".")
    top = [p for p in pats.values() if p["events"] and p["label"] not in ("Risk escalations",)]
    if top:
        biggest = max(top, key=lambda p: p["events"])
        parts.append(f"The most frequent recorded pattern was {biggest['label'].lower()} ({biggest['events']} events in "
                     f"{biggest['sessions']} session(s)).")
    recs = [r for r in aggregate_recommendations(events) if not r["informational"]]
    if recs:
        parts.append("Main review points: " + " ".join(f"({i + 1}) {r['recommendation']}" for i, r in enumerate(recs[:3])))
    parts.append("These are descriptive observations from a prototype; they are not collision predictions, safety "
                 "validations, medical assessments or findings of fault.")
    return " ".join(parts)


def generate_post_event_summary(events: list[dict], backend=None) -> dict:
    """Human-readable post-event summary. Deterministic and offline by default.

    Optional: SAFETY_REPORT_LLM=anthropic with ANTHROPIC_API_KEY and SAFETY_REPORT_LLM_MODEL set (and the
    `anthropic` package installed) rephrases the deterministic summary; any failure falls back silently to the
    deterministic text. `backend` may be any callable(prompt: str) -> str (e.g. for tests)."""
    base = deterministic_summary(events)
    use_env = os.environ.get("SAFETY_REPORT_LLM", "").strip().lower() == "anthropic"
    if backend is None and use_env:
        backend = _anthropic_backend()
    if backend is None:
        return {"text": base, "backend": "deterministic", "deterministic_text": base}
    prompt = ("Rewrite the following safety-event summary for a driver in plain language. Keep every number and "
              "caveat; do not add facts, predictions, diagnoses or blame.\n\n" + base)
    try:
        text = str(backend(prompt)).strip()
        if not text:
            raise ValueError("empty response")
        return {"text": text, "backend": getattr(backend, "name", "custom"), "deterministic_text": base}
    except Exception as exc:  # the hook must never break the offline report
        return {"text": base, "backend": "deterministic", "deterministic_text": base, "llm_error": str(exc)[:200]}


def _anthropic_backend():
    model, key = os.environ.get("SAFETY_REPORT_LLM_MODEL", "").strip(), os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not model or not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    def call(prompt: str) -> str:
        msg = anthropic.Anthropic().messages.create(model=model, max_tokens=500, messages=[{"role": "user", "content": prompt}])
        return "".join(getattr(b, "text", "") for b in msg.content)

    call.name = f"anthropic:{model}"
    return call


# ------------------------------------------------------------------------------------ synthetic (separate)


def synthetic_validation(sim_dir: Path) -> dict:
    """SYNTHETIC scenario validation statistics from data/simulated (never merged with real statistics)."""
    sim_dir = Path(sim_dir)
    rows_path, manifest_path = sim_dir / "scenarios.jsonl", sim_dir / "manifest.json"
    if not rows_path.is_file():
        return {"included": True, "available": False, "message": f"no synthetic dataset at {sim_dir}"}
    rows = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("data_source") == "SYNTHETIC":
            rows.append(r)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    val = manifest.get("validation", {})
    return {
        "included": True, "available": True, "data_source": "SYNTHETIC", "rows": len(rows),
        "scenario_counts": dict(Counter(r.get("scenario_name") for r in rows)),
        "risk_level_rows": {lv: sum(1 for r in rows if r.get("risk_level") == lv) for lv in LEVELS},
        "alarm_recommended_rows": sum(1 for r in rows if r.get("alarm_recommended") is True),
        "instances": val.get("instances"), "expectation_met": val.get("expectation_met"),
        "mismatches": [{"session_id": m.get("session_id"), "mismatch": m.get("mismatch")} for m in val.get("mismatches", [])],
        "gps": synthetic_gps_summary(rows),
        "note": "constructed feature-level scenarios run through the existing engine; validates rule behaviour only",
    }


# ------------------------------------------------------------------------------------ report


def build_safety_report(load, evidence_prefix: str = "", synthetic: dict | None = None) -> dict:
    events = load.events
    summary = summarize_events(events)
    episodes = summary.pop("_episodes")
    patterns = find_patterns(events)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "LOCAL_REAL events only (synthetic data is reported separately only when requested)",
        "scope_note": SCOPE_NOTE,
        "inputs": {"files": load.sources, "missing_files": load.missing_files, "malformed_lines": load.malformed_lines,
                   "excluded_non_real_records": dict(load.excluded_non_real), "duplicates_skipped": load.duplicates},
        "summary": summary,
        "elevated_risk_episodes": episodes,
        "patterns": {"disclaimer": PATTERN_DISCLAIMER, "items": patterns, "descriptions": describe_patterns(patterns)},
        "hotspots": hotspot_analysis(events),
        "recommendations": {"note": RECOMMENDATION_NOTE, "items": aggregate_recommendations(events)},
        "event_entries": [event_entry(e, evidence_prefix) for e in events if is_significant(e)],
        "post_event_summary": generate_post_event_summary(events),
        "synthetic_scenario_validation": synthetic if synthetic is not None else {"included": False},
    }


def _tbl(headers, rows) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |" for r in rows]
    return out


def _c(d) -> str:
    """Compact 'k v, k v' rendering of a count/stat dict (or 'none')."""
    if isinstance(d, str):
        return d
    if not d:
        return "none"
    return ", ".join(f"{k} {v}" for k, v in d.items())


def render_markdown(rep: dict) -> str:
    s, L = rep["summary"], []
    L += ["# Post-event safety summary (LOCAL_REAL)", "", f"> {SCOPE_NOTE}", ""]
    L.append(f"**Input:** {', '.join(f'`{p}`' for p in rep['inputs']['files']) or 'none'}.")
    L.append(f"- Excluded non-real records: {_c(rep['inputs']['excluded_non_real_records'] or 'none')}.")
    L.append(f"- Malformed lines: {rep['inputs']['malformed_lines']}.")
    L.append("")
    L += ["## 1. Event summary", ""]
    if not s["total_events"]:
        L += ["No LOCAL_REAL events found.", ""]
    else:
        L.append(f"- **Total events:** {s['total_events']} (runs {', '.join(s['runs'])}; {s['sessions']} sessions).")
        L.append(f"  - Data sources: {_c(s['data_sources'])}.")
        L.append(f"  - Observation types: {_c(s['observation_types'])}.")
        L.append(f"- **By type:** {_c(s['events_by_type'])}.")
        L.append(f"- **By camera:** {_c(s['events_by_camera'])}.")
        L.append(f"- **By risk level:** {_c(s['events_by_risk_level'])}.")
        L.append(f"  - Events without a risk level (driver-state changes): {s['events_without_risk_level']}.")
        L.append(f"- **By hazard type:** {_c(s['events_by_hazard_type'])}.")
        L.append(f"- **Risk transitions:** {_c(s['risk_transitions'])}.")
        L.append(f"- **Alarm recommended (engine):** {_c(s['alarm_recommended'])}.")
        L.append(f"  - `alarm_triggered` true: {s['alarm_triggered_true']}. The event schema keeps it false; alarms from the demo alert layer are logged separately.")
        ph = s["persistent_hazards"]
        L.append(f"- **Persistent hazards:** {ph['count']}.")
        L.append(f"  - By hazard type: {_c(ph['by_hazard_type'])}.")
        L.append(f"  - By level: {_c(ph['by_risk_level'])}.")
        L.append(f"  - Hazard persistence (s): {_c(ph['hazard_persistence_seconds'])}.")
        d = s["driver_state_events"]
        L.append(f"- **Driver-state events:** {d['count']}. {_c(d['by_transition'])}.")
        L.append(f"- **Risk score** (smoothed, 0–100, not a probability): {_c(s['smoothed_risk_score'])}.")
        L.append(f"  - Raw: {_c(s['raw_risk_score'])}.")
        du = s["durations"]
        ep = du["elevated_risk_episodes"]
        L.append("- **Durations** (stream time):")
        L.append(f"  - Track persistence at the event: {_c(du['track_persistence_seconds'])}.")
        L.append(f"  - Elevated-risk episodes: {ep['count']}, of which {ep['closed']} returned to SAFE (durations {_c(ep['closed_duration_seconds'])}) and {ep['open_at_end_of_clip']} were still open when the clip ended.")
        L.append("")
    L += ["## 2. Hazard patterns (descriptive)", "", f"_{rep['patterns']['disclaimer']}_", ""]
    L += [f"- {line}" for line in rep["patterns"]["descriptions"]]
    L.append("")
    L += ["## 3. Hotspot analysis", ""]
    geo = rep["hotspots"]["geographic"]
    if geo["available"]:
        L.append(f"Real GPS points: {geo['points']}. {geo['note']}.")
        L += _tbl(["lat", "lon", "events"], [[c["lat"], c["lon"], c["events"]] for c in geo["event_counts_by_cell"]])
    else:
        L.append(f"**{GPS_UNAVAILABLE}**")
    ng = rep["hotspots"]["non_geographic"]
    L += ["", f"**{ng['label']}.**", ""]
    L += _tbl(["source file", "events"], list(ng["events_by_source_file"].items()))
    L += ["", f"By camera/source: {_c(ng['events_by_camera'])}.", ""]
    L += [f"Events by stream-time bin ({ng['stream_time_bin_seconds']:g} s; {ng['note']}):", ""]
    L += _tbl(["stream time", "events by type"], [[k, _c(v)] for k, v in ng["events_by_stream_time_bin"].items()])
    L.append("")
    L += ["## 4. Post-event recommendations", "", f"_{rep['recommendations']['note']}_", ""]
    L += _tbl(["recommendation", "events", "type"], [[r["recommendation"], r["events"], "informational" if r["informational"] else "review"]
                                                   for r in rep["recommendations"]["items"]])
    L.append("")
    L += ["## 5. Event report (significant events)", "",
          "Significant events: escalations to HIGH or CRITICAL, persistent hazards and driver-state changes. Missing values are shown as *not recorded*.", ""]
    for en in rep["event_entries"]:
        L.append(f"**{en['event_id']}** ({en['event_type']}, {en['observation_type']}):")
        L.append(f"- **Time:** {en['time']}")
        L.append(f"- **Camera:** {en['camera']}")
        L.append(f"- **Risk:** {en['risk']}")
        L.append(f"- **Hazard:** {en['hazard']}")
        label = "Reason (engine's overall assessment at this time)" if en["event_type"] == "PERSISTENT_HAZARD" else "Reason"
        L.append(f"- **{label}:** {en['reason']}")
        L.append(f"- **Evidence:** `{en['evidence_path']}`")
        L.append(f"- **Recommended review/action:** {en['recommendation']}")
        L.append("")
    if not rep["event_entries"]:
        L += ["No significant events.", ""]
    L += ["## 6. Post-event summary", "", rep["post_event_summary"]["text"], "",
          f"_(summary backend: {rep['post_event_summary']['backend']})_", ""]
    L += ["## 7. Limitations of this analysis", "",
          "- **Scope:** descriptive statistics of what the prototype recorded on short, separate real clips (road-only or driver-only). They are not safety outcomes; there is no collision or near-miss ground truth.",
          "- **Inherited errors:** events inherit errors from the detector, the tracker, the hand-state classifier and the rule thresholds.",
          "  - The hand classifier generalises poorly to the driver-camera setup.",
          "  - A real NO_HANDS event was confirmed as a misclassification (see `event_dataset.md`), so hands-off recommendations need checking against the evidence frame.",
          "- **Ego-motion:** road risk is image-space. Camera ego-motion can look like an approaching object.",
          "- **No location:** there is no GPS and no absolute clock, so there is no geographic or time-of-day analysis.",
          "- **Drowsiness is a proxy:** drowsiness levels are a prototype eye-closure signal, not a medical assessment.", ""]
    syn = rep["synthetic_scenario_validation"]
    if syn.get("included"):
        L += ["---", "", "## SYNTHETIC SCENARIO VALIDATION", "",
              "> **SYNTHETIC data. Kept completely separate from every count above.** Constructed feature-level scenarios; "
              "they validate rule behaviour only and say nothing about real driving.", ""]
        if not syn.get("available"):
            L.append(syn.get("message", "not available"))
        else:
            L.append(f"- **Rows:** {syn['rows']} (data_source {syn['data_source']}).")
            L.append(f"- **Scenarios:** {_c(syn['scenario_counts'])}.")
            L.append(f"- **Risk level rows:** {_c(syn['risk_level_rows'])}.")
            L.append(f"- **Alarm-recommended rows:** {syn['alarm_recommended_rows']}.")
            L.append(f"- **Expectation met:** {syn['expectation_met']} of {syn['instances']} instances.")
            L.append(f"  - Mismatches: {syn['mismatches'] or 'none'}.")
            g = syn["gps"]
            L.append(f"- **Synthetic GPS:** {g.get('points')} points, lat {g.get('lat_range')}, lon {g.get('lon_range')}. {g.get('note', '')}")
        L.append("")
    return "\n".join(L) + "\n"


def statistics_rows(rep: dict) -> list[dict]:
    """Long-format statistics for event_statistics.csv (scope REAL or SYNTHETIC; never mixed)."""
    s, rows = rep["summary"], []

    def add(scope, cat, key, val):
        rows.append({"scope": scope, "category": cat, "key": key, "value": val})

    add("REAL", "total", "events", s["total_events"])
    add("REAL", "total", "sessions", s["sessions"])
    for cat in ("events_by_type", "events_by_camera", "events_by_risk_level", "events_by_hazard_type", "risk_transitions",
                "alarm_recommended"):
        for k, v in s[cat].items():
            add("REAL", cat, k, v)
    add("REAL", "persistent_hazards", "count", s["persistent_hazards"]["count"])
    for k, v in s["driver_state_events"]["by_transition"].items():
        add("REAL", "driver_state_events", k, v)
    for name in ("smoothed_risk_score", "raw_risk_score"):
        for k, v in (s[name] or {}).items():
            add("REAL", name, k, v)
    ep = s["durations"]["elevated_risk_episodes"]
    for k in ("count", "closed", "open_at_end_of_clip"):
        add("REAL", "elevated_risk_episodes", k, ep[k])
    for k, v in (ep["closed_duration_seconds"] or {}).items():
        add("REAL", "elevated_risk_episode_seconds", k, v)
    for p in rep["patterns"]["items"].values():
        add("REAL", "pattern_events", p["label"], p["events"])
    add("REAL", "gps", "real_gps_points", rep["hotspots"]["geographic"]["points"])
    for r in rep["recommendations"]["items"]:
        add("REAL", "recommendation_events", r["key"], r["events"])
    syn = rep["synthetic_scenario_validation"]
    if syn.get("available"):
        add("SYNTHETIC", "total", "rows", syn["rows"])
        for k, v in syn["scenario_counts"].items():
            add("SYNTHETIC", "scenario_rows", k, v)
        for k, v in syn["risk_level_rows"].items():
            add("SYNTHETIC", "risk_level_rows", k, v)
        add("SYNTHETIC", "validation", "expectation_met", syn["expectation_met"])
        add("SYNTHETIC", "validation", "instances", syn["instances"])
    return rows


def write_safety_report(rep: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"md": out_dir / "safety_summary.md", "json": out_dir / "safety_summary.json", "csv": out_dir / "event_statistics.csv"}
    paths["md"].write_text(render_markdown(rep), encoding="utf-8")
    paths["json"].write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    with open(paths["csv"], "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "category", "key", "value"])
        w.writeheader()
        w.writerows(statistics_rows(rep))
    return paths
