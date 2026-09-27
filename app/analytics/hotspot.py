"""Hotspot analysis: geographic only when REAL GPS observations exist; never fabricated.

Without real GPS the result says so and reports clearly-labelled NON-geographic frequencies instead (by source
file, camera, stream time). Synthetic GPS (data/simulated) is summarised separately and is never a hotspot.
"""

from __future__ import annotations

from collections import Counter, defaultdict

GPS_UNAVAILABLE = "GPS hotspot analysis unavailable because no real GPS observations were collected."
GRID_DEG = 0.001  # ~110 m cells for descriptive counting when real GPS exists


def real_gps_points(events: list[dict]) -> list[tuple[float, float, dict]]:
    pts = []
    for e in events:
        lat, lon, src = e.get("gps_lat"), e.get("gps_lon"), e.get("gps_source")
        if src in (None, "UNAVAILABLE", "SYNTHETIC") or e.get("data_source") not in ("LOCAL_REAL", "PUBLIC"):
            continue
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and abs(lat) <= 90 and abs(lon) <= 180:
            pts.append((float(lat), float(lon), e))
    return pts


def hotspot_analysis(events: list[dict], time_bin_seconds: float = 2.0) -> dict:
    pts = real_gps_points(events)
    if pts:
        cells = Counter((round(lat / GRID_DEG) * GRID_DEG, round(lon / GRID_DEG) * GRID_DEG) for lat, lon, _ in pts)
        geographic = {"available": True, "points": len(pts), "grid_degrees": GRID_DEG,
                      "event_counts_by_cell": [{"lat": round(a, 6), "lon": round(b, 6), "events": c} for (a, b), c in cells.most_common(20)],
                      "note": "descriptive event counts per GPS cell; not a measure of objective danger"}
    else:
        geographic = {"available": False, "message": GPS_UNAVAILABLE, "points": 0}
    time_bins = defaultdict(Counter)
    for e in events:
        t = e.get("timestamp")
        if isinstance(t, (int, float)):
            lo = int(t // time_bin_seconds) * time_bin_seconds
            time_bins[f"{lo:g}-{lo + time_bin_seconds:g} s"][e.get("event_type")] += 1
    return {
        "geographic": geographic,
        "non_geographic": {
            "label": "NOT geographic hotspots: event frequencies by recording and by stream time",
            "events_by_source_file": dict(Counter(e.get("source_file") or "not recorded" for e in events).most_common()),
            "events_by_camera": dict(Counter(e.get("camera") or "not recorded" for e in events).most_common()),
            "events_by_stream_time_bin": {k: dict(v) for k, v in sorted(time_bins.items(), key=lambda kv: float(kv[0].split("-")[0]))},
            "stream_time_bin_seconds": time_bin_seconds,
            "note": "stream time = seconds since the start of each clip (no absolute clock); clips differ in length",
        },
    }


def synthetic_gps_summary(rows: list[dict]) -> dict:
    """Bounding box of SYNTHETIC reference coordinates. Explicitly not a hotspot and not a real location."""
    pts = [(r["gps_lat"], r["gps_lon"]) for r in rows if r.get("gps_source") == "SYNTHETIC"
           and isinstance(r.get("gps_lat"), (int, float)) and isinstance(r.get("gps_lon"), (int, float))]
    if not pts:
        return {"points": 0}
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    return {"points": len(pts), "lat_range": [min(lats), max(lats)], "lon_range": [min(lons), max(lons)],
            "location_source": "SYNTHETIC_REFERENCE",
            "note": "synthetic coordinates around a generic reference point; not collection locations and not hotspots"}
