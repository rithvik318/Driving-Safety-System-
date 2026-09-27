"""GPS interface. Coordinates are only ever reported when a real source supplies them.

    provider.fix_at(timestamp) -> GPSFix | None

- NoGPS: always None (events then carry gps_lat/gps_lon = null, gps_source = "UNAVAILABLE").
- GPSLogProvider: fixes from a real GPS log (e.g. a phone logger CSV), matched to the stream
  timestamp; a fix older/newer than max_age_seconds is not used. Nothing is interpolated or
  invented.

CSV format for load_gps_csv: header with at least timestamp, lat, lon (optional accuracy_m),
timestamp in the same seconds clock as the video stream.
"""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class GPSFix:
    latitude: float
    longitude: float
    timestamp: float  # seconds, same clock as the stream
    accuracy_m: float | None = None
    source: str = "GPS_LOG"

    def __post_init__(self):
        for name, v, lim in (("latitude", self.latitude, 90.0), ("longitude", self.longitude, 180.0)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > lim:
                raise ValueError(f"invalid {name}: {v!r}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"invalid GPS timestamp: {self.timestamp!r}")
        if self.accuracy_m is not None and (not math.isfinite(self.accuracy_m) or self.accuracy_m < 0):
            raise ValueError(f"invalid accuracy_m: {self.accuracy_m!r}")
        if not self.source or self.source == UNAVAILABLE:
            raise ValueError("a GPSFix needs a real source name")


class GPSProvider(Protocol):
    name: str

    def fix_at(self, timestamp: float) -> GPSFix | None: ...


class NoGPS:
    """No GPS hardware/log available. Never returns coordinates."""

    name = UNAVAILABLE

    def fix_at(self, timestamp: float) -> GPSFix | None:
        return None


class GPSLogProvider:
    """Nearest real fix within max_age_seconds of the requested time (no interpolation)."""

    def __init__(self, fixes: list[GPSFix], max_age_seconds: float = 2.0, name: str = "GPS_LOG"):
        self.fixes = sorted(fixes, key=lambda f: f.timestamp)
        self._times = [f.timestamp for f in self.fixes]
        self.max_age = max_age_seconds
        self.name = name

    def fix_at(self, timestamp: float) -> GPSFix | None:
        if not self.fixes:
            return None
        i = bisect.bisect_left(self._times, timestamp)
        candidates = [self.fixes[j] for j in (i - 1, i) if 0 <= j < len(self.fixes)]
        best = min(candidates, key=lambda f: abs(f.timestamp - timestamp))
        return best if abs(best.timestamp - timestamp) <= self.max_age + 1e-9 else None


def load_gps_csv(path: Path, source: str = "GPS_LOG", max_age_seconds: float = 2.0) -> GPSLogProvider:
    fixes = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            acc = row.get("accuracy_m")
            fixes.append(GPSFix(float(row["lat"]), float(row["lon"]), float(row["timestamp"]),
                                float(acc) if acc not in (None, "") else None, source))
    return GPSLogProvider(fixes, max_age_seconds, source)
