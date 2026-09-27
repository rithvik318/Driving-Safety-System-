"""Event dataset writer: JSON Lines (always) + Parquet (when pyarrow is installed) + schema.json.

    writer = EventDatasetWriter(root, kind="real")        # accepts LOCAL_REAL / PUBLIC only
    writer = EventDatasetWriter(root, kind="synthetic")   # accepts SYNTHETIC / SYNTHETIC_COMBINATION only
    writer.append(records); writer.finalize()

Files in root: events.jsonl, events.parquet (if pyarrow), schema.json.
Real and synthetic records can never share a dataset: the writer refuses the other kind.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.events.schema import FIELD_NAMES, REAL_SOURCES, SYNTHETIC_SOURCES, arrow_schema, export_schema, validate_record


class EventWriteError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class EventDatasetWriter:
    def __init__(self, root: Path, kind: str = "real", overwrite: bool = False):
        if kind not in ("real", "synthetic"):
            raise ValueError("kind must be 'real' or 'synthetic'")
        self.root = Path(root)
        self.kind = kind
        self.allowed = REAL_SOURCES if kind == "real" else SYNTHETIC_SOURCES
        self.jsonl_path = self.root / "events.jsonl"
        self.parquet_path = self.root / "events.parquet"
        self.schema_path = self.root / "schema.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for p in (self.jsonl_path, self.parquet_path):
                if p.exists():
                    p.unlink()
        self._keys = {(r["run_id"], r["event_id"]) for r in read_jsonl(self.jsonl_path)}

    def append(self, records) -> int:
        rows = []
        for rec in records:
            d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
            errors = validate_record(d)
            if errors:
                raise EventWriteError(f"{d.get('event_id')}: " + "; ".join(errors))
            if d["data_source"] not in self.allowed:
                raise EventWriteError(f"{d['event_id']}: data_source {d['data_source']} cannot be written to the "
                                      f"{self.kind} dataset (allowed: {', '.join(self.allowed)})")
            key = (d["run_id"], d["event_id"])
            if key in self._keys:
                raise EventWriteError(f"duplicate event id {key}")
            self._keys.add(key)
            rows.append({k: d[k] for k in FIELD_NAMES})
        if rows:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    def finalize(self) -> dict:
        """Write schema.json and (re)build events.parquet from events.jsonl."""
        export_schema(self.schema_path)
        rows = read_jsonl(self.jsonl_path)
        out = {"jsonl": str(self.jsonl_path), "schema": str(self.schema_path), "records": len(rows),
               "parquet": None, "parquet_note": None}
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            out["parquet_note"] = "pyarrow not installed: JSON Lines only (pip install pyarrow for Parquet)"
            return out
        table = pa.Table.from_pylist(rows, schema=arrow_schema())
        pq.write_table(table, self.parquet_path)
        out["parquet"] = str(self.parquet_path)
        return out
