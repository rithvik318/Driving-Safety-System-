"""Event layer: event-driven records (risk transitions, persistent hazards, driver-state changes),
evidence (frames / clips), and a provenance-labelled dataset (JSON Lines + Parquet + schema.json).

No dependency on YOLO or OpenCV in the recorder; real and synthetic events never share a dataset.
"""

from app.events.models import DataSource, EventRecord, EventType, EventValidationError, ObservationType
from app.events.recorder import EventRecorder
from app.events.schema import FIELDS, SCHEMA_VERSION, export_schema, validate_record
from app.events.writer import EventDatasetWriter, EventWriteError, read_jsonl

__all__ = ["DataSource", "EventDatasetWriter", "EventRecord", "EventRecorder", "EventType", "EventValidationError",
           "EventWriteError", "FIELDS", "ObservationType", "SCHEMA_VERSION", "export_schema", "read_jsonl", "validate_record"]
