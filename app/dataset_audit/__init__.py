"""Read-only audit of the external local dataset (counts, readability, duplicates)."""

from app.dataset_audit.audit import run_audit
from app.dataset_audit.report import render_report, write_manifest, write_report
from app.dataset_audit.scan import EXPECTED_STRUCTURE, DatasetRootError, validate_root

__all__ = [
    "EXPECTED_STRUCTURE",
    "DatasetRootError",
    "validate_root",
    "run_audit",
    "render_report",
    "write_manifest",
    "write_report",
]
