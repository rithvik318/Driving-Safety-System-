"""Public-release data checks: the published sample, schema, synthetic set and documentation links.

These run on the files committed to the repository (no raw dataset or models needed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config.settings import PROJECT_ROOT
from app.events.schema import FIELD_NAMES, SCHEMA_VERSION, schema_document, validate_record
from app.simulation.models import SIM_FIELD_NAMES, sim_schema_document, validate_sim_row

SAMPLE = PROJECT_ROOT / "data" / "sample"
SCHEMA = PROJECT_ROOT / "data" / "schema"
SYN = PROJECT_ROOT / "data" / "synthetic"

pytestmark = pytest.mark.skipif(not (SAMPLE / "real_events.jsonl").is_file(), reason="public sample not present")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_sample_jsonl_parses_and_conforms_to_schema():
    rows = _jsonl(SAMPLE / "real_events.jsonl")
    assert len(rows) == 66
    for r in rows:
        assert validate_record(r) == [], r.get("event_id")
        assert r["data_source"] == "LOCAL_REAL"
        assert r["gps_lat"] is None and r["gps_source"] == "UNAVAILABLE"
        assert r["alarm_triggered"] is False
    assert len({(r["run_id"], r["event_id"]) for r in rows}) == 66


def test_parquet_loads_and_matches_jsonl():
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(SAMPLE / "real_events.parquet")
    assert table.num_rows == 66 and table.column_names == list(FIELD_NAMES)
    rows = _jsonl(SAMPLE / "real_events.jsonl")
    assert table.column("event_id").to_pylist() == [r["event_id"] for r in rows]


def test_published_schema_matches_code():
    real = json.loads((SCHEMA / "real_event_schema.json").read_text(encoding="utf-8"))
    assert real["schema_version"] == SCHEMA_VERSION
    assert [f["name"] for f in real["fields"]] == list(FIELD_NAMES) and len(FIELD_NAMES) == 60
    assert real == json.loads(json.dumps(schema_document()))
    syn = json.loads((SCHEMA / "synthetic_schema.json").read_text(encoding="utf-8"))
    assert [f["name"] for f in syn["fields"]] == list(SIM_FIELD_NAMES)
    assert syn["fields"] == json.loads(json.dumps(sim_schema_document()))["fields"]


def test_sample_json_is_subset_of_real_events():
    rows = {r["event_id"]: r for r in _jsonl(SAMPLE / "real_events.jsonl")}
    sample = json.loads((SAMPLE / "real_events_sample.json").read_text(encoding="utf-8"))
    assert sample and all(s == rows[s["event_id"]] for s in sample)


def test_published_evidence_links_resolve():
    rows = {r["event_id"]: r for r in _jsonl(SAMPLE / "real_events.jsonl")}
    ev_root = SAMPLE / "evidence"
    dirs = sorted(p for p in ev_root.glob("*/event_*") if p.is_dir())
    assert dirs
    for d in dirs:
        rec = rows[d.name]
        assert rec["evidence_path"] == d.relative_to(ev_root).as_posix()
        for f in json.loads(rec["evidence_files"]):
            assert (ev_root / f).is_file(), f
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        assert meta["record"]["event_id"] == rec["event_id"]
        assert not list(d.glob("driver_frame*")), "driver-camera frames are not published"


def test_synthetic_data_is_labelled_and_separate():
    rows = _jsonl(SYN / "scenarios.jsonl")
    assert len(rows) == 1441
    assert {r["data_source"] for r in rows} == {"SYNTHETIC"}
    for r in rows[:: 97]:
        assert validate_sim_row(r) == []
    real_ids = {r["event_id"] for r in _jsonl(SAMPLE / "real_events.jsonl")}
    assert not ({r["event_id"] for r in rows} & real_ids)
    parents = {r[k] for r in rows for k in ("parent_real_event_id", "parent_real_driver_event_id") if r[k]}
    assert parents <= real_ids
    manifest = json.loads((SYN / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_records"] == 1441 and manifest["validation"]["expectation_met"] == 23


def test_markdown_links_resolve():
    missing = []
    for md in list(PROJECT_ROOT.glob("*.md")) + list((PROJECT_ROOT / "docs").rglob("*.md")) + list((PROJECT_ROOT / "data").rglob("*.md")) + list((PROJECT_ROOT / "results").glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#\s]+)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (md.parent / target).exists():
                missing.append(f"{md.relative_to(PROJECT_ROOT)} -> {target}")
    assert not missing, missing


def test_no_secrets_or_private_paths_in_repo():
    patterns = [r"[A-Z]:\\\\(Users|OneDrive)", r"OneDrive -", r"ghp_[A-Za-z0-9]{20,}", r"sk-ant-[A-Za-z0-9]", r"AKIA[0-9A-Z]{16}",
                r"-----BEGIN [A-Z ]*PRIVATE KEY-----"]
    hits = []
    for p in PROJECT_ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts or p.suffix.lower() not in {".py", ".md", ".json", ".jsonl", ".txt", ".ini", ".example", ".csv", ".yml", ".yaml", ".toml"}:
            continue
        if p.name == "test_public_data.py":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in patterns:
            if re.search(pat, text):
                hits.append(f"{p.relative_to(PROJECT_ROOT)}: {pat}")
    assert not hits, hits
    assert not (PROJECT_ROOT / ".env").exists()
