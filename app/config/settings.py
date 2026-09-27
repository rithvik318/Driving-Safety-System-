"""Project configuration loaded from environment variables.

Values come from (highest priority first):
1. real environment variables
2. a `.env` file in the project root (optional)
3. the defaults below

Relative paths are resolved against the project root, so the project runs
the same way on any machine without hard-coded absolute paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_DEVICES = {"auto", "cpu", "cuda"}  # ROCm builds of PyTorch also use "cuda"

DEFAULTS = {
    "PROJECT_NAME": "Physical AI Driving Safety System",
    "DATA_DIR": "data",
    "MODEL_DIR": "models",
    "OUTPUT_DIR": "outputs",
    "LOG_DIR": "logs",
    "DEVICE": "auto",
    "LOG_LEVEL": "INFO",
    "DATASET_ROOT": "",
}


@dataclass(frozen=True)
class Settings:
    project_name: str
    project_root: Path
    data_dir: Path
    model_dir: Path
    output_dir: Path
    log_dir: Path
    device: str
    log_level: str
    dataset_root: Path | None = None  # external local dataset (e.g. a Google Drive folder)

    def ensure_directories(self) -> None:
        """Create the data/model/output/log directories if they are missing."""
        for path in (self.data_dir, self.model_dir, self.output_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_settings(env_file: Path | None = None) -> Settings:
    """Build a Settings object from the environment.

    `env_file` defaults to `<project root>/.env`. Existing environment
    variables are never overridden by the file.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    def get(key: str) -> str:
        return os.environ.get(key, DEFAULTS[key]).strip()

    log_level = get("LOG_LEVEL").upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ValueError(
            f"LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}, got {log_level!r}"
        )

    device = get("DEVICE").lower()
    if device not in VALID_DEVICES:
        raise ValueError(f"DEVICE must be one of {sorted(VALID_DEVICES)}, got {device!r}")

    dataset_root_raw = get("DATASET_ROOT").strip('"').strip("'")

    return Settings(
        project_name=get("PROJECT_NAME"),
        project_root=PROJECT_ROOT,
        data_dir=_resolve_path(get("DATA_DIR"), PROJECT_ROOT),
        model_dir=_resolve_path(get("MODEL_DIR"), PROJECT_ROOT),
        output_dir=_resolve_path(get("OUTPUT_DIR"), PROJECT_ROOT),
        log_dir=_resolve_path(get("LOG_DIR"), PROJECT_ROOT),
        device=device,
        log_level=log_level,
        dataset_root=_resolve_path(dataset_root_raw, PROJECT_ROOT) if dataset_root_raw else None,
    )
