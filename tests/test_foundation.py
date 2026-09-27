"""Foundation tests: configuration loads and the app starts."""

from __future__ import annotations

from pathlib import Path

import pytest

ENV_KEYS = ["PROJECT_NAME", "DATA_DIR", "MODEL_DIR", "OUTPUT_DIR", "LOG_DIR", "DEVICE", "LOG_LEVEL", "DATASET_ROOT"]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Isolate tests from any real .env file and point outputs at a temp dir."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    return tmp_path


def test_package_imports():
    import app
    import app.config
    import app.main  # noqa: F401

    assert app.__version__


def test_defaults(clean_env):
    from app.config import load_settings

    s = load_settings()
    assert s.project_name == "Physical AI Driving Safety System"
    assert s.device == "auto"
    assert s.log_level == "INFO"
    assert s.data_dir == clean_env / "data"
    assert s.dataset_root is None


def test_relative_paths_resolve_under_project_root(clean_env, monkeypatch):
    from app.config import load_settings
    from app.config.settings import PROJECT_ROOT

    monkeypatch.setenv("DATA_DIR", "data")
    s = load_settings()
    assert s.data_dir == PROJECT_ROOT / "data"
    assert Path(s.data_dir).is_absolute()


def test_env_overrides(clean_env, monkeypatch):
    from app.config import load_settings

    monkeypatch.setenv("DEVICE", "CPU")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("PROJECT_NAME", "Test Project")
    s = load_settings()
    assert (s.device, s.log_level, s.project_name) == ("cpu", "DEBUG", "Test Project")


@pytest.mark.parametrize("key,value", [("LOG_LEVEL", "LOUD"), ("DEVICE", "tpu")])
def test_invalid_values_rejected(clean_env, monkeypatch, key, value):
    from app.config import load_settings

    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        load_settings()


def test_main_starts(clean_env, capsys):
    from app.main import main

    assert main() == 0
    out = capsys.readouterr().out
    assert "Physical AI Driving Safety System" in out
    assert "Environment initialized successfully." in out
    for name in ("data", "models", "outputs", "logs"):
        assert (clean_env / name).is_dir()
    assert (clean_env / "logs" / "app.log").is_file()
