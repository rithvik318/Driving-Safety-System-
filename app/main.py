"""Application entry point.

Run with:  python -m app.main

At this stage it only loads configuration, sets up logging, and creates the
working directories. Perception, tracking, risk and recording modules are
added in later phases.
"""

from __future__ import annotations

import logging
import sys

from app import __version__
from app.config import configure_logging, load_settings

logger = logging.getLogger(__name__)


def main() -> int:
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_dir)
    settings.ensure_directories()

    logger.info("Starting %s v%s", settings.project_name, __version__)
    logger.info("Project root: %s", settings.project_root)
    logger.info("Data dir:     %s", settings.data_dir)
    logger.info("Model dir:    %s", settings.model_dir)
    logger.info("Output dir:   %s", settings.output_dir)
    logger.info("Device:       %s (requested)", settings.device)

    print(settings.project_name)
    print("Environment initialized successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
