"""Feature-level SYNTHETIC scenario generator.

Constructed road-object detections and scripted driver signals are fed through the existing, unmodified
tracker, driver temporal logic, risk engine and event recorder. Output rows are SYNTHETIC
(data_source = observation_type = SYNTHETIC) and are written only to data/simulated/, never to the
real event dataset. See app/simulation/README.md.
"""

from app.simulation.generator import GenerationResult, SyntheticScenarioGenerator, build_manifest, load_real_reference
from app.simulation.models import (SIM_FIELD_NAMES, SIM_FIELDS, SIM_SCHEMA_VERSION, SimulationConfig, SimulationError,
                                   validate_sim_row)
from app.simulation.scenarios import SCENARIO_NAMES, SCENARIOS
from app.simulation.writer import SimulationDatasetWriter, read_rows, render_report

__all__ = ["GenerationResult", "SCENARIOS", "SCENARIO_NAMES", "SIM_FIELDS", "SIM_FIELD_NAMES", "SIM_SCHEMA_VERSION",
           "SimulationConfig", "SimulationDatasetWriter", "SimulationError", "SyntheticScenarioGenerator", "build_manifest",
           "load_real_reference", "read_rows", "render_report", "validate_sim_row"]
