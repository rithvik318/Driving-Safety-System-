"""Contextual risk engine: transparent, deterministic rules over DriverState + tracked road objects.

risk_score is a rule-based prototype score (0-100), not a probability of collision. Levels
(SAFE / CAUTION / HIGH / CRITICAL) are decision-support states, not validated classifications.
"""

from app.risk.engine import RiskEngine
from app.risk.models import HazardType, RiskAssessment, RiskFactor, RiskLevel, RiskSnapshot, TrackHazard

__all__ = ["HazardType", "RiskAssessment", "RiskEngine", "RiskFactor", "RiskLevel", "RiskSnapshot", "TrackHazard"]
