"""trafficev — traffic event detection and accident anticipation for one fixed CCTV view.

Part A (``pipeline.detect_events``): sampled decode -> YOLO -> ByteTrack ->
trajectory/scene rules -> per-class segments.
Part B (``risk.CausalRiskModel``): the same perception run causally at a low
rate -> time-to-collision and braking cues -> calibrated risk.
"""
from .labels import CLASSES

__all__ = ["CLASSES"]
