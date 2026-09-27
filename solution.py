"""
solution.py — WIUT Hackathon 2026, CV track.

The organizers' harness (run_submission.py) imports this module and calls:

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

The implementation lives in src/trafficev (YOLO26 + ByteTrack perception,
scene-layout rules, causal time-to-collision risk). See README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trafficev import pipeline  # noqa: E402
from trafficev.labels import CLASSES as _CLASSES  # noqa: E402
from trafficev.risk import CausalRiskModel  # noqa: E402

# Official class ids (14). Kept in full: removing ids does not change the score.
CLASSES: list[str] = list(_CLASSES)

# Anticipation horizon used by the metric (seconds).
RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A — [[start_sec, end_sec, label], ...] for one .mp4 (labels from CLASSES)."""
    return pipeline.detect_events(video_path)


class RiskEstimator:
    """Part B — causal: step() only ever sees the frames passed to it, in order."""

    def __init__(self) -> None:
        self._model = CausalRiskModel(pipeline.get_detector)

    def reset(self, meta: dict) -> None:
        # meta = {"video_id", "fps", "width", "height", "n_frames"}
        self._model.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        # frame: BGR uint8 (H, W, 3). Returns P(accident starts within 5 s) in [0, 1].
        return float(min(1.0, max(0.0, self._model.step(frame, t_sec))))
