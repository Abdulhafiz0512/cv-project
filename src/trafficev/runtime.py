"""Process-wide setup: offline mode, determinism, device choice, time budget."""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

# Must be set before ultralytics is imported anywhere: no update checks,
# telemetry or font downloads during the offline evaluation run.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "weights"
SEED = 0
TIME_FACTOR = 3.0  # organizers' budget: Part A + Part B <= 3 x video duration
# TRAFFICEV_EXACT=1 disables every time-driven degradation (frame thinning,
# lower CPU rates) so slow machines reproduce the GPU reference outputs.
EXACT = os.environ.get("TRAFFICEV_EXACT", "") == "1"

_configured = False


def configure() -> None:
    """Seed every RNG and pin deterministic kernels (idempotent)."""
    global _configured
    if _configured:
        return
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    _configured = True


def device() -> str:
    """NVIDIA GPU (the organizers' T4) > Apple-silicon GPU (MPS) > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def is_gpu(dev: str) -> bool:
    return dev != "cpu"


@dataclass
class Budget:
    """Wall-clock allowance for one stage, with a projection helper."""
    seconds: float
    start: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def remaining(self) -> float:
        return self.seconds - self.elapsed()

    def on_track(self, fraction_done: float) -> bool:
        """True if finishing at the current pace stays inside the allowance."""
        if fraction_done <= 0.02:
            return True
        return self.elapsed() / fraction_done <= self.seconds
