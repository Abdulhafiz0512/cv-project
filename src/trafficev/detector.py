"""Road-user detector: an Ultralytics YOLO model run on processing-size frames."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import runtime
from .labels import DETECT_CLASSES

DEFAULT_WEIGHTS = "yolo26s.pt"


@dataclass
class Detections:
    """Boxes of one frame in processing-frame pixels."""
    xyxy: np.ndarray   # (N, 4) float32
    conf: np.ndarray   # (N,)
    cls: np.ndarray    # (N,) COCO ids

    def __len__(self) -> int:
        return len(self.conf)


class Detector:
    def __init__(self, weights: str = DEFAULT_WEIGHTS, imgsz: int = 1280, conf: float = 0.10,
                 device: str | None = None):
        runtime.configure()
        from ultralytics import YOLO

        self.device = device or runtime.device()
        # fp32 everywhere: fp16 would make GPU and CPU runs diverge at borderline boxes
        self.half = False
        self.model = YOLO(str(runtime.WEIGHTS_DIR / weights), task="detect")
        self.imgsz = imgsz
        self.conf = conf

    def __call__(self, frames: list[np.ndarray]) -> list[Detections]:
        results = self.model.predict(
            frames, imgsz=self.imgsz, conf=self.conf, classes=DETECT_CLASSES, device=self.device,
            half=self.half, verbose=False,
        )
        out = []
        for r in results:
            b = r.boxes
            out.append(Detections(
                xyxy=b.xyxy.cpu().numpy().astype(np.float32),
                conf=b.conf.cpu().numpy().astype(np.float32),
                cls=b.cls.cpu().numpy().astype(np.int32),
            ))
        return out
