"""Video probing and cheap sampled decoding.

The camera writes 4K H.264 4:2:2 10-bit with an IBBP GOP. Decoding is the
dominant cost of the whole submission, and the harness decodes every frame
again for Part B, so Part A asks the decoder to skip non-reference (B)
frames: that yields ~10 fps of I/P frames at ~1.3x the speed of a full
decode, already downscaled by swscale. OpenCV is the fallback reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

PROC_W, PROC_H = 1280, 720  # processing resolution (detector input, 16:9)


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        # same definition as the organizers' harness
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    info = VideoInfo(
        path=str(path),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def to_proc(frame: np.ndarray) -> np.ndarray:
    """Downscale a full-resolution BGR frame to the processing size."""
    if frame.shape[1] == PROC_W and frame.shape[0] == PROC_H:
        return frame
    interp = cv2.INTER_AREA if frame.shape[1] > 2 * PROC_W else cv2.INTER_LINEAR
    return cv2.resize(frame, (PROC_W, PROC_H), interpolation=interp)


def sample_frames(path: str, target_fps: float = 10.0, skip_bframes: bool = True
                  ) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t_sec, BGR frame at PROC size) at roughly ``target_fps``.

    Timestamps are seconds from the first frame, matching the harness's
    ``frame_index / fps`` convention for constant-frame-rate video.
    """
    try:
        import av  # noqa: F401
    except ImportError:
        yield from sample_frames_cv2(path, target_fps)
        return
    yield from _sample_av(path, target_fps, skip_bframes)


def _sample_av(path: str, target_fps: float, skip_bframes: bool) -> Iterator[tuple[float, np.ndarray]]:
    import av

    min_gap = 1.0 / target_fps - 1e-3
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if skip_bframes:
            stream.codec_context.skip_frame = "NONREF"
        tb = float(stream.time_base)
        # origin = presentation time of the first frame, even if skipped B-frames
        # (open GOP) mean the first *decoded* frame comes later
        t0 = stream.start_time
        last = -1e9
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if t0 is None:
                t0 = frame.pts
            t = (frame.pts - t0) * tb
            if t - last < min_gap:
                continue
            last = t
            img = frame.to_ndarray(width=PROC_W, height=PROC_H, format="bgr24", interpolation="AREA")
            yield float(t), img


def sample_frames_cv2(path: str, target_fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """OpenCV reader: fallback when PyAV is missing or fails."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / target_fps))
    idx = 0
    while cap.grab():
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx / fps, to_proc(frame)
        idx += 1
    cap.release()
