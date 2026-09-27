"""Interface / end-to-end smoke test through the organizers' unchanged harness."""
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest
from helpers import ROOT

import evaluate


def test_classes_are_official():
    import solution
    assert list(solution.CLASSES) == list(evaluate.OFFICIAL_CLASSES)
    assert hasattr(solution, "detect_events") and hasattr(solution, "RiskEstimator")


@pytest.mark.slow
def test_harness_end_to_end(tmp_path):
    """A 3 s synthetic clip must produce a valid predictions.json within the time budget."""
    vids = tmp_path / "videos"
    vids.mkdir()
    path = vids / "synthetic.mp4"
    w, h, fps = 640, 360, 25
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(3 * fps):
        img = np.full((h, w, 3), 60, np.uint8)
        x = 50 + 6 * i
        cv2.rectangle(img, (x, 200), (x + 80, 240), (200, 200, 200), -1)
        vw.write(img)
    vw.release()
    out = tmp_path / "pred.json"
    proc = subprocess.run([sys.executable, str(ROOT / "run_submission.py"), "--videos", str(vids), "--out", str(out),
                           "--solution", str(ROOT / "solution.py"), "--time-factor", "60"],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    pred = json.loads(out.read_text())
    entry = pred["videos"]["synthetic.mp4"]
    assert not pred["log"]["synthetic.mp4"]["errors"], pred["log"]
    assert len(entry["risk"]) == 3 * fps
    errors, _ = evaluate.validate(pred)
    assert not errors, errors
