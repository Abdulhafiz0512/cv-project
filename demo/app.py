"""Live demo (Hugging Face Space, CPU): upload a clip, get the events, an annotated
playback with the event timeline, and the accident-risk curve.

Runs the same package as the submission, tuned for a CPU host: one causal
perception pass at 4 fps with a 960 px detector feeds both the rules (Part A)
and the risk model (Part B), instead of the submission's two passes at
10 fps / 5 fps with a 1280 px detector.

    python demo/app.py          # http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "src").exists() \
    else Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from render import render  # noqa: E402
from trafficev import pipeline  # noqa: E402
from trafficev.detector import Detector  # noqa: E402
from trafficev.risk import CausalRiskModel  # noqa: E402
from trafficev.video import probe  # noqa: E402

MAX_SECONDS = 150
MAX_MB = 600
DEMO_FPS = 4.0

_detector: Detector | None = None


def detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector(imgsz=960)
    return _detector


def analyse(video_path: str | None, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Choose an .mp4 file first.")
    size_mb = Path(video_path).stat().st_size / 1e6
    if size_mb > MAX_MB:
        raise gr.Error(f"The file is {size_mb:.0f} MB; the demo accepts up to {MAX_MB} MB.")
    info = probe(video_path)
    if info.duration > MAX_SECONDS:
        raise gr.Error(f"The clip is {info.duration:.0f} s long; the demo accepts up to {MAX_SECONDS} s. "
                       "Trim it and upload again.")

    model = CausalRiskModel(detector)
    model.reset({"fps": info.fps})
    curve: list[list[float]] = []

    def on_frame(t, rows):
        curve.append([round(t, 2), round(model.cues(t, rows)["risk"], 4)])
        progress(0.75 * t / max(info.duration, 1e-6), desc="Detecting and tracking road users")

    per = pipeline.perceive(video_path, budget_s=1e9, target_fps=DEMO_FPS, detector=detector(), on_frame=on_frame)
    events = pipeline.events_from(per)
    risk = _smooth(curve)

    out_dir = Path(tempfile.mkdtemp(prefix="trafficev_"))
    annotated = out_dir / "annotated.mp4"
    render(video_path, per, events, risk, str(annotated), fps=DEMO_FPS,
           progress=lambda f: progress(0.75 + 0.25 * f, desc="Rendering the annotated video"))

    table = pd.DataFrame([{"class": lab, "start (s)": s, "end (s)": e, "length (s)": round(e - s, 1)}
                          for s, e, lab in events], columns=["class", "start (s)", "end (s)", "length (s)"])
    risk_df = pd.DataFrame(risk, columns=["time (s)", "risk"])
    result = out_dir / "events.json"
    result.write_text(json.dumps({"video": Path(video_path).name, "duration": info.duration,
                                  "events": events, "risk": risk}, indent=1))
    summary = (f"{len(events)} event(s) in {info.duration:.0f} s of video."
               if events else f"No traffic events found in {info.duration:.0f} s of video.")
    return str(annotated), summary, table, risk_df, str(result)


def _smooth(curve: list[list[float]], tau: float = 2.0) -> list[list[float]]:
    """Same rise-instantly / decay-slowly shaping as the submission's RiskEstimator."""
    import math

    out, score, last_t = [], 0.0, 0.0
    for t, raw in curve:
        score = max(raw, score * math.exp(-(t - last_t) / tau))
        last_t = t
        out.append([t, round(score, 4)])
    return out


with gr.Blocks(title="Traffic event detection — live demo") as app:
    gr.Markdown(
        "## Traffic event detection — live demo\n"
        f"Upload an .mp4 from the junction camera (up to {MAX_SECONDS} s and {MAX_MB} MB). "
        "The demo runs on a shared CPU: expect about four to six minutes of processing per minute of video; "
        "the progress bar shows each stage. "
        "You get the detected events, an annotated playback with the event timeline under the video, "
        "and the accident-risk curve.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Video(label="Video", sources=["upload"], format="mp4")
            run = gr.Button("Detect events", variant="primary")
            summary = gr.Markdown()
            download = gr.File(label="Events and risk (JSON)")
        with gr.Column(scale=2):
            out_video = gr.Video(label="Annotated playback")
            table = gr.Dataframe(label="Events", interactive=False,
                                 headers=["class", "start (s)", "end (s)", "length (s)"])
            risk_plot = gr.LinePlot(x="time (s)", y="risk", y_lim=[0, 1], label="Accident risk (next 5 s)", height=220)
    examples = sorted((ROOT / "demo" / "examples").glob("*.mp4")) if (ROOT / "demo" / "examples").exists() else []
    if examples:
        gr.Examples([[str(p)] for p in examples], inputs=[inp])
    run.click(analyse, inputs=[inp], outputs=[out_video, summary, table, risk_plot, download])

if __name__ == "__main__":
    app.queue(max_size=8).launch()
