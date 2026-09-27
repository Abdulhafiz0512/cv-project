# Junction Watch — WIUT Hackathon 2026, computer vision track

Traffic event detection (Part A) and causal accident anticipation (Part B) for the organizers' fixed 4K
camera over a Tashkent T-junction. `solution.py` implements the official interface; the organizers'
`run_submission.py` and `evaluate.py` are included **unchanged**.

- Live demo: `demo/app.py` (Hugging Face Space, see [Website and demo](#website-and-demo))
- Website: `web/` (static; GitHub Pages)
- Our output on the sample videos: `predictions_samples.json`

## Install and run

```bash
pip install -r requirements.txt          # or: docker build -t team .
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

- **Python 3.10+.** On Linux the default `torch` wheel bundles CUDA and uses the T4 automatically; with
  no GPU the same code runs on the CPU.
- **Weights.** `weights/yolo26s.pt` (20 MB) is committed. If a checkout lacks it, run
  `bash weights/download.sh` once with internet; it downloads the file and checks its SHA-256.
- **Offline.** No network access at run time: `YOLO_OFFLINE=1` is set before Ultralytics is imported, the
  tracker's `lap` dependency is installed up front, and nothing is downloaded. The test-suite runs the
  harness with every HTTP request routed to a dead proxy to prove it.
- **Docker.** `docker build -t team .` then
  `docker run --gpus all -v /data/test:/data/test -v $PWD/out:/out team`.

## Approach

```
.mp4 ─► decode (PyAV, B-frames skipped, 10 fps, 1280×720, producer thread)
      ─► YOLO26-S @1280 (COCO: car bus truck motorcycle bicycle person animals)
      ─► ByteTrack  (riders / drivers folded into their vehicle)
      ─► kinematics (ground point = box bottom-centre; speed in box-heights/s)
      ─► rules on trajectories + hand-measured scene layout  ─► segment shaping ─► events

frames one by one ─► RiskEstimator: YOLO26-S + online ByteTrack at 5 Hz
      ─► cues: footprint-overlap conflicts, hard braking, wrong way, red-light crossing
      ─► logistic per cue, fused, rise-instantly / decay τ = 2 s ─► P(accident within 5 s)
```

**Learned vs rule-based.** The only learned component is the detector: YOLO26-S, COCO-pretrained, used
as released (no fine-tuning). Tracking (ByteTrack), the scene layout and every event rule are
hand-written. With no labels for this camera, rules we can read and test beat a classifier we cannot
validate.

**Scene layout** (`src/trafficev/scene.py`, the equivalent of `camera.md`, which we did not receive).
Measured on a 4K frame and stored in a canonical 1920×1080 space:
- the west leg is a divided road: the near carriageway is **inbound** (towards the camera) and ends at a
  stop line and the west zebra; the far carriageway is **outbound** and has a bus bay;
- the east leg has the east zebra, and the south leg (under the camera) has three islands and a long
  diagonal zebra;
- traffic keeps right; the inbound signal heads face away from the camera.

`python tools/draw_scene.py --frame <frame.jpg> --out layout.jpg` draws it.

**Why the details matter:**
- **Decode is the real cost.** The camera writes H.264 4:2:2 10-bit at about 147 Mbit/s with an IBBP
  GOP, and the harness decodes every frame again for Part B. Skipping non-reference frames yields
  exactly the I/P frames, 10 fps, at 1.3× the speed of a full decode. Decoding runs on a producer
  thread, so it overlaps inference.
- **Perspective-normalised speed.** Dividing image speed by an object's own box height cancels most of
  the scale change across this elevated view, so one "stationary" threshold (0.12 bh/s) works
  everywhere.
- **Signal phase from behaviour.** A vehicle waiting at the inbound stop line for 2 s means red.
  `red_light` requires another vehicle waiting both before and after the crossing.
- **Segment shaping.** A strict core condition must hold for a minimum time (for example, a pedestrian
  at least half a body height inside the carriageway for 1.2 s). The reported segment is the looser run
  around it, which matches the annotation conventions (steps onto / leaves the road). Same-class
  segments are merged, as the task requires.
- **Precision first.** The metric adds every predicted class to the macro average, so a false positive of
  a class that never occurs costs a full class. `illegal_turn` and `fire_smoke` are therefore not
  emitted.

| Class | Rule (see `src/trafficev/events.py`) |
|---|---|
| accident | footprints meet, then both road users stop abruptly (> 2.5 bh/s lost within 1 s) and stay still 3 s, outside the signal queue |
| near_miss | hard braking (< −3 bh/s²) with a road user close ahead and no contact afterwards |
| red_light | crossing the inbound stop line while another vehicle waits at it before and after |
| wrong_way | against the one-way direction for ≥ 1.5 s and ≥ 2.5 box heights |
| illegal_u_turn | heading change ≥ 150° on the carriageway, no identity jump |
| stopped_vehicle | still ≥ 10 s on a road link; not in queue / junction / zebra approach / parking / bus bay; not queued behind another vehicle; fragments at the same spot pooled |
| jaywalking | pedestrian ≥ 0.45 body heights inside the carriageway and ≥ 0.6 body heights from every zebra for ≥ 1.2 s |
| failure_to_yield | vehicle (or ridden two-wheeler, not a pushed bike) drives through a zebra while a pedestrian on that zebra is within 3 vehicle heights |
| solid_line_crossing | ground point crosses the solid line along the median |
| stop_line | still ≥ 3 s past the stop line, before the far edge of the zebra |
| congestion | ≥ 6 vehicles in a one-way carriageway, ≥ 80 % crawling, longer than a signal cycle |
| road_obstacle | animal on the carriageway ≥ 2 s |
| illegal_turn, fire_smoke | not emitted (not reliable from this view) |

**Part B cues** (`src/trafficev/risk.py`): pairs of road users on crossing paths whose ground footprints,
extrapolated at constant velocity, overlap within 1.6 s while apart now. We started with point-distance
time-to-collision, but it fired constantly on this oblique view because cars in adjacent lanes look
close. Other cues: hard braking with someone close ahead, wrong-way motion, and crossing the stop line
while another vehicle waits. The estimator runs its own causal detector and tracker. It never opens the
video and never reuses Part A.

## Models, data and licences

| Item | Source | Licence |
|---|---|---|
| YOLO26-S weights (`weights/yolo26s.pt`) | Ultralytics release v8.4.0, trained on COCO | AGPL-3.0 (weights); COCO annotations CC BY 4.0 |
| Ultralytics 8.4.163 (detector runtime, ByteTrack implementation) | github.com/ultralytics/ultralytics | AGPL-3.0 |
| ByteTrack algorithm | Zhang et al., ECCV 2022 | MIT (original code) |
| PyAV, OpenCV, PyTorch, NumPy | PyPI | BSD / Apache-2.0 / BSD |

- **No training was done.** No external dataset was used beyond the detector's COCO pretraining.
- **How the sample videos were used.** Only to measure the scene layout, for EDA, and for our own checks.
- **Licence.** Because of the Ultralytics dependency this repository is AGPL-3.0 (see `LICENSE`).

## Determinism

- Seeds are fixed (Python, NumPy, torch), cuDNN is deterministic, and `cudnn.benchmark` is off.
- Inference is **fp32 on every device**, so GPU and CPU runs agree up to floating-point noise.
- Two runs on the same machine give the same `predictions.json`.
- The only time-driven behaviour is a safety net for machines that are too slow for the budget:
  - Part A thins frames only when projected to exceed 1.1× the video duration.
  - Part B halves its rate only if the whole video is projected to pass 2.6× the duration, and stops
    perceiving at 2.85×.
  - On the T4 neither triggers.
- `TRAFFICEV_EXACT=1` disables both, which is how `predictions_samples.json` is reproduced on slow
  hardware:
  ```bash
  TRAFFICEV_EXACT=1 python run_submission.py --videos samples --out predictions_samples.json --time-factor 30
  ```

## Time budget

The harness allows 3× the video duration for Part A and Part B together, and Part B's loop decodes every
4K frame with OpenCV. Decoding is the dominant cost, so the budget is planned around it. Measured with
`tools/ablation.py` on our 4-core laptop:

| Stage | Video seconds per wall second | Share of the duration |
|---|---|---|
| Part A decode (PyAV, B-frames skipped, producer thread) | 1.61 | 0.62× |
| Part B decode (harness, `cv2.read` every frame) | 0.75 | 1.33× |

- **On the T4.** YOLO26-S at 1280 px adds about 0.2× in Part A (overlapped with decoding) and about 0.1×
  in Part B (5 Hz), for an expected total of about 2.1× on a machine no faster than our laptop.
- **Outside the timer.** The model is loaded and warmed up while the harness imports `solution.py`.
- **Safety nets.** Part A thins frames only if it is projected to exceed 1.1× the duration. Part B
  reads the clock at which Part A started on the same video (a time, never a Part A result) and
  projects the video's total: past 2.6× it halves its perception rate, past 2.85× it stops perceiving
  and lets the score decay. The harness's hard limit is 3×.
- **CPU-only fallback.** Without a GPU, Part A uses 960 px at 5 fps and Part B uses 1 Hz. On our laptop
  the 12 s 4K excerpt finishes in 33.1 s against a 36.1 s budget. The fallback finds fewer events; the
  reference settings are the GPU ones, reproduced anywhere with `TRAFFICEV_EXACT=1`.

## Repository layout

```
solution.py            official interface -> src/trafficev
run_submission.py      organizers' harness (unchanged)
evaluate.py            organizers' metric (unchanged)
src/trafficev/         video.py detector.py tracking.py kinematics.py scene.py signals.py
                       events.py segments.py pipeline.py risk.py runtime.py labels.py
weights/               yolo26s.pt, download.sh
tools/                 draw_scene.py render.py eda.py dev_eval.py build_site.py
scripts/fetch_samples.py  streams the organizers' Drive files through ffmpeg (quota-aware)
demo/app.py            live demo (Gradio / Hugging Face Space)
web/                   website (static)
tests/                 geometry and rule tests on synthetic trajectories, end-to-end harness test
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest                                        # 17 tests, ~20 s on CPU
python scripts/fetch_samples.py --out data/samples      # sample videos -> 1080p
export TRAFFICEV_CACHE=data/cache                       # perception cache for fast rule iteration
python tools/dev_eval.py --videos data/samples --gt labels/dev_labels.json
python tools/render.py --video data/samples/C3905.mp4 --pred predictions_samples.json --out review.mp4
```

## Website and demo

- **Website.** `python tools/build_site.py --videos samples --pred predictions_samples.json --out web
  --demo-url <space-url> --repo-url <repo-url>` generates the EDA figures, the annotated videos and
  `web/data/site.json`. The team section reads `web/data/team.json`. Serve `web/` with GitHub Pages.
- **Demo.** A Hugging Face Space with the Gradio SDK: copy `demo/app.py` to the Space root, together
  with `src/`, `tools/render.py`, `weights/` and `demo/requirements.txt`. On CPU it runs one causal pass
  at 4 fps with a 960 px detector for both the events and the risk curve.

### Running the live demo on another computer

1. Install Python 3.10–3.12 and git, then clone:
   `git clone https://github.com/Abdulhafiz0512/cv-project && cd cv-project`
2. Start it:
   - **Windows:** double-click `demo\run_demo.bat`.
   - **Linux/macOS:** `bash demo/run_demo.sh`.

   The first run creates `.venv-demo` and installs the dependencies.
3. The console prints a public `https://….gradio.live` link (valid 72 h while the computer stays on) and
   `http://<computer-ip>:7860` for the local network. Put the public link on the website:
   `python tools/set_site_links.py --demo-url https://….gradio.live`, then commit and push. GitHub Pages
   redeploys in about a minute.
4. With an NVIDIA GPU and CUDA drivers the demo uses the submission's settings (1280 px, 10 fps)
   automatically; on a CPU it uses 960 px at 4 fps.
5. For a link that outlives 72 h, deploy to a Space (`python demo/make_space.py --push <user>/junction-watch`
   after `hf auth login`) or run a named Cloudflare tunnel to port 7860.

## Known limitations

- The organizers' Drive links hit Google's download quota while we worked. Our checks used a 12 s 4K
  excerpt of `C3905` and synthetic trajectories, not a labelled dev set, so thresholds are hand-set.
- Accident and near-miss rules have not seen a real positive from this camera.
- `camera.md` was not available to us; the layout was measured from the video.

## Team

| Member | Role | Contributions / responsibilities |
|---|---|---|
| Abdulkhafiz Yoqubjonov ([GitHub](https://github.com/Abdulhafiz0512)) | Computer vision pipeline and website | decode/detect/track pipeline, scene layout and rules, Part B risk, evaluation and EDA tools, demo, website |
| Kamronbek Mamaroziqov | ML engineer: data, evaluation, model verification | responsible for the dev-set labels, per-class evaluation and error analysis, and a vision-language check of rare events (accident, fire) |
| *(teammate 3, pending)* | | |
