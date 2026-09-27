# Junction Watch — WIUT Hackathon 2026, CV track (Jira AP-862)

Traffic-event detection (Part A) + causal accident risk (Part B) for one fixed 4K camera over a
Tashkent T-junction. Task PDF: organizers' "CV Track Elimination Task". README.md has the full
approach; this file is the working brief for continuing the project.

## Rules
- `run_submission.py` and `evaluate.py` are the organizers' files: never edit them.
- Commits: no `Co-Authored-By: Claude` trailers (the owner removed them from history).
- Precision first: the metric adds every *predicted* class to the macro-F1, so a false positive
  of a class that never occurs costs a whole class. Don't enable a rule without evidence.
- Part B (`RiskEstimator`) must stay causal: it may not open the video or reuse Part A results.

## Setup (this machine)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt      # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest -q                                # 19 tests must pass
```

## Status (2026-09-27)
Done: package `src/trafficev`, tests, website live at https://abdulhafiz0512.github.io/cv-project/
(GitHub Pages workflow deploys `web/` on push), portable demo (`demo/run_demo.bat|.sh`).
Blocked earlier: the four Drive samples (C3896, C3897, C3902, C3905) hit Google's download quota;
only a 12 s 4K excerpt of C3905 was processed. Thresholds are hand-set, not tuned on labels.

## Remaining work, in order
1. **Put the sample video(s) in `data/samples/`** (raw camera .MP4 is fine; `data/` is gitignored).
2. **Reference predictions + website data** (runs the unchanged harness in exact mode):
   ```bash
   bash scripts/reproduce_samples.sh data/samples        # needs the venv's python first on PATH
   ```
   PowerShell equivalent:
   ```powershell
   $env:TRAFFICEV_CACHE="data/cache"; $env:TRAFFICEV_EXACT="1"
   .venv\Scripts\python run_submission.py --videos data/samples --out predictions_samples.json --team junction-watch --time-factor 30
   .venv\Scripts\python evaluate.py --pred predictions_samples.json --validate-only
   .venv\Scripts\python tools\build_site.py --videos data/samples --pred predictions_samples.json --out web
   ```
   Then delete the excerpt's stale assets (`web/assets/eda/C3905_first12s`, `web/media/C3905_first12s.mp4`)
   unless you keep that clip in `data/samples`.
3. **Time it on this machine's GPU** (official settings, default time factor 3):
   `python run_submission.py --videos data/samples --out data/pred_timing.json` and report
   `part_a_sec`, `part_b_sec`, `total_sec` / duration from its log.
4. **Build a dev set and tune.** `tools/review_events.py` writes a contact sheet per predicted event;
   `tools/render.py` writes an annotated review video. Watch the video, write true events to
   `labels/dev_labels.json` (same format as `examples/ground_truth.json`, with duration and fps),
   then iterate on `src/trafficev/events.py` with `TRAFFICEV_CACHE=data/cache python tools/dev_eval.py
   --videos data/samples --gt labels/dev_labels.json` (seconds per run once perception is cached).
   Check scene geometry first: `python tools/draw_scene.py --frame <frame.jpg> --out layout.jpg`.
5. **Demo link**: `demo\run_demo.bat` prints a gradio.live URL (72 h); publish it with
   `python tools/set_site_links.py --demo-url <url>`, commit, push. For a durable link:
   `hf auth login` then `python demo/make_space.py --push <user>/junction-watch`.
6. **Team**: fill the two placeholder members and LinkedIn URLs in `web/data/team.json` and README.
7. Commit and push; Pages redeploys in about a minute.

## Useful facts
- Samples: 3840x2160, H.264 High 4:2:2 10-bit, 29.97 fps, ~147 Mbit/s, IBBP GOP of 15.
- Decode dominates the budget: the harness's own `cv2.read` of 4K ran at 0.75x real time on a
  4-core laptop; Part A skips B-frames with PyAV (1.61x real time).
- `TRAFFICEV_EXACT=1` = GPU reference settings on any machine (fp32, 1280 px, 10 fps / 5 Hz).
- `TRAFFICEV_CACHE=<dir>` caches perception per video; never set it in the official run.
