# Team tasks — final stretch before submission

Ordered by impact on the elimination score (0.6 model + 0.25 website + 0.15 code).
Tick a box in a commit when it is done; the website's team section lists roles, and
"In this project" should only ever name finished work.

## Abdulkhafiz Yoqubjonov — pipeline, rules, website
- [ ] Publish the live demo: run `demo\run_demo.bat` on the GPU computer, then
      `python tools/set_site_links.py --demo-url <gradio link>`, commit, push (website: 30 % of its score)
- [ ] Run the official command on the GPU computer and record `part_a_sec`, `part_b_sec`,
      `total_sec` / duration from the log (code: "stays inside the time budget with margin")
- [ ] Put every available sample in `data/samples/` and run `scripts/reproduce_samples.sh`
      (predictions_samples.json, annotated videos, timelines, risk curves, EDA)
- [ ] Tune `src/trafficev/events.py` against Kamronbek's labels with `tools/dev_eval.py`;
      keep a class only if its dev F1 justifies predicting it
- [ ] Update the report on the website with the measured numbers and failure cases

## Kamronbek Mamaroziqov — data, evaluation, model verification
- [ ] Dev set: for each sample, render `tools/render.py` output and `tools/review_events.py`
      contact sheets, watch the video and write every true event to `labels/dev_labels.json`
      (format of `examples/ground_truth.json`, with duration and fps). Follow the start/end
      conventions in the task PDF exactly; mark uncertain cases in a separate notes file.
- [ ] Evaluation: `python evaluate.py --pred predictions_samples.json --gt labels/dev_labels.json --per-video`;
      write per-class F1 at IoU 0.3/0.5/0.7 and the main confusions into the website report
- [ ] Error analysis: for each false positive / miss, one line on the cause (detector miss,
      tracking switch, geometry, threshold) so the tuning targets the right stage
- [ ] Rare-event verifier (stretch): prototype a small open VLM (Qwen2.5-VL-3B / Qwen3-VL-2B
      class, fits 5 GB and a T4) that answers yes/no on 4–8 frames around each accident or
      fire_smoke candidate; measure its cost per call against the 3x budget before enabling it

## Teammate 3 — name pending
- [ ] Team section: send name, role, GitHub, LinkedIn, portfolio and previous projects
- [ ] Take one of: EDA write-up for every sample (findings that shaped the solution),
      website polish on phones, or the ablation table (detector A vs B, frame rates)
